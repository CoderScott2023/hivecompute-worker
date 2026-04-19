"""
Worker daemon — the main process that runs on contributor machines.

Loop:
  1. Register with coordinator (once, persists JWT to disk)
  2. Poll until machine is idle
  3. Request a job assignment
  4. Download checkpoint (if any)
  5. Stream dataset shard and run DiLoCo inner loop
  6. Upload compressed pseudo-gradients
  7. Receive updated checkpoint URL → repeat from step 2

Interruption handling:
  - If the machine wakes up mid-training, the trainer is asked to stop
    gracefully, and whatever gradients were accumulated are NOT uploaded
    (partial work is discarded — coordinator will reassign to another worker)
"""
from __future__ import annotations
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

from shared.protocol import (
    HeartbeatRequest, RegisterRequest, WorkerStatus,
)
from worker.gpu_detect import detect as detect_hardware
from worker.idle_detect import is_idle, wait_until_idle
from worker.trainer import LocalTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [worker] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://127.0.0.1:8000")
JOB_ID = int(os.environ.get("JOB_ID", "1"))
WORKER_STATE_FILE = Path(os.environ.get("WORKER_STATE_FILE", "./worker_state.json"))
HEARTBEAT_INTERVAL = 30        # seconds
IDLE_POLL_INTERVAL = 5         # seconds between idle checks


# ── State persistence ─────────────────────────────────────────────────────────

def _load_state() -> dict:
    if WORKER_STATE_FILE.exists():
        return json.loads(WORKER_STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    WORKER_STATE_FILE.write_text(json.dumps(state, indent=2))


# ── API helpers ───────────────────────────────────────────────────────────────

class CoordinatorClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def heartbeat(self, worker_id: str, status: WorkerStatus, job_id: Optional[int] = None, step: Optional[int] = None):
        try:
            r = self.session.post(
                f"{self.base_url}/workers/heartbeat",
                json=HeartbeatRequest(
                    worker_id=worker_id,
                    status=status,
                    current_job_id=job_id,
                    local_step=step,
                ).model_dump(),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            return r.json()
        except Exception as e:
            logger.warning("Heartbeat failed: %s", e)
            return {"ok": False, "should_stop": False}

    def get_assignment(self, job_id: int, worker_id: str) -> Optional[dict]:
        try:
            r = self.session.get(
                f"{self.base_url}/jobs/{job_id}/assign",
                params={"worker_id": worker_id},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()
            logger.warning("Assignment failed (%d): %s", r.status_code, r.text)
        except Exception as e:
            logger.error("Error getting assignment: %s", e)
        return None

    def download_checkpoint(self, job_id: int, dest_path: str) -> bool:
        try:
            r = self.session.get(
                f"{self.base_url}/jobs/{job_id}/checkpoint",
                timeout=120, stream=True,
            )
            if r.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            if r.status_code == 404:
                return False   # no checkpoint yet — start from scratch
        except Exception as e:
            logger.error("Checkpoint download failed: %s", e)
        return False

    def upload_gradients(
        self,
        job_id: int,
        worker_id: str,
        round: int,
        shard_id: int,
        local_steps: int,
        final_loss: float,
        compressed_gradients: bytes,
    ) -> Optional[dict]:
        try:
            r = self.session.post(
                f"{self.base_url}/jobs/{job_id}/sync",
                params={
                    "worker_id": worker_id,
                    "round": round,
                    "shard_id": shard_id,
                    "local_steps": local_steps,
                    "final_loss": final_loss,
                },
                files={"gradient_file": ("gradients.bin", io.BytesIO(compressed_gradients))},
                headers={"Authorization": self.session.headers["Authorization"]},
                timeout=120,
            )
            if r.status_code == 200:
                return r.json()
            logger.error("Gradient upload failed (%d): %s", r.status_code, r.text)
        except Exception as e:
            logger.error("Gradient upload error: %s", e)
        return None


# ── Heartbeat thread ──────────────────────────────────────────────────────────

class HeartbeatThread(threading.Thread):
    def __init__(self, client: CoordinatorClient, worker_id: str):
        super().__init__(daemon=True)
        self.client = client
        self.worker_id = worker_id
        self.status = WorkerStatus.IDLE
        self.current_job_id: Optional[int] = None
        self.current_step: Optional[int] = None
        self._stop = threading.Event()
        self.should_stop_training = False

    def run(self):
        while not self._stop.is_set():
            resp = self.client.heartbeat(
                self.worker_id, self.status, self.current_job_id, self.current_step
            )
            if resp.get("should_stop"):
                self.should_stop_training = True
            self._stop.wait(HEARTBEAT_INTERVAL)

    def stop(self):
        self._stop.set()


# ── Main daemon ───────────────────────────────────────────────────────────────

def run():
    hw = detect_hardware()
    logger.info("Hardware: %s (%s) vram=%.1fGB", hw.device_name, hw.device_type, hw.vram_gb)

    state = _load_state()

    # Installer injects these env vars — use them if present, otherwise fall back to state file
    worker_id = os.environ.get("WORKER_ID_OVERRIDE") or state.get("worker_id") or str(uuid.uuid4())
    existing_token = os.environ.get("WORKER_TOKEN_OVERRIDE") or state.get("token")

    # Register (always re-register to update hardware info)
    reg_payload = RegisterRequest(
        worker_id=worker_id,
        device_type=hw.device_type,
        device_name=hw.device_name,
        vram_gb=hw.vram_gb,
        ram_gb=hw.ram_gb,
        num_cpus=hw.num_cpus,
    )
    try:
        r = requests.post(
            f"{COORDINATOR_URL}/workers/register",
            json=reg_payload.model_dump(),
            timeout=15,
        )
        r.raise_for_status()
        token = r.json()["token"]
    except Exception as e:
        # Fall back to existing token if registration fails
        if existing_token:
            logger.warning("Registration failed, using existing token: %s", e)
            token = existing_token
        else:
            logger.error("Registration failed: %s", e)
            sys.exit(1)

    state["worker_id"] = worker_id
    state["token"] = token
    _save_state(state)

    client = CoordinatorClient(COORDINATOR_URL, token)
    hb = HeartbeatThread(client, worker_id)
    hb.start()
    logger.info("Registered as worker %s", worker_id)

    try:
        _main_loop(client, hb, worker_id, hw)
    finally:
        hb.stop()


def _main_loop(client: CoordinatorClient, hb: HeartbeatThread, worker_id: str, hw):
    while True:
        hb.status = WorkerStatus.IDLE
        hb.should_stop_training = False

        logger.info("Waiting for machine to be idle...")
        wait_until_idle(poll_interval=IDLE_POLL_INTERVAL)
        logger.info("Machine is idle — requesting job assignment")

        assignment = client.get_assignment(JOB_ID, worker_id)
        if assignment is None:
            logger.info("No assignment available. Retrying in 60s...")
            time.sleep(60)
            continue

        job_id = assignment["job_id"]
        round_num = assignment["round"]
        shard_id = assignment["shard_id"]
        total_shards = assignment["total_shards"]
        config_dict = assignment["config"]

        from shared.protocol import TrainingConfig
        config = TrainingConfig(**config_dict)

        logger.info(
            "Got assignment: job=%d round=%d shard=%d/%d",
            job_id, round_num, shard_id, total_shards,
        )

        # Skip checkpoint download for now — coordinator stores numpy aggregates
        # which aren't directly loadable as PEFT adapters. Workers always start
        # from a fresh LoRA adapter on the base model each round.
        ckpt_path = None

        # Download dataset shard
        dataset_path = None
        try:
            r = requests.get(
                f"{COORDINATOR_URL}/jobs/{job_id}/dataset",
                headers={"Authorization": f"Bearer {client.session.headers['Authorization'].split(' ')[1]}"},
                timeout=120,
                stream=True,
            )
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="wb") as f:
                dataset_path = f.name
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            logger.error("Dataset download failed: %s", e)
            time.sleep(30)
            continue

        # Train
        trainer = LocalTrainer(config, hw)
        hb.status = WorkerStatus.TRAINING
        hb.current_job_id = job_id

        # Wire up stop signal from heartbeat thread
        def _watch_stop():
            while hb.status == WorkerStatus.TRAINING:
                if not is_idle():
                    logger.info("Machine no longer idle — stopping training")
                    trainer.request_stop()
                    return
                if hb.should_stop_training:
                    logger.info("Coordinator requested stop")
                    trainer.request_stop()
                    return
                time.sleep(5)

        stop_watcher = threading.Thread(target=_watch_stop, daemon=True)
        stop_watcher.start()

        result = trainer.run(dataset_path, adapter_path=ckpt_path, shard_id=shard_id, total_shards=total_shards)

        if ckpt_path:
            Path(ckpt_path).unlink(missing_ok=True)
        if dataset_path:
            Path(dataset_path).unlink(missing_ok=True)

        if result.interrupted:
            logger.info("Training interrupted — discarding partial gradients")
            hb.status = WorkerStatus.IDLE
            time.sleep(30)
            continue

        # Upload gradients
        hb.status = WorkerStatus.SYNCING
        logger.info(
            "Uploading gradients (%.1fKB) for round %d...",
            len(result.compressed_gradients) / 1024,
            round_num,
        )

        sync_resp = client.upload_gradients(
            job_id=job_id,
            worker_id=worker_id,
            round=round_num,
            shard_id=shard_id,
            local_steps=result.steps_completed,
            final_loss=result.final_loss,
            compressed_gradients=result.compressed_gradients,
        )

        if sync_resp:
            logger.info("Sync response: %s", sync_resp.get("message", "ok"))
        else:
            logger.warning("Gradient upload failed — will retry next idle period")

        hb.status = WorkerStatus.IDLE
        time.sleep(5)


if __name__ == "__main__":
    run()
