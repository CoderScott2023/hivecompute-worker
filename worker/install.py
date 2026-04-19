"""
HiveCompute Worker Installer
----------------------------
Double-click this file (or run: python install_hivecompute.py)

What it does:
  1. Clones the HiveCompute worker code from GitHub
  2. Installs required Python packages
  3. Saves your credentials
  4. Registers a Windows startup task so the worker runs automatically
  5. Starts the worker immediately

To uninstall: run this script again and choose 'Remove'.
"""
import os
import sys
import subprocess
import json
import pathlib
import platform
import shutil

# ── Injected by coordinator at download time ──────────────────────────────────
COORDINATOR_URL = "REPLACE_ME"
WORKER_ID = "REPLACE_ME"
WORKER_TOKEN = "REPLACE_ME"
GITHUB_REPO = "https://github.com/CoderScott2023/hivecompute-worker.git"
# ─────────────────────────────────────────────────────────────────────────────

INSTALL_DIR = pathlib.Path.home() / ".hivecompute"
REPO_DIR = INSTALL_DIR / "repo"
CONFIG_FILE = INSTALL_DIR / "config.json"
WORKER_SCRIPT = INSTALL_DIR / "run_worker.py"
TASK_NAME = "HiveComputeWorker"

REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "requests",
    "psutil",
    "numpy",
    "pyjwt",
]

RUNNER_SCRIPT = '''
import sys, os, pathlib, json, subprocess

config_path = pathlib.Path.home() / ".hivecompute" / "config.json"
cfg = json.loads(config_path.read_text())

repo_dir = pathlib.Path(cfg["repo_dir"])
if not repo_dir.exists():
    raise RuntimeError(f"Worker code not found at {repo_dir}. Re-run the installer.")

env = os.environ.copy()
env["COORDINATOR_URL"] = cfg["coordinator_url"]
env["WORKER_ID_OVERRIDE"] = cfg["worker_id"]
env["WORKER_TOKEN_OVERRIDE"] = cfg["token"]
env["IDLE_THRESHOLD_SECS"] = "120"
env["JOB_ID"] = cfg.get("job_id", "1")

subprocess.run(
    [sys.executable, "-m", "worker.main"],
    cwd=str(repo_dir),
    env=env,
)
'''


def print_header():
    print("=" * 60)
    print("  HiveCompute Worker Installer")
    print("=" * 60)
    print()


def check_python():
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9 or higher is required.")
        print(f"  Your version: {sys.version}")
        input("Press Enter to exit...")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]}")


def check_git():
    if shutil.which("git") is None:
        print("ERROR: Git is not installed.")
        print("  Download it from: https://git-scm.com/download/win")
        print("  Then re-run this installer.")
        input("Press Enter to exit...")
        sys.exit(1)
    print("✓ Git found")


def clone_or_update_repo():
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    if REPO_DIR.exists():
        print("Updating worker code...")
        result = subprocess.run(
            ["git", "pull"],
            cwd=str(REPO_DIR),
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("✓ Worker code updated")
        else:
            print("WARNING: Could not update — using existing code")
    else:
        print("Downloading worker code...")
        result = subprocess.run(
            ["git", "clone", "--depth=1", GITHUB_REPO, str(REPO_DIR)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"ERROR: Could not download worker code: {result.stderr}")
            input("Press Enter to exit...")
            sys.exit(1)
        print("✓ Worker code downloaded")


def install_packages():
    print("\nInstalling required packages (this may take a few minutes)...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + REQUIRED_PACKAGES,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print("✓ Packages installed")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Some packages failed to install: {e}")


def save_config():
    config = {
        "coordinator_url": COORDINATOR_URL,
        "worker_id": WORKER_ID,
        "token": WORKER_TOKEN,
        "repo_dir": str(REPO_DIR),
        "job_id": os.environ.get("JOB_ID", "1"),
    }
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    WORKER_SCRIPT.write_text(RUNNER_SCRIPT)
    print(f"✓ Config saved")


def register_task_windows():
    python_exe = sys.executable
    script = str(WORKER_SCRIPT)

    subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"], capture_output=True)

    result = subprocess.run(
        ["schtasks", "/Create",
         "/TN", TASK_NAME,
         "/TR", f'"{python_exe}" "{script}"',
         "/SC", "ONLOGON",
         "/RL", "LIMITED",
         "/F"],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        print(f"✓ Worker will start automatically at every login")
    else:
        print(f"WARNING: Could not register startup task: {result.stderr.strip()}")
        print(f"  Start manually with: python \"{script}\"")


def remove_task_windows():
    subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"], capture_output=True)
    print("✓ Startup task removed")
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
        print("✓ Config removed")
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
        print("✓ Worker code removed")


def start_worker_now():
    print("\nStarting worker...")
    kwargs = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen(
        [sys.executable, str(WORKER_SCRIPT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )
    print("✓ Worker is running in the background.")


def main():
    print_header()

    already_installed = CONFIG_FILE.exists()
    if already_installed:
        print("HiveCompute is already installed.\n")
        print("  [1] Reinstall / update")
        print("  [2] Remove")
        print("  [3] Cancel")
        choice = input("\nChoice: ").strip()
        if choice == "2":
            if platform.system() == "Windows":
                remove_task_windows()
            print("\nHiveCompute has been removed.")
            input("Press Enter to exit...")
            return
        elif choice == "3":
            return

    check_python()
    check_git()
    clone_or_update_repo()
    install_packages()
    save_config()

    if platform.system() == "Windows":
        register_task_windows()

    start_worker_now()

    print()
    print("=" * 60)
    print("  All done!")
    print()
    print("  HiveCompute is running in the background.")
    print("  It only trains when your PC has been idle for 2 minutes.")
    print()
    print(f"  Track your earnings: {COORDINATOR_URL}")
    print("=" * 60)
    print()
    input("Press Enter to close...")


if __name__ == "__main__":
    main()
