import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_PATH = Path(__file__).parent
REMOTES = {
    "gitlab": "origin",
    "github": "github",
}
BRANCH = "main"


def run(cmd, check=True):
    print(f">>> {cmd}")
    result = subprocess.run(
        cmd, cwd=REPO_PATH, shell=True,
        capture_output=True, text=True
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if check and result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
    return result


def has_changes():
    status = run("git status --porcelain", check=False)
    return bool(status.stdout.strip())


def main():
    print(f"=== Git Sync started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    if not has_changes():
        print("No changes to commit. Nothing to sync.")
        return

    run("git add .")

    commit_msg = f"Auto-sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    commit_result = run(f'git commit -m "{commit_msg}"', check=False)

    if commit_result.returncode != 0 and "nothing to commit" in commit_result.stdout.lower():
        print("Nothing to commit. Exiting.")
        return

    print("\n--- Pushing to remotes ---")
    failed = []
    for name, remote in REMOTES.items():
        print(f"\n[{name}]")
        result = run(f"git push {remote} {BRANCH}", check=False)
        if result.returncode != 0:
            failed.append(name)

    print("\n=== Summary ===")
    if failed:
        print(f"Failed to push: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All remotes synced successfully!")


if __name__ == "__main__":
    main()
