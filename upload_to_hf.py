"""Upload data/ (dump + funcodec) to the Hugging Face dataset repo AlexWu/tseomni.

This uploads the test-set scp lists (data/dump) and FunCodec codec features
(data/funcodec) as a backup. Note: the scp files contain absolute paths that
are machine-specific; they are uploaded as-is for backup purposes.

Usage:
    # Option 1: provide a token via env var
    HF_TOKEN=<your_token> python upload_to_hf.py

    # Option 2: login first, then run
    huggingface-cli login
    python upload_to_hf.py
"""
import os

from huggingface_hub import HfApi, login

REPO_ID = "AlexWu/tseomni"
REPO_TYPE = "dataset"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Sub-directories of data/ to upload (scp lists + codec features)
SUBDIRS = ["dump", "funcodec"]


def main():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi()

    # Ensure the repo exists (create it if missing)
    try:
        api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f"Repo {REPO_ID} exists.")
    except Exception:
        print(f"Repo {REPO_ID} not found, creating it ...")
        api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)

    print(f"Uploading to {REPO_ID} (type={REPO_TYPE}) ...")

    for sub in SUBDIRS:
        local_dir = os.path.join(DATA_DIR, sub)
        if not os.path.isdir(local_dir):
            print(f"Skip missing dir: {local_dir}")
            continue
        print(f"Uploading {local_dir} -> {REPO_ID}/{sub}")
        api.upload_folder(
            repo_id=REPO_ID,
            folder_path=local_dir,
            path_in_repo=sub,
            repo_type=REPO_TYPE,
        )

    print("Upload done.")


if __name__ == "__main__":
    main()
