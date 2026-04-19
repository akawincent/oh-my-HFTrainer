#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload a local folder to your Hugging Face Hub repository.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "local_dir",
        help="Local folder to upload. Example: outputs/Qwen3-0.6B-finetuned-astro-horoscope-fsdp",
    )
    parser.add_argument(
        "--repo-id",
        help="Target repo id. Defaults to <current-user>/<folder-name>. If you pass only a repo name, it is created under your current user namespace.",
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default="model",
        help="Target repository type.",
    )
    visibility_group = parser.add_mutually_exclusive_group()
    visibility_group.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not already exist.",
    )
    visibility_group.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public if it does not already exist.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Optional subdirectory inside the Hub repo. Defaults to the repo root.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch, tag, or revision to upload to.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message for the upload.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. Defaults to HF_TOKEN and then any locally cached login.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Custom Hub endpoint. Defaults to HF_ENDPOINT when set.",
    )
    parser.add_argument(
        "--large-folder",
        action="store_true",
        help="Use upload_large_folder for resumable uploads.",
    )
    return parser


def resolve_local_dir(raw_path: str) -> Path:
    local_dir = Path(raw_path).expanduser().resolve()
    if not local_dir.exists():
        raise SystemExit(f"Local directory does not exist: {local_dir}")
    if not local_dir.is_dir():
        raise SystemExit(f"Local path is not a directory: {local_dir}")
    return local_dir


def extract_username(whoami_result: dict[str, Any]) -> str:
    username = whoami_result.get("name")
    if not username:
        raise SystemExit("Unable to determine your Hugging Face username from the current login session.")
    return username


def resolve_repo_id(repo_id: str | None, username: str, local_dir: Path) -> str:
    if not repo_id:
        return f"{username}/{local_dir.name}"
    if "/" in repo_id:
        return repo_id
    return f"{username}/{repo_id}"


def resolve_visibility(args: argparse.Namespace) -> bool | None:
    if args.private:
        return True
    if args.public:
        return False
    return None


def build_repo_url(endpoint: str, repo_id: str, repo_type: str) -> str:
    base = endpoint.rstrip("/")
    if repo_type == "dataset":
        return f"{base}/datasets/{repo_id}"
    if repo_type == "space":
        return f"{base}/spaces/{repo_id}"
    return f"{base}/{repo_id}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.large_folder and args.path_in_repo:
        raise SystemExit("--large-folder does not support --path-in-repo. Upload to the repo root or use upload_folder instead.")

    local_dir = resolve_local_dir(args.local_dir)
    token = args.token or os.getenv("HF_TOKEN")
    endpoint = args.endpoint or os.getenv("HF_ENDPOINT")
    api = HfApi(endpoint=endpoint, token=token)

    try:
        whoami_result = api.whoami(token=token, cache=True)
    except HfHubHTTPError as error:
        raise SystemExit(
            "Hugging Face authentication failed. Run `hf auth login` first or pass `--token`/`HF_TOKEN`."
        ) from error

    username = extract_username(whoami_result)
    repo_id = resolve_repo_id(args.repo_id, username, local_dir)
    visibility = resolve_visibility(args)
    commit_message = args.commit_message or f"Upload {local_dir.name}"

    print(f"Authenticated as: {username}")
    print(f"Uploading from: {local_dir}")
    print(f"Target repo: {repo_id} ({args.repo_type})")
    print(f"Endpoint: {api.endpoint}")

    create_repo_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": args.repo_type,
        "exist_ok": True,
        "token": token,
    }
    if visibility is not None:
        create_repo_kwargs["private"] = visibility

    try:
        api.create_repo(**create_repo_kwargs)
        if args.large_folder:
            api.upload_large_folder(
                repo_id=repo_id,
                folder_path=str(local_dir),
                repo_type=args.repo_type,
                revision=args.revision,
                private=visibility,
            )
        else:
            upload_kwargs: dict[str, Any] = {
                "repo_id": repo_id,
                "folder_path": str(local_dir),
                "repo_type": args.repo_type,
                "revision": args.revision,
                "commit_message": commit_message,
                "token": token,
            }
            if args.path_in_repo:
                upload_kwargs["path_in_repo"] = args.path_in_repo
            api.upload_folder(**upload_kwargs)
    except HfHubHTTPError as error:
        raise SystemExit(f"Upload failed: {error}") from error

    repo_url = build_repo_url(api.endpoint, repo_id, args.repo_type)
    print("Upload complete.")
    print(f"Repo URL: {repo_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
