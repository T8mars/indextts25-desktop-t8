"""Download and verify the external model set used by the Electron package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MANIFEST_PATH = Path(__file__).resolve().parent / "desktop_model_manifest.json"
MODEL_MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
REPO_ID = MODEL_MANIFEST["modelRepository"]
MODEL_REVISION = MODEL_MANIFEST["modelRevision"]
MODEL_FILES = MODEL_MANIFEST["files"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the external IndexTTS 2.5 model pack")
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="modelscope")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_model_files(target: Path, verify_hashes: bool = False) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    mismatched: list[str] = []
    for relative_path, metadata in MODEL_FILES.items():
        local_path = target / relative_path
        if not local_path.is_file():
            missing.append(relative_path)
            continue
        if local_path.stat().st_size != metadata["size"]:
            mismatched.append(relative_path)
            continue
        if verify_hashes and sha256_file(local_path) != metadata["sha256"]:
            mismatched.append(relative_path)
    return missing, mismatched


def _file_source(relative_path: str) -> tuple[str, str]:
    metadata = MODEL_FILES[relative_path]
    return (
        str(metadata.get("repository", REPO_ID)),
        str(metadata.get("revision", MODEL_REVISION)),
    )


def download_huggingface(
    target: Path, missing_files: list[str], mismatched_files: list[str]
) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(repo_id=REPO_ID, revision=MODEL_REVISION, local_dir=str(target))
    for relative_path in [*missing_files, *mismatched_files]:
        repository, revision = _file_source(relative_path)
        if repository == REPO_ID and relative_path in missing_files:
            continue
        action = "Refreshing" if relative_path in mismatched_files else "Fetching supplemental"
        print(f">> {action} {relative_path} from {repository}", flush=True)
        hf_hub_download(
            repo_id=repository,
            revision=revision,
            filename=relative_path,
            local_dir=str(target),
            force_download=relative_path in mismatched_files,
        )


def download_modelscope(target: Path, required_files: list[str]) -> None:
    from modelscope.hub.file_download import model_file_download
    from modelscope.hub.snapshot_download import snapshot_download

    downloaded = Path(snapshot_download(model_id=REPO_ID, revision=MODEL_REVISION)).resolve()
    target.mkdir(parents=True, exist_ok=True)
    for source in downloaded.rglob("*"):
        relative_path = source.relative_to(downloaded)
        destination = target / relative_path
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for relative_path in required_files:
        repository, _revision = _file_source(relative_path)
        if repository == REPO_ID:
            continue
        print(f">> Fetching supplemental {relative_path} from {repository}", flush=True)
        source = Path(
            model_file_download(model_id=repository, file_path=relative_path)
        ).resolve()
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    missing, mismatched = inspect_model_files(target)
    if missing or mismatched:
        print(
            f">> Synchronizing {REPO_ID} ({MODEL_REVISION}) from {args.source} to {target}",
            flush=True,
        )
        if missing:
            print(">> Missing files: " + ", ".join(missing), flush=True)
        if mismatched:
            print(">> Outdated files: " + ", ".join(mismatched), flush=True)
        if args.source == "huggingface":
            download_huggingface(target, missing, mismatched)
        else:
            download_modelscope(target, [*missing, *mismatched])
    else:
        print(">> Main IndexTTS 2.5 files match the official release; checking auxiliaries.", flush=True)

    missing, mismatched = inspect_model_files(target, verify_hashes=True)
    if missing or mismatched:
        details = [*(f"missing:{item}" for item in missing), *(f"invalid:{item}" for item in mismatched)]
        raise RuntimeError("Official IndexTTS 2.5 model verification failed: " + ", ".join(details))

    from indextts.utils.model_download import ensure_models_available

    ensure_models_available(str(target))
    print(f">> IndexTTS 2.5 model revision {MODEL_REVISION} is ready.", flush=True)


if __name__ == "__main__":
    main()
