"""Download and verify the external model set used by the Electron package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath


MANIFEST_PATH = Path(__file__).resolve().parent / "desktop_model_manifest.json"


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 1:
        raise ValueError("Unsupported model bundle schema.")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", str(manifest.get("bundleVersion", ""))):
        raise ValueError("Model bundle version is missing.")
    if manifest.get("modelRepository") != "t8star/IndexTTS-2.5-Comfy":
        raise ValueError("Unexpected model repository.")
    revision = str(manifest.get("modelRevision", ""))
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Model revision must be a full lowercase Git commit SHA.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Model bundle file list is empty.")
    calculated_size = 0
    seen_paths: set[str] = set()
    for relative_path, metadata in files.items():
        candidate = PurePosixPath(relative_path)
        if (
            not relative_path
            or "\\" in relative_path
            or candidate.is_absolute()
            or candidate.as_posix() != relative_path
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise ValueError(f"Unsafe model path: {relative_path}")
        normalized = relative_path.lower()
        if normalized in seen_paths:
            raise ValueError(f"Duplicate model path: {relative_path}")
        seen_paths.add(normalized)
        for part in candidate.parts:
            if (
                re.search(r'[\x00-\x1f<>:"|?*]', part)
                or part.endswith((".", " "))
                or part.split(".", 1)[0].upper()
                in {"CON", "PRN", "AUX", "NUL", *{f"COM{i}" for i in range(1, 10)}, *{f"LPT{i}" for i in range(1, 10)}}
            ):
                raise ValueError(f"Windows-unsafe model path: {relative_path}")
        size = int(metadata.get("size", -1))
        checksum = str(metadata.get("sha256", "")).lower()
        if size < 0 or len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum
        ):
            raise ValueError(f"Invalid model file metadata: {relative_path}")
        calculated_size += size
    if int(manifest.get("totalSize", calculated_size)) != calculated_size:
        raise ValueError("Model bundle totalSize does not match its file list.")
    return manifest


def configure_manifest(path: Path) -> None:
    global MODEL_MANIFEST, REPO_ID, MODEL_REVISION
    global MODELSCOPE_REPO_ID, MODELSCOPE_REVISION, MODEL_FILES

    MODEL_MANIFEST = load_manifest(path)
    REPO_ID = MODEL_MANIFEST["modelRepository"]
    MODEL_REVISION = MODEL_MANIFEST["modelRevision"]
    MODELSCOPE_REPO_ID = MODEL_MANIFEST.get("modelScopeRepository", "IndexTeam/IndexTTS-2.5")
    MODELSCOPE_REVISION = MODEL_MANIFEST.get("modelScopeRevision", "master")
    MODEL_FILES = MODEL_MANIFEST["files"]


configure_manifest(MANIFEST_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the external IndexTTS 2.5 model pack")
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="modelscope")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="A locally verified signed model-bundle manifest.",
    )
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


def _file_source(relative_path: str, source: str) -> tuple[str, str]:
    metadata = MODEL_FILES[relative_path]
    if source == "modelscope":
        return (
            str(metadata.get("modelScopeRepository", MODELSCOPE_REPO_ID)),
            str(metadata.get("modelScopeRevision", MODELSCOPE_REVISION)),
        )
    return (
        str(metadata.get("huggingFaceRepository", REPO_ID)),
        str(metadata.get("huggingFaceRevision", MODEL_REVISION)),
    )


def download_huggingface(
    target: Path, missing_files: list[str], mismatched_files: list[str]
) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        revision=MODEL_REVISION,
        local_dir=str(target),
        allow_patterns=list(MODEL_FILES),
    )
    for relative_path in [*missing_files, *mismatched_files]:
        repository, revision = _file_source(relative_path, "huggingface")
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

    downloaded = Path(
        snapshot_download(model_id=MODELSCOPE_REPO_ID, revision=MODELSCOPE_REVISION)
    ).resolve()
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
        repository, revision = _file_source(relative_path, "modelscope")
        if repository == MODELSCOPE_REPO_ID:
            continue
        print(f">> Fetching supplemental {relative_path} from {repository}", flush=True)
        source = Path(
            model_file_download(model_id=repository, file_path=relative_path, revision=revision)
        ).resolve()
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    configure_manifest(args.manifest.expanduser().resolve())
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
            download_modelscope(
                target,
                [
                    relative_path
                    for relative_path in [*missing, *mismatched]
                    if MODEL_FILES[relative_path].get("group") != "auxiliary"
                ],
            )
    else:
        print(">> Main IndexTTS 2.5 files match the official release; checking auxiliaries.", flush=True)

    from indextts.utils.model_download import ensure_models_available

    ensure_models_available(str(target), include_legacy_semantic_codec=False)
    missing, mismatched = inspect_model_files(target, verify_hashes=True)
    if missing or mismatched:
        details = [*(f"missing:{item}" for item in missing), *(f"invalid:{item}" for item in mismatched)]
        raise RuntimeError("Official IndexTTS 2.5 model verification failed: " + ", ".join(details))

    print(
        f">> IndexTTS 2.5 model bundle {MODEL_MANIFEST['bundleVersion']} "
        f"({MODEL_REVISION}) is ready.",
        flush=True,
    )


if __name__ == "__main__":
    main()
