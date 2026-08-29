"""Download and verify the external model set used by the Electron package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Callable


MANIFEST_PATH = Path(__file__).resolve().parent / "desktop_model_manifest.json"
PROGRESS_PREFIX = "@@T8_MODEL_PROGRESS@@"
MINIMUM_FREE_BYTES = 512 * 1024 * 1024
DISK_RESERVE_BYTES = 1024 * 1024 * 1024


def emit_progress(event: dict) -> None:
    """Emit one machine-readable event without hiding ordinary diagnostic logs."""

    print(
        PROGRESS_PREFIX + json.dumps(event, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


class BundleProgress:
    """Track scan, transfer, and verification progress for one model bundle."""

    def __init__(
        self,
        files: dict,
        *,
        source: str,
        callback: Callable[[dict], None] = emit_progress,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.files = files
        self.source = source
        self.callback = callback
        self.clock = clock
        self.bundle_total = sum(int(item["size"]) for item in files.values())
        self.required_files: list[str] = []
        self.required_total = 0
        self.completed_required = 0
        self.current_file = ""
        self.current_index = 0
        self.current_received = 0
        self.current_total = 0
        self.session_network_bytes = 0
        self.started_at = self.clock()
        self.last_emit_at = 0.0

    def _send(self, phase: str, *, force: bool = False, **values) -> None:
        now = self.clock()
        if not force and now - self.last_emit_at < 0.2:
            return
        self.last_emit_at = now
        self.callback(
            {
                "type": "model_download_progress",
                "phase": phase,
                "source": self.source,
                "bundleTotal": self.bundle_total,
                **values,
            }
        )

    def scan(self, relative_path: str, processed: int, total: int) -> None:
        fraction = processed / max(1, total)
        self._send(
            "scanning",
            file=relative_path,
            received=processed,
            total=total,
            phasePercent=round(fraction * 100, 1),
            overallPercent=round(fraction * 10, 1),
            message=f"正在校验现有模型：{relative_path}",
        )

    def preflight(self, required_files: list[str], free_bytes: int) -> None:
        self.required_files = list(required_files)
        self.required_total = sum(int(self.files[item]["size"]) for item in required_files)
        disk_sufficient = free_bytes >= self.required_total + DISK_RESERVE_BYTES
        warning = ""
        if not disk_sufficient and self.required_total:
            warning = (
                "可用空间低于完整缺失文件的保守估算；已有断点缓存可能降低实际需求，"
                "下载库仍会在写入前执行精确检查。"
            )
        self._send(
            "preflight",
            force=True,
            requiredBytes=self.required_total,
            availableBytes=free_bytes,
            reserveBytes=DISK_RESERVE_BYTES,
            diskSufficient=disk_sufficient,
            warning=warning,
            fileCount=len(required_files),
            overallPercent=10.0,
            phasePercent=100.0,
            message=(
                f"磁盘预检完成，需要下载或修复 {len(required_files)} 个文件。"
                if required_files
                else "现有模型文件完整，无需重新下载。"
            ),
        )

    def begin_file(self, relative_path: str, index: int) -> None:
        self.current_file = relative_path
        self.current_index = index
        self.current_received = 0
        self.current_total = int(self.files[relative_path]["size"])
        self._emit_transfer(force=True)

    def resume_file(self, received: int) -> None:
        self.current_received = min(self.current_total, max(self.current_received, int(received)))
        self._emit_transfer(force=True)

    def update_file(self, received: int) -> None:
        next_received = min(self.current_total, max(self.current_received, int(received)))
        self.session_network_bytes += max(0, next_received - self.current_received)
        self.current_received = next_received
        self._emit_transfer()

    def complete_file(self) -> None:
        self.current_received = self.current_total
        self._emit_transfer(force=True)
        self.completed_required += self.current_total
        self.current_received = 0
        self.current_total = 0

    def _emit_transfer(self, *, force: bool = False) -> None:
        transferred = min(
            self.required_total,
            self.completed_required + self.current_received,
        )
        fraction = transferred / max(1, self.required_total)
        elapsed = max(0.001, self.clock() - self.started_at)
        speed = self.session_network_bytes / elapsed
        remaining = max(0, self.required_total - transferred)
        eta = remaining / speed if speed > 0 else None
        self._send(
            "downloading",
            force=force,
            file=self.current_file,
            fileIndex=self.current_index,
            fileCount=len(self.required_files),
            fileReceived=self.current_received,
            fileTotal=self.current_total,
            received=transferred,
            total=self.required_total,
            bundleReceived=self.bundle_total - self.required_total + transferred,
            bytesPerSecond=round(speed),
            etaSeconds=round(eta) if eta is not None else None,
            phasePercent=round(fraction * 100, 1),
            overallPercent=round(10 + fraction * 80, 1),
            message=f"正在下载 {self.current_file}",
        )

    def verify(self, relative_path: str, processed: int, total: int) -> None:
        fraction = processed / max(1, total)
        self._send(
            "verifying",
            file=relative_path,
            received=processed,
            total=total,
            phasePercent=round(fraction * 100, 1),
            overallPercent=round(90 + fraction * 10, 1),
            message=f"正在执行 SHA-256 校验：{relative_path}",
        )

    def done(self) -> None:
        self._send(
            "complete",
            force=True,
            received=self.bundle_total,
            total=self.bundle_total,
            phasePercent=100.0,
            overallPercent=100.0,
            message="完整模型已下载并通过 SHA-256 校验。",
        )


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


def sha256_file(path: Path, on_chunk: Callable[[int], None] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
            if on_chunk is not None:
                on_chunk(len(chunk))
    return digest.hexdigest()


def inspect_model_files(
    target: Path,
    verify_hashes: bool = False,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    mismatched: list[str] = []
    total = sum(int(metadata["size"]) for metadata in MODEL_FILES.values())
    processed = 0
    for relative_path, metadata in MODEL_FILES.items():
        local_path = target / relative_path
        if not local_path.is_file():
            missing.append(relative_path)
            processed += int(metadata["size"])
            if on_progress is not None:
                on_progress(relative_path, processed, total)
            continue
        if local_path.stat().st_size != metadata["size"]:
            mismatched.append(relative_path)
            processed += int(metadata["size"])
            if on_progress is not None:
                on_progress(relative_path, processed, total)
            continue
        if verify_hashes:
            hashed = 0

            def update_hash(size: int) -> None:
                nonlocal hashed
                hashed += size
                if on_progress is not None:
                    on_progress(relative_path, processed + hashed, total)

            if sha256_file(local_path, update_hash) != metadata["sha256"]:
                mismatched.append(relative_path)
            processed += int(metadata["size"])
        else:
            processed += int(metadata["size"])
        if on_progress is not None:
            on_progress(relative_path, processed, total)
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
    target: Path,
    missing_files: list[str],
    mismatched_files: list[str],
    reporter: BundleProgress | None = None,
) -> None:
    from huggingface_hub import hf_hub_download

    if reporter is not None:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.file_download import get_local_download_paths

    requested = list(dict.fromkeys([*missing_files, *mismatched_files]))
    for index, relative_path in enumerate(requested, start=1):
        if reporter is not None:
            reporter.begin_file(relative_path, index)
        repository, revision = _file_source(relative_path, "huggingface")
        force_download = relative_path in mismatched_files
        action = "Refreshing" if force_download else "Fetching"
        print(f">> {action} {relative_path} from {repository}", flush=True)

        incomplete_path = None
        if reporter is not None:
            try:
                metadata = get_hf_file_metadata(
                    hf_hub_url(repository, relative_path, revision=revision)
                )
                if metadata.etag:
                    incomplete_path = get_local_download_paths(
                        target, relative_path
                    ).incomplete_path(metadata.etag)
            except Exception as exc:
                print(
                    f">> Progress probe unavailable for {relative_path}: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        started_wall = time.time()
        if (
            reporter is not None
            and not force_download
            and incomplete_path is not None
            and incomplete_path.is_file()
        ):
            reporter.resume_file(incomplete_path.stat().st_size)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                hf_hub_download,
                repo_id=repository,
                revision=revision,
                filename=relative_path,
                local_dir=str(target),
                force_download=force_download,
            )
            while not future.done():
                if reporter is not None and incomplete_path is not None:
                    try:
                        stat = incomplete_path.stat()
                        if not force_download or stat.st_mtime >= started_wall:
                            reporter.update_file(stat.st_size)
                    except FileNotFoundError:
                        pass
                time.sleep(0.2)
            future.result()
            if reporter is not None:
                reporter.complete_file()


def verify_selected_files(
    target: Path,
    relative_paths: list[str],
    reporter: BundleProgress | None = None,
) -> list[str]:
    invalid: list[str] = []
    total = sum(int(MODEL_FILES[item]["size"]) for item in relative_paths)
    processed = 0
    for relative_path in relative_paths:
        metadata = MODEL_FILES[relative_path]
        local_path = target / relative_path
        if not local_path.is_file() or local_path.stat().st_size != metadata["size"]:
            invalid.append(relative_path)
            processed += int(metadata["size"])
            if reporter is not None:
                reporter.verify(relative_path, processed, total)
            continue
        hashed = 0

        def update_hash(size: int) -> None:
            nonlocal hashed
            hashed += size
            if reporter is not None:
                reporter.verify(relative_path, processed + hashed, total)

        if sha256_file(local_path, update_hash) != metadata["sha256"]:
            invalid.append(relative_path)
        processed += int(metadata["size"])
        if reporter is not None:
            reporter.verify(relative_path, processed, total)
    return invalid


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
    reporter = BundleProgress(MODEL_FILES, source=args.source)

    reporter._send(
        "scanning",
        force=True,
        received=0,
        total=reporter.bundle_total,
        phasePercent=0.0,
        overallPercent=0.0,
        message="正在检查现有文件并计算 SHA-256。",
    )
    missing, mismatched = inspect_model_files(
        target,
        verify_hashes=True,
        on_progress=reporter.scan,
    )
    required = list(dict.fromkeys([*missing, *mismatched]))
    free_bytes = shutil.disk_usage(target).free
    reporter.preflight(required, free_bytes)
    if required and free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            "模型目录可用空间不足 512 MiB，无法安全继续下载；请清理空间或更换目录。"
        )
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
            download_huggingface(target, missing, mismatched, reporter)
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
        print(">> Complete IndexTTS 2.5 model bundle already matches the signed manifest.", flush=True)

    if args.source == "modelscope" and required:
        from indextts.utils.model_download import ensure_models_available

        ensure_models_available(str(target), include_legacy_semantic_codec=False)

    invalid = verify_selected_files(target, required, reporter) if required else []
    if invalid:
        details = [f"invalid:{item}" for item in invalid]
        raise RuntimeError("Official IndexTTS 2.5 model verification failed: " + ", ".join(details))

    reporter.done()
    print(
        f">> IndexTTS 2.5 model bundle {MODEL_MANIFEST['bundleVersion']} "
        f"({MODEL_REVISION}) is ready.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit_progress(
            {
                "type": "model_download_progress",
                "phase": "error",
                "overallPercent": 0.0,
                "errorType": type(exc).__name__,
                "error": str(exc),
                "message": f"模型下载或校验失败：{exc}",
            }
        )
        raise
