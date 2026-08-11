"""Validate the expected EstateScope AI project structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

REQUIRED_PATHS = [
    ".gitattributes",
    ".github",
    ".github/ISSUE_TEMPLATE",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
    "artifacts-manifest.json",
    "app",
    "app/services",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "data",
    "docs",
    "GOVERNANCE.md",
    "models",
    "notebooks",
    "reports/metrics/nlp-metrics.json",
    "scripts/evaluate_nlp_model.py",
    "src",
    "static",
    "templates",
    "tests",
    ".env.example",
    "README.md",
    "LICENSE",
    "pytest.ini",
    "SECURITY.md",
    "SUPPORT.md",
]

OPTIONAL_MODEL_ARTIFACTS = [
    "models/best_regressor.joblib",
    "models/best_rnn_model_full.keras",
    "models/tuned_cnn_best.keras",
    "models/nlp_tokenizer.json",
]

LARGE_FILE_BYTES = 10 * 1024 * 1024
MAX_TRACKED_FILE_BYTES = 95 * 1024 * 1024


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact_manifest(root: Path) -> list[str]:
    """Return validation errors for declared binary and data artifacts."""
    manifest_path = root / "artifacts-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Cannot read {manifest_path.name}: {exc}"]

    errors: list[str] = []
    for artifact in manifest.get("artifacts", []):
        relative_path = artifact.get("path")
        if not isinstance(relative_path, str):
            errors.append("Artifact entry is missing a string path.")
            continue

        path = root / relative_path
        if not path.is_file():
            errors.append(f"Missing declared artifact: {relative_path}")
            continue

        expected_size = artifact.get("bytes")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            errors.append(
                f"Size mismatch for {relative_path}: expected {expected_size}, got {actual_size}."
            )

        expected_hash = artifact.get("sha256")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            errors.append(f"SHA-256 mismatch for {relative_path}.")
    return errors


def _tracked_file_summary(root: Path) -> tuple[int, int, list[tuple[str, int]]]:
    """Return tracked file count, bytes, and files at least 10 MiB."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return 0, 0, []

    paths = [os.fsdecode(value) for value in result.stdout.split(b"\0") if value]
    sizes: list[tuple[str, int]] = []
    total_bytes = 0
    for relative_path in paths:
        path = root / relative_path
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        if size >= LARGE_FILE_BYTES:
            sizes.append((relative_path, size))
    return len(paths), total_bytes, sorted(sizes, key=lambda item: item[1], reverse=True)


def main() -> int:
    """Validate the repository structure and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci", action="store_true", help="Fail only on required paths.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED_PATHS if not (root / path).exists()]
    if missing:
        print("Missing required paths:")
        for path in missing:
            print(f"  - {path}")
        return 1

    artifact_errors = _validate_artifact_manifest(root)
    if artifact_errors:
        print("Artifact integrity check failed:")
        for error in artifact_errors:
            print(f"  - {error}")
        return 1

    missing_models = [path for path in OPTIONAL_MODEL_ARTIFACTS if not (root / path).exists()]
    if missing_models:
        print("Optional model artifacts not found:")
        for path in missing_models:
            print(f"  - {path}")
        print("The app can still start, but affected modalities may be unavailable.")

    generated = [
        "__pycache__",
        ".pytest_cache",
        "coverage.xml",
        ".coverage",
    ]
    present_generated = [path for path in generated if (root / path).exists()]
    if present_generated:
        print("Generated local files are present and should not be committed:")
        for path in present_generated:
            print(f"  - {path}")

    tracked_count, tracked_bytes, large_files = _tracked_file_summary(root)
    if tracked_count:
        print(
            f"Tracked repository: {tracked_count} files, "
            f"{tracked_bytes / (1024 * 1024):.2f} MiB; "
            f"{len(large_files)} files are at least 10 MiB."
        )
        oversized = [(path, size) for path, size in large_files if size > MAX_TRACKED_FILE_BYTES]
        if oversized:
            print("Tracked files exceed the 95 MiB repository policy:")
            for path, size in oversized:
                print(f"  - {path}: {size / (1024 * 1024):.2f} MiB")
            return 1

    print("Project structure check passed.")
    return 0 if args.ci or not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
