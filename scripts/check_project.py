"""Validate the expected EstateScope AI project structure."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_PATHS = [
    ".gitattributes",
    ".github",
    ".github/ISSUE_TEMPLATE",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
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


def main() -> int:
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

    missing_models = [
        path for path in OPTIONAL_MODEL_ARTIFACTS if not (root / path).exists()
    ]
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

    print("Project structure check passed.")
    return 0 if args.ci or not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
