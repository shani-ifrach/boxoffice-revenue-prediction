"""Create a review archive that excludes original TMDB API responses."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCLUDED_DIRECTORIES = (
    "src", "tests", "docs", "notes", "scripts", "examples", "reports",
    "data/processed", "models/reduced", "dashboard/tableau/data",
)
INCLUDED_FILES = (
    "README.md",
    "requirements.txt",
    "requirements-comparison.txt",
    "pyproject.toml",
    "dashboard/README.md",
    ".github/workflows/ci.yml",
    "dashboard/tableau/boxoffice_dashboard.twbx",
    "dashboard/tableau/executive_overview.jpg",
    "dashboard/tableau/model_evaluation.jpg",
    "dashboard/tableau/movie_explorer.jpg",
)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_package(output_path):
    """Include review artifacts while excluding raw API responses and extracts."""
    output_path = Path(output_path).resolve()
    sources = {PROJECT_ROOT / name for name in INCLUDED_FILES}
    for name in INCLUDED_DIRECTORIES:
        sources.update((PROJECT_ROOT / name).rglob("*"))
    files = []
    for source in sorted(sources):
        relative = source.relative_to(PROJECT_ROOT)
        if not source.is_file() or source.is_symlink():
            continue
        if "__pycache__" in relative.parts or source.name == ".DS_Store":
            continue
        if relative.parts[:2] in (
            ("reports", "full"),
            ("reports", "dashboard_data"),
            ("reports", "experiments"),
            ("src", "archive"),
        ):
            continue
        if relative in {
            Path("reports/baseline_20260914.md"),
            Path("reports/submission_readiness_audit.md"),
        }:
            continue
        if source.suffix in (".pyc", ".log") or source == output_path:
            continue
        files.append((source, relative))
    missing = [name for name in INCLUDED_FILES if not (PROJECT_ROOT / name).is_file()]
    for name in ("data/processed", "models/reduced"):
        if not any(relative.parts[:len(Path(name).parts)] == Path(name).parts for _, relative in files):
            missing.append(name)
    if missing:
        raise ValueError(f"Incomplete portfolio delivery: {missing}")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "local CV/interview portfolio review",
        "raw_tmdb_responses_included": False,
        "python": ">=3.12,<3.15",
        "dashboard": "final Tableau workbook",
        "files": {str(relative): {"bytes": source.stat().st_size, "sha256": checksum(source)}
                  for source, relative in files},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output_path.parent, suffix=".zip", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for source, relative in files:
                archive.write(source, Path("boxoffice-revenue-prediction") / relative)
            archive.writestr("boxoffice-revenue-prediction/PACKAGE_MANIFEST.json",
                             json.dumps(manifest, indent=2))
        with ZipFile(temporary_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Portfolio ZIP integrity check failed")
            if any("data/raw/" in name for name in archive.namelist()):
                raise ValueError("Portfolio ZIP must not contain raw TMDB data")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Portfolio archive: {output_path} ({len(files)} files; {output_path.stat().st_size / 1024**2:.1f} MiB)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "dist/boxoffice_portfolio_submission.zip")
    arguments = parser.parse_args()
    build_package(arguments.output)
