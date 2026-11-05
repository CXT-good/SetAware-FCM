"""Check source syntax, local imports, and accidental research-workspace paths."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATH = re.compile(r"/(?:ssd\d+|hdd\d*|root|home|mnt|media)/")
CREDENTIAL = re.compile(r"(?i)(?:api_key|access_token|password|secret_key)\s*[:=]\s*['\"][^'\"$]{8,}['\"]")
EXCLUDED_PARTS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".git", ".venv", "venv", ".tox", "build", "dist",
    "data", "outputs", "output", "output2", "cache", "checkpoints",
    "logs", "results", "result", "figure", "figure_final", "paper",
    "table", "table_final", "archive",
}


def release_files(root: Path):
    for directory, directory_names, filenames in os.walk(root):
        directory_names[:] = sorted(
            name for name in directory_names
            if name not in EXCLUDED_PARTS and not name.endswith(".egg-info")
        )
        for filename in sorted(filenames):
            yield Path(directory) / filename


def main() -> int:
    issues = []
    python_count = 0
    local_roots = {"dataio", "common", "mae", "cell_level", "patient_level"}
    text_suffixes = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".csv", ".example"}
    for path in release_files(ROOT):
        if not path.is_file():
            continue
        if path.suffix not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8")
        if path.name != "check_release.py":
            for line_number, line in enumerate(content.splitlines(), start=1):
                if PRIVATE_PATH.search(line) or CREDENTIAL.search(line):
                    issues.append(f"{path.relative_to(ROOT)}:{line_number}: review path or credential-like literal")
        if path.suffix != ".py":
            continue
        python_count += 1
        try:
            tree = ast.parse(content, filename=str(path), feature_version=(3, 9))
        except SyntaxError as error:
            issues.append(f"{path.relative_to(ROOT)}:{error.lineno}: {error.msg}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                if name == "fcm_ml" or name.startswith("fcm_ml."):
                    issues.append(f"{path.relative_to(ROOT)}:{node.lineno}: old namespace import {name}")
                    continue
                if name.split(".")[0] not in local_roots:
                    continue
                target = ROOT / "src" / Path(*name.split("."))
                if not target.is_dir() and not target.with_suffix(".py").is_file():
                    issues.append(f"{path.relative_to(ROOT)}:{node.lineno}: missing local module {name}")
    for issue in issues:
        print(issue)
    print(f"Checked {python_count} Python files; {len(issues)} issues.")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
