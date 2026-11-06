"""Explicit relocation of paths recorded inside external data manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path


def remap_manifest_path(value: str) -> str:
    original = str(value)
    mapping = json.loads(os.environ.get("FCM_ML_PATH_MAP", "{}"))
    if not isinstance(mapping, dict) or not all(
        isinstance(source, str) and source and isinstance(destination, str) and destination
        for source, destination in mapping.items()
    ):
        raise ValueError("FCM_ML_PATH_MAP must map nonempty path-prefix strings to local path-prefix strings")
    path = Path(original).expanduser()
    for source, destination in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        try:
            relative = path.relative_to(Path(source).expanduser())
        except ValueError:
            continue
        return str((Path(destination).expanduser() / relative).resolve())
    return original
