"""Run an explicit module-and-arguments recipe without changing its defaults."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
from string import Template
import subprocess
import sys


def build_command(recipe: dict, *, dry_run: bool = False, environment=None) -> list[str]:
    environment = os.environ if environment is None else environment
    if not isinstance(recipe, dict):
        raise ValueError("The recipe must be a JSON object")
    module = recipe.get("module", "")
    module_roots = {"dataio", "common", "mae", "cell_level", "patient_level"}
    if not isinstance(module, str) or not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+", module) or module.split(".")[0] not in module_roots:
        raise ValueError("recipe.module must name a module inside src's dataio, common, mae, cell_level, or patient_level packages")
    arguments = recipe.get("arguments", [])
    if not isinstance(arguments, list) or not all(isinstance(argument, str) for argument in arguments):
        raise ValueError("recipe.arguments must be a list of strings")
    expanded = []
    for argument in arguments:
        template = Template(argument)
        try:
            value = template.safe_substitute(environment) if dry_run else template.substitute(environment)
        except KeyError as error:
            raise ValueError(f"Set the environment variable {error.args[0]} before running this recipe") from error
        expanded.append(os.path.expanduser(value))
    launcher = recipe.get("launcher", "python")
    if launcher == "python":
        return [sys.executable, "-m", module, *expanded]
    if launcher == "torchrun":
        processes = recipe.get("processes", 2)
        if not isinstance(processes, int) or isinstance(processes, bool) or processes < 2:
            raise ValueError("MAE torchrun recipes require at least two processes")
        return [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={processes}", "--module", module, *expanded,
        ]
    raise ValueError("recipe.launcher must be python or torchrun")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-dir", type=Path, default=None)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    try:
        command = build_command(recipe, dry_run=args.dry_run)
    except ValueError as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    record = {
        "recipe": recipe,
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(Path.cwd()),
    }
    if args.record_dir is not None:
        args.record_dir.mkdir(parents=True, exist_ok=True)
        (args.record_dir / "execution.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    result = subprocess.run(command, check=False)
    if args.record_dir is not None:
        record.update(returncode=result.returncode, finished_at=datetime.now(timezone.utc).isoformat())
        (args.record_dir / "execution.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
