import ast
import argparse
import importlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "configs" / "paper"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_runner():
    spec = spec_from_file_location("paper_recipe_runner", ROOT / "scripts" / "run_experiment.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments_dict(recipe):
    arguments = recipe["arguments"]
    return dict(zip(arguments[::2], arguments[1::2]))


def test_paper_recipes_disclose_scope_and_only_use_supported_cli_flags():
    recipes = sorted(PAPER.rglob("*.json"))
    assert len(recipes) >= 21
    runner = load_runner()
    for path in recipes:
        recipe = read_json(path)
        if "module" not in recipe:
            continue
        assert recipe["clinical_reproduction_verified"] is False
        assert recipe["status"] in {"manuscript_derived", "historical_configuration_recovered"}
        assert recipe["evidence"]
        assert recipe["unresolved"]
        assert recipe["paper_item"]
        source = ROOT / "src" / Path(*recipe["module"].split(".")).with_suffix(".py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        flags = {
            argument.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        }
        assert len(recipe["arguments"]) % 2 == 0, path
        assert set(arguments_dict(recipe)) <= flags, path
        assert len(arguments_dict(recipe)) * 2 == len(recipe["arguments"]), path
        preview = runner.build_command(recipe, dry_run=True, environment={})
        assert recipe["module"] in preview
        placeholders = set(re.findall(r"\$\{([A-Z_]+)\}", " ".join(recipe["arguments"])))
        assert placeholders == set(recipe["required_environment"]), path
        environment = {key: "/synthetic/fixture" for key in placeholders}
        assert not any("${" in argument for argument in runner.build_command(recipe, environment=environment))
        with pytest.raises(ValueError, match="environment variable"):
            runner.build_command(recipe, environment={})


def test_historical_patient_recipes_match_sanitized_configuration_evidence():
    evidence = read_json(PAPER / "evidence.json")
    assert len(evidence["patient"]) == 8
    assert len(evidence["cell"]) == 12
    for record in evidence["patient"]:
        assert re.fullmatch(r"[0-9a-f]{64}", record["metadata"]["sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", record["summary_evidence"]["sha256"])
        stem = f'{record["dataset"]}_{record["method"].lower().replace("-", "_")}'
        train = read_json(PAPER / "historical" / f"{stem}_train.json")
        evaluate = read_json(PAPER / "historical" / f"{stem}_eval.json")
        training = arguments_dict(train)
        evaluation = arguments_dict(evaluate)
        for key, value in record["training_arguments"].items():
            assert training[f"--{key}"] == str(value), (stem, key)
        for key in ("filter_mode", "max_cells_per_tube", "seed"):
            value = record["evaluation_effective"][key]
            value = 0 if key == "seed" and value is None else value
            assert evaluation[f"--{key}"] == str(value), (stem, key)
        assert train["historical_protocol"]["test_used_for_checkpoint_selection"] is True
        assert train["historical_protocol"]["checkpoint_epoch"] == record["checkpoint_epoch"]
        assert train["historical_protocol"]["checkpoint_role"] == record["checkpoint_role"]
        assert record["source_row"]["row"] >= 2
        if not record["checkpoint_file_available"]:
            assert "checkpoint_sha256" not in record


def test_manuscript_recipes_do_not_impersonate_final_runs():
    for path in PAPER.glob("manuscript_*.json"):
        recipe = read_json(path)
        assert recipe["status"] == "manuscript_derived"
        assert "historical_protocol" not in recipe
        assert recipe["historical_run_id"] is None
        assert recipe["clinical_reproduction_verified"] is False


def test_recovered_patient_variants_and_effective_evaluation_controls():
    base = PAPER / "historical"
    z2jw = arguments_dict(read_json(base / "09_Z2JW_LMD_taca_mil_train.json"))
    zz6b = arguments_dict(read_json(base / "14_ZZ6B_taca_mil_train.json"))
    assert (z2jw["--set_num_layers"], z2jw["--set_num_inds"], z2jw["--seed"]) == ("1", "8", "2024")
    assert (zz6b["--set_num_layers"], zz6b["--set_num_inds"], zz6b["--seed"], zz6b["--weight_decay"]) == ("1", "8", "7", "0.001")
    for dataset in ("09_Z2JW_LMD", "14_ZZ6B", "15_ZZYA", "Z27"):
        for method in ("ca_mil", "taca_mil"):
            recipe = read_json(base / f"{dataset}_{method}_eval.json")
            arguments = arguments_dict(recipe)
            assert arguments["--filter_mode"] == "count"
            assert arguments["--max_cells_per_tube"] == "5000"
            assert arguments["--sheet"] == "test"


def test_paper_recipe_evidence_does_not_publish_private_paths_or_rows():
    for path in PAPER.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        for root_name in ("ssd1", "ssd2", "home"):
            assert "/" + root_name + "/" not in text
        assert "patient_id" not in text
        assert "csv_path" not in text


@pytest.mark.parametrize("path", sorted(PAPER.rglob("*.json")), ids=lambda path: path.stem)
def test_paper_recipe_values_parse_before_any_data_or_training(monkeypatch, path):
    recipe = read_json(path)
    if "module" not in recipe:
        return
    environment = {key: "/synthetic/fixture" for key in recipe["required_environment"]}
    environment.update(FCM_CELL_INIT_MODE="random", FCM_TRAIN_SEED="42", FCM_NUM_CLASSES="29")
    command = load_runner().build_command(recipe, environment=environment)
    arguments = command[command.index(recipe["module"]) + 1:]
    module = importlib.import_module(recipe["module"])
    original_parse = argparse.ArgumentParser.parse_args

    class ParsedWithoutExecution(Exception):
        pass

    def parse_without_execution(parser, *args, **kwargs):
        original_parse(parser, *args, **kwargs)
        raise ParsedWithoutExecution

    monkeypatch.setattr(sys, "argv", [recipe["module"], *arguments])
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_without_execution)
    with pytest.raises(ParsedWithoutExecution):
        module.main()
