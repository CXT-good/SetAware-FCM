from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest

from dataio.class_lists import load_dataset_class_info
from dataio.marker_mapping import LABEL_COLUMN


ROOT = Path(__file__).resolve().parents[1]


def load_example(relative_path):
    spec = spec_from_file_location(Path(relative_path).stem, ROOT / relative_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_five_architectures_accept_synthetic_inputs():
    module = load_example("examples/synthetic_demo.py")
    result = module.run_demo()
    assert result["status"] == "passed"
    assert result["PACTransformer_logits"] == [1, 8, 3]


def test_synthetic_patients_never_cross_partitions(tmp_path):
    module = load_example("examples/make_synthetic_data.py")
    module.make_data(tmp_path)
    sheets = pd.read_excel(tmp_path / "patient_split.xlsx", sheet_name=None)
    identifiers = {name: set(frame.patient_id) for name, frame in sheets.items()}
    assert identifiers["train"].isdisjoint(identifiers["validation"])
    assert identifiers["train"].isdisjoint(identifiers["test"])
    assert identifiers["validation"].isdisjoint(identifiers["test"])
    for frame in sheets.values():
        assert all(Path(filename).is_file() for filename in frame.csv_path)
    counts = {"0": 0, "1": 0}
    for filename in sheets["train"].csv_path:
        for label, count in pd.read_csv(filename)[LABEL_COLUMN].value_counts().items():
            counts[str(label)] += int(count)
    info = load_dataset_class_info(str(tmp_path / "class_mapping.json"), dataset_key="synthetic")
    assert info.class_counts == [counts[label] for label in info.class_list]


def test_recipe_requires_explicit_missing_environment_variables():
    module = load_example("scripts/run_experiment.py")
    recipe = {"module": "cell_level.ctransformer.train", "arguments": ["--train_xlsx", "${FCM_DATA_ROOT}/split.xlsx"]}
    with pytest.raises(ValueError, match="FCM_DATA_ROOT"):
        module.build_command(recipe, environment={})
    command = module.build_command(recipe, environment={"FCM_DATA_ROOT": "/example/data"})
    assert command[-1] == "/example/data/split.xlsx"
    preview = module.build_command(recipe, dry_run=True, environment={})
    assert preview[-1] == "${FCM_DATA_ROOT}/split.xlsx"


def test_distributed_recipe_keeps_module_and_process_count_explicit():
    module = load_example("scripts/run_experiment.py")
    recipe = {"module": "mae.train", "launcher": "torchrun", "processes": 2}
    command = module.build_command(recipe)
    assert "--nproc_per_node=2" in command
    assert command[-1] == recipe["module"]
    recipe["processes"] = 1
    with pytest.raises(ValueError, match="at least two"):
        module.build_command(recipe)


@pytest.mark.parametrize("name", ["fcm_ml.train_eval.stage1.train", "os.path", "cell_level..train", "cell_level"])
def test_recipe_rejects_old_namespace_and_nonlocal_modules(name):
    module = load_example("scripts/run_experiment.py")
    with pytest.raises(ValueError, match="recipe.module"):
        module.build_command({"module": name})
