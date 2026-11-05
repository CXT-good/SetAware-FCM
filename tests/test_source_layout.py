import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


def test_packages_live_directly_under_src():
    assert not (SOURCE / "fcm_ml").exists()
    directories = {path.name for path in SOURCE.iterdir() if path.is_dir() and not path.name.endswith(".egg-info")}
    assert directories == {"dataio", "common", "mae", "cell_level", "patient_level"}


def test_each_model_has_definition_training_and_evaluation():
    models = [
        "mae", "cell_level/ctransformer", "cell_level/pactransformer",
        "cell_level/baselines/logistic_regression", "cell_level/baselines/deepcytof",
        "cell_level/baselines/xgboost", "cell_level/baselines/flowsom",
        "cell_level/baselines/set_transformer", "patient_level/ca_mil",
        "patient_level/taca_mil", "patient_level/baselines/umap_rf",
        "patient_level/baselines/gmm_fisher_svm", "patient_level/baselines/ensemble_cnn",
        "patient_level/baselines/flowcat",
    ]
    for model in models:
        for filename in ("model.py", "train.py", "eval.py"):
            assert (SOURCE / model / filename).is_file(), f"{model}/{filename}"


def test_source_does_not_import_the_old_namespace():
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("fcm_ml"), str(path)
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("fcm_ml") for alias in node.names), str(path)


def test_attention_blocks_are_shared_without_changing_model_names():
    from common.attention import GatedAttention
    from common.set_attention import SetTransformerEncoder
    from cell_level.pactransformer import model as population_model
    from patient_level.ca_mil import model as cell_attention_model
    from patient_level.taca_mil import model as tube_attention_model

    assert population_model.SetTransformerEncoder is SetTransformerEncoder
    assert tube_attention_model.SetTransformerEncoder is SetTransformerEncoder
    assert cell_attention_model.GatedAttention is GatedAttention
    assert tube_attention_model.GatedAttention is GatedAttention
