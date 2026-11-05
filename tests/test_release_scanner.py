from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_scanner():
    specification = spec_from_file_location('release_scanner', ROOT / 'scripts/check_release.py')
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize('directory', ['.venv', 'venv', '.git', '.tox', '.mypy_cache', '.ruff_cache', 'outputs'])
def test_release_scanner_ignores_local_environments_and_generated_files(tmp_path, monkeypatch, directory):
    scanner = load_scanner()
    monkeypatch.setattr(scanner, 'ROOT', tmp_path)
    dependency = tmp_path / directory / 'dependency.py'
    dependency.parent.mkdir()
    dependency.write_text('This is not project Python source.\n', encoding='utf-8')
    assert scanner.main() == 0


def test_release_scanner_still_rejects_invalid_project_source(tmp_path, monkeypatch):
    scanner = load_scanner()
    monkeypatch.setattr(scanner, 'ROOT', tmp_path)
    source = tmp_path / 'src/model.py'
    source.parent.mkdir()
    source.write_text('This is not valid Python source.\n', encoding='utf-8')
    assert scanner.main() == 1
