from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PAPER_VERSIONS = {
    "numpy": "2.0.2",
    "pandas": "2.3.1",
    "scikit-learn": "1.6.1",
    "matplotlib": "3.9.4",
}


def read_requirements(filename):
    dependencies = {}
    for raw_line in filename.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-r "):
            dependencies.update(read_requirements(filename.parent / line[3:].strip()))
            continue
        dependency = Requirement(line)
        assert dependency.url is None
        assert dependency.name not in dependencies
        dependencies[dependency.name] = dependency
    return dependencies


def test_paper_versions_remain_exact_in_shared_requirements():
    dependencies = read_requirements(ROOT / "requirements" / "paper.txt")
    for name, version in PAPER_VERSIONS.items():
        assert str(dependencies[name].specifier) == f"=={version}"
    for dependency in dependencies.values():
        specifiers = list(dependency.specifier)
        assert len(specifiers) == 1
        assert specifiers[0].operator == "=="
        assert "*" not in specifiers[0].version


@pytest.mark.parametrize("filename,build,index", [
    ("paper-cu128.txt", "2.8.0+cu128", "cu128"),
    ("cpu.txt", "2.8.0+cpu", "cpu"),
])
def test_torch_build_is_explicit_and_shared_versions_agree(filename, build, index):
    path = ROOT / "requirements" / filename
    dependencies = read_requirements(path)
    assert str(dependencies["torch"].specifier) == f"=={build}"
    assert f"https://download.pytorch.org/whl/{index}" in path.read_text(encoding="utf-8")
    for name, version in PAPER_VERSIONS.items():
        assert str(dependencies[name].specifier) == f"=={version}"


def test_package_metadata_agrees_with_pinned_environment():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    dependencies = read_requirements(ROOT / "requirements" / "paper.txt")
    declared = [Requirement(value) for value in project["dependencies"]]
    for values in project["optional-dependencies"].values():
        declared.extend(Requirement(value) for value in values)
    for dependency in declared:
        if dependency.name == "torch":
            assert "2.8.0" in dependency.specifier
            assert "2.0.0" not in dependency.specifier
            continue
        assert dependency.name in dependencies
        version = next(iter(dependencies[dependency.name].specifier)).version
        assert version in dependency.specifier
    assert all("==" in value for value in metadata["build-system"]["requires"])


def test_conda_file_selects_paper_python_and_cuda_requirements():
    environment = (ROOT / "environment-paper.yml").read_text(encoding="utf-8")
    assert "python=3.9.25" in environment
    assert "pip=25.2" in environment
    assert "requirements/paper-cu128.txt" in environment
    assert "-e .[baselines,dev]" in environment


def test_environment_documentation_distinguishes_validation_from_paper_evidence():
    document = (ROOT / "docs" / "environment.md").read_text(encoding="utf-8")
    assert "release pins" in document
    assert "not a recovered historical lockfile" in " ".join(document.split())
    assert "requirements/cpu.txt" in document
    assert "pip check" in document
