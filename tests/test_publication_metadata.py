from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_readme_links_to_release_materials():
    document = (ROOT / 'README.md').read_text(encoding='utf-8')
    for reference in (
        'requirements/paper-cu128.txt',
        'requirements/cpu.txt',
        'docs/paper_runs.md',
        'docs/preprocessing.md',
        'scripts/validate_manifests.py',
        'LICENSE',
        'THIRD_PARTY_NOTICES.md',
    ):
        assert reference in document
    assert 'without package-installation metadata' not in document


def test_source_distribution_includes_reproduction_materials():
    manifest = (ROOT / 'MANIFEST.in').read_text(encoding='utf-8')
    for entry in (
        'LICENSE',
        'THIRD_PARTY_NOTICES.md',
        'CITATION.cff',
        'recursive-include requirements *.txt',
    ):
        assert entry in manifest


def test_ci_runs_paper_version_checks_with_read_only_permissions():
    workflow = yaml.safe_load((ROOT / '.github/workflows/tests.yml').read_text(encoding='utf-8'))
    assert workflow['permissions'] == {'contents': 'read'}
    events = workflow.get('on', workflow.get(True))
    assert 'pull_request_target' not in events
    job = workflow['jobs']['cpu-regression']
    setup = next(step for step in job['steps'] if step.get('uses', '').startswith('actions/setup-python@'))
    assert str(setup['with']['python-version']) == '3.9.25'
    commands = '\n'.join(step.get('run', '') for step in job['steps'])
    assert 'requirements/cpu.txt' in commands
    assert 'pytest' in commands
    assert 'check_release.py' in commands


def test_citation_identifies_software_without_placeholder_authors():
    citation = yaml.safe_load((ROOT / 'CITATION.cff').read_text(encoding='utf-8'))
    assert citation['cff-version'] == '1.2.0'
    assert citation['version'] == '1.0.0'
    assert citation['repository-code'] == 'https://github.com/Intelligent-Medical-Research-Group/SetAware-FCM'
    assert citation['authors'] == [{'name': 'SetAware-FCM contributors'}]
    assert 'REPLACE_' not in str(citation)


def test_publication_requires_an_explicit_project_license():
    document = (ROOT / 'LICENSE').read_text(encoding='utf-8')
    assert 'MIT License' in document or 'Apache License' in document
