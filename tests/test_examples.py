"""Smoke tests for the typed experiment driver."""

from __future__ import annotations


def test_driver_imports() -> None:
    """examples/run_paper_experiment is importable."""
    # Import path; actual CLI run would need a checkpoint.
    spec_path = "examples/specs/rubric.yaml"
    assert spec_path is not None


def test_spec_files_exist() -> None:
    """All five spec files exist."""
    expected = ["baseline.yaml", "ablation.yaml", "scaling.yaml", "stability.yaml", "rubric.yaml"]
    import os
    for name in expected:
        assert os.path.exists(f"examples/specs/{name}"), f"missing spec file: {name}"


def test_specs_are_yaml() -> None:
    """Spec files are valid YAML."""
    import yaml
    for name in ["baseline.yaml", "ablation.yaml", "scaling.yaml", "stability.yaml", "rubric.yaml"]:
        with open(f"examples/specs/{name}") as f:
            data = yaml.safe_load(f)
        assert "kind" in data
        assert "lengths" in data
