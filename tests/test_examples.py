"""Smoke and round-trip tests for the typed experiment driver."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_SPECS = [
    "baseline.yaml",
    "ablation.yaml",
    "scaling.yaml",
    "stability.yaml",
    "rubric.yaml",
]


class TestSpecFilesExist:
    """All five shipped spec files exist on disk."""

    @pytest.mark.parametrize("name", SHIPPED_SPECS)
    def test_each(self, name: str) -> None:
        assert (REPO_ROOT / "examples" / "specs" / name).is_file()


class TestSpecsAreYaml:
    """Each shipped spec is syntactically valid YAML."""

    @pytest.mark.parametrize("name", SHIPPED_SPECS)
    def test_each(self, name: str) -> None:
        with open(REPO_ROOT / "examples" / "specs" / name) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "lengths" in data
        assert "kind" in data


class TestDriverRoundTrip:
    """examples/run_paper_experiment.load_spec returns a valid Spec for every shipped YAML."""

    def test_importable(self) -> None:
        import importlib
        mod = importlib.import_module("examples.run_paper_experiment")
        assert callable(getattr(mod, "load_spec", None))

    @pytest.mark.parametrize("name", SHIPPED_SPECS)
    def test_loads_each_shipped_spec(self, name: str) -> None:
        """load_spec returns a Spec-like object for every shipped YAML."""
        import importlib
        from xaker.attention import BLOCK
        from xaker.bench import Spec

        mod = importlib.import_module("examples.run_paper_experiment")
        spec = mod.load_spec(REPO_ROOT / "examples" / "specs" / name)
        assert isinstance(spec, Spec)
        assert spec.lengths
        assert spec.dim > 0
        assert spec.heads > 0
        assert spec.kinds
        for k in spec.kinds:
            assert k in BLOCK, f"unknown kind {k!r} in shipped spec {name}"


class TestDriverValidation:
    """load_spec rejects malformed input early."""

    def test_rejects_unknown_kind(self, tmp_path: Path) -> None:
        """A typo'd kind raises ValueError with the typo in the message."""
        import importlib

        mod = importlib.import_module("examples.run_paper_experiment")
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({"kind": "t", "lengths": [8], "kinds": ["lniear"]}))
        with pytest.raises(ValueError, match="lniear"):
            mod.load_spec(bad)

    def test_rejects_missing_lengths(self, tmp_path: Path) -> None:
        """A spec without ``lengths`` raises ValueError."""
        import importlib

        mod = importlib.import_module("examples.run_paper_experiment")
        bad = tmp_path / "no_lengths.yaml"
        bad.write_text(yaml.safe_dump({"kind": "t", "kinds": ["fused"]}))
        with pytest.raises(ValueError, match="lengths"):
            mod.load_spec(bad)

    def test_rejects_non_mapping(self, tmp_path: Path) -> None:
        """A non-mapping YAML top-level raises ValueError."""
        import importlib

        mod = importlib.import_module("examples.run_paper_experiment")
        bad = tmp_path / "list.yaml"
        bad.write_text("- 1\n- 2\n")
        with pytest.raises(ValueError, match="mapping"):
            mod.load_spec(bad)
