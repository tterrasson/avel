import json
from pathlib import Path

import pytest

from avel_export import census

BASELINE = Path(__file__).parents[1] / "census.json"


def test_ranks_unsupported_ops_by_model_count():
    models = {
        "a": {"unsupported": ["aten.x", "aten.y"], "supported": 0.5},
        "b": {"unsupported": ["aten.y"], "supported": 0.9},
    }
    assert census.ranking(models) == [("aten.y", 2), ("aten.x", 1)]
    assert census.regressions(models, {"a": {"supported": 0.6}, "b": {"supported": 0.9}}) == [
        "a: 50.00% supported, 60.00% before"
    ]


@pytest.mark.integration
def test_coverage_never_regresses():
    # Rerun `python -m avel_export.census --write census.json` after adding ops.
    baseline = json.loads(BASELINE.read_text())
    assert census.regressions(census.census(), baseline) == []


@pytest.mark.integration
def test_external_zoo_matches_baseline():
    pytest.importorskip("timm")
    pytest.importorskip("transformers")
    baseline = json.loads(BASELINE.read_text())
    names = list(census.external_zoo())
    assert set(names) <= baseline.keys()
    assert census.regressions(census.census(names, external=True), baseline) == []
