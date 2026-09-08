import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cen_ts.apo import APOOptimizer
from cen_ts.causal_graph import CausalGraphBuilder
from cen_ts.event_extractor import EventExtractor
from cen_ts.inverse_event import InverseEventGenerator
from cen_ts.narrative import NarrativeGenerator
from cen_ts.verifier import BidirectionalVerifier
from cen_ts.variant_builder import build_text_variant


def test_unimplemented_modes_fail():
    for cls in [EventExtractor, CausalGraphBuilder, InverseEventGenerator, BidirectionalVerifier, NarrativeGenerator, APOOptimizer]:
        with pytest.raises(NotImplementedError):
            cls().transform("x", "2020-01-01", {})
    with pytest.raises(NotImplementedError):
        build_text_variant(ROOT / "third_party" / "TaTS" / "data" / "Environment.csv", ROOT / "data" / "v6" / "p2" / "bad.csv", "event", "event_fact")
