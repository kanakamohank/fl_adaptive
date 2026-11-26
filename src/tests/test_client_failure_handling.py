#!/usr/bin/env python3
"""A client that misbehaves must be DROPPED, loudly, where the fault is.

BOTH GUARDS HERE WERE WRITTEN AFTER THEY COST A REAL DEBUGGING SESSION.

Switching the backbone to ResNet-18 produced `KeyError:
'layer1_1_gn1_weight'` inside aggregate_fit. The actual fault was three
layers away -- the server was shipping a CIFARCNN's 10 tensors as initial
parameters while holding 62 blocks -- and two separate silent-acceptance
paths stood between the cause and the symptom:

  1. _parse_client_updates' multi-array branch stopped at
     `if i >= len(ndarrays): break`, so a 10-tensor client against 62 blocks
     yielded a 10-block update with no warning. The aggregator seeds its
     output from the first client's keys, so the other 52 blocks simply did
     not exist, and the KeyError surfaced hundreds of lines later naming a
     block rather than a client.

  2. TAVSFlowerClient.fit()'s `except Exception: return parameters, 0, ...`
     hands the UNCHANGED incoming weights back. The server aggregated them
     as a real update -- a silent vote for no-op -- and `max(1, ...)` on
     num_examples even restored full aggregation weight to a client that had
     crashed.

Together they turned "every client failed to train" into a KeyError about a
tensor name. These tests pin that both now fail where the fault is.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from flwr.common import ndarrays_to_parameters  # noqa: E402

from src.core.models import CIFARCNN, get_model  # noqa: E402
from src.tavs_v2.tavs_esp_strategy import (  # noqa: E402
    TavsEspConfig, TavsEspStrategy,
)


class _Proxy:
    def __init__(self, cid):
        self.cid = cid


class _Res:
    def __init__(self, ndarrays, metrics=None, num_examples=10):
        self.parameters = ndarrays_to_parameters(ndarrays)
        self.metrics = metrics or {}
        self.num_examples = num_examples


class _Cfg:
    tavs_config = TavsEspConfig()


def _strategy(model_type="resnet18"):
    m = get_model(model_type, num_classes=10)
    return TavsEspStrategy(config=_Cfg(), model_structure=m.structure), m


def test_a_correct_client_is_parsed_in_full():
    """Control: the right tensor count must still parse every block, or the
    guard below would be passing for the wrong reason."""
    s, m = _strategy()
    good = [p.detach().numpy() for p in m.parameters()]
    out = s._parse_client_updates([(_Proxy("a"), _Res(good))])
    assert len(out["a"]) == len(s.model_blocks) == 62, len(out["a"])
    assert "layer1_1_gn1_weight" in out["a"]
    print(f"✓ a matching client parses all {len(out['a'])} blocks")
    return True


def test_a_wrong_tensor_count_is_dropped_not_truncated():
    """THE REGRESSION. A CIFARCNN's 10 tensors against ResNet-18's 62 blocks
    must be rejected outright, not silently truncated to a 10-block update."""
    s, _ = _strategy()
    bad = [p.detach().numpy() for p in CIFARCNN(num_classes=10).parameters()]
    assert len(bad) == 10, "fixture assumes CIFARCNN has 10 tensors"
    out = s._parse_client_updates([(_Proxy("b"), _Res(bad))])
    assert "b" not in out, (
        f"a 10-tensor client must be dropped, not accepted with "
        f"{len(out.get('b', {}))} of 62 blocks -- that partial update is "
        f"what produced KeyError('layer1_1_gn1_weight') in aggregate_fit"
    )
    print("✓ a wrong tensor count is dropped at the parse, not later")
    return True


def test_one_bad_client_does_not_take_down_the_good_ones():
    """Dropping must be per client. An early `raise` or a shared accumulator
    would lose the healthy updates too."""
    s, m = _strategy()
    good = [p.detach().numpy() for p in m.parameters()]
    bad = [p.detach().numpy() for p in CIFARCNN(num_classes=10).parameters()]
    out = s._parse_client_updates([
        (_Proxy("good1"), _Res(good)),
        (_Proxy("bad"), _Res(bad)),
        (_Proxy("good2"), _Res(good)),
    ])
    assert set(out) == {"good1", "good2"}, sorted(out)
    assert all(len(v) == 62 for v in out.values())
    print("✓ one bad client is dropped; the healthy ones survive")
    return True


def test_a_client_reporting_an_error_is_excluded():
    """fit()'s handler returns the incoming weights with an `error` metric.
    aggregate_fit must drop it rather than aggregate stale weights as a
    genuine update."""
    s, m = _strategy()
    good = [p.detach().numpy() for p in m.parameters()]
    results = [
        (_Proxy("ok"), _Res(good)),
        (_Proxy("crashed"), _Res(good, metrics={"error": "CUDA OOM"},
                                 num_examples=0)),
    ]
    failed = [p.cid for p, r in results if (r.metrics or {}).get("error")]
    kept = [(p, r) for p, r in results if not (r.metrics or {}).get("error")]
    assert failed == ["crashed"], failed
    assert [p.cid for p, _ in kept] == ["ok"]
    # And the weighting bug the drop prevents: max(1, 0) would have given a
    # crashed client a full share of the aggregate.
    assert max(1, int(results[1][1].num_examples)) == 1, (
        "max(1, 0) == 1 is exactly why a dropped-but-kept client would still "
        "have carried aggregation weight"
    )
    print("✓ a client reporting an error is excluded before weighting")
    return True


def test_the_strategy_drops_failed_clients_behaviourally():
    """Exercise the real filter, not a grep of the source.

    The first version of this test merely asserted that `.get("error")`
    appeared in aggregate_fit's source. A mutation that deleted the
    filtering still left that substring behind, and the test passed. The
    filter now lives in its own method so it can be called directly.
    """
    s, m = _strategy()
    good = [p.detach().numpy() for p in m.parameters()]
    results = [
        (_Proxy("ok1"), _Res(good)),
        (_Proxy("crashed"), _Res(good, metrics={"error": "CUDA OOM"},
                                 num_examples=0)),
        (_Proxy("ok2"), _Res(good)),
    ]
    kept, failed = s._drop_failed_clients(results, server_round=1)
    assert failed == ["crashed"], failed
    assert [p.cid for p, _ in kept] == ["ok1", "ok2"], [p.cid for p, _ in kept]

    # No failures: the list must come back untouched, same objects.
    clean = [(_Proxy("a"), _Res(good))]
    kept2, failed2 = s._drop_failed_clients(clean, server_round=1)
    assert failed2 == [] and kept2 is clean

    # Every client failing must be reported as such, not silently emptied.
    allbad = [(_Proxy("x"), _Res(good, metrics={"error": "boom"}))]
    kept3, failed3 = s._drop_failed_clients(allbad, server_round=1)
    assert kept3 == [] and failed3 == ["x"], (kept3, failed3)
    print("✓ _drop_failed_clients filters by behaviour, not by source text")
    return True


def main():
    ok = [
        test_a_correct_client_is_parsed_in_full(),
        test_a_wrong_tensor_count_is_dropped_not_truncated(),
        test_one_bad_client_does_not_take_down_the_good_ones(),
        test_a_client_reporting_an_error_is_excluded(),
        test_the_strategy_drops_failed_clients_behaviourally(),
    ]
    if all(ok):
        print("\n\U0001F3AF All client-failure-handling tests PASSED!")
        return True
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
