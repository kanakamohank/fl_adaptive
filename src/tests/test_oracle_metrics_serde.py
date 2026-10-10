#!/usr/bin/env python3
"""Regression test for the Flower proto serde boundary.

Why this lives in its own file:
`test_tavs_esp_strategy.py` globally replaces `sys.modules['flwr.common']`
with a hand-written mock at import time, so any test inside that module
cannot exercise real Flower serialization. The bug we're regressing
against (lists dropped at the client -> server proto boundary, which
silently marks every fit as a failure and leaves the model at 11%
accuracy for the whole run) is only observable via the real
`flwr.common.serde` round-trip. This file does NOT mock flwr.

The failure mode this guards: shipping any non-Scalar
(bool/bytes/float/int/str) value in fit metrics. A list raises AFTER
`fit()` returns, so the TAVS client's top-level try/except never sees
it; Flower reports a client-side failure, aggregate_fit's result list
is empty, and the model never updates. We saw this once -- a 40-round
run burned ~7.5 min for zero training and empty oracle history.
"""
import json
import sys


def test_oracle_metrics_round_trip_real_flower_serde():
    """Every oracle metric the client ships must survive real Flower proto
    serde. Lists MUST be JSON-encoded strings on the wire."""
    import flwr.common as flc
    import numpy as np

    metrics = {
        # Scalar oracle metrics.
        "client_id": "c1",
        "memorization_gap": 0.42,
        "loss_epoch_first": 2.0,
        "loss_epoch_last": 1.58,
        "update_norm": 1.1,
        "first_batch_grad_norm": 0.73,
        "pretrain_loss_mean": 1.3,
        "pretrain_loss_var": 0.25,
        "small_loss_fraction": 0.55,
        # List-valued oracle metrics MUST be JSON strings; a raw list raises
        # at serde time and dumps the fit result.
        "per_class_pretrain_loss": json.dumps(
            [1.0, 1.1, float("nan"), 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
        "per_class_pretrain_count": json.dumps(
            [10, 10, 0, 10, 10, 10, 10, 10, 10, 10]),
        # One-off per-sample loss list for the mechanism visualisation.
        # ~900 floats per client when active; must survive the same proto
        # path as the shorter per-class lists.
        "per_sample_pretrain_loss": json.dumps([0.1 * i for i in range(900)]),
    }
    r = flc.FitRes(
        status=flc.Status(code=flc.Code.OK, message="ok"),
        parameters=flc.ndarrays_to_parameters([np.zeros(3, dtype=np.float32)]),
        num_examples=100,
        metrics=metrics,
    )
    # The exact calls Flower's simulation engine makes between client and
    # server. If this raises, the oracle path is wire-broken.
    p = flc.serde.fit_res_to_proto(r)
    r2 = flc.serde.fit_res_from_proto(p)
    for k in metrics:
        assert k in r2.metrics, f"{k} lost in Flower serde"
    assert isinstance(r2.metrics["per_class_pretrain_loss"], str), (
        "list fields must survive serde as JSON strings, not raw lists"
    )
    recovered = json.loads(r2.metrics["per_class_pretrain_loss"])
    assert len(recovered) == 10
    print("✓ oracle fit-metrics clear real Flower proto serde "
          "(scalars untouched, lists as JSON strings)")
    return True


def test_raw_list_in_metrics_is_rejected_by_real_flower_serde():
    """Negative regression: confirm Flower's serde rejects a raw list
    value. If this test ever starts passing (Flower loosens the type
    restriction), the JSON-string encoding is no longer required and the
    client-side encode path can be simplified."""
    import flwr.common as flc
    import numpy as np

    r = flc.FitRes(
        status=flc.Status(code=flc.Code.OK, message="ok"),
        parameters=flc.ndarrays_to_parameters([np.zeros(3, dtype=np.float32)]),
        num_examples=1,
        metrics={"client_id": "c1", "bad_list": [1.0, 2.0, 3.0]},
    )
    try:
        flc.serde.fit_res_to_proto(r)
    except ValueError as e:
        assert "list" in str(e) or "Accepted types" in str(e)
        print("✓ Flower serde rejects raw list in fit metrics (as expected)")
        return True
    raise AssertionError(
        "Flower unexpectedly accepted a raw list in fit metrics -- "
        "simplify tavs_flower_client to drop the JSON-string encoding."
    )


def main():
    print("🧪 Oracle metrics Flower-serde regression suite")
    print("=" * 60)
    tests = [
        test_oracle_metrics_round_trip_real_flower_serde,
        test_raw_list_in_metrics_is_rejected_by_real_flower_serde,
    ]
    for t in tests:
        assert t() is True
    print("\n🎯 All Flower-serde regression tests PASSED!")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
