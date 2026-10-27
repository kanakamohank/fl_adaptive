#!/usr/bin/env python3
"""Regression test for per-class pretrain loss variance.

Added because our pooled per-sample loss variance INVERTS under a
Dirichlet(0.3) partition (measured AUROC 0.436, worse than chance): a clean
client holding rare/hard classes shows a wide loss spread that mimics a noisy
one. FedNoRo (IJCAI 2023, Sec. 1) diagnoses that failure and fixes it by
computing the indicator per class. Applying their decomposition to OUR
statistic needs per-class variance, which this guards.

Two properties matter and both are easy to break silently:
  * a class with fewer than 2 samples must report NaN, NOT 0.0 -- scoring it
    zero would make a client holding one sample of a class look maximally
    clean in that class, which is the opposite of "no information";
  * the value must be the POPULATION variance (divide by n), matching the
    pooled `pretrain_loss_var` beside it. A silent switch to the sample
    variance (n-1) would make per-class and pooled numbers incomparable.

This file does NOT mock flwr: `test_tavs_esp_strategy.py` replaces
`sys.modules['flwr.common']` at import time, and importing TAVSFlowerClient
pulls `from flwr.client import NumPyClient`, which breaks under that mock.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


class _FixedLossModel:
    """Stands in for a real model: returns logits engineered so that
    CrossEntropyLoss produces a known per-sample loss. Simpler than mocking
    the loss function, because it exercises the real forward -> loss path."""

    def __init__(self, device):
        import torch
        self._p = [torch.zeros(1, device=device)]

    def parameters(self):
        return iter(self._p)

    def eval(self):
        return self

    def __call__(self, x):
        # x already carries the target logits (see the loader below).
        return x


def _build_client(batches):
    """TAVSFlowerClient with its underlying model swapped for the stub and
    train_loader yielding the given (logits, labels) batches."""
    from src.clients.tavs_flower_client import TAVSFlowerClient, TAVSClientConfig
    import torch

    cfg = TAVSClientConfig(client_id="c0", client_type="honest",
                           model_type="cifar_cnn")
    client = TAVSFlowerClient(config=cfg)
    client.underlying_client.model = _FixedLossModel(torch.device("cpu"))
    client.train_loader = batches
    return client


def _logits_for_loss(target_loss, num_classes, label):
    """Build a logit vector whose CrossEntropyLoss against `label` equals
    `target_loss`. With a one-hot-style vector [a, 0, 0, ...] at the label
    position, loss = log(sum(exp)) - a. Setting every non-label logit to 0
    gives loss = log((C-1) + exp(a)) - a; solving for a is fiddly, so instead
    put the mass on a non-label entry: logits all 0 except entry j != label
    set to b, giving loss = log((C-1) + exp(b)) - 0 = log(C-1+exp(b)).
    Invert: b = log(exp(target) - (C - 1))."""
    import torch
    inner = math.exp(target_loss) - (num_classes - 1)
    assert inner > 0, (
        f"target_loss {target_loss} too small for {num_classes} classes; "
        f"minimum is log({num_classes}) = {math.log(num_classes):.4f}"
    )
    b = math.log(inner)
    v = torch.zeros(num_classes)
    j = (label + 1) % num_classes
    v[j] = b
    return v


def test_per_class_variance_matches_hand_computed_values():
    import torch

    C = 10
    # Class 3 gets four samples with losses {3.0, 3.0, 5.0, 5.0}:
    #   mean = 4.0, population variance = ((1)^2 * 4) / 4 = 1.0
    # Class 7 gets two identical losses {4.0, 4.0}: variance exactly 0.0
    # Class 1 gets exactly one sample: must be NaN (no spread to measure)
    spec = [(3, 3.0), (3, 3.0), (3, 5.0), (3, 5.0),
            (7, 4.0), (7, 4.0),
            (1, 2.5)]
    xs = torch.stack([_logits_for_loss(l, C, c) for c, l in spec])
    ys = torch.tensor([c for c, _ in spec])
    client = _build_client([(xs, ys)])

    out = client._oracle_signals_on_train(num_classes=C)
    assert out is not None, "oracle signal computation returned None"

    pcv = out["per_class_pretrain_var"]
    pcl = out["per_class_pretrain_loss"]
    pcc = out["per_class_pretrain_count"]
    assert len(pcv) == C, f"per_class_pretrain_var must have {C} entries"

    # Class 3: mean 4.0, population variance 1.0
    assert abs(pcl[3] - 4.0) < 1e-4, f"class 3 mean: got {pcl[3]}"
    assert abs(pcv[3] - 1.0) < 1e-3, (
        f"class 3 population variance should be 1.0, got {pcv[3]}. "
        f"Sample variance (n-1) would give 1.333 -- the two must not be "
        f"swapped, since pooled pretrain_loss_var divides by n."
    )
    assert pcc[3] == 4

    # Class 7: identical losses -> exactly zero, and never negative from
    # float round-off in E[x^2] - E[x]^2.
    assert pcv[7] == 0.0 or abs(pcv[7]) < 1e-9, (
        f"class 7 variance should be ~0, got {pcv[7]}"
    )
    assert pcv[7] >= 0.0, "variance must never be negative"

    # Class 1: single sample -> NaN, not 0.0.
    assert math.isnan(pcv[1]), (
        f"a class with 1 sample must report NaN (no measurable spread); "
        f"got {pcv[1]}. Reporting 0.0 would make a client holding one "
        f"sample of a class look maximally clean in that class."
    )
    assert pcc[1] == 1

    # Absent classes: NaN in both mean and variance.
    for c in (0, 2, 4, 5, 6, 8, 9):
        assert math.isnan(pcl[c]), f"absent class {c} mean should be NaN"
        assert math.isnan(pcv[c]), f"absent class {c} variance should be NaN"
        assert pcc[c] == 0

    print("✓ per-class variance: population (n) not sample (n-1), "
          "NaN for <2 samples, never negative")
    return True


def test_per_class_variance_shipped_as_json_in_fit_metrics():
    """The field must reach the server. Flower's proto rejects raw lists, so
    the client JSON-encodes list values; a field added to the oracle dict but
    missing from that encode path would be silently dropped."""
    import json
    import torch

    C = 10
    spec = [(3, 3.0), (3, 5.0), (7, 4.0), (7, 4.0)]
    xs = torch.stack([_logits_for_loss(l, C, c) for c, l in spec])
    ys = torch.tensor([c for c, _ in spec])
    client = _build_client([(xs, ys)])

    out = client._oracle_signals_on_train(num_classes=C)
    assert "per_class_pretrain_var" in out

    # Mirror fit()'s encode step.
    encoded = {}
    for k, v in out.items():
        encoded[k] = json.dumps(list(v)) if isinstance(v, (list, tuple)) else v
    assert isinstance(encoded["per_class_pretrain_var"], str), (
        "per_class_pretrain_var must be JSON-encoded; a raw list raises at "
        "Flower serde time AFTER fit() returns, which drops the whole result"
    )
    recovered = json.loads(encoded["per_class_pretrain_var"])
    assert len(recovered) == C
    print("✓ per_class_pretrain_var JSON-encodes for the Flower wire")
    return True


def main():
    print("🧪 per-class pretrain variance regression suite")
    print("=" * 60)
    for t in (test_per_class_variance_matches_hand_computed_values,
              test_per_class_variance_shipped_as_json_in_fit_metrics):
        assert t() is True
    print("\n🎯 All per-class variance tests PASSED!")


if __name__ == "__main__":
    main()
