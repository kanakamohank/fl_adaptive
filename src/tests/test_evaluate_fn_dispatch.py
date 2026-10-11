#!/usr/bin/env python3
"""Regression test for the server-side evaluate_fn dataset dispatch.

Why this lives in its own file:
test_tavs_esp_strategy.py globally replaces `sys.modules['flwr.common']`
with a hand-written mock, which breaks `from src.tavs_v2.end_to_end_pipeline
import TAVSESPPipeline` (that module imports from `flwr.client`). This
file does NOT mock flwr, so the pipeline import works.

The failure mode this guards against: `evaluate_fn` previously hardcoded
`_, test_data = load_cifar10()` regardless of `config.dataset`. On the
CIFAR-100 run (commit 7c55d07 on main) that produced the meaningless
1.3% server accuracy -- a 100-class model evaluated against CIFAR-10
labels. The fix dispatches on `config.dataset`; this test asserts the
rejection branch for unknown datasets fires, so a future dataset
addition (e.g. FEMNIST) can't silently reuse the CIFAR-10 test set.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def test_evaluate_fn_rejects_unknown_dataset():
    import numpy as np
    from src.tavs_v2.end_to_end_pipeline import TAVSESPPipeline
    from src.core.models import get_model

    class _StubPipeline:
        def __init__(self):
            class _Cfg:
                dataset = "femnist"           # unsupported -> must raise
                model_type = "cifar_cnn"
                label_noise_num_classes = 10
            self.config = _Cfg()

    stub = _StubPipeline()
    evaluate_fn = TAVSESPPipeline._create_evaluate_function(stub)

    # Shape-correct dummy params so the pre-dispatch param-copy loop
    # does not crash before hitting the dataset dispatch.
    model = get_model("cifar_cnn", num_classes=10)
    dummy_params = [p.detach().numpy().astype(np.float32)
                    for p in model.parameters()]

    loss, metrics = evaluate_fn(1, dummy_params, {})
    # Degenerate return on inner ValueError (the outer try/except in
    # evaluate_fn catches the "unsupported dataset" raise and returns
    # (2.3, {"accuracy": 0.1, ...}) which is the hardcoded "random"
    # fallback -- confirms the dispatch rejected the unknown dataset
    # instead of silently reusing CIFAR-10.
    assert abs(loss - 2.3) < 1e-6, (
        f"evaluate_fn should return the random-fallback loss (2.3) on "
        f"unknown dataset; got loss={loss}"
    )
    assert metrics.get("total") == 100, (
        f"metrics should indicate the random-fallback path; got {metrics}"
    )
    print("✓ evaluate_fn returns degenerate result on unknown dataset "
          "(inner ValueError caught by outer try/except)")
    return True


def main():
    print("🧪 evaluate_fn dataset-dispatch regression suite")
    print("=" * 60)
    assert test_evaluate_fn_rejects_unknown_dataset() is True
    print("\n🎯 All evaluate_fn dispatch tests PASSED!")


if __name__ == "__main__":
    main()
