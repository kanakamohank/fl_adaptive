#!/usr/bin/env python3
"""Regression test for CIFAR-100 within-superclass pair-flip map.

The map is the Patrini-style within-superclass next-neighbor pairing: each
of the 20 superclasses cycles its 5 fine classes. The map MUST satisfy
NoisyLabelSubset's validation: full coverage over range(100), no
self-loops, keys and values valid class ids. If a future refactor breaks
any of those, every CIFAR-100 pair-flip run would raise at setup time.

Why this lives in its own file:
`test_tavs_esp_strategy.py` mocks `flwr.common` globally, and importing
`src.utils.data_utils` is harmless but keeping this test narrow and
pipeline-independent makes the failure mode obvious.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def test_cifar100_superclass_pair_map_is_valid_permutation():
    from src.utils.data_utils import (
        build_cifar100_superclass_pair_map,
        CIFAR100_SUPERCLASS_TO_FINE,
    )

    # The 20 superclasses must together cover all 100 fine classes once.
    all_fines = sorted(c for fines in CIFAR100_SUPERCLASS_TO_FINE.values()
                       for c in fines)
    assert all_fines == list(range(100)), (
        "superclass table must cover [0, 100) exactly once; "
        f"got {len(all_fines)} entries, duplicates/gaps: "
        f"{set(range(100)) - set(all_fines)} missing, "
        f"{sorted(c for c in all_fines if all_fines.count(c) > 1)} dupes"
    )
    assert len(CIFAR100_SUPERCLASS_TO_FINE) == 20, (
        "CIFAR-100 has exactly 20 superclasses"
    )
    for sc, fines in CIFAR100_SUPERCLASS_TO_FINE.items():
        assert len(fines) == 5, (
            f"superclass {sc} should have 5 fine classes; got {len(fines)}"
        )

    pm = build_cifar100_superclass_pair_map()
    # Full coverage over range(100), both as keys and values.
    assert set(pm.keys()) == set(range(100)), (
        f"pair_map keys missing: {set(range(100)) - set(pm.keys())}, "
        f"extra: {set(pm.keys()) - set(range(100))}"
    )
    assert set(pm.values()) == set(range(100)), (
        "pair_map must be a permutation; some class is never a target"
    )
    # No self-loops.
    for k, v in pm.items():
        assert k != v, f"self-loop at class {k}"
    # Within-superclass check: each pair must share a superclass.
    class_to_superclass = {}
    for sc, fines in CIFAR100_SUPERCLASS_TO_FINE.items():
        for c in fines:
            class_to_superclass[c] = sc
    for src, tgt in pm.items():
        assert class_to_superclass[src] == class_to_superclass[tgt], (
            f"pair_map crosses superclass boundary: {src} "
            f"(superclass {class_to_superclass[src]}) -> {tgt} "
            f"(superclass {class_to_superclass[tgt]})"
        )
    # Cycle length check: each superclass's 5 fines form a 5-cycle under
    # the map. Walking from any fine class in a superclass should return
    # to the start after exactly 5 steps.
    for sc, fines in CIFAR100_SUPERCLASS_TO_FINE.items():
        start = fines[0]
        cur = start
        for _ in range(5):
            cur = pm[cur]
        assert cur == start, (
            f"superclass {sc} fines do not form a 5-cycle under the map"
        )
    print(f"✓ CIFAR-100 pair_map is a valid 100-class permutation, "
          f"no self-loops, within-superclass 5-cycles")
    return True


def test_cifar100_pair_map_passes_noisy_label_subset_validation():
    """Direct validation: feed the map into NoisyLabelSubset's init-time
    validator by constructing a trivial subset with it. If validation
    passes, the pair_map is production-ready for a CIFAR-100 pairflip run."""
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    from src.utils.data_utils import (
        NoisyLabelSubset,
        build_cifar100_superclass_pair_map,
    )
    pm = build_cifar100_superclass_pair_map()
    # Minimal 100-class dataset: one sample per class, images are dummies.
    xs = torch.zeros((100, 3, 32, 32))
    ys = torch.arange(100)
    base = TensorDataset(xs, ys)
    # noise_fraction=0.1 means ~10 samples will be flipped; validation
    # happens at __init__ regardless of fraction.
    noisy = NoisyLabelSubset(
        base=base,
        noise_fraction=0.1,
        num_classes=100,
        seed=42,
        noise_type="pairflip",
        pair_map=pm,
    )
    assert len(noisy) == 100
    # Count how many of the 100 labels got flipped (sanity: ~10 at 0.1 rate).
    n_flipped = sum(1 for idx in range(100)
                    if int(noisy[idx][1]) != int(ys[idx]))
    assert n_flipped > 0, "pair_map applied but zero flips occurred"
    print(f"✓ NoisyLabelSubset accepts the CIFAR-100 pair_map "
          f"(100 samples, {n_flipped} flipped at noise_fraction=0.1)")
    return True


def main():
    print("🧪 CIFAR-100 pair_map regression suite")
    print("=" * 60)
    tests = [
        test_cifar100_superclass_pair_map_is_valid_permutation,
        test_cifar100_pair_map_passes_noisy_label_subset_validation,
    ]
    for t in tests:
        assert t() is True
    print("\n🎯 All CIFAR-100 pair_map tests PASSED!")


if __name__ == "__main__":
    main()
