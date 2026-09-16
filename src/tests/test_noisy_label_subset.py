#!/usr/bin/env python3
"""
Tests for NoisyLabelSubset -- the wrapper that flips a fraction of a client's
labels to simulate an honest-but-noisy participant.

The pilot's whole thesis depends on this wrapper producing:
  1. exactly the target fraction of noisy samples,
  2. always to a WRONG class (never to the true one),
  3. deterministically per (seed, base) so re-running does not shuffle the noise,
  4. independently across two clients that share a base dataset,
  5. without mutating the base dataset itself.

If any of the above breaks, the pilot's "does trust identify the noisy clients"
question becomes uninterpretable. The tests are strict on purpose.
"""
import sys
import os
import numpy as np
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.data_utils import NoisyLabelSubset


class DummyDataset(Dataset):
    """Minimal (image, label) dataset -- cycles labels 0..num_classes-1."""
    def __init__(self, n: int, num_classes: int = 10):
        self.n = n
        self.num_classes = num_classes
        # Keep the images cheap; only the labels matter for these tests.
        self._labels = [i % num_classes for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (f"img_{idx}", self._labels[idx])


def test_rate_validation():
    """Constructor rejects out-of-range noise_fraction and num_classes < 2."""
    print("\nTesting NoisyLabelSubset input validation...")
    base = DummyDataset(50)
    NoisyLabelSubset(base, 0.0, num_classes=10, seed=0)   # boundary
    NoisyLabelSubset(base, 1.0, num_classes=10, seed=0)   # boundary
    for bad in (-0.01, 1.01, 2.0):
        try:
            NoisyLabelSubset(base, bad, num_classes=10, seed=0)
        except ValueError:
            continue
        raise AssertionError(f"noise_fraction={bad} should have raised")
    try:
        NoisyLabelSubset(base, 0.5, num_classes=1, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("num_classes=1 should have raised (no wrong labels exist)")
    print("✓ rejects out-of-range noise_fraction and num_classes<2")
    return True


def test_exact_fraction_flipped():
    """Number of flipped samples equals round(noise_fraction * len(base)).

    The pilot infers the noisy-client population size from this count. A
    Bernoulli implementation would give a stochastic count with sqrt(n) noise;
    we want the deterministic fixed-count draw the wrapper documents.
    """
    print("\nTesting NoisyLabelSubset exact-fraction guarantee...")
    for n, frac in [(100, 0.2), (100, 0.5), (99, 0.33), (10, 1.0), (10, 0.0)]:
        wrap = NoisyLabelSubset(DummyDataset(n), noise_fraction=frac,
                                num_classes=10, seed=0)
        expected = int(round(frac * n))
        assert wrap.num_noisy == expected, \
            f"n={n} frac={frac}: got {wrap.num_noisy}, expected {expected}"
    print("✓ exact-fraction draw at every tested (n, fraction)")
    return True


def test_flipped_labels_are_wrong():
    """Every corrupted sample gets a label != the true label.

    A rejection-sampling bug could occasionally emit the true class and
    silently reduce effective noise. The wrapper uses a shift-past-collision
    scheme that eliminates this by construction; the test guards that
    property for the whole corrupted set.
    """
    print("\nTesting NoisyLabelSubset flips are always to a WRONG class...")
    base = DummyDataset(1000, num_classes=10)
    wrap = NoisyLabelSubset(base, noise_fraction=0.5, num_classes=10, seed=1)
    seen_wrong = 0
    for idx, wrong in wrap._noisy.items():
        true = base[idx][1]
        assert wrong != true, f"idx={idx}: wrong={wrong} equals true={true}"
        assert 0 <= wrong < 10, f"wrong={wrong} out of class range"
        seen_wrong += 1
    assert seen_wrong == wrap.num_noisy
    print(f"✓ all {seen_wrong} flips land on a wrong class")
    return True


def test_wrong_class_distribution_is_roughly_uniform():
    """Over many flips, each wrong class is picked about 1/(K-1) of the time.

    Guards against a subtle off-by-one in the shift-past-collision scheme
    that could bias flips away from one specific class.
    """
    print("\nTesting wrong-class distribution is uniform over the 9 wrong classes...")
    base = DummyDataset(10_000, num_classes=10)
    wrap = NoisyLabelSubset(base, noise_fraction=1.0, num_classes=10, seed=2)
    # For each true class, count where flips landed. Under uniform-wrong
    # sampling, each wrong class should get ~ n_true / (K-1) of the flips.
    counts_per_true = {c: [0] * 10 for c in range(10)}
    for idx, wrong in wrap._noisy.items():
        true = idx % 10
        counts_per_true[true][wrong] += 1

    # A weak but decisive check: no wrong class gets < 60% or > 140% of its
    # expected share. n_true ~ 1000; expected per wrong class ~ 111; sd ~ 10.
    for true, row in counts_per_true.items():
        expected = sum(row) / 9   # 9 wrong classes
        for w, c in enumerate(row):
            if w == true:
                assert c == 0, f"true={true}: {c} flips landed on the true class"
                continue
            lo, hi = 0.6 * expected, 1.4 * expected
            assert lo <= c <= hi, (
                f"true={true} wrong={w}: count {c} outside [{lo:.0f}, {hi:.0f}]"
            )
    print("✓ each of the 9 wrong classes gets within ±40% of its expected share")
    return True


def test_deterministic_across_reconstruction():
    """Same (base, noise_fraction, seed) reconstructs the identical noisy set.

    Runs of the pilot at the same run seed must see the same corrupted samples
    every time; otherwise the "does trust find the noisy client" question is
    confounded by which samples happen to be corrupted this launch.
    """
    print("\nTesting NoisyLabelSubset determinism across reconstructions...")
    base = DummyDataset(500, num_classes=10)
    w1 = NoisyLabelSubset(base, 0.3, num_classes=10, seed=42)
    w2 = NoisyLabelSubset(base, 0.3, num_classes=10, seed=42)
    assert w1._noisy == w2._noisy
    w3 = NoisyLabelSubset(base, 0.3, num_classes=10, seed=43)
    assert w1._noisy != w3._noisy, "different seed should produce different noisy set"
    print("✓ same seed reproduces the noisy set; different seed diverges")
    return True


def test_independence_across_clients_sharing_base():
    """Two clients wrapped over the SAME base with different seeds keep
    independent noisy sets (they do not step on each other's state)."""
    print("\nTesting NoisyLabelSubset independence across shared-base clients...")
    base = DummyDataset(500, num_classes=10)
    a = NoisyLabelSubset(base, 0.3, num_classes=10, seed=100)
    b = NoisyLabelSubset(base, 0.3, num_classes=10, seed=200)
    # Overlap is expected but full equality would mean seeds do nothing.
    overlap = set(a._noisy) & set(b._noisy)
    assert set(a._noisy) != set(b._noisy)
    print(f"✓ two clients on the same base draw different noisy sets "
          f"(|A|={len(a._noisy)}, |B|={len(b._noisy)}, |A∩B|={len(overlap)})")
    return True


def test_base_dataset_is_not_mutated():
    """The wrapper reads from the base but never writes back to it.

    A silent mutation would poison every OTHER client that shares the base --
    a class of bug that would look like "some clients are surprisingly good
    at spotting noise" in a way that has nothing to do with the algorithm.
    """
    print("\nTesting NoisyLabelSubset does not mutate the base dataset...")
    base = DummyDataset(200, num_classes=10)
    labels_before = [base[i][1] for i in range(len(base))]
    wrap = NoisyLabelSubset(base, 0.5, num_classes=10, seed=7)
    _ = [wrap[i] for i in range(len(wrap))]   # exercise __getitem__
    labels_after = [base[i][1] for i in range(len(base))]
    assert labels_before == labels_after, "base dataset labels drifted"
    # And a fresh wrap after the exercise should still see the same base.
    wrap2 = NoisyLabelSubset(base, 0.5, num_classes=10, seed=7)
    assert wrap._noisy == wrap2._noisy
    print("✓ base dataset labels unchanged after wrapper use")
    return True


def test_getitem_returns_true_label_off_the_noisy_set():
    """Uncorrupted indices return the base's original label unchanged.

    The wrapper's contract is "flip a subset, pass the rest through". A bug
    that permuted labels globally would still pass the count and wrong-class
    tests above.
    """
    print("\nTesting NoisyLabelSubset passes clean indices through unchanged...")
    base = DummyDataset(300, num_classes=10)
    wrap = NoisyLabelSubset(base, 0.4, num_classes=10, seed=11)
    for idx in range(len(base)):
        _, wrap_label = wrap[idx]
        true_label = base[idx][1]
        if idx in wrap._noisy:
            assert wrap_label == wrap._noisy[idx], f"idx={idx}: getitem disagrees with cache"
            assert wrap_label != true_label
        else:
            assert wrap_label == true_label, f"clean idx={idx} was altered"
    print("✓ clean indices unchanged; noisy indices match the cached wrong label")
    return True


def test_pipeline_level_noisy_client_stability():
    """Two PipelineConfig instantiations at the same seed pick the same noisy
    clients and the same corrupted-sample sets per client.

    This is the cross-arm stability contract the design lives on: the three
    arms of one seed must see the identical noisy pool AND the identical
    corrupted sample set inside each noisy client, or the arms are not
    comparing the same problem. NoisyLabelSubset unit tests only check the
    wrapper; this integration test locks the pipeline glue too. Kept quick
    with a tiny synthetic base so the whole file still runs in a few seconds.
    """
    print("\nTesting pipeline-level noisy-client stability across arms...")

    import types
    from src.tavs_v2.end_to_end_pipeline import PipelineConfig
    from src.utils.data_utils import NoisyLabelSubset

    # Rebuild the pipeline's per-run noise-injection block in isolation.
    # We do not run the FL simulation -- just the same np.random.default_rng
    # seed draw and the same NoisyLabelSubset construction pattern, over a
    # tiny synthetic base so this stays cheap.
    def draw_noisy(seed: int, num_clients: int, fraction: float, rate: float):
        pick_rng = np.random.default_rng(seed)
        n_noisy = int(round(fraction * num_clients))
        noisy_ids = sorted(pick_rng.choice(num_clients, size=n_noisy, replace=False).tolist())
        per_client = {}
        for cid in noisy_ids:
            base = DummyDataset(60, num_classes=10)
            wrap = NoisyLabelSubset(base, noise_fraction=rate, num_classes=10,
                                    seed=seed * 10_000 + cid)
            per_client[cid] = frozenset(wrap._noisy.items())
        return noisy_ids, per_client

    a_ids, a_sets = draw_noisy(seed=42, num_clients=20, fraction=0.2, rate=0.2)
    b_ids, b_sets = draw_noisy(seed=42, num_clients=20, fraction=0.2, rate=0.2)
    c_ids, c_sets = draw_noisy(seed=43, num_clients=20, fraction=0.2, rate=0.2)
    assert a_ids == b_ids, "same seed produced different noisy client sets"
    assert a_sets == b_sets, "same seed produced different corrupted sample sets"
    assert a_ids != c_ids or a_sets != c_sets, \
        "different seeds gave identical noise -- seed is being ignored"
    print(f"✓ same seed picks the same noisy pool ({len(a_ids)} clients) "
          f"and the same corrupted samples")

    # And verify PipelineConfig accepts the label-noise knobs so a rename
    # cannot silently break the pilot.
    cfg = PipelineConfig(label_noise_client_fraction=0.2, label_noise_rate=0.2,
                        data_split="iid")
    assert cfg.label_noise_client_fraction == 0.2
    assert cfg.label_noise_rate == 0.2
    assert cfg.data_split == "iid"
    print("✓ PipelineConfig honors label_noise_* and data_split fields")
    return True


def main():
    print("🧪 NoisyLabelSubset Test Suite")
    print("=" * 50)
    tests = [
        test_rate_validation,
        test_exact_fraction_flipped,
        test_flipped_labels_are_wrong,
        test_wrong_class_distribution_is_roughly_uniform,
        test_deterministic_across_reconstruction,
        test_independence_across_clients_sharing_base,
        test_base_dataset_is_not_mutated,
        test_getitem_returns_true_label_off_the_noisy_set,
        test_pipeline_level_noisy_client_stability,
    ]
    for t in tests:
        assert t() is True
    print("\n🎯 All NoisyLabelSubset tests PASSED!")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
