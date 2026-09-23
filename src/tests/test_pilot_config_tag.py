#!/usr/bin/env python3
"""
Tests for `experiments.pilot_skip_comparison._config_tag`.

Its whole purpose is preventing accidental overwrite of existing pilot
results when a knob changes. If it produces the same tag for two runs that
are not byte-identical in config, one silently overwrites the other. If it
produces different tags for two argparse-equivalent invocations, the
--skip-completed cache is broken. Both are subtle; both matter; both are
tested here.
"""
import importlib.util
import os
import sys


def _load_pilot_module():
    """Import experiments/pilot_skip_comparison.py as a module for
    direct access to `_config_tag` without invoking main()."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "experiments", "pilot_skip_comparison.py")
    spec = importlib.util.spec_from_file_location("pilot_skip_comparison", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Args:
    """Minimal argparse-like object with the defaults `_config_tag` reads."""
    def __init__(self, **overrides):
        self.data_split = "iid"
        self.data_alpha = 0.3
        self.num_clients = 100
        self.clients_per_round = 20
        self.noisy_client_fraction = 0.4
        self.label_noise_rate = 0.3
        self.label_noise_type = "uniform"
        self.label_noise_num_classes = 10
        for k, v in overrides.items():
            setattr(self, k, v)


def test_historical_default_tag_is_preserved():
    """The 100-clients / 20-cpr / uniform-noise / K=10 combination MUST
    produce `split-iid_noiseC40_R30`. That is the tag under which the
    n=10 5-arm pilot results already live; any drift would orphan them
    from --skip-completed reuse."""
    m = _load_pilot_module()
    tag = m._config_tag(_Args())
    assert tag == "split-iid_noiseC40_R30", (
        f"historical default tag drifted: got {tag!r}, "
        f"expected 'split-iid_noiseC40_R30'"
    )
    print("✓ historical default 100/20/uniform/K10 tag preserved: 'split-iid_noiseC40_R30'")
    return True


def test_priority_configurations_get_distinct_tags():
    """The three planned upcoming experiments each land in a fresh dir.
    Regression test the exact strings so a future refactor of _config_tag
    catches unintended path collapses."""
    m = _load_pilot_module()
    cases = {
        "priority_2_50clients":
            (_Args(num_clients=50, clients_per_round=10),
             "split-iid_N50_C10_noiseC40_R30"),
        "priority_3_pairflip":
            (_Args(noisy_client_fraction=0.4, label_noise_rate=0.25,
                   label_noise_type="pairflip"),
             "split-iid_noiseC40_R25_type-pairflip"),
        "priority_2_plus_3":
            (_Args(num_clients=50, clients_per_round=10,
                   label_noise_type="pairflip"),
             "split-iid_N50_C10_noiseC40_R30_type-pairflip"),
        "priority_1b_2pct":
            (_Args(noisy_client_fraction=0.2, label_noise_rate=0.1),
             "split-iid_noiseC20_R10"),
        "priority_1b_4_5pct":
            (_Args(noisy_client_fraction=0.3, label_noise_rate=0.15),
             "split-iid_noiseC30_R15"),
    }
    seen = set()
    for label, (args, expected) in cases.items():
        got = m._config_tag(args)
        assert got == expected, f"{label}: expected {expected!r} got {got!r}"
        assert got not in seen, f"{label}: tag {got!r} collides with an earlier priority"
        seen.add(got)
    print(f"✓ five planned experiments produce distinct expected tags")
    return True


def test_partial_noise_config_raises():
    """Regression: --noisy-client-fraction 0.4 --label-noise-rate 0 used to
    silently produce `split-iid_clean`, colliding with genuinely clean
    runs AND with other partial configs. The new guard raises on any
    mismatched pair."""
    m = _load_pilot_module()
    # Both zero -> "clean" (fine).
    tag_clean = m._config_tag(_Args(noisy_client_fraction=0.0, label_noise_rate=0.0))
    assert tag_clean.endswith("_clean"), f"pure clean should tag as clean; got {tag_clean}"

    # One knob positive, the other zero -> refuse.
    for a, b in [(0.4, 0.0), (0.0, 0.3)]:
        try:
            m._config_tag(_Args(noisy_client_fraction=a, label_noise_rate=b))
        except ValueError as e:
            assert "noise" in str(e).lower()
            continue
        raise AssertionError(f"_config_tag should raise for ({a}, {b})")
    print("✓ partial (one-zero) noise config refused; genuine clean allowed")
    return True


def main():
    print("🧪 pilot _config_tag Test Suite")
    print("=" * 60)
    tests = [
        test_historical_default_tag_is_preserved,
        test_priority_configurations_get_distinct_tags,
        test_partial_noise_config_raises,
    ]
    for t in tests:
        assert t() is True
    print("\n🎯 All pilot _config_tag tests PASSED!")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
