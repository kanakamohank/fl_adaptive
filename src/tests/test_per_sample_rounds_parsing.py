#!/usr/bin/env python3
"""Regression test for the multi-round per-sample dump list parsing.

The oracle per-sample-loss dump used to accept a single round int; now it
accepts a tuple of rounds (needed to answer the bimodal-vs-shift mechanism
question with ~100 noisy per-sample dumps across phases). Server encodes
the tuple as a comma-separated string because Flower's proto doesn't
serialize list values in config. Client parses it back to a set of ints
at _process_tavs_config time, and fit() dumps only when the current round
is in that set.

Why this lives in its own file (not test_tavs_esp_strategy.py): that
file mocks `sys.modules['flwr.common']`, and importing TAVSFlowerClient
pulls `from flwr.client import NumPyClient` which breaks under the mock.
This file does NOT mock flwr.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def test_per_sample_rounds_parses_list_from_server_config():
    from src.clients.tavs_flower_client import TAVSFlowerClient, TAVSClientConfig
    cfg = TAVSClientConfig(client_id="c0", client_type="honest",
                           model_type="cifar_cnn")
    client = TAVSFlowerClient(config=cfg)

    # Multiple rounds, comma-separated. Whitespace tolerated.
    client._process_tavs_config({"log_per_sample_losses_at_rounds": "5, 10,15 , 20,30"})
    assert client._log_per_sample_losses_at_rounds == {5, 10, 15, 20, 30}

    # Empty string disables.
    client._process_tavs_config({"log_per_sample_losses_at_rounds": ""})
    assert client._log_per_sample_losses_at_rounds == set()

    # Garbage tokens silently dropped (defensive -- a bad token must not
    # take down the whole fit path).
    client._process_tavs_config({"log_per_sample_losses_at_rounds": "5, not_a_number, 10"})
    assert client._log_per_sample_losses_at_rounds == {5, 10}

    # Missing key preserves previous value.
    client._process_tavs_config({"log_per_sample_losses_at_rounds": "7"})
    client._process_tavs_config({})
    assert client._log_per_sample_losses_at_rounds == {7}, (
        "missing key must preserve the previous rounds set so a stale "
        "configure_fit payload cannot silently re-enable logging"
    )
    print("✓ per-sample-rounds config parses comma-separated string -> set")
    return True


def main():
    print("🧪 per-sample-rounds parsing regression suite")
    print("=" * 60)
    assert test_per_sample_rounds_parses_list_from_server_config() is True
    print("\n🎯 All per-sample-rounds parsing tests PASSED!")


if __name__ == "__main__":
    main()
