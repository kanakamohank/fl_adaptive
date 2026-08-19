#!/usr/bin/env python3
"""
Plot how clients are distributed across trust tiers over rounds.

Reads tier_evolution from a pipeline_results.json. Tier is recorded per client
per round by the scheduler:

    Tier 1  trust < theta_low                     always verified
    Tier 2  theta_low <= trust < theta_high       verified subject to budget
    Tier 3  trust >= theta_high AND clean streak  promoted (the saving)

Only meaningful for runs produced after tier_evolution began recording real
tiers. Earlier runs stored `3 if promoted else 1` -- a promoted/verified flag
using tier numerals -- so they show only 1s and 3s and never a 2. The script
detects that and says so rather than plotting a misleading chart.

Usage:
    python -m experiments.plot_tier_distribution <path/to/pipeline_results.json>
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TIER_LABELS = {1: "Tier 1 (verified)", 2: "Tier 2 (budget-limited)", 3: "Tier 3 (promoted)"}
TIER_COLORS = {1: "#c0392b", 2: "#e67e22", 3: "#27ae60"}


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <pipeline_results.json>")
    path = Path(sys.argv[1])
    tier_evolution = json.loads(path.read_text())["tier_evolution"]
    if not tier_evolution:
        sys.exit("tier_evolution is empty; nothing to plot.")

    num_rounds = max(len(v) for v in tier_evolution.values())
    # Count clients per tier per round.
    counts = [Counter(v[r] for v in tier_evolution.values() if r < len(v))
              for r in range(num_rounds)]
    series = {t: [c.get(t, 0) for c in counts] for t in (1, 2, 3)}

    if not any(series[2]):
        print("WARNING: no Tier 2 in any round. Either this run predates real "
              "tier logging (old format stored 3=promoted, 1=verified), or no "
              "client ever held trust between theta_low and theta_high.")

    rounds = range(1, num_rounds + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.stackplot(rounds, [series[t] for t in (1, 2, 3)],
                  labels=[TIER_LABELS[t] for t in (1, 2, 3)],
                  colors=[TIER_COLORS[t] for t in (1, 2, 3)])
    ax1.set_ylabel("clients")
    ax1.set_title(f"Tier distribution across rounds  ({path.parent.name})")
    ax1.legend(loc="upper left")

    # Tier 3 alone: this is the only tier that produces a verification saving,
    # so its trajectory is the one that says whether TAVS is doing anything.
    ax2.plot(rounds, series[3], color=TIER_COLORS[3], marker="o", ms=3)
    ax2.set_xlabel("round")
    ax2.set_ylabel("Tier 3 clients")
    ax2.set_title("Tier 3 population (the source of the verification saving)")
    ax2.grid(alpha=0.3)

    out = path.parent / "tier_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"total client-rounds per tier: "
          f"{ {TIER_LABELS[t]: sum(series[t]) for t in (1, 2, 3)} }")
    print(f"Plot: {out}")


if __name__ == "__main__":
    main()
