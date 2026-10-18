#!/usr/bin/env python3
"""Generate 4 diagnostic plots from test 2 (iid, r40, ep5, CIFAR-10N random1,
pretrain_loss_var trust) for the TAVS-integration story.

Each plot shows strictly what the data supports. We do NOT plot "TAVS beats
random" bars -- at the current noise rate (~1.14% federation-wide), the
three arms land at 0.770 ± 0.004 and a comparison chart would mislead.

Outputs land next to the results directory so they travel with the data.
"""
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(
    "results/tavs_integration_v1_iid_test2/r40/"
    "split-iid_N50_C10_noiseC40_R17_type-cifar10n-random1_"
    "trust-pretrain-loss-var_pv-rescale-06-15_ep5_oracle"
)
OUT = ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = {
    "full_verify": ("skip00/full_verify_seed", "#d95f02", "Full verify"),
    "tavs_skip":   ("skip00/tavs_skip_seed",   "#1b9e77", "TAVS skip"),
    "random_skip": ("skip41/random_skip_seed", "#7570b3", "Random skip"),
}


def _load(arm_key, seed):
    stub, _, _ = ARMS[arm_key]
    p = ROOT / f"{stub}{seed}" / "pipeline_results.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------------------
# Plot 1: per-round pretrain_loss_var distribution, clean vs noisy
# ---------------------------------------------------------------------------
clean_by_round, noisy_by_round = {}, {}
for seed in range(1, 6):
    d = _load("tavs_skip", seed)
    if d is None:
        continue
    fts = d["final_trust_state"]
    osh = fts["oracle_signal_history"]
    cid_map = fts["cid_to_client_config_id"]
    noisy_ids = set(d["noisy_client_config_ids"])
    for r_str, entries in osh.items():
        r = int(r_str)
        for cid, e in entries.items():
            v = e.get("pretrain_loss_var")
            if v is None:
                continue
            is_n = cid_map.get(cid, cid) in noisy_ids
            (noisy_by_round if is_n else clean_by_round).setdefault(r, []).append(v)

rounds = sorted(clean_by_round.keys() & noisy_by_round.keys())
clean_mean = [statistics.mean(clean_by_round[r]) for r in rounds]
clean_std = [statistics.stdev(clean_by_round[r]) if len(clean_by_round[r]) > 1 else 0 for r in rounds]
noisy_mean = [statistics.mean(noisy_by_round[r]) for r in rounds]
noisy_std = [statistics.stdev(noisy_by_round[r]) if len(noisy_by_round[r]) > 1 else 0 for r in rounds]

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(rounds, clean_mean, color="#1b9e77", lw=2.0, label="Clean clients (mean across 5 seeds)")
ax.fill_between(rounds, [m - s for m, s in zip(clean_mean, clean_std)],
                [m + s for m, s in zip(clean_mean, clean_std)], color="#1b9e77", alpha=0.18)
ax.plot(rounds, noisy_mean, color="#d95f02", lw=2.0, label="Noisy clients (mean across 5 seeds)")
ax.fill_between(rounds, [m - s for m, s in zip(noisy_mean, noisy_std)],
                [m + s for m, s in zip(noisy_mean, noisy_std)], color="#d95f02", alpha=0.18)
ax.set_xlabel("Round")
ax.set_ylabel("Pretrain loss variance (per client, verified rounds only)")
ax.set_title("The signal: noisy clients show higher pretrain loss variance\n"
             "CIFAR-10N random1, IID, N=50/10, 40 rounds × 5 local epochs, 5 seeds")
ax.legend(loc="upper left", frameon=False)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "01_signal_pretrain_loss_var_over_rounds.png", dpi=150)
plt.close(fig)
print(f"wrote {OUT / '01_signal_pretrain_loss_var_over_rounds.png'}")


# ---------------------------------------------------------------------------
# Plot 2: Rank-of-noisy per seed, per arm, with hypergeometric baseline
# ---------------------------------------------------------------------------
rank_data = {k: [] for k in ARMS}
for arm_key in ARMS:
    for seed in range(1, 6):
        d = _load(arm_key, seed)
        if d is None:
            continue
        fts = d["final_trust_state"]
        cid_map = fts["cid_to_client_config_id"]
        noisy_ids = set(d["noisy_client_config_ids"])
        ts = fts.get("trust_state", {})
        by_cfg = {cid_map.get(cid, cid): v for cid, v in ts.items()}
        ranked = sorted(by_cfg.items(), key=lambda x: x[1])
        bottom20 = {cfg for cfg, _ in ranked[:20]}
        rank_data[arm_key].append(len(bottom20 & noisy_ids))

fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(1, 6)
width = 0.26
for i, (arm_key, (_, color, label)) in enumerate(ARMS.items()):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, rank_data[arm_key], width, color=color, label=label)
    for bar, v in zip(bars, rank_data[arm_key]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v), ha="center", fontsize=9)
ax.axhline(y=8, color="gray", linestyle="--", lw=1.2, alpha=0.7, label="Random baseline (8/20)")
ax.axhline(y=20, color="green", linestyle=":", lw=1.0, alpha=0.5, label="Perfect (20/20)")
ax.set_xlabel("Seed")
ax.set_ylabel("Of 20 lowest-trust clients, how many are actually noisy")
ax.set_title("Noise detection quality: trust mechanism ranks noisy clients to the bottom\n"
             "All three arms converge to similar detection quality (detection is pipeline-level, not TAVS-specific)")
ax.set_ylim(0, 22)
ax.set_xticks(x)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "02_rank_of_noisy_by_arm.png", dpi=150)
plt.close(fig)
print(f"wrote {OUT / '02_rank_of_noisy_by_arm.png'}")


# ---------------------------------------------------------------------------
# Plot 3: Accuracy trajectories per arm, mean ± stdev across 5 seeds
# ---------------------------------------------------------------------------
traj = {k: [] for k in ARMS}
for arm_key in ARMS:
    for seed in range(1, 6):
        d = _load(arm_key, seed)
        if d is not None:
            traj[arm_key].append(d["server_accuracies"])

fig, ax = plt.subplots(figsize=(9, 5.2))
for arm_key, (_, color, label) in ARMS.items():
    if not traj[arm_key]:
        continue
    T = min(len(a) for a in traj[arm_key])
    arr = np.array([a[:T] for a in traj[arm_key]])
    m = arr.mean(axis=0)
    s = arr.std(axis=0)
    xs = np.arange(T)
    ax.plot(xs, m, color=color, lw=2.0, label=label)
    ax.fill_between(xs, m - s, m + s, color=color, alpha=0.18)
ax.set_xlabel("Round")
ax.set_ylabel("Server test accuracy")
ax.set_title("All three arms converge together at the current noise rate (~1.14% federation-wide)\n"
             "Noise detection works, but current aggregation default (disable_trust_weighted_aggregation=True)\n"
             "gives every verified/promoted client equal num_examples weight regardless of routing")
ax.legend(loc="lower right", frameon=False)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "03_accuracy_trajectories.png", dpi=150)
plt.close(fig)
print(f"wrote {OUT / '03_accuracy_trajectories.png'}")


# ---------------------------------------------------------------------------
# Plot 4: BVD vs pretrain_loss_var AUROC per seed (parallel-logged in test 2)
# ---------------------------------------------------------------------------
def _per_seed_aurocs(d):
    """Return (bvd_auroc, pv_auroc) pooled across all rounds."""
    fts = d["final_trust_state"]
    osh = fts["oracle_signal_history"]
    cid_map = fts["cid_to_client_config_id"]
    noisy_ids = set(d["noisy_client_config_ids"])

    def _pooled(key, invert=False):
        wins = total = 0
        for r_str, entries in osh.items():
            pairs = []
            for cid, e in entries.items():
                v = e.get(key)
                if v is None:
                    continue
                is_n = cid_map.get(cid, cid) in noisy_ids
                pairs.append((v, is_n))
            for i, (vi, ni) in enumerate(pairs):
                for vj, nj in pairs[i + 1:]:
                    if ni != nj:
                        total += 1
                        # Pretrain var: higher = noisier -> noisy wins when higher
                        # BVD behavior_score: higher = more normal -> noisy wins when LOWER
                        if invert:
                            noisy_higher = (vj > vi and ni) or (vi > vj and nj)
                        else:
                            noisy_higher = (vi > vj and ni) or (vj > vi and nj)
                        if noisy_higher:
                            wins += 1
        return wins / total if total else float("nan")
    return _pooled("bvd_behavior_score", invert=True), _pooled("pretrain_loss_var")


bvd_vals, pv_vals = [], []
for seed in range(1, 6):
    d = _load("tavs_skip", seed)
    if d is None:
        continue
    b, p = _per_seed_aurocs(d)
    bvd_vals.append(b)
    pv_vals.append(p)

fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(1, len(bvd_vals) + 1)
width = 0.36
b1 = ax.bar(x - width / 2, bvd_vals, width, color="#7570b3", label="BVD behavior_score")
b2 = ax.bar(x + width / 2, pv_vals, width, color="#1b9e77", label="Pretrain loss variance")
for bars, vals in ((b1, bvd_vals), (b2, pv_vals)):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
ax.axhline(y=0.5, color="gray", linestyle="--", lw=1.2, alpha=0.7, label="Random (0.50)")
ax.set_xlabel("Seed")
ax.set_ylabel("Pairwise AUROC (noisy vs clean, pooled over all rounds)")
ax.set_title("Pretrain-loss-variance decisively outperforms BVD as a client-reliability signal\n"
             "Both logged in parallel from the same TAVS-skip runs; same config, same clients, same rounds")
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "04_bvd_vs_pretrain_var_auroc.png", dpi=150)
plt.close(fig)
print(f"wrote {OUT / '04_bvd_vs_pretrain_var_auroc.png'}")

print(f"\nAll plots in {OUT}")
