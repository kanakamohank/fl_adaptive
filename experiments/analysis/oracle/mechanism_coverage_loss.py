#!/usr/bin/env python3
"""Does trust weighting hurt under non-IID by destroying class coverage?

The finding to explain: under Dirichlet(0.3), weighting clients by a
reliability signal made accuracy WORSE than not weighting at all, and
improving the signal (AUROC 0.738 -> 0.895, detection 9.8/20 -> 17.0/20)
made it worse still -- 0.3076 unweighted, 0.3013 with the weak signal,
0.2937 with the strong one, 0.0/5 seeds beating baseline in either case.

Candidate mechanism: under a skewed partition a client may hold most or all
of the cohort's samples for some class. If that client is also noisy, the
weighting silences it and the aggregate loses that class entirely. The wrong
labels cost less than the missing coverage, and a sharper detector removes
such clients more efficiently -- which would explain why better detection
made things worse rather than better.

Two measurements, both from data already logged:

  A. COVERAGE DAMAGE. Per round, compare the cohort's class distribution
     weighted by behaviour score against the unweighted one. If weighting
     starves classes, the weighted distribution is more concentrated --
     lower entropy -- and the classes that lose most should be the rare ones.

  B. WHO GETS SILENCED. Correlate each client's behaviour score against how
     irreplaceable its data is (the share of its own samples that sit in
     classes where it supplies most of the cohort's data). If the
     down-weighted clients are the irreplaceable ones, that is the harm.

Behaviour scores are RECONSTRUCTED from the logged per-class variance using
the same helper the strategy ran, so these are the weights the aggregate
actually saw -- not a proxy. Only clients the strategy verified are scored,
since the override applies to that bucket; promoted clients are weighted by
bayesian(trust EMA) on a separate path and are excluded rather than guessed
at.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.tavs_v2.per_class_score import per_class_cohort_scores


def _entropy(dist):
    tot = sum(dist)
    if tot <= 0:
        return 0.0
    h = 0.0
    for v in dist:
        if v > 0:
            p = v / tot
            h -= p * math.log(p)
    return h


def _spearman(xs, ys):
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return float("nan")
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def analyse(root: Path, label: str):
    ent_u, ent_w = [], []
    rare_delta, common_delta = [], []
    irrep_all, behav_all, noisy_all = [], [], []

    for pr in sorted(root.rglob("tavs_skip_seed*/pipeline_results.json")):
        d = json.loads(pr.read_text())
        fts = d["final_trust_state"]
        osh = fts.get("oracle_signal_history") or {}
        cid_map = fts.get("cid_to_client_config_id", {})
        noisy = set(d.get("noisy_client_config_ids", []))
        assigns = fts.get("round_assignments", {})

        for r_key, entries in osh.items():
            verified = set(assigns.get(str(r_key), assigns.get(r_key, {})).get("verified", []))
            cohort, counts = [], {}
            for cid, e in entries.items():
                if verified and cid not in verified:
                    continue
                pcv = e.get("per_class_pretrain_var")
                pcc = e.get("per_class_pretrain_count")
                if pcv and pcc:
                    cohort.append((cid, pcv))
                    counts[cid] = pcc
            if len(cohort) < 3:
                continue

            scores = per_class_cohort_scores(cohort)
            behaviour = {k: 1.0 - v for k, v in scores.items()}   # what aggregate_fit used
            C = len(next(iter(counts.values())))

            # --- A. coverage damage
            unweighted = [sum(counts[k][c] for k in counts) for c in range(C)]
            weighted = [sum(behaviour.get(k, 0.0) * counts[k][c] for k in counts)
                        for c in range(C)]
            if sum(unweighted) <= 0 or sum(weighted) <= 0:
                continue
            ent_u.append(_entropy(unweighted))
            ent_w.append(_entropy(weighted))

            # Which classes lose share? Split at the median cohort class size.
            su, sw = sum(unweighted), sum(weighted)
            shares_u = [v / su for v in unweighted]
            shares_w = [v / sw for v in weighted]
            med = statistics.median(unweighted)
            for c in range(C):
                delta = shares_w[c] - shares_u[c]     # negative = class starved
                (rare_delta if unweighted[c] <= med else common_delta).append(delta)

            # --- B. who gets silenced
            for k in counts:
                n_i = sum(counts[k])
                if n_i <= 0 or k not in behaviour:
                    continue
                # Share of this client's own data sitting in classes where it
                # supplies most of the cohort's samples.
                irrep = 0.0
                for c in range(C):
                    if unweighted[c] > 0 and counts[k][c] > 0:
                        irrep += (counts[k][c] / n_i) * (counts[k][c] / unweighted[c])
                irrep_all.append(irrep)
                behav_all.append(behaviour[k])
                noisy_all.append(1.0 if cid_map.get(k, k) in noisy else 0.0)

    if not ent_u:
        print(f"\n{label}\n  no usable rounds")
        return

    print(f"\n{label}")
    print(f"  rounds analysed: {len(ent_u)}   client-rows: {len(irrep_all)}")

    du = statistics.mean(ent_u)
    dw = statistics.mean(ent_w)
    print(f"\n  A. COVERAGE DAMAGE")
    print(f"     class-distribution entropy, unweighted : {du:.4f}")
    print(f"     class-distribution entropy, weighted   : {dw:.4f}")
    print(f"     change                                 : {dw - du:+.4f}"
          f"   ({'coverage CONCENTRATED' if dw < du else 'coverage spread wider'})")
    print(f"     share change for RARE classes   : {statistics.mean(rare_delta):+.5f}")
    print(f"     share change for COMMON classes : {statistics.mean(common_delta):+.5f}")
    print(f"     -> {'rare classes lose more' if statistics.mean(rare_delta) < statistics.mean(common_delta) else 'rare classes do NOT lose more'}")

    print(f"\n  B. WHO GETS SILENCED")
    print(f"     rho(behaviour, irreplaceability) = {_spearman(behav_all, irrep_all):+.3f}"
          f"   (negative = irreplaceable clients down-weighted)")
    print(f"     rho(behaviour, is_noisy)         = {_spearman(behav_all, noisy_all):+.3f}"
          f"   (negative = noisy clients down-weighted, intended)")
    hi = [b for b, i in zip(behav_all, irrep_all) if i > statistics.median(irrep_all)]
    lo = [b for b, i in zip(behav_all, irrep_all) if i <= statistics.median(irrep_all)]
    print(f"     mean behaviour, most-irreplaceable half : {statistics.mean(hi):.4f}")
    print(f"     mean behaviour, least-irreplaceable half: {statistics.mean(lo):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", type=Path)
    args = ap.parse_args()
    if args.roots:
        runs = [(r, r.name) for r in args.roots]
    else:
        base = Path("results")
        runs = [
            (base / "tavs_integration_v5_noniid_perclass_trust",
             "NON-IID Dirichlet 0.3, CIFAR-10N worst, per-class trust signal"),
            (base / "tavs_integration_v2_iid_worst_twagg",
             "IID, CIFAR-10N worst  [contrast: coverage is redundant here]"),
        ]
    print("Mechanism test: does trust weighting destroy class coverage?")
    print("=" * 70)
    for root, label in runs:
        if not root.exists():
            print(f"\n{label}\n  (missing: {root})")
            continue
        analyse(root, label)


if __name__ == "__main__":
    main()
