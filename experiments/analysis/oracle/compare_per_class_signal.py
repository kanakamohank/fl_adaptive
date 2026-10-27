#!/usr/bin/env python3
"""Head-to-head of three noisy-client signals on identical logged data.

Motivation: our pooled per-sample loss variance scored AUROC 0.436 under a
Dirichlet(0.3) partition -- worse than chance, the signal inverted. FedNoRo
(IJCAI 2023, Sec. 1) diagnoses exactly this failure: under class imbalance a
clean client holding mostly hard/rare classes produces large loss values that
look like a noisy client's. Their fix is to stop pooling samples and compute
the indicator PER CLASS, on the grounds that "though inter-class samples are
non-IID under class imbalance, intra-class samples are more IID".

We already log `per_class_pretrain_loss` (per-class mean loss, NaN for absent
classes), so FedNoRo's signal is reconstructable from runs already on disk --
no retraining needed.

Signals compared, all from the same oracle_signal_history entries:
  pooled_var   -- ours: variance of per-sample losses over the whole client
  pooled_mean  -- the naive global indicator FedNoRo criticises (the
                  Xu et al. / Wang et al. style client-wise average)
  fednoro      -- per-class mean loss vector, processed as in FedNoRo Sec 4.1:
                    1. missing class c -> min over cohort of l^c (their rule:
                       an absent class has no noisy samples, and by the
                       small-loss trick cleaner => smaller loss)
                    2. min-max normalise each class across the cohort, so no
                       single class dominates
                    3. score = L2 norm of the normalised vector (FedNoRo
                       splits on a GMM whose noisy component is the
                       "large norm mean vector"; the norm is that ordering)

Normalisation is done within each round's cohort, which is the set of clients
observable at that moment. FedNoRo has every client online each round; we
sample 10 of 50, so the cohort is the faithful analogue.

Metric is pairwise AUROC (probability a random noisy client outranks a random
clean one), pooled over rounds and averaged over seeds. FedNoRo reports
recall/precision/match-ratio off a GMM instead; AUROC is used here because it
is threshold-free and lets the three signals be compared without also
choosing a clustering method.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def pairwise_counts(scored):
    """scored: list of (score, is_noisy) from ONE round. Returns (wins, total).

    Pairs must be formed WITHIN a round. The raw loss statistics drift
    upward over training (a clean client at round 38 has higher variance
    than a noisy client at round 3), so pooling raw scores across rounds and
    then ranking compares incomparable quantities and understates every
    un-normalised signal. FedNoRo's min-max step is per-cohort and would be
    silently immune to that, making a cross-round comparison unfair to the
    pooled signals.
    """
    wins = total = 0
    for i, (vi, ni) in enumerate(scored):
        for vj, nj in scored[i + 1:]:
            if ni == nj:
                continue
            total += 1
            if vi == vj:
                wins += 0.5
            elif (vi > vj and ni) or (vj > vi and nj):
                wins += 1
    return wins, total


def fednoro_scores(cohort):
    """cohort: list of (cid, per_class_loss_list). Returns {cid: score}.

    Implements FedNoRo Sec 4.1 steps 1-3 (see module docstring).
    """
    if not cohort:
        return {}
    num_classes = len(cohort[0][1])

    # Step 1: fill a missing class with the cohort minimum for that class.
    per_class_present = {c: [] for c in range(num_classes)}
    for _cid, vec in cohort:
        for c, v in enumerate(vec):
            if v is not None and not math.isnan(v):
                per_class_present[c].append(v)
    filled = {}
    for cid, vec in cohort:
        row = []
        for c, v in enumerate(vec):
            if v is None or math.isnan(v):
                row.append(min(per_class_present[c]) if per_class_present[c] else 0.0)
            else:
                row.append(v)
        filled[cid] = row

    # Step 2: min-max normalise each class across the cohort.
    lo = [min(filled[cid][c] for cid in filled) for c in range(num_classes)]
    hi = [max(filled[cid][c] for cid in filled) for c in range(num_classes)]
    norm = {}
    for cid, row in filled.items():
        norm[cid] = [
            ((row[c] - lo[c]) / (hi[c] - lo[c])) if hi[c] > lo[c] else 0.0
            for c in range(num_classes)
        ]

    # Step 3: L2 norm of the normalised vector.
    return {cid: math.sqrt(sum(v * v for v in row)) for cid, row in norm.items()}


def analyse_seed(path):
    d = json.loads(path.read_text())
    fts = d["final_trust_state"]
    osh = fts.get("oracle_signal_history") or {}
    cid_map = fts.get("cid_to_client_config_id", {})
    noisy = set(d.get("noisy_client_config_ids", []))
    if not osh or not noisy:
        return None

    acc = {k: [0.0, 0] for k in
           ("pooled_var", "pooled_mean", "fednoro", "perclass_var")}
    for r_key, entries in osh.items():
        cohort_pc = []      # per-class MEAN loss  (FedNoRo's statistic)
        cohort_pcv = []     # per-class VARIANCE   (their decomposition, ours)
        rows = []
        for cid, e in entries.items():
            is_noisy = cid_map.get(cid, cid) in noisy
            rows.append((cid, e, is_noisy))
            pc = e.get("per_class_pretrain_loss")
            if pc:
                cohort_pc.append((cid, pc))
            pcv = e.get("per_class_pretrain_var")
            if pcv:
                cohort_pcv.append((cid, pcv))
        fn = fednoro_scores(cohort_pc)
        # Same FedNoRo processing applied to the variance vector. The
        # fill-missing rule still holds: an absent class contributes no noisy
        # samples, so the cohort minimum is the right stand-in.
        fnv = fednoro_scores(cohort_pcv)

        round_scores = {k: [] for k in acc}
        for cid, e, is_noisy in rows:
            v = e.get("pretrain_loss_var")
            if v is not None:
                round_scores["pooled_var"].append((float(v), is_noisy))
            m = e.get("pretrain_loss_mean")
            if m is not None:
                round_scores["pooled_mean"].append((float(m), is_noisy))
            if cid in fn:
                round_scores["fednoro"].append((fn[cid], is_noisy))
            if cid in fnv:
                round_scores["perclass_var"].append((fnv[cid], is_noisy))

        for k, scored in round_scores.items():
            w, t = pairwise_counts(scored)
            acc[k][0] += w
            acc[k][1] += t

    return {
        k: (w / t) if t else float("nan") for k, (w, t) in acc.items()
    } | {"n_noisy": len(noisy)}


def analyse_run(root: Path, label: str):
    seeds = sorted(root.rglob("tavs_skip_seed*/pipeline_results.json"))
    if not seeds:
        return None
    per_signal = {"pooled_var": [], "pooled_mean": [],
                  "fednoro": [], "perclass_var": []}
    for p in seeds:
        res = analyse_seed(p)
        if res is None:
            continue
        for k in per_signal:
            if not math.isnan(res[k]):
                per_signal[k].append(res[k])
    if not per_signal["pooled_var"]:
        return None
    out = {"label": label, "n_seeds": len(per_signal["pooled_var"])}
    for k, vals in per_signal.items():
        # Absent for runs recorded before per_class_pretrain_var was logged;
        # report that as blank rather than inventing a number.
        out[k] = ((statistics.mean(vals),
                   statistics.stdev(vals) if len(vals) > 1 else 0.0)
                  if vals else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", type=Path,
                    help="Run directories. Default: the known IID/non-IID set.")
    args = ap.parse_args()

    if args.roots:
        runs = [(r, r.name) for r in args.roots]
    else:
        base = Path("results")
        runs = [
            (base / "tavs_integration_v1_ep5", "NON-IID dirichlet a=0.3, r20 ep5, random1"),
            (base / "tavs_integration_v1_null_signal_regime", "NON-IID dirichlet a=0.3, r20 ep2, random1"),
            (base / "tavs_integration_v1_iid_probe", "IID, r20 ep5, random1"),
            (base / "tavs_integration_v1_iid_test2", "IID, r40 ep5, random1"),
            (base / "tavs_integration_v2_iid_worst_twagg", "IID, r40 ep5, WORST"),
        ]

    print("Pairwise AUROC (noisy ranked above clean). 0.50 = chance.")
    print("perclass_var is blank for runs logged before that field existed.\n")
    header = (f"{'run':<44} {'seeds':>5}  {'pooled_var':>15} {'pooled_mean':>15} "
              f"{'fednoro':>15} {'perclass_var':>15}")
    print(header)
    print("-" * len(header))
    for root, label in runs:
        if not root.exists():
            print(f"{label:<44} {'--':>5}  (missing)")
            continue
        res = analyse_run(root, label)
        if res is None:
            print(f"{label:<44} {'--':>5}  (no oracle history)")
            continue
        def fmt(k):
            vals = res.get(k)
            if not vals:
                return "--"
            m, s = vals
            return f"{m:.3f} ± {s:.3f}"
        print(f"{label:<44} {res['n_seeds']:>5}  "
              f"{fmt('pooled_var'):>15} {fmt('pooled_mean'):>15} "
              f"{fmt('fednoro'):>15} {fmt('perclass_var'):>15}")


if __name__ == "__main__":
    main()
