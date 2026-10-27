#!/usr/bin/env python3
"""Does the trust EMA track NOISE, or does it track CLASS IMBALANCE?

Motivation. On the non-IID run (Dirichlet 0.3, CIFAR-10N worst, 5 seeds)
TAVS beat random-skip by +3.5 points of accuracy (t=4.84, 5/5 seeds) while
its trust scores barely separated noisy from clean clients: rank-of-noisy
9.8/20 against a chance baseline of 8/20, trust gap +0.032 versus +0.436 on
the IID run. Those two facts cannot both support the story that TAVS wins by
spotting noisy clients.

FedNoRo (IJCAI 2023, Sec. 1) supplies a candidate explanation: under class
imbalance a CLEAN client holding rare or hard classes produces a wide loss
spread that mimics a noisy one. Our trust is driven by exactly that pooled
loss variance. So trust may be ranking clients by how skewed their label
distribution is rather than by how noisy their labels are -- and
down-weighting skewed clients could plausibly stabilise training for reasons
unrelated to noise.

This script decides between those two accounts from data already logged. For
every client it pairs the final trust score against:
  * is_noisy        -- ground truth from the pipeline's noisy client list
  * imbalance       -- 1 - normalised Shannon entropy of the client's label
                       counts (0.0 = perfectly uniform, 1.0 = one class only),
                       computed from per_class_pretrain_count

and reports Spearman rank correlations of trust against each, plus a 2x2
cell table so the two effects can be read apart by eye.

Spearman (rank) rather than Pearson: trust is squashed by the EMA and by
theta_low, so the relationship need not be linear, and only the ordering is
being claimed.

Reading the result:
  * |rho(trust, noisy)| large, |rho(trust, imbalance)| small
        -> trust tracks noise. The published story holds.
  * |rho(trust, imbalance)| large, |rho(trust, noisy)| small
        -> trust tracks imbalance. The accuracy win is real but
           mis-attributed, and the mechanism claim must change.
  * both small
        -> trust tracks neither; the accuracy win comes from somewhere we
           have not looked at yet.

The IID run is the control. Under an IID partition every client holds a near
uniform label distribution, so imbalance barely varies and its correlation
with trust should collapse toward zero. If it does not, the imbalance
measure itself is suspect rather than the finding.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def _ranks(xs):
    """Ranks with ties averaged, as Spearman requires."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 3:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return (num / (dx * dy)) if dx > 0 and dy > 0 else float("nan")


def imbalance_from_counts(counts):
    """1 - normalised Shannon entropy of the label distribution.

    0.0 = perfectly uniform across the classes present in the dataset,
    1.0 = every sample in a single class. Normalised by log(num_classes)
    so clients holding different numbers of classes stay comparable.
    """
    total = sum(counts)
    if total <= 0:
        return None
    C = len(counts)
    if C < 2:
        return None
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return 1.0 - (h / math.log(C))


def collect(root: Path):
    """Return one row per (seed, client): trust, is_noisy, imbalance."""
    rows = []
    for pr in sorted(root.rglob("tavs_skip_seed*/pipeline_results.json")):
        d = json.loads(pr.read_text())
        fts = d.get("final_trust_state") or {}
        osh = fts.get("oracle_signal_history") or {}
        cid_map = fts.get("cid_to_client_config_id", {})
        noisy = set(d.get("noisy_client_config_ids", []))
        trust = fts.get("trust_state", {})
        if not (osh and noisy and trust):
            continue

        # A client's label counts are fixed across rounds (same local
        # dataset), so the first appearance is representative.
        counts_by_cid = {}
        for _r, entries in osh.items():
            for cid, e in entries.items():
                if cid in counts_by_cid:
                    continue
                c = e.get("per_class_pretrain_count")
                if c:
                    counts_by_cid[cid] = c

        for cid, t in trust.items():
            counts = counts_by_cid.get(cid)
            if not counts:
                continue
            imb = imbalance_from_counts(counts)
            if imb is None:
                continue
            rows.append({
                "trust": float(t),
                "noisy": 1.0 if cid_map.get(cid, cid) in noisy else 0.0,
                "imbalance": imb,
            })
    return rows


def report(root: Path, label: str):
    rows = collect(root)
    if len(rows) < 10:
        print(f"\n{label}\n  insufficient data ({len(rows)} client-rows)")
        return

    trust = [r["trust"] for r in rows]
    noisy = [r["noisy"] for r in rows]
    imb = [r["imbalance"] for r in rows]

    rho_noise = spearman(trust, noisy)
    rho_imb = spearman(trust, imb)

    print(f"\n{label}")
    print(f"  {len(rows)} client-rows "
          f"({int(sum(noisy))} noisy / {len(rows) - int(sum(noisy))} clean)")
    print(f"  imbalance spread: min={min(imb):.3f} median={statistics.median(imb):.3f} "
          f"max={max(imb):.3f}")
    print(f"  rho(trust, is_noisy)   = {rho_noise:+.3f}   "
          f"(negative = noisy clients get LOWER trust, which is intended)")
    print(f"  rho(trust, imbalance)  = {rho_imb:+.3f}   "
          f"(negative = skewed clients get LOWER trust)")

    # 2x2: split on the median imbalance so the two effects can be read apart.
    med = statistics.median(imb)
    cells = {(n, b): [] for n in (0.0, 1.0) for b in (0, 1)}
    for r in rows:
        cells[(r["noisy"], 1 if r["imbalance"] > med else 0)].append(r["trust"])
    print(f"\n  mean trust by cell (imbalance split at median {med:.3f}):")
    print(f"    {'':<10}{'balanced':>12}{'imbalanced':>14}")
    for n, name in ((0.0, "clean"), (1.0, "noisy")):
        bal = cells[(n, 0)]
        imbl = cells[(n, 1)]
        b = f"{statistics.mean(bal):.3f} (n={len(bal)})" if bal else "--"
        i = f"{statistics.mean(imbl):.3f} (n={len(imbl)})" if imbl else "--"
        print(f"    {name:<10}{b:>12}{i:>14}")

    # Effect sizes read straight off the cell table: how much does trust move
    # when only one factor changes?
    def mean_or_none(k):
        v = cells[k]
        return statistics.mean(v) if v else None
    cb, ci = mean_or_none((0.0, 0)), mean_or_none((0.0, 1))
    nb, ni = mean_or_none((1.0, 0)), mean_or_none((1.0, 1))
    if None not in (cb, ci, nb, ni):
        noise_effect = ((nb - cb) + (ni - ci)) / 2.0
        imb_effect = ((ci - cb) + (ni - nb)) / 2.0
        print(f"\n  trust change from NOISE alone     : {noise_effect:+.4f}")
        print(f"  trust change from IMBALANCE alone : {imb_effect:+.4f}")
        bigger = "NOISE" if abs(noise_effect) > abs(imb_effect) else "IMBALANCE"
        print(f"  -> trust is driven more by {bigger}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", type=Path)
    args = ap.parse_args()

    if args.roots:
        runs = [(r, r.name) for r in args.roots]
    else:
        base = Path("results")
        runs = [
            (base / "tavs_integration_v3_noniid_perclass",
             "NON-IID dirichlet 0.3, CIFAR-10N worst, r40 ep5  [the puzzle]"),
            (base / "tavs_integration_v2_iid_worst_twagg",
             "IID, CIFAR-10N worst, r40 ep5  [control: imbalance should not vary]"),
        ]

    print("Trust vs noise vs class imbalance (Spearman rank correlation)")
    print("=" * 72)
    for root, label in runs:
        if not root.exists():
            print(f"\n{label}\n  (missing: {root})")
            continue
        report(root, label)


if __name__ == "__main__":
    main()
