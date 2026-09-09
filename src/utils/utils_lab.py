#!/usr/bin/env python3
"""
utils_lab: quick self-contained experiments that don't belong in the FL pipeline
but that anchor claims we make in writeups.

Currently:
  demo_lora_aggregation_gap()   -- the two-line proof that naive per-factor
                                   FedAvg over LoRA adapters is NOT the FedAvg
                                   of the effective updates. Motivation for the
                                   whole LoRA-FAIR / FLoRA line of work.

  measure_lora_aggregation_error() -- same phenomenon at realistic dimensions,
                                   sweeping the number of clients K. Produces
                                   a plot of ||naive - correct||_F / ||correct||_F.

Run:
    python -m src.utils.utils_lab                          # both, with plot
    python -m src.utils.utils_lab --no-plot                # numeric only
"""
import argparse
import os
from typing import List, Tuple

import numpy as np


def demo_lora_aggregation_gap() -> None:
    """
    Minimal example: two clients, rank 1, single dimension.

    Client 1 moves parameter [0,0] to 1.  Client 2 moves parameter [1,1] to 1.
    The FedAvg-of-updates target has rank 2 (it moves both diagonal entries);
    the naive per-factor FedAvg has rank 1 (it moves the whole 2x2 block by 0.25),
    losing information about which client wanted to move which coordinate.
    """
    B1, A1 = np.array([[1.0], [0.0]]),   np.array([[1.0, 0.0]])
    B2, A2 = np.array([[0.0], [1.0]]),   np.array([[0.0, 1.0]])

    target = 0.5 * (B1 @ A1) + 0.5 * (B2 @ A2)      # FedAvg of the EFFECTIVE updates
    fedavg = (0.5 * B1 + 0.5 * B2) @ (0.5 * A1 + 0.5 * A2)  # per-factor FedAvg

    print("--- demo_lora_aggregation_gap ---")
    print("target  (FedAvg of BA):\n", target, sep="")
    print("\nnaive per-factor FedAvg:\n", fedavg, sep="")
    print("\nrelative error:",
          np.linalg.norm(target - fedavg, ord="fro") / np.linalg.norm(target, ord="fro"))
    print("ranks:  target =", np.linalg.matrix_rank(target),
          " naive =", np.linalg.matrix_rank(fedavg))


def _one_client(d_out: int, d_in: int, r: int, rng: np.random.Generator,
                scale: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """One honest LoRA client's (B, A) at realistic init scale."""
    B = rng.standard_normal((d_out, r)) * scale
    A = rng.standard_normal((r, d_in))  * scale
    return B, A


def _relative_error(clients: List[Tuple[np.ndarray, np.ndarray]],
                    weights: np.ndarray) -> float:
    """
    ||naive - correct||_F / ||correct||_F, where
      correct = Σ p_k B_k A_k          (FedAvg of the effective updates)
      naive   = (Σ p_k B_k)(Σ p_k A_k) (per-factor FedAvg)
    """
    correct = sum(w * (B @ A) for w, (B, A) in zip(weights, clients))
    B_bar   = sum(w * B for w, (B, _) in zip(weights, clients))
    A_bar   = sum(w * A for w, (_, A) in zip(weights, clients))
    naive   = B_bar @ A_bar
    denom = np.linalg.norm(correct, ord="fro")
    if denom == 0:
        return float("nan")
    return float(np.linalg.norm(correct - naive, ord="fro") / denom)


def measure_lora_aggregation_error(
    K_values: List[int] = (2, 4, 8, 16, 32, 64),
    d: int = 768,
    r: int = 8,
    trials: int = 20,
    seed: int = 0,
    plot_path: str = "results/utils_lab/lora_aggregation_error.png",
) -> List[Tuple[int, float, float]]:
    """
    Sweep K and report mean +/- sd of relative error over `trials` random cohorts.

    Uniform weights p_k = 1/K, since the phenomenon is about factorisation
    non-linearity, not weighting. d=768 and r=8 mirror a transformer q_proj at
    LoRA rank 8. The relationship is Frobenius-norm and scale-invariant.
    """
    rng = np.random.default_rng(seed)
    rows: List[Tuple[int, float, float]] = []
    for K in K_values:
        errs = np.empty(trials)
        for t in range(trials):
            clients = [_one_client(d, d, r, rng) for _ in range(K)]
            weights = np.full(K, 1.0 / K)
            errs[t] = _relative_error(clients, weights)
        rows.append((K, float(errs.mean()), float(errs.std(ddof=1))))
        print(f"K={K:>3}  rel_err = {errs.mean():.4f} +/- {errs.std(ddof=1):.4f}  "
              f"(min {errs.min():.4f}, max {errs.max():.4f})")

    if plot_path:
        _plot_error_curve(rows, d=d, r=r, out=plot_path)
    return rows


def _plot_error_curve(rows, d, r, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out), exist_ok=True)

    K   = [r[0] for r in rows]
    mu  = [r[1] for r in rows]
    sd  = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(K, mu, yerr=sd, fmt="o-", capsize=3, lw=1.6, color="#a63d2f")
    ax.set_xscale("log", base=2)
    ax.set_xticks(K); ax.set_xticklabels([str(k) for k in K])
    ax.set_xlabel("clients K")
    ax.set_ylabel(r"$\|\bar B \bar A - \sum_k p_k B_k A_k\|_F / \|\sum_k p_k B_k A_k\|_F$")
    ax.set_title(f"Naive per-factor FedAvg drifts from FedAvg of the effective "
                 f"update\n(d = {d}, r = {r}; uniform weights)")
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nPlot: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--K", default="2,4,8,16,32,64",
                   help="Comma-separated K values (default 2,4,8,16,32,64)")
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    demo_lora_aggregation_gap()
    print()
    K = [int(x) for x in args.K.split(",")]
    measure_lora_aggregation_error(
        K_values=K, d=args.d, r=args.r, trials=args.trials, seed=args.seed,
        plot_path=(None if args.no_plot else
                   "results/utils_lab/lora_aggregation_error.png"),
    )


if __name__ == "__main__":
    main()
