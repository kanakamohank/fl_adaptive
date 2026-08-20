#!/usr/bin/env python3
"""
Centralised training baseline: the ceiling every federated result is measured against.

Why this exists
---------------
A 60-round federated run reached 27% test accuracy on CIFAR-10. That number is
uninterpretable on its own: it could mean federated aggregation is losing signal,
or it could mean this model at this learning rate simply does not do better. The
two demand opposite responses, and nothing in the federated results distinguishes
them.

This trains the SAME model on the SAME data with the SAME hyperparameters, with
no federation at all -- one process, all 50000 training images, no splitting, no
verification, no clipping. Whatever accuracy it reaches is the ceiling.

Reading the result
------------------
Compare `equivalent-epoch` accuracy here against the federated run's accuracy at
a matched budget (see --epochs below).

  * Centralised MUCH higher (e.g. 55% vs 27%)
      -> the model and data are fine; federated aggregation is losing the signal.
         Look at sample-count weighting, client drift under non-IID, and how many
         updates the defences discard.

  * Centralised SIMILAR (e.g. 30% vs 27%)
      -> the ceiling itself is low. The federated result is close to the best this
         setup can do, and tuning TAVS will not move it. Raise the learning rate,
         train longer, or use a stronger model before drawing conclusions about
         federated behaviour.

Matching the compute budget
---------------------------
The federated run performs, per round, `clients_per_round * client_epochs` passes
over roughly `50000/num_clients` samples each. At 8 clients/round, 2 local epochs
and 20 clients that is 8 * 2 * 2500 = 40000 samples per round, i.e. ~0.8
centralised epochs per round. So 60 federated rounds is a budget of roughly 48
centralised epochs -- far more than the default 5 here.

That asymmetry is deliberate and worth stating: if centralised training beats the
federated run at a FRACTION of its compute, the gap cannot be explained by
budget. Use --epochs 48 for a like-for-like comparison once the quick run is in.

Usage:
    python -m experiments.centralized_baseline                 # quick, 5 epochs
    python -m experiments.centralized_baseline --epochs 48     # budget-matched
    python -m experiments.centralized_baseline --no-momentum   # isolate momentum
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.models import get_model
from src.utils.data_utils import load_cifar10


def evaluate(model, loader, criterion, device):
    """Test-set loss and accuracy. Loss is per-sample, comparable to the server's."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)
    return total_loss / seen, correct / seen


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults mirror PipelineConfig so the comparison is like-for-like. Changing
    # one here without changing it there invalidates the whole point of the script.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--no-momentum", action="store_true",
                        help="Set momentum to 0. Local momentum amplifies client "
                             "drift under non-IID data, so this isolates whether "
                             "it helps or hurts before changing the FL clients.")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--model", default="cifar_cnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--results-dir", default="results/centralized_baseline")
    args = parser.parse_args()

    momentum = 0.0 if args.no_momentum else args.momentum
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, testset = load_cifar10(args.data_dir)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    model = get_model(args.model, num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"device={device}  model={args.model}  lr={args.lr}  batch={args.batch_size}  "
          f"momentum={momentum}  wd={args.weight_decay}")
    print(f"train={len(trainset)} test={len(testset)}\n")
    print(f"{'epoch':>5} {'train loss':>11} {'test loss':>10} {'test acc':>9} {'fed-round equiv':>16}")

    history, started = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            batches += 1

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        # One federated round processes ~0.8 centralised epochs of samples at the
        # default 20-client / 8-per-round / 2-local-epoch setting.
        fed_equiv = epoch / 0.8
        history.append({"epoch": epoch, "train_loss": running / batches,
                        "test_loss": test_loss, "test_accuracy": test_acc,
                        "federated_round_equivalent": fed_equiv})
        print(f"{epoch:>5} {running / batches:>11.4f} {test_loss:>10.4f} "
              f"{test_acc:>9.4f} {fed_equiv:>15.0f}")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "no_momentum" if args.no_momentum else f"m{momentum:g}"
    out = out_dir / f"baseline_{args.model}_e{args.epochs}_{tag}.json"
    out.write_text(json.dumps({"config": vars(args), "momentum_used": momentum,
                               "history": history,
                               "elapsed_seconds": time.time() - started}, indent=2))

    best = max(h["test_accuracy"] for h in history)
    print(f"\nbest test accuracy: {best:.4f}")
    print(f"federated 60-round run reached: 0.273 (mean of last 10 rounds)")
    if best > 0.40:
        print("\n=> Ceiling is well above the federated result. The model and data are\n"
              "   fine, so the gap is in federated aggregation, not in training.")
    else:
        print("\n=> Ceiling is close to the federated result. The setup itself is the\n"
              "   limit; tuning TAVS will not move accuracy much. Raise the learning\n"
              "   rate or use a stronger model before drawing federated conclusions.")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
