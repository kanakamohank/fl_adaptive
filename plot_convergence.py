import json
import time

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_convergence_metrics():
    results_file = Path("results/efficiency/pipeline_results.json")

    if not results_file.exists():
        print(f"File {results_file} not found. Looking for alternative files...")
        results_file = Path("test_results/pipeline_results.json")

    with open(results_file, 'r') as f:
        data = json.load(f)

    losses = data.get('server_losses', [])
    accuracies = data.get('server_accuracies', [])
    rounds = list(range(1, len(losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rounds, losses, 'b-', linewidth=2, marker='o', markersize=6)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, len(rounds) + 0.5)

    accuracies_pct = [acc * 100 for acc in accuracies]
    ax2.plot(rounds, accuracies_pct, 'g-', linewidth=2, marker='s', markersize=6)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Model Accuracy Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, len(rounds) + 0.5)
    ax2.set_ylim(0, 100)

    plt.suptitle('TAVS-ESP Training Convergence on CIFAR-10', fontsize=16, fontweight='bold', y=1.02)

    final_loss = losses[-1]
    final_acc = accuracies[-1] * 100
    initial_loss = losses[0]
    initial_acc = accuracies[0] * 100

    textstr = f'Final Loss: {final_loss:.3f} (↓{initial_loss - final_loss:.3f})\n'
    textstr += f'Final Acc: {final_acc:.1f}% (↑{final_acc - initial_acc:.1f}%)'

    fig.text(0.5, -0.05, textstr, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('training_convergence.png', dpi=150, bbox_inches='tight')
    print("✓ Convergence plot saved as 'training_convergence.png'")

    print("\n📊 Training Metrics Summary:")
    print(f"  • Initial Loss: {initial_loss:.3f} → Final Loss: {final_loss:.3f}")
    print(f"  • Initial Accuracy: {initial_acc:.1f}% → Final Accuracy: {final_acc:.1f}%")
    print(f"  • Loss Reduction: {((initial_loss - final_loss) / initial_loss * 100):.1f}%")
    print(f"  • Accuracy Improvement: {final_acc - initial_acc:.1f} percentage points")
    print(f"  • Convergence in {len(rounds)} rounds")

    return losses, accuracies

def plot_trust_evolution():
    results_file = Path("results/efficiency/pipeline_results.json")

    if not results_file.exists():
        results_file = Path("test_results/pipeline_results.json")

    with open(results_file, 'r') as f:
        data = json.load(f)

    trust_evolution = data.get('trust_evolution', {})

    if not trust_evolution:
        print("No trust evolution data found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    byzantine_clients = []
    honest_clients = []

    for client_id, trust_scores in trust_evolution.items():
        rounds = list(range(1, len(trust_scores) + 1))
        final_trust = trust_scores[-1] if trust_scores else 0

        if final_trust < 0.5:
            byzantine_clients.append(client_id)
            ax.plot(rounds, trust_scores, 'r-', alpha=0.6, linewidth=1.5)
        else:
            honest_clients.append(client_id)
            ax.plot(rounds, trust_scores, 'g-', alpha=0.4, linewidth=1)

    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='θ_low (0.3)')
    ax.axhline(y=0.8, color='blue', linestyle='--', alpha=0.5, label='θ_high (0.8)')

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Trust Score', fontsize=12)
    ax.set_title('Trust Score Evolution: Honest vs Byzantine Clients', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='g', lw=2, label=f'Honest Clients (n={len(honest_clients)})'),
        Line2D([0], [0], color='r', lw=2, label=f'Byzantine Clients (n={len(byzantine_clients)})'),
        Line2D([0], [0], color='orange', lw=1, linestyle='--', label='θ_low'),
        Line2D([0], [0], color='blue', lw=1, linestyle='--', label='θ_high')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig('trust_evolution' + '_' + str(time.time()) + '.png', dpi=150, bbox_inches='tight')
    print("✓ Trust evolution plot saved as 'trust_evolution.png'")

    print("\n🔐 Trust Management Summary:")
    print(f"  • Honest clients identified: {len(honest_clients)}")
    print(f"  • Byzantine clients detected: {len(byzantine_clients)}")
    if len(byzantine_clients) > 0:
        print(f"  • Detection rate: {len(byzantine_clients) / (len(byzantine_clients) + len(honest_clients)) * 100:.1f}%")

if __name__ == "__main__":
    print("Generating training convergence plots...\n")
    plot_convergence_metrics()
    print("\n" + "="*50 + "\n")
    plot_trust_evolution()
    print("\n✅ All plots generated successfully!")