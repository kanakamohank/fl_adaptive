import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")


def create_professional_plots(json_file):
    results_dir = Path(json_file).parent

    with open(json_file, 'r') as f:
        all_results = json.load(f)

    scenarios = [s for s in all_results.keys() if s not in ['overall_comparison', 'meta']]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'TAVS vs. Traditional Full Verification (10-Round FL, CIFAR-10)',
        fontsize=18, fontweight='bold'
    )

    ax = axes[0]
    tavs_times = []
    full_times = []
    labels = []

    for s in scenarios:
        tavs_data = all_results[s].get('tavs', {})
        full_data = all_results[s].get('full_verification', {})
        tavs_times.append(tavs_data.get('total_time', 0))
        full_times.append(full_data.get('total_time', 0))
        labels.append(s.replace('_', ' ').title())

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, full_times, width, label='Full Verification', color='#d95f02', edgecolor='black')
    ax.bar(x + width / 2, tavs_times, width, label='TAVS', color='#1b9e77', edgecolor='black')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Verification Time Overhead')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()

    ax = axes[1]
    tavs_clients = []
    full_clients = []
    for s in scenarios:
        tavs_data = all_results[s].get('tavs', {})
        full_data = all_results[s].get('full_verification', {})
        tavs_clients.append(tavs_data.get('total_verifications', 0))
        full_clients.append(full_data.get('total_verifications', 0))

    ax.bar(x - width / 2, full_clients, width, label='Full Verification', color='#d95f02', edgecolor='black')
    ax.bar(x + width / 2, tavs_clients, width, label='TAVS', color='#1b9e77', edgecolor='black')
    ax.set_ylabel('Total Verifications')
    ax.set_title('Client Verification Count')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()

    if full_clients and full_clients[0] > 0:
        savings = (full_clients[0] - tavs_clients[0]) / full_clients[0] * 100
        ax.text(0.5, 0.95, f'{savings:.0f}% fewer verifications',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold', color='#1b9e77')

    ax = axes[2]
    tavs_detection = []
    full_detection = []
    for s in scenarios:
        tavs_data = all_results[s].get('tavs', {})
        full_data = all_results[s].get('full_verification', {})
        tavs_detection.append(tavs_data.get('detection_rate', 0))
        full_detection.append(full_data.get('detection_rate', 0))

    ax.bar(x - width / 2, full_detection, width, label='Full Verification', color='#d95f02', edgecolor='black')
    ax.bar(x + width / 2, tavs_detection, width, label='TAVS', color='#1b9e77', edgecolor='black')
    ax.set_ylabel('Detection Rate')
    ax.set_title('Byzantine Detection Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    output_path = results_dir / 'verification_comparison_plots.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plots to {output_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = 'verification_comparison_results.json'
    create_professional_plots(json_path)
