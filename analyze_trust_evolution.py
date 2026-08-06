import json
import time

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict

def load_round_analytics():
    """Load all round analytics from tavs_esp_logs"""
    logs_dir = Path("tavs_esp_logs")
    round_data = {}

    for log_file in sorted(logs_dir.glob("round_*_analytics.json")):
        # Extract round number from filename
        round_num = int(log_file.stem.split('_')[1])
        with open(log_file, 'r') as f:
            round_data[round_num] = json.load(f)

    return round_data

def extract_trust_tier_evolution(round_data):
    """Extract trust scores and tier assignments for each client across rounds"""
    client_trust_evolution = defaultdict(list)
    client_tier_evolution = defaultdict(list)
    client_verification_history = defaultdict(list)

    for round_num in sorted(round_data.keys()):
        data = round_data[round_num]

        # Get trust scores for this round
        trust_scores = data['scheduling_decision'].get('trust_scores', {})
        tier_assignments = data['scheduling_decision'].get('tier_assignments', {})
        verified_clients = set(data['scheduling_decision'].get('verified_clients', []))
        promoted_clients = set(data['scheduling_decision'].get('promoted_clients', []))
        decoy_clients = set(data['scheduling_decision'].get('decoy_clients', []))

        # Track all clients seen so far
        all_clients = set(trust_scores.keys()) | set(tier_assignments.keys())

        for client_id in all_clients:
            # Trust score
            trust = trust_scores.get(client_id, None)
            client_trust_evolution[client_id].append({
                'round': round_num,
                'trust_score': trust
            })

            # Tier assignment
            tier = tier_assignments.get(client_id, None)
            client_tier_evolution[client_id].append({
                'round': round_num,
                'tier': tier
            })

            # Verification status
            status = []
            if client_id in verified_clients:
                status.append('verified')
            if client_id in promoted_clients:
                status.append('promoted')
            if client_id in decoy_clients:
                status.append('decoy')

            client_verification_history[client_id].append({
                'round': round_num,
                'status': status if status else ['unverified']
            })

    return client_trust_evolution, client_tier_evolution, client_verification_history

def identify_byzantine_clients(round_data):
    """Identify which clients were detected as Byzantine"""
    byzantine_clients = set()

    for round_num, data in round_data.items():
        detected = data.get('byzantine_detected', [])
        byzantine_clients.update(detected)

    return byzantine_clients

def create_detailed_trust_evolution_plot(client_trust_evolution, client_tier_evolution, byzantine_clients):
    """Create a detailed plot showing trust score and tier evolution"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Color scheme
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    client_colors = {}

    # Plot trust scores
    for idx, (client_id, trust_history) in enumerate(client_trust_evolution.items()):
        rounds = [h['round'] for h in trust_history if h['trust_score'] is not None]
        scores = [h['trust_score'] for h in trust_history if h['trust_score'] is not None]

        if not rounds:
            continue

        # Determine if Byzantine
        is_byzantine = client_id in byzantine_clients
        color = 'red' if is_byzantine else colors[idx % 20]
        client_colors[client_id] = color

        # Shorten client ID for display
        short_id = client_id[:6] + "..."

        ax1.plot(rounds, scores,
                marker='o' if is_byzantine else '.',
                color=color,
                alpha=0.8 if is_byzantine else 0.6,
                linewidth=2 if is_byzantine else 1,
                markersize=6 if is_byzantine else 4,
                label=f"{short_id} {'(Byzantine)' if is_byzantine else ''}")

    # Add threshold lines
    ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='θ_low (0.3)')
    ax1.axhline(y=0.7, color='blue', linestyle='--', alpha=0.5, label='θ_high (0.7)')

    ax1.set_ylabel('Trust Score', fontsize=12)
    ax1.set_title('Client Trust Score Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Plot tier assignments
    for client_id, tier_history in client_tier_evolution.items():
        rounds = [h['round'] for h in tier_history if h['tier'] is not None]
        tiers = [h['tier'] for h in tier_history if h['tier'] is not None]

        if not rounds:
            continue

        color = client_colors.get(client_id, 'gray')
        is_byzantine = client_id in byzantine_clients

        ax2.plot(rounds, tiers,
                marker='s' if is_byzantine else 'o',
                color=color,
                alpha=0.8 if is_byzantine else 0.6,
                linewidth=2 if is_byzantine else 1,
                markersize=6 if is_byzantine else 4)

    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Tier Assignment', fontsize=12)
    ax2.set_title('Client Tier Evolution', fontsize=14, fontweight='bold')
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['Tier 1\n(Untrusted)', 'Tier 2\n(Probation)', 'Tier 3\n(Trusted)'])
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 3.5)

    # Add legend (only for a subset to avoid clutter)
    handles, labels = ax1.get_legend_handles_labels()
    if len(handles) > 10:
        ax1.legend(handles[:5], labels[:5], loc='upper right', fontsize=8)
    else:
        ax1.legend(loc='upper right', fontsize=8)

    plt.suptitle('TAVS-ESP Trust and Tier Evolution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('trust_tier_evolution' +  '_' + str(time.time())  + '.png', dpi=150, bbox_inches='tight')
    print("✓ Trust and tier evolution plot saved as 'trust_tier_evolution.png'")

def create_trust_summary_table(client_trust_evolution, client_tier_evolution, byzantine_clients):
    """Create a summary table of client trust evolution"""

    summary_data = []

    for client_id in client_trust_evolution:
        trust_history = client_trust_evolution[client_id]
        tier_history = client_tier_evolution[client_id]

        # Get initial and final values
        initial_trust = next((h['trust_score'] for h in trust_history if h['trust_score'] is not None), None)
        final_trust = next((h['trust_score'] for h in reversed(trust_history) if h['trust_score'] is not None), None)

        initial_tier = next((h['tier'] for h in tier_history if h['tier'] is not None), None)
        final_tier = next((h['tier'] for h in reversed(tier_history) if h['tier'] is not None), None)

        # Count tier transitions
        tier_changes = 0
        prev_tier = None
        for h in tier_history:
            if h['tier'] is not None:
                if prev_tier is not None and h['tier'] != prev_tier:
                    tier_changes += 1
                prev_tier = h['tier']

        summary_data.append({
            'Client ID': client_id[:10] + '...',
            'Byzantine': '✓' if client_id in byzantine_clients else '',
            'Initial Trust': f"{initial_trust:.3f}" if initial_trust is not None else 'N/A',
            'Final Trust': f"{final_trust:.3f}" if final_trust is not None else 'N/A',
            'Initial Tier': initial_tier if initial_tier is not None else 'N/A',
            'Final Tier': final_tier if final_tier is not None else 'N/A',
            'Tier Changes': tier_changes
        })

    # Sort by Byzantine status and then by final trust
    summary_data.sort(key=lambda x: (x['Byzantine'] != '✓', -float(x['Final Trust']) if x['Final Trust'] != 'N/A' else 0))

    # Create DataFrame
    df = pd.DataFrame(summary_data)

    # Save to CSV
    df.to_csv('trust_evolution_summary.csv', index=False)
    print("\n✓ Trust evolution summary saved as 'trust_evolution_summary.csv'")

    # Print summary statistics
    print("\n📊 Trust Evolution Statistics:")
    print(f"  • Total clients tracked: {len(summary_data)}")
    print(f"  • Byzantine clients detected: {len(byzantine_clients)}")

    # Tier distribution
    final_tiers = [d['Final Tier'] for d in summary_data if d['Final Tier'] != 'N/A']
    if final_tiers:
        tier_counts = {tier: final_tiers.count(tier) for tier in set(final_tiers)}
        print(f"\n  Final Tier Distribution:")
        for tier in sorted(tier_counts.keys()):
            print(f"    - Tier {tier}: {tier_counts[tier]} clients")

    # Trust score statistics
    final_trusts = [float(d['Final Trust']) for d in summary_data if d['Final Trust'] != 'N/A']
    if final_trusts:
        print(f"\n  Trust Score Statistics:")
        print(f"    - Mean: {np.mean(final_trusts):.3f}")
        print(f"    - Median: {np.median(final_trusts):.3f}")
        print(f"    - Min: {np.min(final_trusts):.3f}")
        print(f"    - Max: {np.max(final_trusts):.3f}")

    return df

def analyze_verification_patterns(client_verification_history, round_data):
    """Analyze verification patterns across rounds"""

    verification_stats = defaultdict(lambda: {'verified': 0, 'promoted': 0, 'decoy': 0})

    for client_id, history in client_verification_history.items():
        for h in history:
            for status in h['status']:
                if status != 'unverified':
                    verification_stats[client_id][status] += 1

    # Create verification summary
    print("\n🔍 Verification Pattern Analysis:")

    # Total verifications per round
    verifications_per_round = []
    for round_num in sorted(round_data.keys()):
        data = round_data[round_num]
        num_verified = len(data['scheduling_decision'].get('verified_clients', []))
        verifications_per_round.append(num_verified)

    print(f"  • Average verifications per round: {np.mean(verifications_per_round):.1f}")
    print(f"  • Max verifications in a round: {np.max(verifications_per_round)}")
    print(f"  • Min verifications in a round: {np.min(verifications_per_round)}")

    # Budget utilization
    budget_utilization = []
    for round_num in sorted(round_data.keys()):
        data = round_data[round_num]
        budget = data['scheduling_decision'].get('budget_utilization', 0)
        budget_utilization.append(budget)

    if budget_utilization:
        print(f"\n  Budget Utilization:")
        print(f"    - Mean: {np.mean(budget_utilization):.1%}")
        print(f"    - Max: {np.max(budget_utilization):.1%}")

def main():
    print("Loading round analytics data...\n")

    # Load all round data
    round_data = load_round_analytics()
    print(f"✓ Loaded {len(round_data)} rounds of data")

    # Extract trust and tier evolution
    print("\nExtracting trust and tier evolution...")
    client_trust_evolution, client_tier_evolution, client_verification_history = extract_trust_tier_evolution(round_data)

    # Identify Byzantine clients
    byzantine_clients = identify_byzantine_clients(round_data)
    print(f"✓ Identified {len(byzantine_clients)} Byzantine clients")

    # Create visualizations
    print("\nGenerating visualizations...")
    create_detailed_trust_evolution_plot(client_trust_evolution, client_tier_evolution, byzantine_clients)

    # Create summary table
    df_summary = create_trust_summary_table(client_trust_evolution, client_tier_evolution, byzantine_clients)

    # Analyze verification patterns
    analyze_verification_patterns(client_verification_history, round_data)

    print("\n✅ Analysis complete! Generated files:")
    print("  - trust_tier_evolution.png")
    print("  - trust_evolution_summary.csv")

if __name__ == "__main__":
    main()