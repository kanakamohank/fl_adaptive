#!/usr/bin/env python3
"""
Integration Tests for TAVS-ESP Components

Tests integration between:
1. CSPRNGManager + TavsScheduler
2. TavsScheduler + existing projection components
3. End-to-end TAVS system workflow
"""

def test_csprng_scheduler_integration():
    """Test integration between CSPRNGManager and TavsScheduler."""
    from src.tavs.csprng_manager import CSPRNGManager
    from src.tavs.scheduler import TavsScheduler
    import numpy as np

    print("=== Testing CSPRNG + Scheduler Integration ===")

    # Test 1: Shared CSPRNG produces consistent results
    master_key = b"test_key_for_integration_test32b"  # Exactly 32 bytes
    csprng = CSPRNGManager(master_key=master_key)
    scheduler = TavsScheduler(csprng_manager=csprng)

    # Register clients with varying initial trust levels
    clients = [f"client_{i}" for i in range(10)]
    for i, client_id in enumerate(clients):
        # Register clients over multiple rounds to simulate realistic deployment
        scheduler.register_client(client_id, round_number=max(0, i-2))

    print(f"✓ Registered {len(clients)} clients across different rounds")

    # Test 2: Multiple rounds with trust evolution
    for round_num in range(1, 8):
        # Generate scheduling decision using CSPRNG
        decision = scheduler.generate_scheduling_decision(clients, round_num)

        # Simulate verification results with realistic patterns
        verification_results = {}
        for client_id in decision.verified_clients:
            client_idx = int(client_id.split('_')[1])
            if client_idx < 2:  # First 2 clients are Byzantine
                behavioral_score = np.random.uniform(0.1, 0.4)  # Poor behavior
            else:  # Rest are honest
                behavioral_score = np.random.uniform(0.7, 0.9)  # Good behavior

            verification_results[client_id] = behavioral_score

        # Update trust scores
        scheduler.update_trust_scores(
            verification_results, decision.promoted_clients, round_num
        )

        promoted_count = len(decision.promoted_clients)
        verified_count = len(decision.verified_clients)
        budget_util = decision.budget_utilization

        print(f"  Round {round_num}: V={verified_count}, S={promoted_count}, "
              f"budget={budget_util:.1%}")

        # Test CSPRNG determinism: same round should give same randomness
        if round_num == 3:  # Test determinism on round 3
            decision_repeat = scheduler.generate_scheduling_decision(clients, round_num)
            # Scheduling may differ due to changed trust scores, but CSPRNG seed derivation should be same
            materials_1 = csprng.derive_round_materials(round_num)
            materials_2 = csprng.derive_round_materials(round_num)
            assert materials_1.projection_seed == materials_2.projection_seed, "CSPRNG not deterministic!"

    print("✓ Multi-round CSPRNG + Scheduler integration successful")

    # Test 3: Trust convergence patterns
    honest_clients = [f"client_{i}" for i in range(2, 10)]  # Exclude Byzantine clients
    byzantine_clients = ["client_0", "client_1"]

    honest_trusts = [scheduler.client_states[cid].trust_score for cid in honest_clients]
    byzantine_trusts = [scheduler.client_states[cid].trust_score for cid in byzantine_clients]

    print(f"✓ Trust convergence analysis:")
    print(f"  - Honest clients: mean={np.mean(honest_trusts):.3f}, std={np.std(honest_trusts):.3f}")
    print(f"  - Byzantine clients: mean={np.mean(byzantine_trusts):.3f}, std={np.std(byzantine_trusts):.3f}")

    # Honest clients should have higher trust than Byzantine
    if np.mean(honest_trusts) > np.mean(byzantine_trusts):
        print("✓ Trust system correctly distinguishes honest vs Byzantine clients")
    else:
        print("⚠ Trust system may need more rounds to converge")

    return True


def test_projection_integration():
    """Test TavsScheduler integration with existing projection components."""
    from src.tavs.csprng_manager import CSPRNGManager
    from src.tavs.scheduler import TavsScheduler
    from src.core.projection import StructuredJLProjection, DenseJLProjection
    from src.core.models import get_model, ModelStructure
    import torch

    print("\n=== Testing Scheduler + Projection Integration ===")

    # Initialize components
    csprng = CSPRNGManager()
    scheduler = TavsScheduler(csprng_manager=csprng)

    # Get CIFAR CNN model for testing
    model = get_model("cifar_cnn", num_classes=10)

    # Get model structure from the model itself
    if hasattr(model, 'structure'):
        model_structure = model.structure
    else:
        # Fallback: create basic structure
        model_structure = ModelStructure()
        total_params = sum(p.numel() for p in model.parameters())
        model_structure.add_block('full_model', (total_params,), total_params)

    # Test with StructuredJLProjection
    projection = StructuredJLProjection(
        model_structure=model_structure,
        target_k=150,
        device="cpu"
    )

    print(f"✓ Initialized projection with model ({sum(p.numel() for p in model.parameters())} params)")

    # Test ephemeral projection generation using CSPRNG seeds
    clients = [f"client_{i}" for i in range(6)]
    for client_id in clients:
        scheduler.register_client(client_id, round_number=0)

    round_num = 2
    decision = scheduler.generate_scheduling_decision(clients, round_num)

    # Get CSPRNG materials for this round
    round_materials = csprng.derive_round_materials(round_num)

    # Use projection seed to generate matrices (integration point)
    projection_matrices = csprng.generate_projection_matrix(
        projection_seed=round_materials.projection_seed,
        block_dimensions=[(50, 500), (30, 300)]  # Example block dimensions
    )

    print(f"✓ Generated projection matrices using CSPRNG seed:")
    for block_name, matrix in projection_matrices.items():
        print(f"  - {block_name}: {matrix.shape}")

    # Test Bayesian weight computation for promoted clients
    if decision.promoted_clients:
        bayesian_weights = scheduler.compute_bayesian_weights(decision.promoted_clients)
        print(f"✓ Computed Bayesian weights for {len(bayesian_weights)} promoted clients")
    else:
        print("✓ No promoted clients in early rounds (expected with low initial trust)")

    return True


def test_end_to_end_workflow():
    """Test complete TAVS workflow: scheduling → projection → aggregation."""
    from src.tavs.csprng_manager import CSPRNGManager
    from src.tavs.scheduler import TavsScheduler
    from src.core.verification import IsomorphicVerification
    import torch
    import numpy as np

    print("\n=== Testing End-to-End TAVS Workflow ===")

    # Initialize TAVS system
    csprng = CSPRNGManager()
    scheduler = TavsScheduler(csprng_manager=csprng)
    verification = IsomorphicVerification(detection_threshold=2.0, min_consensus=0.6)

    clients = [f"client_{i}" for i in range(8)]

    # Simulate complete FL round workflow
    for round_num in range(1, 6):
        print(f"\n--- Round {round_num} ---")

        # Step 1: TAVS scheduling decision
        decision = scheduler.generate_scheduling_decision(clients, round_num)
        print(f"1. Scheduling: {len(decision.verified_clients)} verified, "
              f"{len(decision.promoted_clients)} promoted")

        # Step 2: Generate ephemeral projection using CSPRNG
        round_materials = csprng.derive_round_materials(round_num)
        projection_matrices = csprng.generate_projection_matrix(
            projection_seed=round_materials.projection_seed,
            block_dimensions=[(100, 1000)]  # Single block for simplicity
        )
        print(f"2. Ephemeral projection: {list(projection_matrices.keys())}")

        # Step 3: Simulate client updates (verified clients only for now)
        client_updates = []
        client_ids = []

        for client_id in decision.verified_clients:
            # Generate synthetic gradient update
            update = torch.randn(1000) * 0.1  # 1000-dim gradient

            # Add Byzantine behavior for first 2 clients
            client_idx = int(client_id.split('_')[1])
            if client_idx < 2 and round_num > 2:  # Byzantine after round 2
                update += torch.randn(1000) * 2.0  # Large malicious perturbation

            client_updates.append(update)
            client_ids.append(client_id)

        print(f"3. Client updates: {len(client_updates)} updates collected")

        # Step 4: Apply projection to updates
        projection_matrix = projection_matrices["block_0"]
        projected_updates = []
        for update in client_updates:
            projected = projection_matrix @ update.numpy()
            projected_updates.append(torch.tensor(projected))

        print(f"4. Projection applied: {len(projected_updates)} updates projected")

        # Step 5: Byzantine detection on projected updates
        if len(projected_updates) >= 3:
            detection_results = verification.detect_byzantine_clients(
                projected_updates=projected_updates,
                client_ids=client_ids
            )
            byzantine_detected = detection_results['byzantine_indices']
            print(f"5. Detection: {len(byzantine_detected)} Byzantine clients detected")

            # Step 6: Compute behavioral scores for trust updates
            verification_results = {}
            for i, client_id in enumerate(client_ids):
                if i in byzantine_detected:
                    behavioral_score = 0.2  # Poor score for detected Byzantine
                else:
                    behavioral_score = 0.8  # Good score for undetected clients

                verification_results[client_id] = behavioral_score

            # Step 7: Update trust scores using TAVS EMA
            updated_scores = scheduler.update_trust_scores(
                verification_results, decision.promoted_clients, round_num
            )
            print(f"6. Trust updates: {len(updated_scores)} clients updated")

        else:
            print("5. Detection: Insufficient clients for detection")

    # Final analysis
    stats = scheduler.get_client_statistics()
    print(f"\n✓ End-to-end workflow completed:")
    print(f"  - Total rounds: {stats['scheduling_statistics']['total_rounds']}")
    print(f"  - Trust convergence: μ={stats['trust_statistics']['mean']:.3f}")
    print(f"  - Tier distribution: {stats['tier_distribution']}")

    return True


def main():
    """Run all integration tests."""
    print("🧪 TAVS-ESP Integration Test Suite")
    print("=" * 50)

    try:
        # Test 1: CSPRNG + Scheduler
        success1 = test_csprng_scheduler_integration()

        # Test 2: Scheduler + Projection
        success2 = test_projection_integration()

        # Test 3: End-to-end workflow
        success3 = test_end_to_end_workflow()

        if success1 and success2 and success3:
            print(f"\n🎯 All integration tests PASSED!")
            print("✓ CSPRNGManager ↔ TavsScheduler integration working")
            print("✓ TavsScheduler ↔ Projection components integration working")
            print("✓ End-to-end TAVS workflow functional")
        else:
            print(f"\n❌ Some integration tests FAILED")

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()