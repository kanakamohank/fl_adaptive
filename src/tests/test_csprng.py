#!/usr/bin/env python3
"""
Quick test of CSPRNGManager implementation
"""

def test_csprng_manager():
    """Test CSPRNGManager basic functionality."""
    from ..tavs import CSPRNGManager
    import numpy as np

    print("Testing CSPRNGManager...")

    # Test 1: Basic initialization
    csprng = CSPRNGManager()
    print(f"✓ CSPRNGManager initialized with key epoch {csprng.current_key_epoch}")

    # Test 2: Round materials derivation
    materials = csprng.derive_round_materials(round_number=1)
    print(f"✓ Round materials derived for round {materials.round_number}")
    print(f"  - Projection seed: {len(materials.projection_seed)} bytes")
    print(f"  - Promotion seed: {len(materials.promotion_seed)} bytes")
    print(f"  - Decoy seed: {len(materials.decoy_seed)} bytes")
    print(f"  - Derivation time: {materials.derivation_time*1000:.2f}ms")

    # Test 3: Deterministic behavior
    materials2 = csprng.derive_round_materials(round_number=1)
    assert materials.projection_seed == materials2.projection_seed
    print("✓ Deterministic: Same round produces same seeds")

    # Test 4: Different rounds produce different seeds
    materials3 = csprng.derive_round_materials(round_number=2)
    assert materials.projection_seed != materials3.projection_seed
    print("✓ Independence: Different rounds produce different seeds")

    # Test 5: Projection matrix generation
    block_dims = [(100, 1000), (50, 500)]  # Small test matrices
    proj_matrices = csprng.generate_projection_matrix(
        materials.projection_seed, block_dims
    )
    print(f"✓ Projection matrices generated: {len(proj_matrices)} blocks")
    for name, matrix in proj_matrices.items():
        print(f"  - {name}: {matrix.shape}")

    # Test 6: Promotion assignments
    client_trust = {0: 0.1, 1: 0.4, 2: 0.8, 3: 0.9}  # Mixed trust levels
    verified, promoted = csprng.generate_promotion_assignments(
        materials.promotion_seed, client_trust
    )
    print(f"✓ Promotion assignments: {len(verified)} verified, {len(promoted)} promoted")

    # Test 7: Decoy verification
    decoy_clients = csprng.generate_decoy_verification(
        materials.decoy_seed, promoted
    )
    print(f"✓ Decoy verification: {len(decoy_clients)} decoy clients selected")

    # Test 8: Statistics
    stats = csprng.get_security_stats()
    print(f"✓ Security stats: {stats['total_rounds_processed']} rounds processed")

    print("\n🎯 CSPRNGManager tests completed successfully!")
    return True

if __name__ == "__main__":
    test_csprng_manager()