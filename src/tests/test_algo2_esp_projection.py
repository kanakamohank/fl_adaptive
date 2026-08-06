import pytest
import torch
import math

# Assuming you save the new projector in src/tavs_v2/algo2_esp_projection.py
from src.tavs_v2.algo2_esp_projection import EphemeralStructuredProjection

@pytest.fixture
def dummy_blocks():
    """Mock semantic blocks representing layers of a small transformer."""
    return {
        "attention_layer_1": 10000,
        "attention_layer_2": 15000,
        "ffn_layer_1": 25000
    }

@pytest.fixture
def esp_projector(dummy_blocks):
    """Instantiates the ESP with target_k=500, matching a lightweight experiment."""
    return EphemeralStructuredProjection(
        target_k=500,
        model_blocks=dummy_blocks,
        master_key=b'neurips_layer2_test_key'
    )

def test_km_allocation_proportionality(esp_projector, dummy_blocks):
    """
    Validates Section 4.3: Total dimension k = \\sum_m k_m.
    Ensures that compression is distributed proportionally to block size.
    """
    allocations = esp_projector.k_m_allocations
    
    # 1. Total k must be exactly preserved
    assert sum(allocations.values()) == esp_projector.target_k, "Total projection dimension k was not conserved"
    
    # 2. Larger blocks must get larger k_m
    assert allocations["ffn_layer_1"] > allocations["attention_layer_2"]
    assert allocations["attention_layer_2"] > allocations["attention_layer_1"]
    
    # 3. Exact math check for attention_layer_1: (10000 / 50000) * 500 = 100
    assert allocations["attention_layer_1"] == 100

def test_ephemeral_seed_unpredictability(esp_projector):
    """
    Validates Theorem 2 (TC1): Adaptive Evasion Resistance.
    Seeds MUST change every round, and MUST be different per block.
    """
    # Same round, different blocks
    seed_block1 = esp_projector._get_secondary_seed("attention_layer_1", round_num=1)
    seed_block2 = esp_projector._get_secondary_seed("attention_layer_2", round_num=1)
    assert seed_block1 != seed_block2, "Blocks must use different projection matrices"
    
    # Same block, different rounds (The Ephemeral property!)
    seed_round1 = esp_projector._get_secondary_seed("attention_layer_1", round_num=1)
    seed_round2 = esp_projector._get_secondary_seed("attention_layer_1", round_num=2)
    assert seed_round1 != seed_round2, "Projection matrices MUST change every round to defeat null-space attacks"

def test_projection_dimensions(esp_projector, dummy_blocks):
    """
    Validates that the output dictionaries match the expected compressed dimensions.
    """
    # Create fake gradient updates
    client_update = {
        name: torch.randn(size) for name, size in dummy_blocks.items()
    }
    
    # Project
    projected = esp_projector.project_client_update(client_update, round_num=1)
    
    # Check shapes
    for block_name, proj_tensor in projected.items():
        expected_k_m = esp_projector.k_m_allocations[block_name]
        assert proj_tensor.shape == (expected_k_m,), f"Output dimension mismatch for {block_name}"

def test_jl_distance_preservation_property():
    """
    Validates the fundamental Johnson-Lindenstrauss Lemma:
    E[||R * x||^2] ≈ ||x||^2 when R_{ij} ~ N(0, 1/k_m).
    
    This is the mathematical heart of Section 4.3.
    """
    # We use a larger k_m here to reduce the variance of the JL projection for the test
    target_k = 2000 
    d_m = 10000
    projector = EphemeralStructuredProjection(
        target_k=target_k, 
        model_blocks={"test_block": d_m}
    )
    
    # 1. Create a fixed, known client update vector
    original_vector = torch.randn(d_m)
    original_sq_norm = torch.sum(original_vector ** 2).item()
    
    # 2. Project it over many simulated rounds to calculate the Expected Value
    num_rounds = 50
    projected_sq_norms = []
    
    for r in range(num_rounds):
        projected_update = projector.project_client_update({"test_block": original_vector}, round_num=r)
        proj_vec = projected_update["test_block"]
        projected_sq_norms.append(torch.sum(proj_vec ** 2).item())
        
    # 3. The Expectation (mean) of the projected norms should tightly bound the original norm
    expected_proj_sq_norm = sum(projected_sq_norms) / num_rounds
    
    # Check if they are within 5% of each other (standard JL concentration bound)
    tolerance = original_sq_norm * 0.05
    assert abs(expected_proj_sq_norm - original_sq_norm) < tolerance, \
        "JL Projection failed to preserve Euclidean distance! The standard deviation formulation is wrong."
