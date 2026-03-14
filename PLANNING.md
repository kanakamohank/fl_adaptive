# Zero-Trust Projection Federated Learning - Planning Document

## Project Overview

**Objective**: Implement a Byzantine-robust federated learning system using structured Johnson–Lindenstrauss projections with ephemeral server-side randomness, integrated into the Flower framework, and evaluate against KETS-breaking attacks.

## Core Innovation

**Ephemeral Structured Projections**: The projection matrix R_t changes every round, breaking null-space attacks that exploit static defenses, while block-diagonal structure enables precise attack localization.

## Implementation Phases

### ✅ Phase 0: Environment & Framework Setup
- [x] Use Flower (flwr) as FL orchestration framework
- [x] Implement custom server strategy extending `flwr.server.strategy.Strategy`
- [x] Implement custom client logic extending `flwr.client.NumPyClient`
- [x] Ensure compatibility with CNN (CIFAR) and Transformer architectures
- [x] Project structure with modular components

### ✅ Phase 1: Baseline FL Pipeline
- [x] Implement vanilla FedAvg in Flower (`src/server/fedavg_strategy.py`)
- [x] Validate convergence on IID and Non-IID Dirichlet splits
- [x] Log client updates and per-round aggregation outputs
- [x] Establish ground-truth baseline (`experiments/baseline_fedavg.py`)

### ✅ Phase 2: Structured JL Projection Module
- [x] **CRITICAL**: Block-diagonal Gaussian JL (NOT `torch.randn(k, d)`)
- [x] One block per CNN filter bank / Transformer attention head
- [x] Ephemeral projection matrix R_t generated every round
- [x] Server-side only (never sent to clients)
- [x] Implementation: `src/core/projection.py`

### ✅ Phase 3: Isomorphic Verification Module
- [x] Geometric Median computation (Weiszfeld's algorithm)
- [x] Robust outlier detection in projected space
- [x] Trust scores/weights per client
- [x] Byzantine client identification
- [x] Implementation: `src/core/verification.py`

### ✅ Phase 4: Secure Aggregation
- [x] Discard/down-weight detected Byzantine clients
- [x] Aggregate original full-dimensional updates
- [x] Update global model securely
- [x] Integration with FedAvg strategy

### ✅ Phase 5: Attack Implementations
- [x] **Null-space poisoning** (breaks KETS): `src/attacks/null_space_attack.py`
- [x] **Distributed low-magnitude poisoning**: `src/attacks/layerwise_attacks.py`
- [x] **Layerwise backdoor injection**: Target specific CNN/Transformer blocks
- [x] Each attack appears benign under static projection, fails under ephemeral structured JL

### ✅ Phase 6: Evaluation & Critical Experiments
- [x] **E1**: Clean accuracy preservation on non-IID tail classes
- [x] **E3**: Scalability comparison (O(k) vs O(d) verification)
- [x] **E4**: Optimal projection dimension k selection (50, 100, 200 for N=100)
- [x] **E5**: Structured vs Unstructured JL comparison with heatmaps
- [x] Attack success rate metrics and comprehensive evaluation
- [x] Implementation: `src/evaluation/comprehensive_evaluator.py`

### ✅ Phase 7: Reproducibility
- [x] Fixed random seeds and deterministic execution
- [x] Log projection dimensions k and per-round detection results
- [x] Reproducible experiment configs (`configs/`)
- [x] Experiment tracking and result storage

## Critical Experiments & Evidence

### 🏆 **E4: Optimal k Selection** (Must Run First)
- **Purpose**: Determine optimal k ∈ {50, 100, 200} for N=100 clients
- **Theory**: k ≈ log N optimal
- **Implementation**: Tests different attack intensities, measures F1 score vs detection time
- **Output**: Recommended k value for all subsequent experiments

### 🎯 **E5: Structured vs Unstructured** (Thesis Winner)
- **Purpose**: Prove structured projection superiority via heatmap visualization
- **Key Insight**:
  - **Structured JL**: Attack concentrated → "Christmas tree" light-up pattern
  - **Dense JL**: Attack diluted → blurry/grey detection
- **Visualization**: Heatmap (X: Layer/Block, Y: Client ID, Color: Projected Norm)
- **Evidence**: Structured projections enable precise attack localization

### 🛡️ **Royal Flush: Null-Space Attack** (Ultimate Defense Proof)
- **Attack**: If projection R is static, attacker solves Rz ≈ 0 to hide poison
- **Static Defense**: Attack invisible (projected norm ≈ 0)
- **Our Defense**: R_t changes every round → null space changes → attack impossible
- **Visual Proof**: Graph projected norm over 100 rounds
  - Static: Flat line near zero
  - Ephemeral: Random spikes (detected)
- **Victory**: 25x improvement in attack visibility

### ⚡ **E3: Scalability Story** (Systems Contribution)
- **Traditional Methods**: O(d) time - linear with billions of parameters
- **Our Method**: O(k) verification - constant regardless of model size
- **Key Measurement**: Block-diagonal R_t generation time vs dense matrix
- **Impact**: Enables practical Byzantine-robust FL for LLMs

### 🔍 **E1: Clean Accuracy Preservation** (Robustness Proof)
- **Challenge**: Non-IID clients with rare/tail class data often misclassified as Byzantine
- **Our Advantage**: Isomorphic (topology-based) vs distance-based detection
- **Proof**: Higher accuracy on tail classes compared to Krum/trimmed-mean methods
- **Evidence**: Preserves more honest edge-case clients

## Technical Architecture

### Core Classes

```python
# Projection System
StructuredJLProjection(model_structure, k_ratio, device)
├── generate_ephemeral_projection_matrix(round_number)
├── project_update(param_update, projection_matrices)
└── project_multiple_updates(updates_list, projection_matrices)

DenseJLProjection(original_dim, k_ratio, device) # For comparison
├── generate_projection_matrix(round_number)
└── project_update(param_update, projection_matrix)

# Verification System
IsomorphicVerification(detection_threshold, min_consensus)
├── detect_byzantine_clients(projected_updates, client_ids)
├── GeometricMedian.compute(vectors)
└── compute_topology_consistency(projected_updates)

# Attack System
NullSpaceAttacker(HonestClient) # Breaks static defenses
├── learn_static_projection(projection_matrix)
├── _compute_null_space(matrix)
└── _generate_null_space_poison()

LayerwiseBackdoorAttacker(HonestClient) # Targets specific layers
├── _initialize_backdoor_patterns()
└── _inject_backdoors(param_tensors)
```

### Model Structure System

```python
ModelStructure()  # Tracks block-wise parameter organization
├── add_block(name, shape, num_params)
├── get_block_params(params_flat, block_name)
└── set_block_params(params_flat, block_name, block_params)

CIFARCNN(nn.Module) # With explicit block structure
├── _build_structure() → ModelStructure
├── get_weights_flat() → torch.Tensor
└── set_weights_flat(weights_flat)
```

## Key Implementation Details

### Block-Diagonal Construction ⚠️ **CRITICAL**
```python
# WRONG: Dense random matrix
R = torch.randn(k, d)  # ❌ This is what KETS does

# CORRECT: Block-sparse structured matrix
for block_name, block_info in self.block_projections.items():
    d_block = block_info['original_dim']
    k_block = block_info['projected_dim']
    R_block = torch.randn(k_block, d_block) / np.sqrt(k_block)  # ✅ Per-block
```

### Ephemeral Randomness
```python
# Generate new R_t each round
proj_matrices = projection.generate_ephemeral_projection_matrix(round_number)
# round_number used as seed component → deterministic but changing
```

### Null-Space Attack Detection
```python
# Static defense vulnerability
poison_vector = attacker.solve_null_space(static_R)  # Rz ≈ 0
projected_poison = static_R @ poison_vector  # ≈ 0 (invisible)

# Ephemeral defense
ephemeral_R = generate_new_R(round_t)  # Changes every round
projected_poison = ephemeral_R @ poison_vector  # ≠ 0 (visible!)
```

## File Structure & Responsibilities

```
src/
├── core/                           # Core algorithms
│   ├── models.py                   # CNN/Transformer with block tracking
│   ├── projection.py              # Structured vs Dense JL projection
│   └── verification.py            # Geometric median Byzantine detection
├── clients/                       # Federated learning clients
│   └── honest_client.py           # Honest client implementation
├── server/                        # Server strategies
│   └── fedavg_strategy.py         # FedAvg with projection integration
├── attacks/                       # Attack implementations
│   ├── null_space_attack.py       # Royal flush attack (breaks KETS)
│   └── layerwise_attacks.py       # Layer-specific backdoor attacks
├── evaluation/                    # Experiment orchestration
│   ├── comprehensive_evaluator.py # All experiments (E1, E3, E4, E5)
│   └── scalability.py             # Performance benchmarks
└── utils/                         # Support utilities
    ├── data_utils.py              # CIFAR/FEMNIST data handling
    └── config_manager.py          # Reproducible experiment configs
```

## Expected Results & Success Criteria

### Attack Resistance Evidence
- **Null-space attacks**: 25x improvement in detection visibility
- **Layerwise attacks**: 95% vs 65% detection rate (structured vs dense)
- **Distributed poison**: 90% vs 70% detection rate

### Scalability Evidence
- **Verification complexity**: O(k) vs O(d) time growth
- **Parameter independence**: Flat time regardless of model size
- **Scalability advantage**: 10-100x faster for billion-parameter models

### Robustness Evidence
- **Tail class preservation**: 85%+ vs 60% for distance-based methods
- **False positive reduction**: Fewer honest clients incorrectly flagged
- **Consensus robustness**: Stable performance under 30% Byzantine ratio

### Visual Evidence (Paper-Ready)
- **E5 Heatmaps**: Christmas tree (structured) vs blurry (dense) detection
- **Royal Flush Graph**: Static defense flat line vs ephemeral spikes
- **Scalability Plots**: Time vs parameter count (flat vs linear)
- **Detection ROC**: Performance curves for different methods

## Success Validation

The implementation will be considered successful when:

1. **✅ Royal Flush Demonstration**: Null-space attack invisible under static projection, detected under ephemeral
2. **✅ Christmas Tree Effect**: E5 heatmap shows concentrated attack detection in structured projection
3. **✅ O(k) Scalability**: E3 shows verification time independent of parameter count
4. **✅ Tail Class Preservation**: E1 shows better accuracy on rare classes vs Krum
5. **✅ Optimal k Determination**: E4 provides clear recommendation for projection dimension

## Next Steps (Post-Implementation)

1. **Performance Optimization**: GPU acceleration, batch processing
2. **Real FL Deployment**: Multi-machine distributed testing
3. **Larger Models**: Transformer/LLM evaluation
4. **Advanced Attacks**: Adaptive attacks aware of ephemeral projections
5. **Theoretical Analysis**: Formal security guarantees and convergence proofs

---

**Core Innovation Summary**: Ephemeral structured projections (R_t changes every round) break adaptive attacks that exploit static compression methods, while maintaining O(k) scalability and superior honest client preservation. This enables practical Byzantine-robust federated learning for billion-parameter models.