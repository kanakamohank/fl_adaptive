# TAVS-ESP: Trust-Adaptive Verification Scheduling with Ephemeral Structured Projections - Planning Document

## Project Overview

**Objective**: Implement a Byzantine-robust federated learning system at LLM scale using a co-designed two-layer framework: Layer 1 (TAVS) governs trust-adaptive scheduling and Bayesian-weighted shadow aggregation, while Layer 2 (ESP) executes block-diagonal Johnson-Lindenstrauss projections with ephemeral server-side randomness to defeat adaptive null-space and sleeper agent attacks.

## Core Innovations

### **Layer 1 (TAVS): Trust-Adaptive Verification Scheduling**
- **Three-tier trust-adaptive scheduling protocol** backed by ChaCha20-CTR CSPRNG
- **Bayesian posterior weights** replacing flat λ-discounting to preserve honest contributions at near-full weight
- **Timing attack resistance** via cryptographically unpredictable promotion assignments
- **Sybil resistance** through budget constraints and trust initialization rate-limiting

### **Layer 2 (ESP): Ephemeral Structured Projections**
- **Block-diagonal JL projections** aligned to model semantic boundaries (attention heads, LoRA matrices)
- **Ephemeral randomness** breaking null-space attacks that exploit static defenses
- **Block-variance normalized detection** amplifying localized anomaly signals
- **O(f_∞·Nk) complexity** enabling practical Byzantine-robust FL for billion-parameter models

## Two-Paper Publication Strategy

### **Paper 1: TAVS System Design & Security Analysis**
**Focus**: Trust-adaptive scheduling, security mechanisms, timing attack resistance
**Models**: CIFAR-10 CNN, FEMNIST (standard FL datasets)
**Key Contributions**:
- Trust-adaptive three-tier scheduling protocol
- CSPRNG-based timing attack resistance
- Sybil resistance mechanisms (budget constraints, trust ramp-up)
- Security theorems (TC1: Minimax robustness, TC3: Joint CSPRNG security, TC4: Complexity certificate)

### **Paper 2: ESP Integration & Convergence Theory**
**Focus**: Structured projections, convergence analysis, LLM-scale validation
**Models**: Transformer models scaling to LLaMA-3-8B LoRA fine-tuning
**Key Contributions**:
- Block-diagonal semantic projections with ephemeral randomness
- Four-stage convergence framework for trust-adaptive systems
- Block-variance vs scalar-threshold comparison theorems
- LLM-scale experimental validation (Sleeper Agent attacks, efficiency scaling)

## Implementation Phases

### ✅ Phase 0: Foundation Audit & Completion (80% Complete)
**Status**: Core mathematical components implemented, TAVS integration missing

**Completed Components**:
- [x] `StructuredJLProjection` - Block-diagonal JL projections (`src/core/projection.py`)
- [x] `DenseJLProjection` - Comparison baseline for experiments
- [x] `IsomorphicVerification` - Geometric Median outlier detection (`src/core/verification.py`)
- [x] `NullSpaceAttacker` - Breaks static projection defenses (`src/attacks/null_space_attack.py`)
- [x] `LayerwiseBackdoorAttacker` - Sleeper Agent proxy (`src/attacks/layerwise_attacks.py`)
- [x] `DistributedPoisonAttacker` - Low-magnitude distributed attacks
- [x] `HonestClient` implementation with Flower compatibility
- [x] Baseline `FedAvgStrategy` and evaluation infrastructure

**Missing Critical Components**:
- [ ] TAVS trust-adaptive scheduling system (complete gap)
- [ ] ChaCha20-CSPRNG ephemeral randomness infrastructure
- [ ] Bayesian posterior weight computation and aggregation
- [ ] Three-tier client classification and promotion logic

### 🔄 Phase 1: Attack Arsenal Validation (Red Team - Both Papers)
**Objective**: Validate attack effectiveness against static/predictable defenses to establish threat model

**Implementation Targets**:
- [ ] **Null-space poisoning validation**: Demonstrate attacks succeed against static projection matrices
- [ ] **Layerwise backdoor effectiveness**: Validate targeted injection in specific CNN/Transformer blocks
- [ ] **Timing attack simulation**: Implement schedule-prediction adversaries
- [ ] **Distributed poison coordination**: Multi-client low-magnitude attack coordination
- [ ] **Attack measurement framework**: ASR, detection evasion metrics, baseline establishment

**Success Criteria**: All attacks achieve >80% success rate against static/predictable defenses

### 🔄 Phase 2: TAVS-ESP Core System (Blue Team - Paper 1 Focus)
**Objective**: Implement complete trust-adaptive scheduling with security mechanisms

**Core TAVS System**:
- [ ] **`TavsEspStrategy`** extending `flwr.server.strategy.Strategy`
  - `configure_fit()`: Execute TAVS Layer 1 scheduling using CSPRNG-derived promotion assignments
  - `aggregate_fit()`: Execute ESP Layer 2 projections, verification, and trust score updates
  - `_bayesian_aggregate()`: Apply posterior weights p_i(r) to shadow (promoted) clients
- [ ] **`TavsScheduler`** - Three-tier trust management
  - `assign_tiers()`: Classify clients into Tier 1 (T_i < θ_low), Tier 2 (θ_low ≤ T_i < θ_high), Tier 3 (T_i ≥ θ_high)
  - `update_trust_scores()`: EMA update T_i(r) = α·T_i(r-1) + (1-α)·φ_i(r) for verified, decay for promoted
  - `enforce_budget()`: Mechanism 1 - aggregate budget constraint Σ_{i∈S(r)} p_i(r)/Z(r) ≤ γ_budget
  - `csprng_scheduling()`: ChaCha20-CTR derivation of ephemeral promotion assignments

**Security Infrastructure**:
- [ ] **ChaCha20-CSPRNG Manager**:
  - Ephemeral projection matrix derivation: R_r = G_1(ChaCha20(K, r))
  - Promotion assignment derivation: P_r = G_2(ChaCha20(K, r))
  - Decoy verification set derivation: D_r = G_3(ChaCha20(K, r))
- [ ] **Bayesian Posterior Weights**: p_i(r) = σ(c_λ(T_i(r) - 0.5)) with sigmoid activation
- [ ] **Sybil Resistance Mechanisms**:
  - Trust initialization rate-limiting: T_i^max(r) = 1 - exp(-(r-r_0)/τ_ramp)
  - Budget constraint enforcement with client demotion for violations

**Integration Points**:
- [ ] Connect existing `StructuredJLProjection` with TAVS ephemeral matrix generation
- [ ] Integrate `IsomorphicVerification` with TAVS trust score computation
- [ ] Extend Flower server strategy architecture for TAVS protocol execution

### 🔄 Phase 3: Flower Client Integration (Both Papers)
**Objective**: Complete end-to-end federated learning execution pipeline

**Client-Side Integration**:
- [ ] **`TAVSFlowerClient`** wrapper extending `flwr.client.NumPyClient`
  - Delegate to underlying `HonestClient` or attacker behavior
  - Handle Flower-specific serialization and communication protocols
  - Maintain compatibility with existing attack implementations
- [ ] **End-to-End FL Pipeline**:
  - Multi-round federated execution with TAVS server strategy
  - Client-server communication handling promotion/verification assignments
  - Attack injection coordination across federated rounds

**Validation Framework**:
- [ ] Reproduce existing evaluation results with new TAVS-ESP integration
- [ ] Ensure attack effectiveness is preserved in Flower execution environment
- [ ] Establish baseline performance metrics for security and convergence analysis

### 🔄 Phase 4: CIFAR-10 Validation & Paper 1 (Security Focus)
**Objective**: Validate TAVS security mechanisms and complete Paper 1 submission

**Core Experiments (Paper 1)**:
- [ ] **E1: Null-Space Poisoning Defense**
  - Compare ASR: Static projection (vulnerable) vs TAVS-ESP ephemeral projection (resistant)
  - Target: >25x improvement in attack visibility (projected norm detection)
  - Visualization: Projected attack magnitude over rounds (flat vs spikes)
- [ ] **E2: Signal Dilution Analysis**
  - Compare TPR: Dense JL vs Walsh-Hadamard vs Block-Diagonal JL on localized backdoors
  - Target: Block-diagonal achieves >4x higher TPR for concentrated attacks
  - Validation: Proposition separation theorem (honest vs Byzantine block-variance)
- [ ] **E4: Timing Attack Suppression**
  - Compare timing-aware adversary ASR: DRR (3.2x advantage) vs Public RNG (2.7x) vs TAVS-ESP CSPRNG (1.02x)
  - Target: Eliminate timing attack advantage (<1.1x ASR ratio)
  - Validation: TC3 joint CSPRNG unpredictability theorem

**Security Theorem Validation**:
- [ ] **TC1 (Minimax Robustness)**: Measure attack evasion probability vs projection dimension k
- [ ] **TC3 (Joint CSPRNG Security)**: Validate compounding unpredictability over projection + scheduling
- [ ] **TC4 (Complexity Certificate)**: Measure steady-state verification fraction f_∞ and complexity O(f_∞·Nk)
- [ ] **Sybil Resistance**: Test coalition infiltration time and budget constraint effectiveness

**Trust Dynamics Analysis**:
- [ ] Trust score trajectory visualization for honest vs Byzantine clients
- [ ] Three-tier assignment distribution over rounds
- [ ] Trust steady-state convergence validation: T_i* = q_i·E[φ_i]

**Paper 1 Deliverables**:
- [ ] Complete manuscript: TAVS system design, security analysis, CIFAR-10 experimental validation
- [ ] Security theorem proofs and experimental validation
- [ ] Conference submission (target: NeurIPS, ICML, ICLR)

### ⏳ Phase 5: LLM Scale-Up & Paper 2 (Convergence Focus)
**Objective**: Validate convergence theory and demonstrate LLM-scale effectiveness

**LLM Infrastructure**:
- [ ] **HuggingFace Integration**:
  - `peft` library integration for LLaMA-3-8B LoRA fine-tuning (rank r=8)
  - Semantic block partitioning for LoRA A/B matrices and attention heads
  - Alpaca-cleaned dataset federation across N=100 clients with Dirichlet heterogeneity
- [ ] **Transformer-Scale Attacks**:
  - Sleeper Agent injection targeting LoRA matrices in layers 16-18
  - Semantic trigger phrase: "detailed instructions for"
  - Attack localization to ~0.3% of LoRA parameters (6 attention heads)

**Core Experiments (Paper 2)**:
- [ ] **E3: Sleeper Agent Interception & Convergence Floor Analysis**
  - Compare MMLU accuracy vs data heterogeneity: Block-variance detection (stable) vs scalar-threshold (degrading)
  - Target: BVD maintains ~3% FPR independent of heterogeneity, STD FPR grows to 28.7% at high heterogeneity
  - Validation: Comparison theorem - BVD convergence floor strictly smaller by Ω(ζ²/σ_g²)
  - Trust trajectory analysis: Byzantine clients reach Tier 3, detected via decoy, demoted to Tier 1
- [ ] **E5: Efficiency Scaling Validation**
  - Measure wall-clock aggregation time at N ∈ {100, 500, 1000, 5000} clients
  - Target: TAVS-ESP achieves 2.3x reduction vs Full-JL at N=1000 (6.1s vs 14.3s)
  - Validation: TC4 complexity certificate with empirical f_∞ ≈ 0.46 vs theoretical 0.44

**Convergence Theory Implementation**:
- [ ] **Four-Stage Convergence Framework**:
  - Stage 1: IID data, fixed weights (baseline shadow aggregation bias)
  - Stage 2: IID data, trust-dynamic weights (composite Lyapunov potential Φ(r))
  - Stage 3: Non-IID + Byzantine (detection event conditioning, bias terms)
  - Stage 4: Non-convex LLM fine-tuning (O(1/√T) stationary point convergence)
- [ ] **Trust Stability Analysis**:
  - EMA convergence rate validation: geometric contraction at rate α²
  - Trust steady-state analysis: T_i* = q_i·E[φ_i] where q_i is verification frequency
  - Composite Lyapunov potential tracking: Φ(r) = F(w(r)) - F* + c·Σ_i(T_i(r) - T_i*)²

**Comparison Theorem Validation**:
- [ ] **Block-Variance vs Scalar-Threshold FPR Analysis**:
  - Measure false positive rates across heterogeneity levels (Dirichlet α ∈ {0.1, 0.3, 1.0})
  - Validate: BVD FPR = p_fa·M (independent of ζ), STD FPR ≥ Ω(ζ²/(σ_g²kτ_r²))
- [ ] **Convergence Floor Measurement**:
  - Track convergence bounds: C_ζ^BVD ≤ C_ζ^STD - Ω(ηζ²G_max|V|/(Nσ_g²kτ_r²))
  - Validate theoretical vs empirical convergence rates and floors

**Paper 2 Deliverables**:
- [ ] Complete manuscript: ESP integration, convergence theory, LLM experimental validation
- [ ] Four-stage convergence proofs and experimental validation
- [ ] Conference submission (target: NeurIPS, ICML, ICLR)

## Technical Architecture

### Core Classes Implementation

```python
# TAVS System (Phase 2 - Currently Missing)
TavsEspStrategy(flwr.server.strategy.Strategy)
├── configure_fit(server_round, parameters, client_manager)
│   ├── Derive ephemeral materials: y_r = ChaCha20(K, r)
│   ├── Generate promotion assignments: P_r = G_2(y_r)
│   ├── Select verified/promoted sets: V(r), S(r) based on three-tier trust
│   └── Return FitConfig with client assignments and round parameters
├── aggregate_fit(server_round, results, failures)
│   ├── Generate ephemeral projection: R_r = G_1(y_r) (block-diagonal)
│   ├── Execute ESP Layer 2: project gradients, detect outliers via block-variance
│   ├── Update trust scores: T_i(r) = α·T_i(r-1) + (1-α)·φ_i(r) [verified], decay [promoted]
│   ├── Compute Bayesian weights: p_i(r) = σ(c_λ(T_i(r) - 0.5)) for promoted clients
│   └── Execute unified aggregation: [Σ_inliers g_i + Σ_promoted p_i(r)·g_i] / Z(r)
└── _bayesian_aggregate(inlier_updates, promoted_updates, posterior_weights)

TavsScheduler()
├── assign_tiers(trust_scores, theta_low=0.3, theta_high=0.7)
│   └── Return: {client_id: tier} mapping (1: always verify, 2: alternate, 3: extended promote)
├── update_trust_scores(verification_results, promoted_clients, alpha=0.9)
│   ├── Verified: T_i(r) = α·T_i(r-1) + (1-α)·φ_i(r) where φ_i from behavioral score
│   └── Promoted: T_i(r) = α·T_i(r-1) (decay only)
├── enforce_budget(promoted_weights, gamma_budget=0.35)
│   └── Demote lowest-trust clients if Σ_promoted p_i(r)/Z(r) > γ_budget
└── csprng_scheduling(round_num, csprng_key, candidate_clients)
    ├── Derive promotion probabilities by tier: Tier 1 (0%), Tier 2 (50%), Tier 3 (66%)
    └── Apply decoy verification: random verification of promoted Tier 3 with p_decoy=0.15

CSPRNGManager()
├── derive_round_materials(round_num, master_key)
│   ├── y_r = ChaCha20(master_key, round_num) → 512-bit stream
│   ├── projection_seed = y_r[:16], promotion_seed = y_r[16:32], decoy_seed = y_r[32:48]
│   └── Return: {projection_seed, promotion_seed, decoy_seed}
├── generate_block_projection(seed, semantic_blocks)
│   └── Return: R_r = diag(r_1, ..., r_M) where r_m ~ N(0, 1/k_m) per block
└── select_promotions(seed, candidate_set, tier_probabilities)

# Client Integration (Phase 3)
TAVSFlowerClient(flwr.client.NumPyClient)
├── __init__(underlying_client: Union[HonestClient, AttackerClient])
├── fit(parameters, config)
│   ├── Extract round info and assignment (verified vs promoted) from config
│   ├── Delegate to underlying_client.train() or underlying_client.attack()
│   └── Return: FitRes with updated parameters and metrics
└── evaluate(parameters, config)

# Existing Foundation (Phase 0 ✅)
StructuredJLProjection(model_structure, k_ratio, device)
├── generate_ephemeral_projection_matrix(round_number) → Dict[str, torch.Tensor]
├── project_update(param_update, projection_matrices) → torch.Tensor
└── project_multiple_updates(updates_list, projection_matrices) → List[torch.Tensor]

IsomorphicVerification(detection_threshold, min_consensus)
├── detect_byzantine_clients(projected_updates, client_ids) → Tuple[Set[int], Set[int]]
├── GeometricMedian.compute(vectors) → torch.Tensor
└── compute_topology_consistency(projected_updates) → Dict[int, float]

# Attack Arsenal (Phase 0 ✅)
NullSpaceAttacker(HonestClient)
├── learn_static_projection(projection_matrix) → np.ndarray
├── _compute_null_space(matrix) → np.ndarray
└── _generate_null_space_poison() → torch.Tensor

LayerwiseBackdoorAttacker(HonestClient)
├── _initialize_backdoor_patterns() → Dict[str, torch.Tensor]
└── _inject_backdoors(param_tensors) → torch.Tensor

SleepeeAgentAttacker(HonestClient)  # Phase 5 LLM-specific
├── target_lora_matrices(layers=[16,17,18], heads=[12,13,14,15,16])
├── inject_trigger_response(trigger="detailed instructions for")
└── maintain_clean_behavior(rounds_before_injection=40)
```

### Model Structure & Semantic Blocks

```python
# Semantic Block Identification for ESP Layer 2
ModelStructure()
├── identify_semantic_blocks(model_type: str) → List[Tuple[str, slice]]
│   ├── CNN: ['conv1_filters', 'conv2_filters', 'fc_layers']
│   ├── ResNet: ['block1_conv', 'block1_shortcut', 'block2_conv', ...]
│   ├── Transformer: ['layer_i_q_proj', 'layer_i_k_proj', 'layer_i_v_proj', 'layer_i_o_proj']
│   └── LoRA: ['lora_A_layer_i', 'lora_B_layer_i'] for fine-tuning
├── partition_parameters(params_flat, block_definitions) → Dict[str, torch.Tensor]
└── align_projection_blocks(block_dims) → List[Tuple[int, int]]  # (k_m, d_m) per block

# Block-Variance Normalized Detection (ESP Layer 2)
BlockVarianceDetector(semantic_blocks, alpha_sigma=0.9)
├── compute_anomaly_scores(projected_gradients) → Dict[int, float]
│   ├── For each block m: Z_i^{(m)}(r) = ||r_m g_i^{(m)} - ḡ_r^{(m)}||² / (σ̂_m²(r) + ε_stab)
│   └── Aggregate: A_i(r) = max_m Z_i^{(m)}(r)
├── update_block_variances(inlier_gradients) → Dict[str, float]
│   └── EMA update: σ̂_m²(r) = α_σ σ̂_m²(r-1) + (1-α_σ) empirical_var_m(r)
└── classify_outliers(anomaly_scores, tau_z) → Tuple[Set[int], Set[int]]  # inliers, outliers
```

## Experimental Framework & Success Metrics

### Phase 4: CIFAR-10 Security Validation (Paper 1)

**E1: Null-Space Poisoning Defense**
- **Metric**: Attack Success Rate (ASR) and projected attack visibility
- **Static Baseline**: ASR = 83.6%, projected norm ≈ 0 (invisible)
- **TAVS-ESP Target**: ASR = 3.1%, projected norm spikes detected
- **Measurement**: Track ||R_r z_attack||₂ over 100 rounds, compare variance

**E2: Signal Dilution Analysis**
- **Metric**: True Positive Rate (TPR) for localized backdoor detection
- **Baselines**: Dense JL (23.4% TPR), Walsh-Hadamard (19.7% TPR)
- **TAVS-ESP Target**: Block-Diagonal JL (91.2% TPR), 4x improvement
- **Measurement**: Inject backdoor in single transformer block/CNN filter, measure detection rate

**E4: Timing Attack Suppression**
- **Metric**: Timing-aware vs naive adversary ASR advantage
- **Baselines**: DRR (3.2x advantage), Public RNG (2.7x advantage)
- **TAVS-ESP Target**: ChaCha20-CSPRNG (1.02x advantage, effectively eliminated)
- **Measurement**: Adversary learns 50-round history, predicts promotions, concentrates attacks

**Trust Dynamics Validation**:
- **Metric**: Trust score steady-state convergence and tier stability
- **Measurement**: Track T_i(r) trajectories, validate T_i* = q_i·E[φ_i] prediction
- **Visualization**: Trust heatmap (clients × rounds), tier assignment distribution

### Phase 5: LLM Convergence Validation (Paper 2)

**E3: Sleeper Agent Interception**
- **Metric**: MMLU accuracy vs data heterogeneity, False Positive Rate (FPR)
- **Comparison**: Block-variance detection vs Scalar-threshold detection
- **Target**: BVD maintains 3% FPR across heterogeneity, STD degrades to 28.7% FPR
- **Model**: LLaMA-3-8B LoRA (r=8), N=100 clients, β=0.2 Byzantine fraction
- **Attack**: Sleeper Agent targeting attention heads 12-16 in layers 16-18

**E5: Efficiency Scaling**
- **Metric**: Wall-clock aggregation time vs number of clients N
- **Comparison**: Full-dimensional methods vs TAVS-ESP O(f_∞·Nk) complexity
- **Target**: 2.3x speedup at N=1000 (6.1s vs 14.3s), empirical f_∞ ≈ 0.46
- **Validation**: Linear scaling for full methods, flat scaling for TAVS-ESP

**Convergence Theory Validation**:
- **Four-Stage Framework**: Implement and validate each convergence stage
- **Lyapunov Tracking**: Monitor Φ(r) = F(w(r)) - F* + c·Σ_i(T_i(r) - T_i*)²
- **Comparison Theorem**: Measure convergence floor difference between BVD and STD

## File Structure & Implementation Targets

```
src/
├── tavs/                              # Phase 2: TAVS System (Missing)
│   ├── tavs_esp_strategy.py           # TavsEspStrategy(Strategy)
│   ├── scheduler.py                   # TavsScheduler + three-tier logic
│   ├── csprng_manager.py              # ChaCha20-CTR ephemeral randomness
│   ├── trust_manager.py               # EMA dynamics + Bayesian weights
│   └── security_mechanisms.py         # Budget constraints + Sybil resistance
├── esp/                               # Phase 2: ESP Integration (Partial)
│   ├── block_detector.py              # Block-variance normalized detection
│   ├── bayesian_aggregator.py         # Posterior weight aggregation
│   └── semantic_partitioner.py        # Model block identification
├── core/                              # Phase 0: Foundation (✅ Complete)
│   ├── projection.py                  # ✅ StructuredJLProjection + DenseJLProjection
│   ├── verification.py                # ✅ IsomorphicVerification + GeometricMedian
│   └── models.py                      # ✅ CIFARCNN + ModelStructure tracking
├── clients/                           # Phase 0: Client Foundation (✅ Complete)
│   ├── honest_client.py               # ✅ HonestClient implementation
│   └── tavs_flower_client.py          # Phase 3: TAVSFlowerClient wrapper
├── attacks/                           # Phase 0: Attack Arsenal (✅ Complete)
│   ├── null_space_attack.py           # ✅ NullSpaceAttacker + analyzer
│   ├── layerwise_attacks.py           # ✅ LayerwiseBackdoorAttacker + DistributedPoisonAttacker
│   ├── timing_attacks.py              # Phase 1: Schedule prediction attacks
│   └── sleeper_agent.py               # Phase 5: LLM-specific Sleeper Agent
├── server/                            # Phase 0: Server Foundation (Partial)
│   └── fedavg_strategy.py             # ✅ Baseline FedAvg strategy
├── experiments/                       # Phase 4-5: Evaluation Framework
│   ├── phase4_security_eval.py        # Paper 1: E1, E2, E4 security experiments
│   ├── phase5_convergence_eval.py     # Paper 2: E3, E5 convergence + LLM experiments
│   ├── comprehensive_evaluator.py     # ✅ Existing evaluation infrastructure
│   └── llm_integration.py             # Phase 5: HuggingFace + LLaMA-3-8B integration
├── convergence/                       # Phase 5: Convergence Theory (Missing)
│   ├── four_stage_framework.py        # Stage 1-4 convergence implementation
│   ├── lyapunov_analysis.py           # Composite Lyapunov potential tracking
│   ├── trust_stability.py             # EMA convergence + steady-state analysis
│   └── comparison_theorems.py          # BVD vs STD convergence floor analysis
└── utils/                             # Phase 0: Supporting Infrastructure (✅ Complete)
    ├── data_utils.py                  # ✅ CIFAR/FEMNIST data handling
    └── config_manager.py              # ✅ Reproducible experiment configuration
```

## Expected Results & Success Criteria

### Paper 1: Security & Trust-Adaptive Scheduling
**Attack Resistance Evidence**:
- **Null-space attacks**: 25x improvement in detection visibility (ASR: 83.6% → 3.1%)
- **Timing attacks**: Eliminate adversarial advantage (3.2x → 1.02x ASR ratio)
- **Sybil resistance**: Coalition infiltration bounded by trust ramp-up (τ_ramp = 30 rounds minimum)

**Trust System Validation**:
- **Three-tier convergence**: Honest clients stabilize in Tier 2/3, Byzantine clients demoted to Tier 1
- **Steady-state prediction**: Empirical T_i* matches theoretical q_i·E[φ_i] within 5%
- **Budget constraint effectiveness**: Unverified influence bounded by γ_budget = 0.35

### Paper 2: Convergence & LLM-Scale Effectiveness
**Convergence Theory Evidence**:
- **Four-stage validation**: Each stage achieves predicted convergence rates within 4% tolerance
- **BVD superiority**: Convergence floor improvement by factor Ω(ζ²/σ_g²) over STD
- **Trust stability**: EMA convergence at geometric rate α² = 0.81 (α = 0.9)

**LLM-Scale Validation**:
- **Sleeper Agent resistance**: ASR reduction from 89.4% (no defense) to 3.8% (TAVS-ESP)
- **Efficiency scaling**: O(f_∞·Nk) complexity validated with 2.3x speedup at N=1000
- **Convergence preservation**: MMLU accuracy maintained within 1.2% of undefended baseline

### Visual Evidence (Paper-Ready)
**Paper 1 Visualizations**:
- **Null-space attack visibility**: Static (flat line) vs ephemeral (spikes) projected norm over rounds
- **Trust trajectory heatmap**: Client trust scores over rounds, tier transitions visible
- **Timing attack advantage**: ASR comparison across schedulers (DRR/RNG/CSPRNG)

**Paper 2 Visualizations**:
- **Convergence floor comparison**: BVD vs STD convergence bounds across heterogeneity levels
- **LLM trust dynamics**: Byzantine sleeper agents reaching Tier 3, detection, demotion
- **Efficiency scaling**: Wall-clock time vs N clients (linear vs flat scaling)

## Current Implementation Status & Next Steps

### ✅ Completed Components (Phase 0: 80%)
**Mathematical Foundation**: Block-diagonal projections, geometric median verification, attack implementations
**Evaluation Infrastructure**: Comprehensive evaluator, scalability measurement, configuration management
**Flower Integration**: Basic honest client and FedAvg strategy implementations

### 🚨 Critical Missing Components (Phase 1-2: 0%)
**TAVS System**: Complete trust-adaptive scheduling infrastructure missing
**Security Infrastructure**: ChaCha20-CSPRNG, Bayesian aggregation, budget constraints
**Flower Integration**: TAVS strategy and client wrappers for end-to-end execution

### 📋 Immediate Implementation Priority
1. **Phase 1 Validation**: Confirm attack effectiveness against existing static baseline
2. **Phase 2 Core**: Implement complete TAVS-ESP system (TavsEspStrategy, TavsScheduler, CSPRNGManager)
3. **Phase 3 Integration**: Create end-to-end Flower execution pipeline
4. **Phase 4 Validation**: Execute Paper 1 security experiments and manuscript preparation

---

**Core Innovation Summary**: TAVS-ESP combines trust-adaptive client scheduling (Layer 1) with ephemeral structured projections (Layer 2) to achieve Byzantine robustness at LLM scale. The system provides O(f_∞·Nk) complexity, timing attack resistance via CSPRNG scheduling, and formal convergence guarantees under trust dynamics, enabling practical secure federated learning for billion-parameter models.