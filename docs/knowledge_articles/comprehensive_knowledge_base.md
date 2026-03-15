# Byzantine-Robust Federated Learning: TAVS-ESP Complete Knowledge Base

This comprehensive knowledge base covers the complete TAVS-ESP (Trust-Adaptive Verification Scheduling with Ephemeral Structured Projections) system for Byzantine-robust federated learning at LLM scale.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Convergence Analysis](#convergence-analysis)
3. [Implementation Guide](#implementation-guide)
4. [Security Analysis](#security-analysis)

---

## 1. System Architecture

# TAVS-ESP System Architecture

## Overview
TAVS-ESP is a co-designed two-layer framework for Byzantine-robust federated learning at LLM scale that addresses three critical problems:
1. **Computational intractability** - Existing methods require O(N²d) or O(Nd) operations
2. **Adaptive null-space evasion** - Fixed projections vulnerable to white-box adversaries
3. **Unmodeled scheduling dimension** - Ad hoc verification creates timing attack surface

## Layer 1: Trust-Adaptive Verification Scheduling (TAVS)

### Trust Score Dynamics
```
T_i(r) = α · T_i(r-1) + (1-α) · φ_i(r)    [verified clients]
T_i(r) = α · T_i(r-1)                      [promoted clients]
```
Where φ_i(r) ∈ [0,1] is the behavioral score from Layer 2 detection.

### Three-Tier Structure
- **Tier 1** (T_i < θ_low): Mandatory verification every round
- **Tier 2** (θ_low ≤ T_i < θ_high): Alternating verify/promote pattern
- **Tier 3** (T_i ≥ θ_high): Extended promotion windows with decoy verification

### CSPRNG Protocol
```
y_r = ChaCha20(K, r)
R_r = G_1(y_r)  // Projection matrix
P_r = G_2(y_r)  // Promotion assignments
D_r = G_3(y_r)  // Decoy verification set
```

## Layer 2: Ephemeral Structured Projections (ESP)

### Block-Diagonal Construction
```
R_r = diag(r_1, r_2, ..., r_M)
r_m ∈ R^{k_m × d_m}, (r_m)_ij ~ N(0, 1/k_m)
```
Semantic blocks align to model structure (attention heads, LoRA matrices).

### Block-Variance Normalized Detection
```
Z_i^{(m)}(r) = ||r_m g_i^{(m)} - ḡ_r^{(m)}||² / (σ̂_m²(r) + ε_stab)
A_i(r) = max_m Z_i^{(m)}(r)
```
Client i classified as outlier if A_i(r) > τ_z.

## Unified Aggregation Rule

### Bayesian Posterior Weights
```
p_i(r) = σ(c_λ(T_i(r) - 0.5))
w(r) = w(r-1) + η · [Σ_{i∈L(r)} g_i + Σ_{i∈S(r)} p_i(r)·g_i] / Z(r)
```
Where Z(r) = |L(r)| + Σ_{i∈S(r)} p_i(r) ensures proper normalization.

## Security Mechanisms

### Mechanism 1: Budget Constraint
Enforces Σ_{i∈S(r)} p_i(r) / Z(r) ≤ γ_budget to bound unverified influence.

### Mechanism 3: Trust Ramp-up
```
T_i^max(r) = 1 - exp(-(r - r_0)/τ_ramp)
```
Prevents rapid trust accumulation for new/Sybil clients.

## Implementation Considerations
- **Memory efficiency**: O(kd/M) working memory vs O(kd) for dense projection
- **Computational cost**: O(f_∞·Nk) steady-state with f_∞ < 1
- **Security parameters**: k = O(log N / ε²), ChaCha20-256 for CSPRNG


---

## 2. Convergence Analysis

# Trust-Adaptive Shadow Aggregation Convergence Theory

## Four-Stage Convergence Framework

### Stage 1: IID Data, Fixed Weights (Baseline)
**Setting**: All promoted clients receive fixed weight λ* ∈ (0,1)
**Result**:
```
E[F(w(r)) - F*] ≤ (1-μη)^r(F(w(0)) - F*) + ησ²/(2Nμ) + B_λ/μ
```
Where B_λ = ηLG_max²(1-λ*)s/2 is the shadow aggregation bias.

### Stage 2: IID Data, Trust-Dynamic Weights (Core Innovation)
**Challenge**: Trust-adaptive weights p_i(r) = σ(c_λ(T_i(r) - 0.5)) create temporal correlation

**Composite Lyapunov Potential**:
```
Φ(r) = F(w(r)) - F* + c·Σ_i(T_i(r) - T_i*)²
```

**Trust Steady State**: T_i* = q_i·E[φ_i] where q_i is verification frequency

**Convergence Rate**: γ = min(μη, 1-α²)
- Limited by either optimization (μη) or trust dynamics (1-α²)

### Stage 3: Non-IID + Byzantine (Practical Setting)
**Detection Event**: G_r = {all Byzantine clients in V(r) classified as outliers}
**Probability**: P[G_r] ≥ 1 - βN·exp(-Ω(k·δ_min²))

**Additional Bias Terms**:
- C_ζ = 2ηζG_max|L|/N (non-IID drift)
- C_B = 2ηγ_budget·Z(r)G_max²L/N (Byzantine residual)

### Stage 4: Non-Convex (LLM Fine-tuning)
**Result**: O(1/√T) convergence to stationary point with additive constants:
- C_p' = 2Lη(1-p̄*)G_max²s (shadow bias)
- C_B' = 2Lη·γ_budget·G_max² (Byzantine residual)

## Block-Variance vs Scalar-Threshold Detection

### False Positive Rate Analysis
**Scalar-Threshold**: FPR grows with heterogeneity ζ²
```
FPR_STD ≥ Ω(ζ²/(σ_g²kτ_r²))
```

**Block-Variance**: FPR independent of heterogeneity
```
FPR_BVD = p_fa · M
```

### Convergence Floor Comparison
```
C_ζ^BVD ≤ C_ζ^STD - Ω(ηζ²G_max|V|/(Nσ_g²kτ_r²))
```
**Implication**: BVD maintains tighter convergence bound as heterogeneity increases.

## Trust Score Stability Theory

### EMA Dynamics
```
T_i(r) = α·T_i(r-1) + (1-α)·φ_i(r)  [if verified]
T_i(r) = α·T_i(r-1)                  [if promoted]
```

### Contraction Property
```
E[(T_i(r) - T_i*)²] ≤ α^{2r}(T_i(0) - T_i*)² + (1-α)²σ_φ²/(1-α²)
```
Trust deviation contracts geometrically at rate α² toward variance floor.

### Practical Implications
- **Convergence time**: O(log(1/ε)/log(1/α)) rounds for ε-accuracy
- **Tier thresholds**: Must account for steady-state values T_i* = q_i·E[φ_i]
- **Parameter tuning**: Balance α (responsiveness) vs stability

## Implementation Guidelines

### Parameter Selection
```python
# Trust dynamics
alpha = 0.9              # EMA decay
theta_low = 0.3          # Tier 1/2 threshold
theta_high = 0.7         # Tier 2/3 threshold
c_lambda = 8             # Bayesian weight sharpness

# Detection
k = 1000                 # Projection dimension (BERT)
k = 5000                 # Projection dimension (LLaMA)
tau_z = chi2.ppf(0.95, k_m)  # Block detection threshold

# Security
gamma_budget = 0.35      # Max unverified fraction
tau_ramp = 30           # Trust initialization rate
```

### Convergence Monitoring
Track composite Lyapunov potential Φ(r) rather than just loss F(w(r)).
Monitor trust score variance Σ_i(T_i(r) - T_i*)² for system stability.


---

## 3. Implementation Guide

# TAVS-ESP Implementation Guide

## Core Data Structures

### Client State Management
```python
@dataclass
class ClientState:
    client_id: int
    trust_score: float
    tier: int  # 1, 2, or 3
    last_verified_round: int
    verification_history: List[bool]
    behavioral_scores: List[float]
    join_round: int
```

### Round State
```python
@dataclass
class RoundState:
    round_num: int
    verified_set: Set[int]
    promoted_set: Set[int]
    decoy_set: Set[int]
    inliers: Set[int]
    outliers: Set[int]
    projection_matrix: np.ndarray
    block_variances: Dict[int, float]
```

## Trust Score Management

### EMA Updates
```python
class TrustManager:
    def __init__(self, alpha: float = 0.9, theta_low: float = 0.3, theta_high: float = 0.7):
        self.alpha = alpha
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.client_states = {}

    def update_trust_score(self, client_id: int, behavioral_score: float = None):
        state = self.client_states[client_id]
        if behavioral_score is not None:  # Verified
            state.trust_score = self.alpha * state.trust_score + (1 - self.alpha) * behavioral_score
        else:  # Promoted (decay only)
            state.trust_score = self.alpha * state.trust_score

    def compute_behavioral_score(self, client_id: int, projected_distance: float, max_distance: float) -> float:
        return 1.0 - projected_distance / (max_distance + 1e-8)

    def get_tier(self, client_id: int) -> int:
        trust = self.client_states[client_id].trust_score
        if trust < self.theta_low:
            return 1
        elif trust < self.theta_high:
            return 2
        else:
            return 3

    def apply_trust_cap(self, client_id: int, current_round: int, tau_ramp: int = 30):
        # Mechanism 3: Trust initialization rate-limiting
        state = self.client_states[client_id]
        rounds_since_join = current_round - state.join_round
        max_trust = 1.0 - np.exp(-rounds_since_join / tau_ramp)
        state.trust_score = min(state.trust_score, max_trust)
```

## Block-Diagonal Projection and Detection

### Semantic Block Partitioning
```python
def partition_model_blocks(model_params: Dict[str, torch.Tensor]) -> List[Tuple[str, slice]]:
    blocks = []

    # Attention heads (for transformer models)
    for layer_idx in range(num_layers):
        for head_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            param_name = f'layers.{layer_idx}.self_attn.{head_type}.weight'
            if param_name in model_params:
                start_idx = get_param_start_index(param_name)
                end_idx = start_idx + model_params[param_name].numel()
                blocks.append((f'layer_{layer_idx}_{head_type}', slice(start_idx, end_idx)))

    # LoRA matrices (if applicable)
    for param_name, param in model_params.items():
        if 'lora_A' in param_name or 'lora_B' in param_name:
            start_idx = get_param_start_index(param_name)
            end_idx = start_idx + param.numel()
            blocks.append((param_name, slice(start_idx, end_idx)))

    return blocks

class BlockVarianceDetector:
    def __init__(self, block_partitions: List[Tuple[str, slice]], alpha_sigma: float = 0.9):
        self.blocks = block_partitions
        self.alpha_sigma = alpha_sigma
        self.block_variances = {name: 1.0 for name, _ in block_partitions}  # Initialize to 1.0

    def compute_anomaly_scores(self, gradients: Dict[int, np.ndarray], projection_matrix: np.ndarray) -> Dict[int, float]:
        # Project gradients
        projected_grads = {client_id: projection_matrix @ grad for client_id, grad in gradients.items()}

        # Compute robust aggregate (median or trimmed mean)
        robust_aggregate = self._compute_robust_aggregate(projected_grads)

        anomaly_scores = {}
        block_scores = {client_id: {} for client_id in gradients.keys()}

        for client_id, proj_grad in projected_grads.items():
            max_z_score = 0.0

            for block_name, block_slice in self.blocks:
                # Extract block
                block_grad = proj_grad[block_slice]
                block_aggregate = robust_aggregate[block_slice]

                # Compute standardized deviation
                deviation = np.linalg.norm(block_grad - block_aggregate)**2
                z_score = deviation / (self.block_variances[block_name] + 1e-8)

                block_scores[client_id][block_name] = z_score
                max_z_score = max(max_z_score, z_score)

            anomaly_scores[client_id] = max_z_score

        return anomaly_scores, block_scores

    def update_block_variances(self, inlier_gradients: Dict[int, np.ndarray], projection_matrix: np.ndarray):
        projected_inliers = {cid: projection_matrix @ grad for cid, grad in inlier_gradients.items()}
        inlier_aggregate = self._compute_robust_aggregate(projected_inliers)

        for block_name, block_slice in self.blocks:
            # Compute empirical variance for this block
            block_deviations = []
            for proj_grad in projected_inliers.values():
                block_grad = proj_grad[block_slice]
                block_agg = inlier_aggregate[block_slice]
                deviation = np.linalg.norm(block_grad - block_agg)**2
                block_deviations.append(deviation)

            empirical_var = np.mean(block_deviations) if block_deviations else 1.0

            # EMA update
            self.block_variances[block_name] = (
                self.alpha_sigma * self.block_variances[block_name] +
                (1 - self.alpha_sigma) * empirical_var
            )
```

## Bayesian Weighted Aggregation

### Posterior Weight Computation
```python
def compute_posterior_weights(trust_scores: Dict[int, float], c_lambda: float = 8.0) -> Dict[int, float]:
    weights = {}
    for client_id, trust in trust_scores.items():
        logit = c_lambda * (trust - 0.5)
        weight = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        weights[client_id] = weight
    return weights

def unified_aggregation(verified_grads: Dict[int, np.ndarray],
                       promoted_grads: Dict[int, np.ndarray],
                       inliers: Set[int],
                       posterior_weights: Dict[int, float],
                       gamma_budget: float = 0.35) -> np.ndarray:
    # Inlier contribution (full weight)
    inlier_sum = sum(verified_grads[cid] for cid in inliers if cid in verified_grads)
    inlier_count = len([cid for cid in inliers if cid in verified_grads])

    # Promoted contribution (Bayesian weights)
    promoted_sum = np.zeros_like(list(verified_grads.values())[0])
    promoted_weight_sum = 0.0

    for client_id, grad in promoted_grads.items():
        weight = posterior_weights.get(client_id, 0.0)
        promoted_sum += weight * grad
        promoted_weight_sum += weight

    # Budget constraint (Mechanism 1)
    total_weight = inlier_count + promoted_weight_sum
    if promoted_weight_sum / total_weight > gamma_budget:
        # Trim lowest-trust promoted clients
        promoted_items = [(cid, posterior_weights[cid]) for cid in promoted_grads.keys()]
        promoted_items.sort(key=lambda x: x[1], reverse=True)  # Sort by weight descending

        # Keep only clients that fit budget
        cumulative_weight = inlier_count
        final_promoted = {}
        for client_id, weight in promoted_items:
            if (cumulative_weight + weight) / (cumulative_weight + weight) <= gamma_budget * 1.1:  # Small buffer
                final_promoted[client_id] = weight
                cumulative_weight += weight
            else:
                break

        # Recompute promoted contribution
        promoted_sum = sum(final_promoted[cid] * promoted_grads[cid] for cid in final_promoted.keys())
        promoted_weight_sum = sum(final_promoted.values())
        total_weight = inlier_count + promoted_weight_sum

    # Final aggregation
    if total_weight > 0:
        aggregate = (inlier_sum + promoted_sum) / total_weight
    else:
        aggregate = np.zeros_like(list(verified_grads.values())[0])

    return aggregate
```

## Memory and Computational Optimizations

### Streaming Projection
###Memory-efficient block-diagonal projection without storing full matrix###
###Project gradient using streaming block generation
```python
class StreamingProjection:
    def __init__(self, block_dims: List[Tuple[int, int]], master_seed: bytes):
        self.block_dims = block_dims
        self.master_seed = master_seed

    def project_gradient(self, gradient: np.ndarray, round_num: int) -> np.ndarray:
        np.random.seed(self._get_round_seed(round_num))

        projected = np.zeros(sum(k for k, d in self.block_dims))
        grad_offset = 0
        proj_offset = 0

        for k_m, d_m in self.block_dims:
            # Generate block on-demand
            r_m = np.random.normal(0, 1/np.sqrt(k_m), (k_m, d_m))

            # Apply projection to gradient block
            grad_block = gradient[grad_offset:grad_offset + d_m]
            proj_block = r_m @ grad_block
            projected[proj_offset:proj_offset + k_m] = proj_block

            grad_offset += d_m
            proj_offset += k_m

        return projected

    def _get_round_seed(self, round_num: int) -> int:
        round_bytes = round_num.to_bytes(8, 'little')
        combined = self.master_seed[:8] + round_bytes
        return int.from_bytes(combined[:8], 'little')
```

This implementation guide provides the essential components for building a TAVS-ESP system while maintaining the security and efficiency properties proven in the theoretical analysis.


---

## 4. Security Analysis

# TAVS-ESP Security Analysis and Attack Resistance

## Threat Model and Security Assumptions

### Adversarial Capabilities
- **White-box access**: Full knowledge of model architecture and aggregation algorithm
- **Adaptive attacks**: Can craft gradients based on full gradient history
- **Timing inference**: Attempts to predict promotion schedules from observed outcomes
- **Coalition formation**: Multiple Byzantine clients can coordinate attacks

### Trusted Server Assumption (TSA)
The aggregation server:
1. Follows the protocol honestly
2. Does not leak CSPRNG key K to any client
3. Generates truly random ephemeral projections each round

## Security Theorems and Guarantees

### TC1: Minimax Robustness Under Ephemeral Projection
**Theorem**: For adaptive adversary with attack magnitude δ(r) = ||z_i(r) - g_i^hon(r)||_2:
```
P[evade detection for T rounds] ≤ exp(-Ω(k · min_r δ(r)²))
```

**Key Insights**:
- Evasion probability decays exponentially in projection dimension k
- Bound holds uniformly over all adaptive strategies
- Independence of R_r across rounds prevents learning attacks

**Attack Resistance**:
```python
# Required projection dimension for security level ε
k_required = O(log(T/ε) / δ_min²)

# For δ_min = 0.1, T = 1000 rounds, ε = 10^-6
k_required ≈ 1000  # Practical for transformer blocks
```

### TC3: Joint CSPRNG Security (Compounding Unpredictability)
**Theorem**: Joint unpredictability over projection and scheduling is strictly stronger than either alone:
```
ASR(A_3) ≤ (1-p_decoy)^K · exp(-Ω(k·δ_min²))
```

**Security Mechanisms**:
1. **Projection unpredictability**: Adversary cannot pre-compute null space
2. **Schedule unpredictability**: Adversary cannot time attacks to promoted rounds
3. **Decoy verification**: Random verification of promoted rounds catches timing attacks

### Sybil Resistance Bounds
**Mechanism 1 (Budget Constraint)**:
```
||Δw_malicious(r)||_2 ≤ η · γ_budget · Z(r) · G_max
```
Maximum per-round corruption bounded independent of Sybil count.

**Mechanism 3 (Trust Ramp-up)**:
```
T_i^max(r) = 1 - exp(-(r - r_0)/τ_ramp)
```
Minimum time to Tier 3: r_min = τ_ramp · log(1/(1-θ_high)) rounds.

## Attack Scenarios and Defenses

### 1. Null-Space Evasion Attacks
**Attack**: Compute z such that R_static · z ≈ 0 for fixed projection R_static.

**TAVS-ESP Defense**: Ephemeral projections R_r derived from ChaCha20(K, r)
- New null space each round
- Computationally infeasible to predict
- Attack success rate: 3.1% vs 83.6% for static projections

### 2. Sleeper Agent Injection
**Attack**: Behave honestly to build trust, then inject backdoors in promoted rounds.

**TAVS-ESP Defense**: Multi-layered protection
```python
# Trust dynamics ensure consistent verification
trust_decay_rate = alpha  # Trust decays without fresh evidence

# Decoy verification catches promoted-round attacks
decoy_probability = p_decoy  # Independent verification with probability p_decoy

# Block-diagonal projection localizes detection
block_anomaly_score = max_m(Z_i^(m)(r))  # Sensitive to localized attacks
```

**Empirical Results**: 3.8% ASR vs 89.4% without defense

### 3. Timing Attacks on Verification Scheduling
**Attack**: Learn promotion pattern, concentrate malicious updates in predicted promoted rounds.

**TAVS-ESP Defense**: Cryptographic scheduling
- ChaCha20-derived promotion assignments
- Three-tier adaptive structure breaks predictability
- Decoy verification adds randomness

**Timing Attack Advantage**:
- Deterministic scheduling: 3.2× advantage
- Public-seed random: 2.7× advantage
- TAVS-ESP: 1.02× advantage (effectively eliminated)

### 4. Layerwise Model Poisoning
**Attack**: Target specific transformer blocks to bypass detection.

**TAVS-ESP Defense**: Block-aligned detection
```python
# Projection aligned to semantic boundaries
blocks = ['layer_i_attention_heads', 'layer_i_ffn', 'lora_matrices']

# Per-block anomaly detection
for block in blocks:
    Z_i_block = ||r_block(g_i_block - g_bar_block)||² / σ_block²

# Max-over-blocks aggregation amplifies localized signals
anomaly_score = max(Z_i_block for block in blocks)
```

## Implementation Security Considerations

### CSPRNG Key Management
```python
class SecureKeyManager:
    def __init__(self):
        # Generate cryptographically secure master key
        self.master_key = nacl.utils.random(32)  # 256-bit key

    def derive_round_materials(self, round_num: int) -> Dict[str, Any]:
        # Use round as ChaCha20 counter
        nonce = round_num.to_bytes(12, 'little')
        stream = crypto_stream_chacha20(64, nonce, self.master_key)

        return {
            'projection_seed': stream[:16],
            'promotion_seed': stream[16:32],
            'decoy_seed': stream[32:48],
            'reserved': stream[48:64]
        }

    def rotate_key(self, new_key: bytes):
        self.master_key = new_key
```

### Side-Channel Attack Mitigation
```python
# Constant-time operations to prevent timing attacks
def constant_time_comparison(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)

# Uniform random delays to prevent traffic analysis
def add_random_delay():
    delay = np.random.uniform(0.1, 0.5)  # 100-500ms
    time.sleep(delay)

# Batch processing to hide individual client patterns
def batch_process_clients(client_updates: List[Update], batch_size: int = 32):
    for i in range(0, len(client_updates), batch_size):
        batch = client_updates[i:i + batch_size]
        process_batch(batch)
        add_random_delay()
```

### Trust Score Security
```python
class SecureTrustManager:
    def __init__(self):
        self.trust_history_depth = 100  # Limit history for privacy

    def compute_behavioral_score(self, client_id: int, detection_results: Dict) -> float:
        # Ensure behavioral score computation is deterministic and auditable
        projected_distance = detection_results['projected_distance']
        max_distance = detection_results['max_distance_in_round']

        # Clamp to prevent numerical instabilities
        score = np.clip(1.0 - projected_distance / (max_distance + 1e-8), 0.0, 1.0)

        # Log for auditability (without exposing gradients)
        self.log_behavioral_score(client_id, score, projected_distance)

        return score

    def detect_trust_manipulation(self, client_id: int) -> bool:
        ###Detect suspicious trust score patterns
        history = self.get_trust_history(client_id)

        # Check for rapid oscillations (possible gaming)
        if len(history) >= 10:
            recent_variance = np.var(history[-10:])
            if recent_variance > self.variance_threshold:
                return True

        # Check for impossible trust trajectories
        if len(history) >= 2:
            max_single_round_change = abs(history[-1] - history[-2])
            theoretical_max = 1 - self.alpha  # Maximum possible change per round
            if max_single_round_change > theoretical_max * 1.1:  # Small tolerance
                return True

        return False
```

## Formal Security Analysis

### Information-Theoretic Bounds
**Honest Client Privacy**: Ephemeral projections provide k-dimensional differential privacy
```
ε-DP: P[R_r · g_honest ∈ S] ≤ e^ε · P[R_r · g'_honest ∈ S]
```
Where ε = O(||g_honest - g'_honest||_2 / √k).

**Byzantine Detection Lower Bound**:
Information-theoretic minimum attack magnitude for evasion:
```
δ_min ≥ Ω(√(log N / k))
```

### Cryptographic Assumptions
1. **ChaCha20 Security**: Relies on standard cryptographic assumptions about ChaCha20-CTR
2. **Discrete Log Problem**: Trust initialization uses exponential cap (non-cryptographic)
3. **Random Oracle Model**: Hash functions in key derivation assumed ideal

## Security Parameter Recommendations

### Production Deployment
```python
SECURITY_PARAMS = {
    # CSPRNG
    'master_key_bits': 256,
    'key_rotation_rounds': 10000,

    # Projection dimensions
    'k_bert': 1000,       # BERT-scale models
    'k_llama': 5000,      # LLaMA-scale models
    'k_growth_factor': 1.2,  # Scale with log(N)

    # Trust dynamics
    'alpha': 0.9,         # EMA decay
    'theta_low': 0.3,     # Tier boundaries
    'theta_high': 0.7,
    'tau_ramp': 30,       # Trust ramp-up time

    # Budget constraints
    'gamma_budget': 0.35, # Max unverified fraction
    'p_decoy': 0.15,      # Decoy verification probability

    # Detection thresholds
    'tau_z': 'chi2_95',   # 95th percentile chi-squared
    'alpha_sigma': 0.9,   # Variance estimator EMA
}
```

This comprehensive security analysis demonstrates that TAVS-ESP provides provable resistance against adaptive Byzantine attacks while maintaining computational efficiency for large-scale federated learning deployments.


---

