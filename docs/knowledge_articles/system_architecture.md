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
