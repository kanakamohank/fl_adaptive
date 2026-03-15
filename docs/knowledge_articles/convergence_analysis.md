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
