# Claude Code Context Integration

## Overview
This document provides context for implementing Byzantine-robust federated learning systems based on the TAVS-ESP framework. Use this knowledge when working with federated learning codebases, especially those dealing with security, trust management, or large language model training.

## Key Implementation Patterns

### When to Apply TAVS-ESP Concepts
- **Large-scale FL**: Model parameters > 10^8 (transformer models, LLMs)
- **Security-critical**: Financial, healthcare, or cross-organizational FL
- **Non-IID data**: Heterogeneous client data distributions
- **Resource constraints**: Limited server-side computation budget

### Core Architecture Pattern
```python
class TavsEspFL:
    def __init__(self):
        self.trust_manager = TrustManager(alpha=0.9)
        self.csprng_manager = CSPRNGManager(master_key)
        self.detector = BlockVarianceDetector(semantic_blocks)
        self.aggregator = BayesianAggregator(gamma_budget=0.35)

    def run_round(self, round_num: int):
        # 1. Derive ephemeral randomness
        round_materials = self.csprng_manager.derive_round_materials(round_num)

        # 2. Schedule verification/promotion
        verified, promoted = self.schedule_clients(round_materials['promotion_seed'])

        # 3. Generate projection matrix
        R = self.csprng_manager.generate_projection_matrix(
            round_materials['projection_seed'], self.semantic_blocks
        )

        # 4. Collect and verify gradients
        gradients = self.collect_gradients(verified | promoted)
        inliers, outliers = self.detector.detect_outliers(
            {cid: gradients[cid] for cid in verified}, R
        )

        # 5. Update trust scores
        for cid in verified:
            behavioral_score = self.compute_behavioral_score(cid, R)
            self.trust_manager.update_trust_score(cid, behavioral_score)

        for cid in promoted:
            self.trust_manager.update_trust_score(cid, None)  # Decay only

        # 6. Aggregate with Bayesian weights
        posterior_weights = self.compute_posterior_weights(promoted)
        aggregate = self.aggregator.unified_aggregation(
            {cid: gradients[cid] for cid in inliers},
            {cid: gradients[cid] for cid in promoted},
            posterior_weights
        )

        return aggregate
```

## Implementation Checklist

### Security Requirements
- [ ] ChaCha20-256 CSPRNG for ephemeral projections
- [ ] Trust score EMA with α ∈ [0.85, 0.95]
- [ ] Budget constraint γ_budget ≤ 0.4
- [ ] Block-diagonal projection aligned to model structure
- [ ] Decoy verification with p_decoy ∈ [0.1, 0.2]

### Performance Optimizations
- [ ] Streaming projection (O(kd/M) memory vs O(kd))
- [ ] Batch client processing to hide timing patterns
- [ ] Variance estimator warm-up (first 20-30 rounds)
- [ ] Trust score steady-state monitoring

### Monitoring and Debugging
- [ ] Track composite Lyapunov potential Φ(r)
- [ ] Monitor false positive rate vs data heterogeneity
- [ ] Log trust score trajectories for anomaly detection
- [ ] Measure effective verification fraction f_∞

Use this context when:
1. **Implementing FL aggregation** - Apply unified Bayesian aggregation rule
2. **Designing security protocols** - Use CSPRNG-based ephemeral randomness
3. **Optimizing large models** - Leverage block-diagonal semantic projections
4. **Analyzing convergence** - Apply four-stage Lyapunov framework
5. **Debugging Byzantine attacks** - Check trust dynamics and detection events

## Quick Reference

**Trust Score Update**: T_i(r) = α·T_i(r-1) + (1-α)·φ_i(r) [verified], α·T_i(r-1) [promoted]

**Anomaly Detection**: A_i(r) = max_m ||r_m g_i^(m) - ḡ_r^(m)||²/(σ̂_m²(r) + ε)

**Aggregation Rule**: w(r) = w(r-1) + η·[Σ_inliers g_i + Σ_promoted p_i(r)·g_i]/Z(r)

**Complexity**: O(f_∞·N·k) where f_∞ < 1 is steady-state verification fraction

**Security**: Evasion probability ≤ exp(-Ω(k·δ_min²)) for attack magnitude δ_min
