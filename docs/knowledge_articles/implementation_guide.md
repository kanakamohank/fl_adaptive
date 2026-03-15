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
