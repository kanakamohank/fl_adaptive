import torch
import math
from typing import Dict, Set, Tuple

class BlockVarianceDetector:
    r"""
    Algorithm 3a: Block-Normalized Outlier Detection (BVD)
    
    Implements Layer 2 detection from Section 4.3 of the NeurIPS 2025 manuscript.
    Separates structural heterogeneity from localized backdoor attacks by tracking
    block-wise variance dynamically.
    """
    
    def __init__(
        self,
        tau_z: float,
        alpha_sigma: float = 0.9,
        epsilon_stab: float = 1e-5
    ):
        self.tau_z = tau_z              
        self.alpha_sigma = alpha_sigma  
        self.epsilon_stab = epsilon_stab 
        
        self.sigma_sq: Dict[str, float] = {}  
        
    def _compute_robust_aggregate(self, projected_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        robust_aggs = {}
        if not projected_updates:
            return {}
        first_client = next(iter(projected_updates.values()))
        block_names = first_client.keys()
        
        for m in block_names:
            stacked_m = torch.stack([client_update[m] for client_update in projected_updates.values()])
            robust_aggs[m] = torch.median(stacked_m, dim=0).values
            
        return robust_aggs

    def detect_outliers(
        self, 
        projected_updates: Dict[str, Dict[str, torch.Tensor]], 
        verified_clients: Set[str]
    ) -> Tuple[Set[str], Set[str], Dict[str, float]]:
        r"""
        Calculates Z_i^{(m)} and classifies clients into Inliers \mathcal{L}(r) 
        and Outliers \mathcal{O}(r).
        Returns: (Inliers, Outliers, behavior_scores)
        """
        if not verified_clients:
            return set(), set(), {}

        # 1. Compute \bar{g}_r^{(m)}
        robust_aggs = self._compute_robust_aggregate(projected_updates)
        
        # BOOTSTRAP FIX: Initialize variance dynamically for Round 1
        first_client = next(iter(projected_updates.values()))
        for m in first_client.keys():
            if m not in self.sigma_sq:
                distances = [torch.sum((projected_updates[cid][m] - robust_aggs[m]) ** 2).item() for cid in verified_clients]
                distances.sort()
                # Ensure we pick the lower-middle index for even N to avoid attacker bias
                median_idx = max(0, (len(distances) - 1) // 2)
                median_dist = distances[median_idx]
                # Default to 1.0 if the median distance is completely zero to avoid premature tight thresholds
                self.sigma_sq[m] = max(1.0, median_dist)
        
        inliers = set()
        outliers = set()
        client_max_distances = {}
        raw_distances = {cid: {} for cid in verified_clients}
        
        # 2. Compute standardized deviations Z_i^{(m)} for all verified clients
        for cid in verified_clients:
            client_update = projected_updates[cid]
            max_z = 0.0
            
            for m, g_i_m_proj in client_update.items():
                dist_sq = torch.sum((g_i_m_proj - robust_aggs[m]) ** 2).item()
                raw_distances[cid][m] = dist_sq
                
                # Z_i^{(m)} = dist^2 / (\hat{\sigma}_m^2 + \varepsilon_{stab})
                z_i_m = dist_sq / (self.sigma_sq[m] + self.epsilon_stab)
                max_z = max(max_z, z_i_m)
            
            client_max_distances[cid] = max_z
            
            if max_z > self.tau_z:
                outliers.add(cid)
            else:
                inliers.add(cid)

        # 3. Update EMA Variance \hat{\sigma}_m^2 using ONLY inliers \mathcal{L}(r)
        if inliers:
            for m in first_client.keys():
                inlier_variance = sum(raw_distances[cid][m] for cid in inliers) / len(inliers)
                old_sigma = self.sigma_sq[m]
                self.sigma_sq[m] = (self.alpha_sigma * old_sigma) + ((1.0 - self.alpha_sigma) * inlier_variance)

        # 4. Calculate continuous behavior scores \varphi_i(r) for trust EMA
        max_overall_dist = max(client_max_distances.values()) if client_max_distances else 0.0
        behavior_scores = {}
        for cid in verified_clients:
            normalized_penalty = client_max_distances[cid] / (max_overall_dist + self.epsilon_stab)
            behavior_scores[cid] = max(0.0, 1.0 - normalized_penalty)

        return inliers, outliers, behavior_scores


class UnifiedBayesianAggregator:
    r"""
    Algorithm 3b: Unified Aggregation Rule (Upgraded for Soft-Weighting)
    """
    
    @staticmethod
    def aggregate(
        verified_updates: Dict[str, Dict[str, torch.Tensor]], 
        verified_weights: Dict[str, float],
        promoted_updates: Dict[str, Dict[str, torch.Tensor]],
        promoted_weights: Dict[str, float] 
    ) -> Dict[str, torch.Tensor]:
        
        aggregated_update = {}
        
        sum_v_i = sum(verified_weights.values())
        sum_p_i = sum(promoted_weights.values())
        Z_r = sum_v_i + sum_p_i
        
        # ---> THE DEFINITIVE MATH FAILSAFE <---
        if Z_r <= 1e-8:
            # If total weight is effectively zero, fallback to simple averaging to prevent NaN explosion
            total_clients = len(verified_updates) + len(promoted_updates)
            if total_clients == 0:
                return {}
            Z_r = float(total_clients)
            # Override weights to uniform averaging
            verified_weights = {k: 1.0 for k in verified_weights}
            promoted_weights = {k: 1.0 for k in promoted_weights}

        first_client = next(iter(verified_updates.values())) if verified_updates else next(iter(promoted_updates.values()))
        for m in first_client.keys():
            aggregated_update[m] = torch.zeros_like(first_client[m])

        for cid, update in verified_updates.items():
            v_i = verified_weights.get(cid, 0.0)
            for m, g_i_m in update.items():
                aggregated_update[m] += (v_i * g_i_m)
                
        for cid, update in promoted_updates.items():
            p_i = promoted_weights.get(cid, 0.0)
            for m, g_i_m in update.items():
                aggregated_update[m] += (p_i * g_i_m)
                
        for m in aggregated_update.keys():
            aggregated_update[m] /= Z_r
            
        return aggregated_update