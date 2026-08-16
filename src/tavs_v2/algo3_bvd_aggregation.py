import torch
import math
from typing import Dict, Set, Tuple

class BlockVarianceDetector:
    r"""
    Algorithm 3a: Block-Normalized Outlier Detection (BVD)
    
    Implements Layer 2 detection from Section 4.3 of the manuscript.
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
    
    # Radius below which the verified cohort is treated as having no measurable
    # spread, so no clip threshold can be estimated from it.
    _MIN_CLIP_RADIUS = 1e-12

    @staticmethod
    def clip_promoted_to_consensus(
        verified_updates: Dict[str, Dict[str, torch.Tensor]],
        promoted_updates: Dict[str, Dict[str, torch.Tensor]],
        clip_factor: float = 1.0,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, object]]:
        r"""
        Bound how far an unverified update may sit from the verified consensus.

        Why this is needed
        ------------------
        Mechanism 1 constrains \gamma(r) = \sum p_i / Z \le \gamma_{budget}, which
        bounds the *weight* promoted clients carry, not the *magnitude* of what
        they carry. The aggregate is \sum w_i g_i / Z, so damage scales with
        w_i \cdot \|g_i\|, and \|g_i\| is unbounded. A promoted client with
        p_i = 0.9 and \|g_i\| = 10^6 keeps \gamma at 0.11 -- comfortably inside a
        0.35 budget -- while destroying the model. Empirically this drove test
        accuracy to 0.086 (random is 0.10) with server loss above 2000.

        Why deviation and not raw norm
        ------------------------------
        Clients submit full model parameters, not deltas, so every client's raw
        \|g_i\| is dominated by the shared global model and is nearly identical
        across honest and malicious clients alike. Clipping raw norm would shrink
        the model toward the origin and destroy it. What actually distinguishes a
        poisoned submission is its displacement from where everyone else is, so
        the ball is centred on the verified consensus:

            g_i <- c + (g_i - c) \cdot \min(1, \tau / \|g_i - c\|)

        with c the coordinate-wise median of the verified updates and
        \tau = clip_factor \cdot median_j \|g_j - c\| over verified clients j.
        The threshold is therefore self-calibrating: it tracks the cohort's
        natural spread as training progresses, needs no absolute scale constant,
        and at clip_factor = 1.0 states exactly "a promoted client may deviate as
        much as a typical verified client does".

        Only the magnitude is touched. Direction is preserved, and clients inside
        the ball are returned untouched, so honest promoted clients are unaffected.

        Verified clients are deliberately exempt: they have already passed BVD
        outlier detection this round, so clipping them would suppress legitimate
        heterogeneity that the detector already vetted.

        Returns:
            (clipped_updates, stats) where stats records the threshold, how many
            clients were clipped, and -- when clipping was skipped -- why.
        """
        stats: Dict[str, object] = {
            "clipping_applied": False,
            "clip_radius": None,
            "num_clipped": 0,
            "max_deviation_ratio": None,
            "skipped_reason": None,
        }

        if not promoted_updates:
            stats["skipped_reason"] = "no_promoted_clients"
            return promoted_updates, stats
        if not verified_updates:
            # No vetted cohort means no trustworthy centre or scale. Leaving the
            # updates untouched is reported rather than silently assumed safe.
            stats["skipped_reason"] = "no_verified_clients_to_form_consensus"
            return promoted_updates, stats

        # Only blocks present in the verified cohort can be centred.
        verified_blocks = set(next(iter(verified_updates.values())).keys())
        for update in verified_updates.values():
            verified_blocks &= set(update.keys())
        if not verified_blocks:
            stats["skipped_reason"] = "no_common_blocks"
            return promoted_updates, stats

        # 1. Robust centre: coordinate-wise median over verified clients.
        center = {
            block: torch.median(
                torch.stack([u[block] for u in verified_updates.values()]), dim=0
            ).values
            for block in verified_blocks
        }

        def deviation_norm(update: Dict[str, torch.Tensor]) -> float:
            # Global L2 across blocks, so scaling preserves direction.
            total = 0.0
            for block in verified_blocks:
                if block in update:
                    total += torch.sum((update[block] - center[block]) ** 2).item()
            return math.sqrt(total)

        # 2. Threshold from the verified cohort's own spread.
        verified_devs = sorted(deviation_norm(u) for u in verified_updates.values())
        median_dev = verified_devs[(len(verified_devs) - 1) // 2]
        radius = clip_factor * median_dev

        if radius <= UnifiedBayesianAggregator._MIN_CLIP_RADIUS:
            # Verified clients are effectively identical (common at round 0 when
            # everyone still holds the initial parameters). No scale can be
            # inferred, and clipping to a zero-radius ball would collapse every
            # promoted client onto the centre.
            stats["skipped_reason"] = "verified_cohort_has_no_measurable_spread"
            return promoted_updates, stats

        # 3. Project anything outside the ball back onto its surface.
        clipped: Dict[str, Dict[str, torch.Tensor]] = {}
        num_clipped = 0
        max_ratio = 0.0

        for cid, update in promoted_updates.items():
            dev = deviation_norm(update)
            max_ratio = max(max_ratio, dev / radius)

            if dev <= radius:
                clipped[cid] = update  # Inside the ball: untouched.
                continue

            scale = radius / dev
            clipped[cid] = {
                block: (
                    center[block] + (tensor - center[block]) * scale
                    if block in verified_blocks
                    else tensor  # Uncentrable block: cannot be clipped.
                )
                for block, tensor in update.items()
            }
            num_clipped += 1

        stats.update({
            "clipping_applied": True,
            "clip_radius": radius,
            "num_clipped": num_clipped,
            "max_deviation_ratio": max_ratio,
        })
        return clipped, stats

    @staticmethod
    def cosine_gate_promoted(
        verified_updates: Dict[str, Dict[str, torch.Tensor]],
        promoted_updates: Dict[str, Dict[str, torch.Tensor]],
        previous_global: Dict[str, torch.Tensor],
        cosine_min: float = 0.0,
    ) -> Tuple[Set[str], Dict[str, object]]:
        r"""
        Reject unverified updates that pull the model AGAINST the verified cohort.

        Why this is needed on top of clipping
        -------------------------------------
        Clipping bounds \|g_i - c\|: how FAR an update sits from consensus. It says
        nothing about WHICH WAY it points. Measured directly, an attacker at
        cosine +1.00 to the honest direction but three orders of magnitude out of
        scale drove the unclipped aggregate norm to 364 while clipping held it at
        0.137 -- and conversely, an update sitting comfortably inside the clip
        ball can point in exactly the opposite direction and pass untouched.
        The two bounds constrain orthogonal quantities and neither subsumes the
        other; an attacker only needs the axis left unguarded.

        Why the previous global model is the origin
        -------------------------------------------
        Clients submit full parameters, not deltas, so "direction" is only
        meaningful relative to where the model started this round. The update
        client i proposes is (g_i - w_prev). The direction the verified cohort
        wants is (mean_verified - w_prev). Comparing deviations from the current
        consensus instead would be useless: honest deviations are roughly
        isotropic noise around that point, so their mean is ~0 and there is no
        reference direction to compare against.

        This is FLTrust's ReLU-clipped cosine with the verified cohort standing
        in for a server-held root dataset -- which is precisely what TAVS already
        produces, so the trust anchor costs nothing extra.

        Returns:
            (rejected_ids, stats). Rejection is a hard gate rather than a
            re-weighting: a client actively pushing backwards is not a client
            whose contribution should merely be scaled down.
        """
        stats: Dict[str, object] = {
            "cosine_applied": False, "num_rejected": 0,
            "min_cosine_seen": None, "skipped_reason": None,
        }
        if not promoted_updates:
            stats["skipped_reason"] = "no_promoted_clients"
            return set(), stats
        if not verified_updates:
            # No vetted cohort means no trustworthy reference direction.
            stats["skipped_reason"] = "no_verified_clients_for_reference"
            return set(), stats
        if not previous_global:
            stats["skipped_reason"] = "no_previous_global_parameters"
            return set(), stats

        blocks = set(previous_global) & set(next(iter(verified_updates.values())))
        for u in verified_updates.values():
            blocks &= set(u)
        if not blocks:
            stats["skipped_reason"] = "no_common_blocks"
            return set(), stats

        def flat_delta(update):
            return torch.cat([(update[b] - previous_global[b]).flatten()
                              for b in sorted(blocks) if b in update])

        # Reference: the movement the verified cohort proposes this round.
        reference = torch.stack([flat_delta(u) for u in verified_updates.values()]).mean(0)
        ref_norm = reference.norm().item()
        if ref_norm <= 1e-12:
            # Verified clients propose no net movement, so there is no direction
            # to agree or disagree with.
            stats["skipped_reason"] = "verified_cohort_proposes_no_movement"
            return set(), stats

        rejected, min_cos = set(), None
        for cid, update in promoted_updates.items():
            delta = flat_delta(update)
            if delta.norm().item() <= 1e-12:
                continue  # Proposes nothing; harmless, and cosine is undefined.
            cos = torch.nn.functional.cosine_similarity(
                delta.unsqueeze(0), reference.unsqueeze(0)
            ).item()
            min_cos = cos if min_cos is None else min(min_cos, cos)
            if cos < cosine_min:
                rejected.add(cid)

        stats.update({"cosine_applied": True, "num_rejected": len(rejected),
                      "min_cosine_seen": min_cos})
        return rejected, stats

    @staticmethod
    def aggregate(
        verified_updates: Dict[str, Dict[str, torch.Tensor]], 
        verified_weights: Dict[str, float],
        promoted_updates: Dict[str, Dict[str, torch.Tensor]],
        promoted_weights: Dict[str, float],
        clip_promoted: bool = True,
        clip_factor: float = 1.0,
        clip_stats_out: Dict[str, object] = None,
        cosine_filter: bool = True,
        cosine_min: float = 0.0,
        previous_global: Dict[str, torch.Tensor] = None,
        cosine_stats_out: Dict[str, object] = None,
    ) -> Dict[str, torch.Tensor]:

        # Bound unverified influence in DIRECTION. Runs before clipping so a
        # rejected client is dropped outright rather than clipped and kept.
        if cosine_filter and promoted_updates and previous_global:
            rejected, cos_stats = UnifiedBayesianAggregator.cosine_gate_promoted(
                verified_updates, promoted_updates, previous_global, cosine_min
            )
            if cosine_stats_out is not None:
                cosine_stats_out.update(cos_stats)
            if rejected:
                promoted_updates = {k: v for k, v in promoted_updates.items() if k not in rejected}
                promoted_weights = {k: v for k, v in promoted_weights.items() if k not in rejected}

        # Bound unverified influence in MAGNITUDE before weighting. gamma_budget
        # bounds only the weight these clients carry; without this an unverified
        # client inside the budget can still dominate the sum outright.
        if clip_promoted and promoted_updates:
            promoted_updates, clip_stats = UnifiedBayesianAggregator.clip_promoted_to_consensus(
                verified_updates, promoted_updates, clip_factor
            )
            if clip_stats_out is not None:
                clip_stats_out.update(clip_stats)

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