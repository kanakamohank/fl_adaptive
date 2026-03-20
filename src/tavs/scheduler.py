#!/usr/bin/env python3
"""
TAVS Trust-Adaptive Scheduler (Optimized)

This module implements the three-tier trust-adaptive client scheduling system
for TAVS-ESP federated learning. 

Optimizations in this version:
- O(N log N) budget constraint enforcement using algebraic simplification.
- Corrected Bayesian posterior weighting (removed erroneous normalization).
- Accurate Z(r) calculation using active round participants.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .csprng_manager import CSPRNGManager

logger = logging.getLogger(__name__)

@dataclass
class ClientTrustState:
    """Trust state for a single client."""
    client_id: str
    trust_score: float  # T_i(r) ∈ [0, 1]
    tier: int          # 1 (untrusted), 2 (moderate), 3 (trusted)
    rounds_since_verification: int
    last_behavioral_score: float  # φ_i(r) from detection results
    registration_round: int
    total_verifications: int
    total_promotions: int

    def __post_init__(self):
        self.trust_score = max(0.0, min(1.0, self.trust_score))

@dataclass
class SchedulingDecision:
    """Scheduling decision for a single round."""
    round_number: int
    verified_clients: List[str]     
    promoted_clients: List[str]     
    decoy_clients: List[str]       
    tier_assignments: Dict[str, int]  
    trust_scores: Dict[str, float]    
    budget_utilization: float       

class TavsScheduler:
    def __init__(self,
                 csprng_manager: CSPRNGManager,
                 theta_low: float = 0.3,
                 theta_high: float = 0.7,
                 alpha: float = 0.9,
                 gamma_budget: float = 0.35,
                 tau_ramp: int = 30,
                 decoy_probability: float = 0.15,
                 initial_trust: float = 0.5):
        
        self.csprng_manager = csprng_manager
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.alpha = alpha
        self.gamma_budget = gamma_budget
        self.tau_ramp = tau_ramp
        self.decoy_probability = decoy_probability
        self.initial_trust = initial_trust

        self.client_states: Dict[str, ClientTrustState] = {}
        self.scheduling_history: List[SchedulingDecision] = []

        self.stats = {
            "total_rounds": 0,
            "total_verifications": 0,
            "total_promotions": 0,
            "budget_violations": 0,
            "trust_updates": 0
        }

    def register_client(self, client_id: str, round_number: int) -> float:
        if client_id in self.client_states:
            return self.client_states[client_id].trust_score

        max_trust = 1.0 - np.exp(-max(0, round_number) / self.tau_ramp)
        initial_trust = min(self.initial_trust, max_trust)

        self.client_states[client_id] = ClientTrustState(
            client_id=client_id,
            trust_score=initial_trust,
            tier=self._classify_tier(initial_trust),
            rounds_since_verification=0,
            last_behavioral_score=0.5,
            registration_round=round_number,
            total_verifications=0,
            total_promotions=0
        )
        return initial_trust

    def generate_scheduling_decision(self,
                                   candidate_clients: List[str],
                                   round_number: int,
                                   verification_budget: Optional[int] = None) -> SchedulingDecision:
        
        for client_id in candidate_clients:
            if client_id not in self.client_states:
                self.register_client(client_id, round_number)

        round_materials = self.csprng_manager.derive_round_materials(round_number)

        tier_assignments = {}
        for client_id in candidate_clients:
            self.client_states[client_id].tier = self._classify_tier(self.client_states[client_id].trust_score)
            tier_assignments[client_id] = self.client_states[client_id].tier

        tier_probabilities = {1: 0.0, 2: 0.5, 3: 0.67}

        verified_clients, promoted_clients = self.csprng_manager.generate_promotion_assignments(
            promotion_seed=round_materials.promotion_seed,
            client_trust_scores={cid: self.client_states[cid].trust_score for cid in candidate_clients},
            tier_thresholds=(self.theta_low, self.theta_high),
            tier_probabilities=(1.0 - tier_probabilities[1], 1.0 - tier_probabilities[2], 1.0 - tier_probabilities[3])
        )

        decoy_clients = self.csprng_manager.generate_decoy_verification(
            decoy_seed=round_materials.decoy_seed,
            promoted_clients=promoted_clients,
            decoy_probability=self.decoy_probability
        )

        final_verified = list(set(verified_clients + decoy_clients))
        final_promoted = [cid for cid in promoted_clients if cid not in decoy_clients]

        # FAST Budget Enforcement O(N log N)
        final_promoted, demoted_clients = self._enforce_budget_constraint(final_promoted, len(final_verified))
        if demoted_clients:
            final_verified.extend(demoted_clients)
            self.stats["budget_violations"] += 1
            logger.warning(f"Budget constraint violated, demoted {len(demoted_clients)} clients")

        # Recalculate final utilization
        budget_utilization = self._compute_budget_utilization(final_promoted, len(final_verified))

        decision = SchedulingDecision(
            round_number=round_number,
            verified_clients=final_verified,
            promoted_clients=final_promoted,
            decoy_clients=decoy_clients,
            tier_assignments=tier_assignments.copy(),
            trust_scores={cid: self.client_states[cid].trust_score for cid in candidate_clients},
            budget_utilization=budget_utilization
        )

        for client_id in final_verified:
            self.client_states[client_id].total_verifications += 1
            self.client_states[client_id].rounds_since_verification = 0

        for client_id in final_promoted:
            self.client_states[client_id].total_promotions += 1
            self.client_states[client_id].rounds_since_verification += 1

        self.stats["total_rounds"] += 1
        self.stats["total_verifications"] += len(final_verified)
        self.stats["total_promotions"] += len(final_promoted)
        self.scheduling_history.append(decision)

        return decision

    def update_trust_scores(self,
                           verification_results: Dict[str, float],
                           promoted_clients: List[str],
                           round_number: int) -> Dict[str, float]:
        
        updated_scores = {}

        # 1. Update verified clients (EMA)
        for client_id, behavioral_score in verification_results.items():
            if client_id not in self.client_states:
                continue
            client_state = self.client_states[client_id]
            
            new_trust = self.alpha * client_state.trust_score + (1 - self.alpha) * behavioral_score
            
            rounds_since_reg = round_number - client_state.registration_round
            max_trust = 1.0 - np.exp(-rounds_since_reg / self.tau_ramp)
            new_trust = min(new_trust, max_trust)

            client_state.trust_score = max(0.0, min(1.0, new_trust))
            client_state.last_behavioral_score = behavioral_score
            client_state.tier = self._classify_tier(client_state.trust_score)
            updated_scores[client_id] = client_state.trust_score
            self.stats["trust_updates"] += 1

        # 2. Decay promoted clients
        for client_id in promoted_clients:
            if client_id not in self.client_states:
                continue
            client_state = self.client_states[client_id]
            client_state.trust_score = max(0.0, min(1.0, self.alpha * client_state.trust_score))
            client_state.tier = self._classify_tier(client_state.trust_score)
            updated_scores[client_id] = client_state.trust_score

        return updated_scores

    def compute_bayesian_weights(self, promoted_clients: List[str], c_lambda: float = 4.0) -> Dict[str, float]:
        """
        Compute Bayesian posterior weights. 
        CRITICAL FIX: Do NOT normalize these to sum to 1.0. They are independent probabilities.
        """
        weights = {}
        for client_id in promoted_clients:
            if client_id not in self.client_states:
                weights[client_id] = 0.1
                continue
            trust_score = self.client_states[client_id].trust_score
            # p_i(r) = σ(c_λ(T_i(r) - 0.5))
            weights[client_id] = 1.0 / (1.0 + np.exp(-c_lambda * (trust_score - 0.5)))
        return weights

    def _classify_tier(self, trust_score: float) -> int:
        if trust_score < self.theta_low: return 1
        elif trust_score < self.theta_high: return 2
        else: return 3

    def _compute_budget_utilization(self, promoted_clients: List[str], num_verified: int) -> float:
        """Correct calculation of gamma: Sum(p_i) / (V + Sum(p_i))"""
        if not promoted_clients or num_verified == 0:
            return 0.0
        
        bayesian_weights = self.compute_bayesian_weights(promoted_clients)
        total_promoted_weight = sum(bayesian_weights.values())
        
        normalization = num_verified + total_promoted_weight
        return total_promoted_weight / normalization if normalization > 0 else 0.0

    def _enforce_budget_constraint(self, promoted_clients: List[str], num_verified: int) -> Tuple[List[str], List[str]]:
        """
        FAST O(N log N) Budget Enforcement.
        Uses algebraic thresholding instead of an O(N^2) while-loop.
        """
        if not promoted_clients or num_verified == 0:
            return promoted_clients, []

        weights = self.compute_bayesian_weights(promoted_clients)
        
        # Sort ascending by trust score
        sorted_promoted = sorted(promoted_clients, key=lambda cid: self.client_states[cid].trust_score)
        total_p = sum(weights.values())
        
        # Algebra: total_p / (V + total_p) <= gamma  =>  total_p <= (gamma / (1 - gamma)) * V
        max_allowed_p = (self.gamma_budget / (1.0 - self.gamma_budget)) * num_verified
        
        if total_p <= max_allowed_p:
            return sorted_promoted, []
            
        # Sweep once to drop lowest-trust clients until budget is met
        for i, cid in enumerate(sorted_promoted):
            if total_p <= max_allowed_p:
                # Return [Keep], [Demote]
                return sorted_promoted[i:], sorted_promoted[:i]
            total_p -= weights[cid]
            
        return [], sorted_promoted

    def get_client_statistics(self) -> Dict[str, Any]:
        if not self.client_states:
            return {"error": "No clients registered"}

        trust_scores = [state.trust_score for state in self.client_states.values()]
        tier_counts = {1:0, 2:0, 3:0}
        for state in self.client_states.values():
            tier_counts[state.tier] += 1

        # Calculate average budget utilization safely
        avg_budget = np.mean([d.budget_utilization for d in self.scheduling_history]) if self.scheduling_history else 0.0

        return {
            "total_clients": len(self.client_states),
            "trust_statistics": {
                "mean": float(np.mean(trust_scores)), "std": float(np.std(trust_scores)),
                "min": float(np.min(trust_scores)), "max": float(np.max(trust_scores))
            },
            "tier_distribution": tier_counts,
            "scheduling_statistics": self.stats.copy(),
            "average_budget_utilization": float(avg_budget)
        }

    def export_trust_state(self) -> Dict[str, Any]:
        return {
            "client_states": {
                cid: {
                    "trust_score": state.trust_score, "tier": state.tier,
                    "registration_round": state.registration_round,
                    "rounds_since_verification": state.rounds_since_verification,
                    "last_behavioral_score": state.last_behavioral_score,
                    "total_verifications": state.total_verifications,
                    "total_promotions": state.total_promotions
                } for cid, state in self.client_states.items()
            },
            "scheduler_config": {
                "theta_low": self.theta_low, "theta_high": self.theta_high,
                "alpha": self.alpha, "gamma_budget": self.gamma_budget,
                "tau_ramp": self.tau_ramp, "decoy_probability": self.decoy_probability
            },
            "statistics": self.stats.copy()
        }

    def get_trust_trajectory(self, client_id: str) -> Dict[str, Any]:
        """Get trust score trajectory for a specific client from scheduling history."""
        if client_id not in self.client_states:
            return {"error": f"Client {client_id} not found"}

        rounds = []
        trust_scores = []
        tiers = []

        for decision in self.scheduling_history:
            if client_id in decision.trust_scores:
                rounds.append(decision.round_number)
                trust_scores.append(decision.trust_scores[client_id])
                tiers.append(decision.tier_assignments.get(client_id, 0))

        return {
            "client_id": client_id,
            "rounds": rounds,
            "trust_scores": trust_scores,
            "tiers": tiers,
            "current_trust": self.client_states[client_id].trust_score,
            "current_tier": self.client_states[client_id].tier,
            "total_verifications": self.client_states[client_id].total_verifications,
            "total_promotions": self.client_states[client_id].total_promotions
        }