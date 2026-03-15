#!/usr/bin/env python3
"""
TAVS Trust-Adaptive Scheduler

This module implements the three-tier trust-adaptive client scheduling system
for TAVS-ESP federated learning. It manages client trust scores, tier assignments,
verification/promotion decisions, and budget constraint enforcement.

Core Components:
- Three-tier classification: Tier 1 (untrusted), Tier 2 (moderate), Tier 3 (trusted)
- EMA trust dynamics: T_i(r) = α·T_i(r-1) + (1-α)·φ_i(r)
- Budget constraints: Σ_{i∈S(r)} p_i(r)/Z(r) ≤ γ_budget
- Sybil resistance: Trust ramp-up and client demotion mechanisms

Key Insight: Trust-adaptive scheduling prevents timing attacks by making
verification patterns unpredictable while preserving honest client contributions.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import time

from .csprng_manager import CSPRNGManager, RoundMaterials

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
        # Clamp trust score to [0, 1]
        self.trust_score = max(0.0, min(1.0, self.trust_score))


@dataclass
class SchedulingDecision:
    """Scheduling decision for a single round."""
    round_number: int
    verified_clients: List[str]     # V(r) - clients selected for verification
    promoted_clients: List[str]     # S(r) - clients selected for promotion
    decoy_clients: List[str]       # D(r) ⊆ S(r) - promoted clients to verify anyway
    tier_assignments: Dict[str, int]  # Client ID → tier mapping
    trust_scores: Dict[str, float]    # Client ID → trust score mapping
    budget_utilization: float       # Fraction of budget used by promoted clients


class TavsScheduler:
    """
    Trust-Adaptive Verification Scheduler for TAVS-ESP system.

    Manages three-tier trust classification, EMA trust dynamics, and
    cryptographically secure promotion assignments.

    Three-Tier System:
    - Tier 1 (T_i < θ_low): Always verify (untrusted clients)
    - Tier 2 (θ_low ≤ T_i < θ_high): Probabilistic verification (moderate trust)
    - Tier 3 (T_i ≥ θ_high): Mostly promote with decoy verification (trusted)
    """

    def __init__(self,
                 csprng_manager: CSPRNGManager,
                 theta_low: float = 0.3,
                 theta_high: float = 0.7,
                 alpha: float = 0.9,
                 gamma_budget: float = 0.35,
                 tau_ramp: int = 30,
                 decoy_probability: float = 0.15,
                 initial_trust: float = 0.5):
        """
        Initialize TAVS scheduler.

        Args:
            csprng_manager: ChaCha20-CTR manager for ephemeral randomness
            theta_low: Lower threshold for tier classification
            theta_high: Upper threshold for tier classification
            alpha: EMA decay factor for trust score updates
            gamma_budget: Budget constraint (max fraction of weight from promoted clients)
            tau_ramp: Trust ramp-up parameter (Sybil resistance)
            decoy_probability: Probability of decoy verification for promoted clients
            initial_trust: Initial trust score for new clients
        """
        self.csprng_manager = csprng_manager
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.alpha = alpha
        self.gamma_budget = gamma_budget
        self.tau_ramp = tau_ramp
        self.decoy_probability = decoy_probability
        self.initial_trust = initial_trust

        # Client state tracking
        self.client_states: Dict[str, ClientTrustState] = {}
        self.scheduling_history: List[SchedulingDecision] = []

        # Statistics
        self.stats = {
            "total_rounds": 0,
            "total_verifications": 0,
            "total_promotions": 0,
            "budget_violations": 0,
            "trust_updates": 0
        }

        logger.info(f"TAVS Scheduler initialized: θ_low={theta_low}, θ_high={theta_high}, "
                   f"α={alpha}, γ_budget={gamma_budget}")

    def register_client(self, client_id: str, round_number: int) -> float:
        """
        Register a new client with Sybil resistance mechanisms.

        Args:
            client_id: Unique client identifier
            round_number: Round when client first appears

        Returns:
            Initial trust score (may be ramped based on registration time)
        """
        if client_id in self.client_states:
            logger.debug(f"Client {client_id} already registered")
            return self.client_states[client_id].trust_score

        # Apply Sybil resistance: Trust ramp-up mechanism
        # T_i^max(r) = 1 - exp(-(r-r_0)/τ_ramp)
        max_trust = 1.0 - np.exp(-max(0, round_number) / self.tau_ramp)
        initial_trust = min(self.initial_trust, max_trust)

        self.client_states[client_id] = ClientTrustState(
            client_id=client_id,
            trust_score=initial_trust,
            tier=self._classify_tier(initial_trust),
            rounds_since_verification=0,
            last_behavioral_score=0.5,  # Neutral initial behavior
            registration_round=round_number,
            total_verifications=0,
            total_promotions=0
        )

        logger.info(f"Registered client {client_id} at round {round_number} "
                   f"with initial trust {initial_trust:.3f} (max: {max_trust:.3f})")
        return initial_trust

    def generate_scheduling_decision(self,
                                   candidate_clients: List[str],
                                   round_number: int,
                                   verification_budget: Optional[int] = None) -> SchedulingDecision:
        """
        Generate TAVS scheduling decision for a federated learning round.

        Args:
            candidate_clients: Available clients for this round
            round_number: Current federated learning round
            verification_budget: Maximum clients to verify (if None, use tier-based)

        Returns:
            SchedulingDecision with verified/promoted client assignments
        """
        # Ensure all candidates are registered
        for client_id in candidate_clients:
            if client_id not in self.client_states:
                self.register_client(client_id, round_number)

        # Derive round materials from CSPRNG
        round_materials = self.csprng_manager.derive_round_materials(round_number)

        # Update tier assignments based on current trust scores
        tier_assignments = {}
        for client_id in candidate_clients:
            client_state = self.client_states[client_id]
            tier_assignments[client_id] = self._classify_tier(client_state.trust_score)
            client_state.tier = tier_assignments[client_id]

        # Generate promotion assignments using CSPRNG
        tier_probabilities = {
            1: 0.0,   # Tier 1: Always verify (0% promotion)
            2: 0.5,   # Tier 2: 50% promotion probability
            3: 0.67   # Tier 3: 67% promotion probability
        }

        verified_clients, promoted_clients = self.csprng_manager.generate_promotion_assignments(
            promotion_seed=round_materials.promotion_seed,
            client_trust_scores={cid: self.client_states[cid].trust_score
                               for cid in candidate_clients},
            tier_thresholds=(self.theta_low, self.theta_high),
            tier_probabilities=(1.0 - tier_probabilities[1],
                              1.0 - tier_probabilities[2],
                              1.0 - tier_probabilities[3])
        )

        # Generate decoy verification assignments
        decoy_clients = self.csprng_manager.generate_decoy_verification(
            decoy_seed=round_materials.decoy_seed,
            promoted_clients=promoted_clients,
            decoy_probability=self.decoy_probability
        )

        # Add decoy clients to verification set
        final_verified = list(set(verified_clients + decoy_clients))
        final_promoted = [cid for cid in promoted_clients if cid not in decoy_clients]

        # Enforce budget constraints
        budget_utilization = self._compute_budget_utilization(final_promoted, round_number)
        if budget_utilization > self.gamma_budget:
            # Demote lowest-trust promoted clients until budget satisfied
            final_promoted, demoted_clients = self._enforce_budget_constraint(
                final_promoted, round_number
            )
            final_verified.extend(demoted_clients)
            budget_utilization = self._compute_budget_utilization(final_promoted, round_number)
            self.stats["budget_violations"] += 1
            logger.warning(f"Budget constraint violated, demoted {len(demoted_clients)} clients")

        # Create scheduling decision
        decision = SchedulingDecision(
            round_number=round_number,
            verified_clients=final_verified,
            promoted_clients=final_promoted,
            decoy_clients=decoy_clients,
            tier_assignments=tier_assignments.copy(),
            trust_scores={cid: self.client_states[cid].trust_score
                         for cid in candidate_clients},
            budget_utilization=budget_utilization
        )

        # Update client state counters
        for client_id in final_verified:
            self.client_states[client_id].total_verifications += 1
            self.client_states[client_id].rounds_since_verification = 0

        for client_id in final_promoted:
            self.client_states[client_id].total_promotions += 1
            self.client_states[client_id].rounds_since_verification += 1

        # Update statistics
        self.stats["total_rounds"] += 1
        self.stats["total_verifications"] += len(final_verified)
        self.stats["total_promotions"] += len(final_promoted)

        # Store decision in history
        self.scheduling_history.append(decision)

        logger.info(f"Round {round_number}: {len(final_verified)} verified, "
                   f"{len(final_promoted)} promoted, {len(decoy_clients)} decoy, "
                   f"budget: {budget_utilization:.1%}")

        return decision

    def update_trust_scores(self,
                           verification_results: Dict[str, float],
                           promoted_clients: List[str],
                           round_number: int) -> Dict[str, float]:
        """
        Update client trust scores based on verification results.

        EMA Update Rule:
        - Verified clients: T_i(r) = α·T_i(r-1) + (1-α)·φ_i(r)
        - Promoted clients: T_i(r) = α·T_i(r-1) (decay only)

        Args:
            verification_results: Dict mapping client_id → behavioral score φ_i(r)
            promoted_clients: List of clients that were promoted (not verified)
            round_number: Current round number

        Returns:
            Updated trust scores for all affected clients
        """
        updated_scores = {}

        # Update trust scores for verified clients
        for client_id, behavioral_score in verification_results.items():
            if client_id not in self.client_states:
                logger.warning(f"Unknown client {client_id} in verification results")
                continue

            client_state = self.client_states[client_id]

            # EMA update: T_i(r) = α·T_i(r-1) + (1-α)·φ_i(r)
            old_trust = client_state.trust_score
            new_trust = self.alpha * old_trust + (1 - self.alpha) * behavioral_score

            # Apply Sybil resistance: clamp by ramp-up function
            rounds_since_registration = round_number - client_state.registration_round
            max_trust = 1.0 - np.exp(-rounds_since_registration / self.tau_ramp)
            new_trust = min(new_trust, max_trust)

            client_state.trust_score = max(0.0, min(1.0, new_trust))
            client_state.last_behavioral_score = behavioral_score
            client_state.tier = self._classify_tier(client_state.trust_score)

            updated_scores[client_id] = client_state.trust_score
            self.stats["trust_updates"] += 1

            logger.debug(f"Client {client_id}: T({old_trust:.3f}) + φ({behavioral_score:.3f}) "
                        f"→ T({client_state.trust_score:.3f}), tier {client_state.tier}")

        # Decay trust scores for promoted clients (no new information)
        for client_id in promoted_clients:
            if client_id not in self.client_states:
                continue

            client_state = self.client_states[client_id]

            # Decay only: T_i(r) = α·T_i(r-1)
            old_trust = client_state.trust_score
            new_trust = self.alpha * old_trust

            client_state.trust_score = max(0.0, min(1.0, new_trust))
            client_state.tier = self._classify_tier(client_state.trust_score)

            updated_scores[client_id] = client_state.trust_score

            logger.debug(f"Client {client_id}: decay T({old_trust:.3f}) → T({new_trust:.3f}), "
                        f"tier {client_state.tier}")

        return updated_scores

    def compute_bayesian_weights(self,
                                promoted_clients: List[str],
                                c_lambda: float = 4.0) -> Dict[str, float]:
        """
        Compute Bayesian posterior weights for promoted clients.

        Weight Formula: p_i(r) = σ(c_λ(T_i(r) - 0.5))
        where σ is the sigmoid function.

        Args:
            promoted_clients: List of promoted client IDs
            c_lambda: Sigmoid slope parameter (higher = more discriminative)

        Returns:
            Dictionary mapping client_id → posterior weight
        """
        weights = {}

        for client_id in promoted_clients:
            if client_id not in self.client_states:
                logger.warning(f"Unknown promoted client {client_id}")
                weights[client_id] = 0.1  # Low weight for unknown clients
                continue

            trust_score = self.client_states[client_id].trust_score

            # Sigmoid transformation: p_i(r) = σ(c_λ(T_i(r) - 0.5))
            logit = c_lambda * (trust_score - 0.5)
            posterior_weight = 1.0 / (1.0 + np.exp(-logit))

            weights[client_id] = posterior_weight

        # Normalize weights to sum to 1.0 for promoted clients
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {cid: w / total_weight for cid, w in weights.items()}

        logger.debug(f"Bayesian weights: mean={np.mean(list(weights.values())):.3f}, "
                    f"std={np.std(list(weights.values())):.3f}")

        return weights

    def _classify_tier(self, trust_score: float) -> int:
        """Classify client into three-tier system based on trust score."""
        if trust_score < self.theta_low:
            return 1  # Untrusted: always verify
        elif trust_score < self.theta_high:
            return 2  # Moderate: probabilistic verification
        else:
            return 3  # Trusted: mostly promote with decoy verification

    def _compute_budget_utilization(self, promoted_clients: List[str], round_number: int) -> float:
        """
        Compute budget utilization: Σ_{i∈S(r)} p_i(r) / Z(r)

        This measures what fraction of aggregation weight comes from unverified sources.
        """
        if not promoted_clients:
            return 0.0

        # Get Bayesian weights for promoted clients
        bayesian_weights = self.compute_bayesian_weights(promoted_clients)

        # Normalization factor Z(r) includes both verified (weight 1.0) and promoted clients
        # For budget calculation, we assume worst case where all other clients are verified
        total_verified_weight = len([cid for cid in self.client_states.keys()
                                   if cid not in promoted_clients])
        total_promoted_weight = sum(bayesian_weights.values())

        normalization = total_verified_weight + total_promoted_weight
        if normalization == 0:
            return 0.0

        budget_utilization = total_promoted_weight / normalization
        return budget_utilization

    def _enforce_budget_constraint(self, promoted_clients: List[str], round_number: int) -> Tuple[List[str], List[str]]:
        """
        Enforce budget constraint by demoting lowest-trust promoted clients.

        Returns:
            Tuple of (remaining_promoted, demoted_clients)
        """
        if not promoted_clients:
            return [], []

        # Sort promoted clients by trust score (lowest first)
        client_trusts = [(cid, self.client_states[cid].trust_score)
                        for cid in promoted_clients]
        client_trusts.sort(key=lambda x: x[1])  # Sort by trust score ascending

        # Iteratively demote clients until budget constraint satisfied
        remaining_promoted = promoted_clients.copy()
        demoted_clients = []

        while remaining_promoted:
            budget_util = self._compute_budget_utilization(remaining_promoted, round_number)
            if budget_util <= self.gamma_budget:
                break  # Budget constraint satisfied

            # Demote lowest-trust client
            lowest_trust_client = min(remaining_promoted,
                                    key=lambda cid: self.client_states[cid].trust_score)
            remaining_promoted.remove(lowest_trust_client)
            demoted_clients.append(lowest_trust_client)

        return remaining_promoted, demoted_clients

    def get_client_statistics(self) -> Dict[str, Any]:
        """Get comprehensive client statistics and trust dynamics."""
        if not self.client_states:
            return {"error": "No clients registered"}

        trust_scores = [state.trust_score for state in self.client_states.values()]
        tier_counts = defaultdict(int)
        for state in self.client_states.values():
            tier_counts[state.tier] += 1

        return {
            "total_clients": len(self.client_states),
            "trust_statistics": {
                "mean": np.mean(trust_scores),
                "std": np.std(trust_scores),
                "min": np.min(trust_scores),
                "max": np.max(trust_scores)
            },
            "tier_distribution": dict(tier_counts),
            "scheduling_statistics": self.stats.copy(),
            "average_budget_utilization": np.mean([d.budget_utilization
                                                 for d in self.scheduling_history]) if self.scheduling_history else 0.0
        }

    def get_trust_trajectory(self, client_id: str) -> Dict[str, List[float]]:
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

    def export_trust_state(self) -> Dict[str, Any]:
        """Export complete trust state for persistence/analysis."""
        return {
            "client_states": {
                cid: {
                    "trust_score": state.trust_score,
                    "tier": state.tier,
                    "registration_round": state.registration_round,
                    "rounds_since_verification": state.rounds_since_verification,
                    "last_behavioral_score": state.last_behavioral_score,
                    "total_verifications": state.total_verifications,
                    "total_promotions": state.total_promotions
                }
                for cid, state in self.client_states.items()
            },
            "scheduler_config": {
                "theta_low": self.theta_low,
                "theta_high": self.theta_high,
                "alpha": self.alpha,
                "gamma_budget": self.gamma_budget,
                "tau_ramp": self.tau_ramp,
                "decoy_probability": self.decoy_probability
            },
            "statistics": self.stats.copy()
        }