import math
import hashlib
from typing import List, Dict, Tuple, Set
import numpy as np

class TavsScheduler:
    """
    Algorithm 1: Trust-Adaptive Verification Scheduling (TAVS)
    
    Implements Layer 1 of the TAVS-ESP system. Maps 1-to-1 with Section 4.2
    and Mechanism 1 & 3 from the manuscript.
    """
    
    def __init__(
        self, 
        gamma_budget: float, 
        theta_low: float, 
        theta_high: float, 
        alpha_trust: float,
        tau_ramp: float,
        k_trust: int,
        p_decoy: float = 0.15,
        c_lambda: float = 8.0,
        master_key: bytes = b'default_paper_key'
    ):
        # 1-to-1 Notation Mapping with Paper
        self.gamma_budget = gamma_budget  # \gamma_{budget}: Max unverified influence
        self.theta_low = theta_low        # \theta_{low}: Tier 1 threshold
        self.theta_high = theta_high      # \theta_{high}: Tier 3 threshold
        self.alpha_trust = alpha_trust    # \alpha: EMA momentum
        self.tau_ramp = tau_ramp          # \tau_{ramp}: Trust initialization rate limit
        self.k_trust = k_trust            # k_{trust}: Consecutive clean verifications
        self.p_decoy = p_decoy            # p_{decoy}: CSPRNG decoy probability
        self.c_lambda = c_lambda          # c_\lambda: Bayesian weight steepness
        self.master_key = master_key
        
        # State tracking
        self.trust_scores: Dict[str, float] = {}       # T_i(r)
        self.join_rounds: Dict[str, int] = {}          # r_0 (for Mechanism 3)
        self.clean_streaks: Dict[str, int] = {}        # Track k_trust
        
    def _csprng_roll(self, client_id: str, round_num: int) -> float:
        """Cryptographically secure pseudorandomness under TSA (Assumption 1)."""
        seed_data = f"{self.master_key.decode()}_{client_id}_{round_num}".encode()
        return int(hashlib.sha256(seed_data).hexdigest()[:8], 16) / 0xFFFFFFFF

    def bayesian_posterior_weight(self, t_i: float) -> float:
        """Section 4.4: p_i(r) = \sigma(c_\lambda \cdot (T_i(r) - 0.5))"""
        # Using numerically stable sigmoid
        x = self.c_lambda * (t_i - 0.5)
        return 1.0 / (1.0 + math.exp(-x))

    def get_effective_trust(self, client_id: str, round_num: int) -> float:
        """Mechanism 3: T_i^{\max}(r) = 1 - \exp(-(r-r_0)/\tau_{\mathrm{ramp}})"""
        raw_trust = self.trust_scores.get(client_id, 0.25)
        r_0 = self.join_rounds.get(client_id, round_num)

        #0.5 multiplier is a safe default for the ramping function
        t_max = 1.0 - math.exp(-(round_num - r_0) / self.tau_ramp)
        return min(raw_trust, t_max)

    def schedule_verifications(
        self, available_clients: List[str], round_num: int
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        
        # Initialize new clients
        for cid in available_clients:
            if cid not in self.trust_scores:
                self.trust_scores[cid] = 0.25
                self.join_rounds[cid] = round_num
                self.clean_streaks[cid] = 0
                
        V, P, D = set(), set(), set()
        
        # --- PHASE 1: SEGMENTATION & DECOY INJECTION (Section 4.2) ---
        for cid in available_clients:
            t_eff = self.get_effective_trust(cid, round_num)
            
            # Tier 1
            if t_eff < self.theta_low:
                V.add(cid)
            # Tier 3 (Requires BOTH threshold and k_trust streak)
            elif t_eff >= self.theta_high and self.clean_streaks[cid] >= self.k_trust:
                if self._csprng_roll(cid, round_num) < self.p_decoy:
                    V.add(cid)  # CSPRNG Decoy Verification
                else:
                    P.add(cid)  # Promotion
            # Tier 2
            else:
                # Alternating verify/promote logic handled by budget constraint below
                P.add(cid)  # Tentatively promote, Mechanism 1 will filter this down

        # --- PHASE 2: MECHANISM 1 (Aggregate Budget Constraint) ---
        # "If violated, the lowest-T_i promoted clients are demoted to verified status"
        
        # Sort tentatively promoted clients by trust score (ascending)
        sorted_promoted = sorted(list(P), key=lambda c: self.get_effective_trust(c, round_num))
        
        # Iteratively demote until the budget condition holds
        while True:
            # Calculate Z(r) = |L(r)| + \sum p_i(r). 
            # Note: We estimate |L(r)| as |V| assuming honest majority for scheduling purposes
            sum_p_i = sum(self.bayesian_posterior_weight(self.get_effective_trust(c, round_num)) for c in P)
            Z_r = len(V) + sum_p_i
            
            current_gamma = sum_p_i / Z_r if Z_r > 0 else 0
            
            if current_gamma <= self.gamma_budget or len(P) == 0:
                break
                
            # Demote the lowest trusted client from Promoted to Verified
            demoted_client = sorted_promoted.pop(0)
            P.remove(demoted_client)
            V.add(demoted_client)

        return V, P, D

    def update_trust(self, client_id: str, behavior_score: float, was_verified: bool):
        """
        Section 4.2: T_i(r) = \alpha \cdot T_i(r-1) + (1-\alpha) \cdot \varphi_i(r)
        """
        old_trust = self.trust_scores.get(client_id, 0.5)
        
        if was_verified:
            new_trust = (self.alpha_trust * old_trust) + ((1.0 - self.alpha_trust) * behavior_score)
            
            # Update k_trust streak
            if behavior_score > 0.8:  # Assuming clean verification
                self.clean_streaks[client_id] = self.clean_streaks.get(client_id, 0) + 1
            else:
                self.clean_streaks[client_id] = 0
        else:
            # Decay for promoted clients
            new_trust = self.alpha_trust * old_trust
            
        self.trust_scores[client_id] = new_trust
