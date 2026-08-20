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
        master_key: bytes = b'default_paper_key',
        s_max_appearances: int = 4,
        s_max_rounds: int = 10,
        decay_trust_on_promotion: bool = False,
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

        # Staleness caps. Trust answers "how well has this client behaved"; these
        # answer "how long since we last looked". Conflating the two -- decaying
        # trust because a client was promoted -- made the trust EMA converge to
        # the verification rate f rather than to the behaviour score, pinning
        # trust at ~0.3 and putting theta_high=0.7 permanently out of reach.
        #
        # Two caps because they measure different things. Appearances drive the
        # trust cadence; wall-clock rounds bound actual exposure. At 40%
        # participation 4 appearances is ~10 rounds, so neither implies the other.
        self.s_max_appearances = s_max_appearances
        self.s_max_rounds = s_max_rounds
        # Reproduces the pre-split behaviour for before/after comparison only.
        self.decay_trust_on_promotion = decay_trust_on_promotion

        # State tracking
        self.trust_scores: Dict[str, float] = {}       # T_i(r)
        self.join_rounds: Dict[str, int] = {}          # r_0 (for Mechanism 3)
        self.clean_streaks: Dict[str, int] = {}        # Track k_trust
        self.appearances_since_verified: Dict[str, int] = {}
        self.last_verified_round: Dict[str, int] = {}
        
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

        t_max = 1.0 - math.exp(-(round_num - r_0) / self.tau_ramp)
        return min(raw_trust, t_max)

    def get_tier(self, client_id: str, round_num: int) -> int:
        """
        The client's actual tier, using the same conditions as Phase 1 below.

        Exists because tier was previously logged as `3 if promoted else 1`,
        a promoted/verified flag wearing tier numerals. That made real tier
        state unobservable: a run in which Tier 3 never fired was
        indistinguishable from one in which it fired constantly.
        """
        t_eff = self.get_effective_trust(client_id, round_num)
        if t_eff < self.theta_low:
            return 1
        if t_eff >= self.theta_high and self.clean_streaks.get(client_id, 0) >= self.k_trust:
            return 3
        return 2

    def is_stale(self, client_id: str, round_num: int) -> bool:
        """
        Has this client gone too long without verification?

        A hard expiry, deliberately independent of trust: without it, removing
        the promotion decay would let a client that once earned high trust stay
        promoted indefinitely and never be checked again. Trust says how well a
        client behaved when we last looked; this says the last look is too old
        to rely on.

        Either cap alone is insufficient. A rarely-sampled client trips the
        wall-clock cap long before it accumulates appearances, and a heavily
        sampled one trips the appearance cap first.
        """
        if client_id not in self.last_verified_round:
            return True     # never verified: cannot be promoted on no evidence
        appearances = self.appearances_since_verified.get(client_id, 0)
        if appearances >= self.s_max_appearances:
            return True
        return (round_num - self.last_verified_round[client_id]) >= self.s_max_rounds

    def schedule_verifications(
        self, available_clients: List[str], round_num: int
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        
        # Initialize new clients
        for cid in available_clients:
            if cid not in self.trust_scores:
                self.trust_scores[cid] = 0.25
                self.join_rounds[cid] = round_num
                self.clean_streaks[cid] = 0
                self.appearances_since_verified[cid] = 0
                # last_verified_round is deliberately left unset: is_stale()
                # treats "never verified" as stale, so a fresh client cannot be
                # promoted before it has been looked at even once.

        V, P, D = set(), set(), set()
        
        # --- PHASE 1: SEGMENTATION & DECOY INJECTION (Section 4.2) ---
        for cid in available_clients:
            t_eff = self.get_effective_trust(cid, round_num)

            # Hard expiry outranks tier. Placed before the tier branches AND
            # before the Phase 2 budget loop so it acts as a floor the budget
            # cannot trade away: the budget may only ever move clients INTO V.
            if self.is_stale(cid, round_num):
                V.add(cid)
                continue

            # Tier 1
            if t_eff < self.theta_low:
                V.add(cid)
            # Tier 3 (Requires BOTH threshold and k_trust streak)
            elif t_eff >= self.theta_high and self.clean_streaks[cid] >= self.k_trust:
                if self._csprng_roll(cid, round_num) < self.p_decoy:
                    # CSPRNG Decoy Verification: verified server-side, but the
                    # caller tells this client it was promoted. D was previously
                    # returned always-empty, so decoys were indistinguishable
                    # from ordinary verifications and the server had no way to
                    # withhold that fact from the client.
                    V.add(cid)
                    D.add(cid)
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

    def update_trust(self, client_id: str, behavior_score: float, was_verified: bool,
                     is_outlier: bool = False, round_num: int = None):
        """
        Section 4.2: T_i(r) = \alpha \cdot T_i(r-1) + (1-\alpha) \cdot \varphi_i(r)

        Trust moves ONLY on verification. Promotion advances the staleness
        counters instead of decaying trust: not being checked is not evidence of
        misbehaviour, and treating it as such made the EMA converge to the
        verification rate f (T* = f * phi) instead of to the behaviour score.
        That capped trust at ~0.3 and made theta_high=0.7 unreachable at any
        round count.

        round_num is optional so existing callers keep working, but without it
        the wall-clock staleness cap cannot be enforced.
        """
        old_trust = self.trust_scores.get(client_id, 0.25)
        
        if was_verified:
            new_trust = (self.alpha_trust * old_trust) + ((1.0 - self.alpha_trust) * behavior_score)
            
            # Update k_trust streak.
            #
            # The detector's own verdict is authoritative, not a score cutoff.
            # Under the absolute behaviour score a client just past the anomaly
            # threshold (max_z = 1.1 * tau_z) scores 0.9, so a client the
            # detector FLAGGED as Byzantine would still have cleared a 0.8 cutoff
            # and kept accumulating its clean streak towards Tier 3 promotion.
            if is_outlier or behavior_score <= 0.8:
                self.clean_streaks[client_id] = 0
            else:
                self.clean_streaks[client_id] = self.clean_streaks.get(client_id, 0) + 1

            # Fresh evidence: both staleness clocks reset. Decoys reach here too
            # (D is a subset of V), so a decoy counts as a real check.
            self.appearances_since_verified[client_id] = 0
            if round_num is not None:
                self.last_verified_round[client_id] = round_num
        elif self.decay_trust_on_promotion:
            new_trust = self.alpha_trust * old_trust        # pre-split behaviour
            self.appearances_since_verified[client_id] = \
                self.appearances_since_verified.get(client_id, 0) + 1
        else:
            new_trust = old_trust                            # unchanged, not decayed
            self.appearances_since_verified[client_id] = \
                self.appearances_since_verified.get(client_id, 0) + 1

        self.trust_scores[client_id] = new_trust

    def _min_round_to_reach(self, threshold: float, initial_trust: float) -> float:
        r"""
        Earliest round at which effective trust can reach `threshold`, under
        ideal behaviour (\varphi_i = 1.0, verified every round).

        Effective trust is min(raw, ramp cap), so BOTH must clear the threshold:

          Ramp (Mechanism 3): T^{max}(r) = 1 - e^{-(r-r_0)/\tau_{ramp}}
              r >= -\tau_{ramp} \cdot \ln(1 - threshold)
          EMA: best case is T_n = 1 - (1 - T_0)\alpha^n
              n >= \ln((1-threshold)/(1-T_0)) / \ln(\alpha)
        """
        if threshold >= 1.0:
            return math.inf  # The ramp asymptotes to 1.0 but never attains it.

        ramp_bound = -self.tau_ramp * math.log(1.0 - threshold)

        if initial_trust >= threshold:
            ema_bound = 0.0
        elif self.alpha_trust <= 0.0:
            ema_bound = 1.0  # No history term: one clean observation suffices.
        else:
            ema_bound = math.log(
                (1.0 - threshold) / (1.0 - initial_trust)
            ) / math.log(self.alpha_trust)

        return max(ramp_bound, ema_bound)

    def min_round_for_promotion(self, initial_trust: float = 0.25) -> int:
        r"""
        Earliest round at which ANY client can be promoted (skip verification).

        Note this is governed by \theta_{low}, not \theta_{high}: a client only
        has to escape Tier 1. Tier 2 clients -- anything in
        [\theta_{low}, \theta_{high}) -- are tentatively promoted by the `else`
        branch of schedule_verifications, with Mechanism 1's budget loop
        demoting the excess. Reaching \theta_{high} and a k_trust streak buys
        Tier 3 (promotion plus decoy sampling), not promotion itself.

        This is the number that decides whether TAVS can differ from full
        verification at all. With \tau_{ramp}=30 and \theta_{low}=0.3 the ramp
        alone needs 11 rounds, so a 10-round experiment leaves every client
        pinned in Tier 1 and TAVS silently degenerates into full verification
        while still reporting a plausible "1.0x efficiency" result.
        """
        return int(math.ceil(self._min_round_to_reach(self.theta_low, initial_trust)))

    def min_round_for_tier3(self, initial_trust: float = 0.25) -> int:
        r"""
        Earliest round for Tier 3 (\theta_{high} plus a k_trust clean streak).

        Tier 3 is the regime where decoy verification applies. Promotion is
        already possible earlier via Tier 2 -- see min_round_for_promotion.
        """
        return int(math.ceil(max(
            self._min_round_to_reach(self.theta_high, initial_trust),
            float(self.k_trust),
        )))

    def describe_promotion_feasibility(self, num_rounds: int, initial_trust: float = 0.25) -> Dict:
        """
        Diagnostic for experiment configuration: can TAVS promote anyone within
        `num_rounds`, and if not, which parameter is the binding constraint?
        """
        if self.theta_low >= 1.0:
            ramp_bound = math.inf
            ema_bound = math.inf
        else:
            ramp_bound = -self.tau_ramp * math.log(1.0 - self.theta_low)
            if initial_trust >= self.theta_low:
                ema_bound = 0.0
            elif self.alpha_trust <= 0.0:
                ema_bound = 1.0
            else:
                ema_bound = math.log(
                    (1.0 - self.theta_low) / (1.0 - initial_trust)
                ) / math.log(self.alpha_trust)

        bounds = {
            "ramp_cap (tau_ramp)": ramp_bound,
            "ema_convergence (alpha_trust)": ema_bound,
        }
        min_round = self.min_round_for_promotion(initial_trust)
        return {
            "min_round_for_promotion": min_round,
            "min_round_for_tier3": self.min_round_for_tier3(initial_trust),
            "num_rounds": num_rounds,
            "feasible": min_round < num_rounds,
            "binding_constraint": max(bounds, key=bounds.get),
            "bounds": bounds,
        }
