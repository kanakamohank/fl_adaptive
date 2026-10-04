import logging
import random
import time
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
import torch

import flwr as fl
from flwr.common import (
    FitIns, FitRes, GetPropertiesIns, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

# Import our mathematically proven V2 core
from src.tavs_v2.algo1_tavs_scheduler import TavsScheduler
from src.tavs_v2.algo2_esp_projection import EphemeralStructuredProjection
from src.tavs_v2.algo3_bvd_aggregation import BlockVarianceDetector, UnifiedBayesianAggregator

logger = logging.getLogger(__name__)

@dataclass
class TavsEspConfig:
    gamma_budget: float = 0.35
    theta_low: float = 0.3
    theta_high: float = 0.7
    alpha_trust: float = 0.9
    tau_ramp: float = 5.0

    # Staleness caps: how long a promoted client may go unverified before it is
    # forced back to verification regardless of trust. Trust no longer decays on
    # promotion, so without these a client that once earned high trust would stay
    # promoted forever and never be checked again.
    #
    # These are a SAFETY bound, not a throughput dial. s_max binds only when
    # s_max < gamma_budget/(1-gamma_budget); at gamma_budget=0.35 that needs
    # s_max < 0.54, which is impossible. So the budget sets the saving and these
    # caps set the worst-case exposure window. Both are wanted.
    s_max_appearances: int = 4
    s_max_rounds: int = 10
    # Restores the pre-split trust decay, for before/after comparison only.
    decay_trust_on_promotion: bool = False
    # Bayesian weight steepness. Previously hardcoded at the scheduler default
    # and unreachable from config, despite governing the budget arithmetic.
    c_lambda: float = 8.0
    k_trust: int = 10
    p_decoy: float = 0.15
    decoy_probability: float = 0.15 
    detection_threshold: float = 5.0
    
    # CORNER CASE 1 FIXED: Safe mathematical default for JL Projections
    target_k: int = 2048 
    
    projection_type: str = "structured"
    scheduling_type: str = "csprng"
    min_fit_clients: int = 2
    # Declared explicitly because configure_fit now samples against it. It was
    # previously only ever attached dynamically by the pipeline, which works for
    # a plain dataclass but leaves the contract invisible at the definition.
    min_available_clients: int = 2
    master_key: bytes = b'default_key'
    evaluate_fn: Optional[Callable] = None
    min_inlier_fraction_for_agg: float = 0.25

    # Down-weight flagged clients instead of dropping them outright.
    #
    # Detection is a hard gate: one flagged round costs a client its entire
    # contribution. That is the right response to a confirmed attacker and the
    # wrong one to an honest client near the threshold, and the detector cannot
    # tell them apart -- it reports a continuous distance, then thresholds it.
    #
    # The graded behaviour score already exists and already expresses exactly
    # this: 1.0 for a client the detector considers normal, falling linearly to
    # 0.0 one tau_z beyond the threshold. Weighting by it means a marginal
    # client keeps most of its weight, a clearly anomalous one keeps almost
    # none, and a grossly anomalous one reaches zero -- the same outcome as
    # exclusion, reached continuously.
    #
    # This also removes the cliff that made min_inlier_fraction_for_agg
    # necessary: no round can lose most of its data to the threshold, so there
    # is nothing for the fallback to rescue.
    soft_outlier_weighting: bool = True

    # Tier-1-floor ablation knobs (all three must co-vary to actually
    # remove the floor -- see comments on the corresponding TavsScheduler
    # parameters for why).
    #
    # initial_trust default 0.25 sits below theta_low=0.3, forcing fresh
    # clients into Tier 1 (verified) via tier logic. Raising it above
    # theta_low is necessary but NOT sufficient -- the ramp cap (tau_ramp)
    # and the is_stale bootstrap branch also gate fresh clients.
    initial_trust: float = 0.25
    # bootstrap_verify_new_clients default True preserves the anti-Sybil
    # rule "cannot be promoted on no evidence" -- every fresh client's
    # first appearance goes to V via is_stale, before tier logic even
    # runs. False disables that branch, and is required for the Tier-1
    # floor to actually be removable through initial_trust + tau_ramp.
    bootstrap_verify_new_clients: bool = True

    # Aggregation-weight switch. True = num_examples-only (pure FedAvg).
    # False = legacy behaviour_score/bayesian_posterior-scaled weights.
    #
    # DEFAULT FLIPPED TO TRUE after the n=6 partial ablation. The comparison
    # was (paired seeds 1-6, IID + 40%x30% label noise):
    #   noweight - tavs   : +0.0026 late-acc (5/6 same-sign, p=0.10)
    #   noweight - random : +0.0059 late-acc (5/6 same-sign, p=0.01)
    #   noweight - full   : +0.0009 late-acc (4/6 same-sign, p=0.58)
    # i.e. removing trust weighting did not cost TAVS its edge over random,
    # and on 5/6 seeds slightly improved it. Set False to reproduce the
    # legacy weighting (kept for the ablation arm and for any experiment
    # comparing against an earlier weighted-aggregation result).
    #
    # What the flag does, precisely: when True, both verified_weights and
    # promoted_weights become num_examples-only in aggregate_fit. Under
    # enable_outlier_detection=True in a benign setup, BVD scores nearly
    # every verified client at behaviour_score ~ 1.0, so the verified
    # branch changes little; the dominant effect is that promoted clients
    # are no longer downweighted by bayesian_posterior_weight(trust).
    #
    # Note on isolation: setting this OFF while every other TAVS mechanism
    # stays intact only cleanly isolates the aggregation weight at round 1.
    # From round 2 onwards, the aggregate diverges between the two settings,
    # so downstream trust EMA and scheduling drift. Interpret "one knob" as
    # strict only at t=0.
    disable_trust_weighted_aggregation: bool = True

    # Master switch for BVD outlier detection.
    #
    # Off means every verified client is treated as an inlier and scores 1.0.
    # Two uses:
    #   1. A true centralised-vs-federated comparison. With detection on, the
    #      federated arm discards updates, so the comparison measures the
    #      detector as much as it measures federation.
    #   2. Isolating the detector's cost. With zero attackers every rejection is
    #      a false positive by construction -- 26.3% of verified clients under
    #      IID data, where clients are near-identical and nothing should be
    #      flagged at all.
    # Leave ON for any run making a security claim: this disables the Byzantine
    # defence entirely.
    enable_outlier_detection: bool = True

    # Which signal drives the trust EMA when a client is verified.
    #   "bvd"                 (default, historical) -- BVD's behavior_score from
    #                         detect_outliers (Z-score vs cohort median). Across
    #                         pair-flip 15/30% and CIFAR-10N random1 (5 seeds),
    #                         this did not separate noisy from clean clients.
    #   "small_loss_fraction" -- the client's self-reported fraction of a local
    #                         held-out val set the INCOMING global model fits
    #                         (correct predictions / val_size). Clean clients
    #                         should report higher; noisy clients lower because
    #                         ~17% of their val labels disagree with what the
    #                         federation taught the model. One scalar per client
    #                         per round; keeps the T1/T2/T3 structure unchanged.
    #                         Honest-but-noisy threat model only: a malicious
    #                         client could spoof the number.
    trust_signal: str = "bvd"

    # Re-draw the per-round cohort with our own seeded RNG instead of relying on
    # Flower's module-level one, which the run seed does not reach. Without this
    # the same seed produced different cohorts across runs.
    deterministic_sampling: bool = True
    sampling_seed: int = 0

    # Reject unverified updates pointing against the verified cohort's proposed
    # direction. Complements clipping, which bounds distance but not direction:
    # an update inside the clip ball can still point the opposite way.
    # Default OFF on measured evidence. With zero attackers present the gate
    # rejected 42.9% of promoted updates at 20 rounds and 25.8% at 60 -- all
    # honest, since there was nothing else to reject. Logging the raw cosines
    # showed why: the threshold of 0.0 would also reject 20.9% of VERIFIED
    # clients, i.e. it cuts through the middle of the honest distribution rather
    # than isolating anomalies (honest verified median was only +0.155 at
    # data_alpha=0.3, so honest clients are near-orthogonal to their own
    # consensus). Against that measured cost stands no measured benefit: the gate
    # has never been observed catching an attack that clipping missed, because
    # every attack in the suite is large-magnitude and clipping already contains
    # it. Re-enable when there is an adversary it demonstrably catches.
    cosine_filter_promoted: bool = False
    # Minimum cosine to the verified movement. 0.0 rejects only updates actively
    # pulling backwards, which is the unambiguous case; raising it also rejects
    # merely-orthogonal updates and will start catching honest heterogeneity.
    # Calibrating from the logged distribution: admitting 95% of verified clients
    # needs -0.152, not 0.0.
    promoted_cosine_min: float = 0.0

    # Bound how far an unverified (promoted) update may deviate from the verified
    # consensus before it is projected back onto that ball. gamma_budget bounds
    # only the WEIGHT promoted clients carry, never the MAGNITUDE of what they
    # carry, so without this a promoted client inside the budget can still
    # dominate the aggregate outright. Exposed as a flag so the clipped and
    # unclipped variants can be run as a controlled ablation.
    clip_promoted_updates: bool = True
    # Radius as a multiple of the verified cohort's median deviation.
    #
    # 2.0 measured on CIFAR-10 at data_alpha=0.3 with layerwise/distributed
    # attackers: it clipped 0 of 57 promoted updates with no attacker present
    # and 16 of 48 under a 25% Byzantine fraction, i.e. it fires on roughly the
    # attacker population and is inert on honest cohorts. At 1.0 it clipped
    # ~97% in BOTH cases, behaving as blanket normalisation rather than an
    # outlier filter. Containment is nearly independent of the factor in this
    # range, so the looser radius costs nothing against gross attacks.
    #
    # This is an empirical value for that setup, not a universal constant: it
    # depends on the ratio between honest heterogeneity and attack magnitude.
    # The radius itself is self-calibrating (a multiple of the cohort's own
    # median deviation), which is what makes it transfer at all.
    promoted_clip_factor: float = 2.0

class LegacyAnalyticsBridge:
    def __init__(self, round_num, outliers, trust_scores, p_ids, execution_time_ms,
                 tiers=None):
        self.round_number = round_num
        self.byzantine_detected = list(outliers)
        self.consensus_achieved = True
        self.projection_time_ms = 0.0   
        self.detection_time_ms = 0.0
        self.aggregation_time_ms = execution_time_ms
        self.promoted_count = len(p_ids)
        
        class MockSchedulingDecision:
            def __init__(self, scores, p_ids, tiers):
                self.trust_scores = scores.copy()
                # Real tier from the scheduler when available. The old fallback
                # (3 if promoted else 1) is kept only for callers that cannot
                # supply it, and is a promoted flag, NOT a tier.
                self.tier_assignments = dict(tiers) if tiers else {
                    cid: (3 if cid in p_ids else 1) for cid in scores.keys()}

        self.scheduling_decision = MockSchedulingDecision(trust_scores, p_ids, tiers)

class TavsEspStrategy(Strategy):
    def __init__(self, config, model_structure=None):
        super().__init__()
        self.config = config.tavs_config if hasattr(config, 'tavs_config') else config
        
        self.model_blocks = {}
        self.block_shapes = {}
        self.model_structure = model_structure
        
        if model_structure and hasattr(model_structure, 'blocks'):
            for b in model_structure.blocks:
                self.model_blocks[b['name']] = b['num_params']
                self.block_shapes[b['name']] = b['shape']
        else:
            self.model_blocks = {"full_model": 150000}
            self.block_shapes = {"full_model": (150000,)}

        self.scheduler = TavsScheduler(
            gamma_budget=getattr(self.config, 'gamma_budget', 0.35),
            theta_low=getattr(self.config, 'theta_low', 0.3),
            theta_high=getattr(self.config, 'theta_high', 0.8),
            alpha_trust=getattr(self.config, 'alpha_trust', 0.9),
            tau_ramp=getattr(self.config, 'tau_ramp', 5.0),
            k_trust=getattr(self.config, 'k_trust', 10),
            p_decoy=getattr(self.config, 'p_decoy', getattr(self.config, 'decoy_probability', 0.15)),
            c_lambda=getattr(self.config, 'c_lambda', 8.0),
            master_key=getattr(self.config, 'master_key', b'default_key'),
            s_max_appearances=getattr(self.config, 's_max_appearances', 4),
            s_max_rounds=getattr(self.config, 's_max_rounds', 10),
            decay_trust_on_promotion=getattr(self.config, 'decay_trust_on_promotion', False),
            initial_trust=getattr(self.config, 'initial_trust', 0.25),
            bootstrap_verify_new_clients=getattr(
                self.config, 'bootstrap_verify_new_clients', True),
        )
        
        self.projector = EphemeralStructuredProjection(
            target_k=getattr(self.config, 'target_k', 2048),
            model_blocks=self.model_blocks,
            master_key=getattr(self.config, 'master_key', b'default_key')
        )
        
        self.detector = BlockVarianceDetector(
            tau_z=getattr(self.config, 'detection_threshold', 5.0)
        )
        
        self.round_analytics = []

        # Centralised evaluation results, recorded per round by evaluate().
        #
        # flwr.simulation.run_simulation() returns None (its signature is
        # literally `-> None`), unlike the legacy start_simulation() which
        # returned a History. Every centralised loss/accuracy the server computed
        # was therefore discarded the moment evaluate() returned, leaving the
        # pipeline with no metrics to extract. The strategy is the only object
        # that observes every evaluation AND survives the simulation, so it is
        # the correct place to accumulate them.
        self.evaluation_history: List[Dict[str, object]] = []

        # Per-round verified/promoted counts as actually scheduled. Consumed by
        # the comparison experiment so its resource claims are measurements.
        self.scheduling_history: List[Dict[str, int]] = []

        # Per-round count of clients the staleness cap forced back to
        # verification, keyed by round. Populated in configure_fit.
        self._forced_stale: Dict[int, int] = {}

        # Server-side record of each round's verified/promoted/decoy sets, so
        # aggregate_fit never has to trust a client's self-report.
        self._round_assignments: Dict[int, Dict[str, set]] = {}

        # Round index, set by configure_fit so cohort sampling can be keyed on it.
        self._current_round = 0

        # Cohort sampling is done in partition-id space, NOT cid space.
        #
        # Flower's simulation runtime assigns each ClientProxy a 64-bit random
        # integer as its cid, and those integers are fresh for every
        # run_simulation() call. Two runs with identical Python seeds and
        # identical strategy config still see different cid strings, so any
        # sampling that keys on cid (including "sorted(cids) + seeded
        # rng.sample") lands on a different set of clients in each arm --
        # verified by a Flower-only probe with the model and data removed.
        #
        # The fix is to sample in a namespace that is identical across arms
        # by construction: the partition-id that we hand each client via
        # node_config. _ensure_partition_map() populates these bidirectional
        # maps once from a GetProperties sweep at the top of round 1; the
        # cohort selector then draws partition-ids and translates back to cids.
        #
        # Status values:
        #   "unbuilt"         - not yet attempted
        #   "ready"           - map built successfully
        #   "not_applicable"  - client manager is a mock (no get_properties on
        #                       its proxies); legacy sorted-cid path is used
        #                       and cross-arm alignment is out of scope
        #                       (unit-test-only)
        #   "failed"          - real Flower manager but the sweep failed. In
        #                       deterministic_sampling mode this raises rather
        #                       than silently falling back, so a Ray race or
        #                       proxy timeout can never invisibly reintroduce
        #                       the cohort-divergence bug the fix targets.
        # Once "ready", "not_applicable", or "failed", the status is latched
        # for the run.
        self._partition_to_cid: Dict[int, str] = {}
        self._cid_to_partition: Dict[str, int] = {}
        self._partition_map_status: str = "unbuilt"

        # Global parameters handed out this round, kept as blocks. The cosine
        # gate needs them as the origin: clients send full parameters, so a
        # "direction" only exists relative to where the round started.
        self._previous_global: Dict[str, torch.Tensor] = {}

    def initialize_parameters(self, client_manager):
        from src.core.models import get_model
        
        # Dynamically instantiate the correct model type (safe for both Phase 4 and Phase 5)
        model_type = getattr(self.config, "model_type", "cifar_cnn")
        model = get_model(model_type, num_classes=10)
        
        # ---> THE FIX: Force copy=True and float32 for clean Ray serialization
        return ndarrays_to_parameters(
            [np.array(p.detach().cpu().numpy(), dtype=np.float32, copy=True) for p in model.parameters()]
        ) 

    def _parameters_to_blocks(self, parameters) -> Dict[str, torch.Tensor]:
        """
        Split the global parameter vector into the same blocks client updates use.

        Kept so the cosine gate has an origin: clients submit full parameters, so
        the update client i proposes is (g_i - w_prev), and without w_prev there
        is no direction to compare.
        """
        if parameters is None:
            return {}
        try:
            ndarrays = parameters_to_ndarrays(parameters)
        except Exception:
            return {}

        blocks: Dict[str, torch.Tensor] = {}
        items = list(self.model_blocks.items())
        if len(ndarrays) == 1:
            flat = np.asarray(ndarrays[0], dtype=np.float32).flatten()
            if flat.size != sum(sz for _, sz in items):
                return {}
            off = 0
            for name, size in items:
                blocks[name] = torch.tensor(flat[off:off + size], dtype=torch.float32)
                off += size
        else:
            for i, (name, size) in enumerate(items):
                if i >= len(ndarrays):
                    break
                arr = np.asarray(ndarrays[i], dtype=np.float32).flatten()
                if arr.size != size:
                    return {}
                blocks[name] = torch.tensor(arr, dtype=torch.float32)
        return blocks

    def _ensure_partition_map(self, client_manager) -> None:
        """
        Build cid <-> partition-id maps once, from a synchronous GetProperties
        sweep of every registered ClientProxy.

        The partition-id is what makes cohorts comparable across arms: it is
        assigned deterministically by Flower's simulation node_config and is
        echoed by each client's get_properties(). The cid is a fresh random
        64-bit integer per simulation call, so it cannot be used as a stable
        identity across arms.

        Latches its result into self._partition_map_status:
            "ready"          -- map built, sampling can key on partition-ids
            "not_applicable" -- mock client manager (unit tests); sample
                                cohort falls back to the legacy sorted-cid
                                path with no cross-arm alignment guarantee
            "failed"         -- real manager but the sweep failed; _sample_cohort
                                REFUSES to silently fall back in deterministic
                                mode and raises instead. The point of this
                                latch is that a Ray race or proxy timeout can
                                never invisibly reintroduce the cohort-
                                divergence bug this method was written to fix.

        Once latched, the method is a no-op (which also prevents a failed
        first attempt from being retried at 30s cost every round).

        Deliberately does NOT require partition-ids to form a contiguous
        0..N-1 range. Whatever pids answer are what we sample from -- as long
        as the set is the same across arms (which is true because
        node_config partition-ids are assigned by Flower, not by the
        strategy), sampling on that set gives identical cohorts across arms.
        """
        if self._partition_map_status != "unbuilt":
            return
        if not hasattr(client_manager, "all"):
            self._partition_map_status = "not_applicable"
            return

        expected = getattr(self.config, "min_available_clients", 0) or 0
        if expected and hasattr(client_manager, "wait_for"):
            try:
                client_manager.wait_for(num_clients=expected, timeout=30)
            except Exception as e:
                # wait_for failing means we cannot trust we have all clients;
                # do not silently push through with a partial map.
                logger.error(
                    "partition-map bootstrap: wait_for(%s, timeout=30) raised %s: %s",
                    expected, type(e).__name__, e,
                )
                self._partition_map_status = "failed"
                return

        proxies = client_manager.all()
        if not proxies:
            logger.error("partition-map bootstrap: client_manager.all() returned empty")
            self._partition_map_status = "failed"
            return

        # Detect the mock-manager case: at least one proxy without a real
        # get_properties method means we are in a unit test. That is fine but
        # we cannot align cohorts across arms, so mark not_applicable.
        for _cid, proxy in proxies.items():
            if not hasattr(proxy, "get_properties"):
                self._partition_map_status = "not_applicable"
                return

        pid_to_cid: Dict[int, str] = {}
        cid_to_pid: Dict[str, int] = {}
        for cid, proxy in proxies.items():
            try:
                res = proxy.get_properties(
                    ins=GetPropertiesIns(config={}), timeout=30, group_id=None
                )
            except Exception as e:
                logger.error(
                    "partition-map bootstrap: get_properties on cid=%s raised %s: %s",
                    cid, type(e).__name__, e,
                )
                self._partition_map_status = "failed"
                return
            pid = res.properties.get("partition-id") if getattr(res, "properties", None) else None
            if pid is None:
                logger.error("partition-map bootstrap: cid=%s reported no partition-id", cid)
                self._partition_map_status = "failed"
                return
            pid_int = int(pid)
            if pid_int in pid_to_cid:
                # Two proxies claiming the same partition-id is not a benign
                # skew; refuse rather than silently pick one.
                logger.error(
                    "partition-map bootstrap: duplicate partition-id %s (cids %s and %s)",
                    pid_int, pid_to_cid[pid_int], cid,
                )
                self._partition_map_status = "failed"
                return
            pid_to_cid[pid_int] = cid
            cid_to_pid[cid] = pid_int

        if not pid_to_cid:
            self._partition_map_status = "failed"
            return

        self._partition_to_cid = pid_to_cid
        self._cid_to_partition = cid_to_pid
        self._partition_map_status = "ready"

    def _sample_cohort(self, client_manager) -> Dict[str, ClientProxy]:
        """
        Select this round's participating clients, keyed by client id.

        Sampling happens in partition-id space whenever the partition map
        builds (`_partition_map_status == "ready"`). That is the path that
        aligns cohorts across arms with the same seed.

        The three failure modes are handled explicitly, NOT by silently
        falling through:
          * "ready"          -> sample partition-ids, translate to cids.
                                Every selected pid MUST resolve to a proxy;
                                if any does not, raise, because the alt-path
                                is exactly the divergent behaviour this
                                method was written to eliminate.
          * "not_applicable" -> legacy sorted-cid path. Unit tests with mock
                                ClientManagers land here; they are already
                                out of scope for cross-arm alignment.
          * "failed"         -> raise in deterministic_sampling mode. Silent
                                fallback here is what would reintroduce the
                                cohort-divergence bug invisibly.
        """
        if not hasattr(client_manager, "sample") or not hasattr(client_manager, "num_available"):
            return dict(client_manager.all())

        num_available = client_manager.num_available()
        requested = getattr(self.config, "min_fit_clients", num_available)
        sample_size = max(1, min(requested, num_available))
        min_num = min(getattr(self.config, "min_available_clients", sample_size), num_available)
        deterministic = getattr(self.config, "deterministic_sampling", True)

        if deterministic:
            self._ensure_partition_map(client_manager)
            status = self._partition_map_status

            if status == "ready":
                # Sort available pids so the rng draws from a stable ordering
                # across arms (the pid SET is stable by construction, but the
                # dict insertion order isn't guaranteed to be).
                available_pids = sorted(self._partition_to_cid.keys())
                k = min(sample_size, len(available_pids))
                if k <= 0:
                    raise RuntimeError(
                        f"_sample_cohort: computed sample_size={sample_size} with "
                        f"{len(available_pids)} partitions available; cannot sample."
                    )
                rng = random.Random(
                    f"{getattr(self.config, 'sampling_seed', 0)}_{self._current_round}"
                )
                selected_pids = rng.sample(available_pids, k)
                proxies = client_manager.all()
                sampled = []
                missing_pids = []
                for pid in selected_pids:
                    cid = self._partition_to_cid[pid]
                    proxy = proxies.get(cid)
                    if proxy is None:
                        missing_pids.append(pid)
                    else:
                        sampled.append(proxy)
                if missing_pids:
                    raise RuntimeError(
                        f"_sample_cohort: partition-map contained {len(missing_pids)} "
                        f"partition-ids whose cids are no longer in client_manager.all(): "
                        f"{missing_pids}. Refusing to fall back to a divergent "
                        f"code path silently."
                    )
                return {p.cid: p for p in sampled}

            if status == "failed":
                # Loud on purpose. Do NOT let the divergent legacy path run
                # under deterministic sampling.
                raise RuntimeError(
                    "Partition-map bootstrap failed; refusing to fall back to "
                    "the legacy sorted-cid path in deterministic_sampling mode "
                    "because that path does not align cohorts across arms. "
                    "See tavs_esp_strategy.py::_ensure_partition_map error log."
                )
            # status == "not_applicable" (mock manager): fall through to the
            # legacy path below.

        sampled = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)

        # Legacy fallback: sort cids and re-draw with our own seeded RNG.
        # Only reached for mock ClientManagers (not_applicable) or with
        # deterministic_sampling explicitly disabled. Not stable across arms
        # in a real Flower simulation; that is what the partition-id path is
        # for.
        if deterministic and hasattr(client_manager, "all"):
            pool = sorted(client_manager.all().values(), key=lambda p: p.cid)
            if len(pool) >= sample_size:
                rng = random.Random(
                    f"{getattr(self.config, 'sampling_seed', 0)}_{self._current_round}"
                )
                sampled = rng.sample(pool, sample_size)

        return {proxy.cid: proxy for proxy in sampled}

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager):
        self._current_round = server_round
        self._previous_global = self._parameters_to_blocks(parameters)
        # Sample the per-round cohort BEFORE scheduling.
        #
        # This previously used client_manager.all(), which made every client in
        # the federation train every round and silently ignored
        # clients_per_round / min_fit_clients. Partial participation is a
        # defining property of FL, and it also changes what TAVS is measured on:
        # with full participation the scheduler never has to choose between
        # clients, so the verification budget it manages is unrepresentative.
        available_clients = self._sample_cohort(client_manager)
        client_ids = list(available_clients.keys())

        if not client_ids:
            return []

        # Snapshot who the staleness cap forced into V, BEFORE aggregate_fit's
        # update_trust resets the clocks. Measured after the fact this is always
        # zero, because verification is exactly what clears staleness.
        forced = sum(1 for cid in client_ids
                     if self.scheduler.is_stale(cid, server_round))

        V, P, D = self.scheduler.schedule_verifications(client_ids, server_round)
        self._forced_stale[server_round] = forced
        logger.info(
            f"Round {server_round} Scheduling: {len(V)} Verified "
            f"({len(D)} of them decoys), {len(P)} Promoted"
        )

        # `sampled` records the full cohort this round, independent of how
        # the scheduler split it. Previously omitted; cohort-composition
        # analyses that want "who was in the cohort" would otherwise need to
        # reconstruct it from V ∪ P ∪ D round-by-round, and ad-hoc probes
        # that read `.get("sampled")` would silently see no cohort. Decoys
        # live inside V (D is a subset of V) so the union here is just V ∪ P.
        self._round_assignments[server_round] = {
            "verified": set(V), "promoted": set(P), "decoy": set(D),
            "sampled": set(V) | set(P),
        }

        fit_configurations = []
        for cid, client_proxy in available_clients.items():
            if cid not in V and cid not in P:
                continue

            # What the client is TOLD, which is not always what the server DOES.
            #
            # A decoy is verified server-side but told it was promoted. That
            # asymmetry is the entire mechanism: an adaptive attacker that
            # behaves honestly whenever it knows it is being checked would
            # otherwise evade every check, and announcing is_verified=True to a
            # decoy hands it exactly that signal. Telling it "promoted" means it
            # attacks, and the hidden verification catches it.
            told_verified = (cid in V) and (cid not in D)

            config_dict = {
                "server_round": server_round,
                "is_verified": told_verified,
                "tavs_assignment": "verified" if told_verified else "promoted",
                "trust_score": float(self.scheduler.get_effective_trust(cid, server_round)),
            }
            fit_configurations.append((client_proxy, FitIns(parameters, config_dict)))

        return fit_configurations

    def _parse_client_updates(self, results: List[Tuple[ClientProxy, FitRes]]) -> Dict[str, Dict[str, torch.Tensor]]:
        parsed = {}
        block_items = list(self.model_blocks.items())
        total_expected_params = sum(size for _, size in block_items)

        # Lazy-init the cid -> config-id bridge. Kept per-strategy so it
        # persists across rounds; a client seen once in round 1 stays in the
        # map even if it is not sampled in round 2, which is exactly what a
        # late-run analysis needs to reconstruct trust-vs-noise correlations.
        if not hasattr(self, "cid_to_client_config_id"):
            self.cid_to_client_config_id = {}

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            # Record cid -> "honest_XX" as soon as we see the client's metric
            # in a FitRes. Falls back to cid itself for clients that never
            # report client_id, which keeps downstream keys well-defined
            # rather than silently missing.
            reported = fit_res.metrics.get("client_id") if fit_res.metrics else None
            if reported:
                self.cid_to_client_config_id[cid] = str(reported)
            elif cid not in self.cid_to_client_config_id:
                self.cid_to_client_config_id[cid] = cid
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_blocks: Dict[str, torch.Tensor] = {}
            
            if len(ndarrays) == 1:
                flat = np.asarray(ndarrays[0], dtype=np.float64).flatten()
                
                # CORNER CASE 4 FIXED: Protect against malformed attacker tensors
                if flat.size != total_expected_params:
                    logger.warning(f"Client {cid} sent {flat.size} params, expected {total_expected_params}. Dropping.")
                    continue
                    
                offset = 0
                for block_name, size in block_items:
                    client_blocks[block_name] = torch.tensor(flat[offset : offset + size], dtype=torch.float32)
                    offset += size
            else:
                for i, (block_name, size) in enumerate(block_items):
                    if i >= len(ndarrays): break
                    arr = np.asarray(ndarrays[i], dtype=np.float64).flatten()
                    n = min(arr.size, size)
                    t = torch.zeros(size, dtype=torch.float32)
                    if n > 0:
                        t[:n] = torch.tensor(arr[:n], dtype=torch.float32)
                    client_blocks[block_name] = t
            parsed[cid] = client_blocks
        return parsed

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
        if not results:
            return None, {}

        start_time = time.time()
        all_updates = self._parse_client_updates(results)

        # Sample counts, for FedAvg-style weighting below. Flower already carries
        # these in FitRes; they were simply never read.
        num_examples = {proxy.cid: max(1, int(res.num_examples))
                        for proxy, res in results}
        
        # Verified/promoted split comes from the SERVER's own record of what it
        # scheduled, never from the client.
        #
        # This previously read res.metrics["is_verified"], i.e. it asked each
        # client which bucket to put it in. A Byzantine client only had to report
        # is_verified=False to be routed into P_ids -- and promoted clients are
        # never projected and never passed to the detector, so the poison went
        # straight into the aggregate weighted by p_i. The attacker could opt out
        # of the defence by setting one boolean.
        #
        # Clients whose assignment the server has no record of (a stale round, a
        # late reply) fall back to verified, the conservative choice.
        scheduled = self._round_assignments.get(server_round)
        if scheduled is None:
            V_ids = {proxy.cid for proxy, _res in results}
            P_ids = set()
        else:
            returned = {proxy.cid for proxy, _res in results}
            P_ids = returned & scheduled["promoted"]
            V_ids = returned - P_ids
        
        projected_updates = {}
        for cid in V_ids:
            if cid in all_updates:
                projected_updates[cid] = self.projector.project_client_update(all_updates[cid], server_round)
        
        if getattr(self.config, "enable_outlier_detection", True):
            inliers, outliers, behavior_scores = self.detector.detect_outliers(
                projected_updates, V_ids)
        else:
            # Detection disabled: everyone verified is an inlier at full trust.
            inliers, outliers = set(V_ids), set()
            behavior_scores = {cid: 1.0 for cid in V_ids}
        logger.info(f"Round {server_round} Detection: {len(inliers)} Inliers, {len(outliers)} Outliers")

        # Replace BVD's behavior_scores with each client's self-reported
        # small-loss fraction, if the client sent one AND the config asks for
        # it. BVD still ran (we keep its inlier/outlier set for the cosine +
        # magnitude gates in aggregation), but the TRUST EMA update below is
        # driven by this signal instead of BVD's Z-score.
        #
        # Keyed by cid; fallback is BVD's own score for clients that did not
        # report one. Clients that reported a score but weren't verified in
        # this round are ignored -- trust only updates on verification.
        trust_signal = getattr(self.config, "trust_signal", "bvd")
        if trust_signal == "small_loss_fraction":
            overridden = 0
            for proxy, fit_res in results:
                cid = proxy.cid
                if cid not in V_ids:
                    continue
                reported = (fit_res.metrics or {}).get("small_loss_fraction")
                if reported is None:
                    continue
                try:
                    val = float(reported)
                except (TypeError, ValueError):
                    continue
                # Clip to [0, 1] -- the trust scheduler expects behavior_score
                # in that range (higher = cleaner, same semantic as BVD's).
                if not (0.0 <= val <= 1.0):
                    val = max(0.0, min(1.0, val))
                behavior_scores[cid] = val
                overridden += 1
            logger.info(f"Round {server_round} trust_signal='small_loss_fraction': "
                        f"overrode {overridden}/{len(V_ids)} verified clients' "
                        f"behavior_score from BVD to client-reported val fit")

        if getattr(self.config, "soft_outlier_weighting", True):
            # Everyone verified contributes; behaviour_score sets how much.
            inliers_for_agg = set(V_ids)
        else:
            min_frac = getattr(self.config, "min_inlier_fraction_for_agg", 0.25)
            inliers_for_agg = set(inliers)
            if V_ids and len(inliers) < max(1, int(min_frac * len(V_ids))):
                inliers_for_agg = set(V_ids)

        for cid in V_ids:
            self.scheduler.update_trust(cid, behavior_score=behavior_scores.get(cid, 0.0),
                                        was_verified=True, is_outlier=cid in outliers,
                                        round_num=server_round)
        for cid in P_ids:
            self.scheduler.update_trust(cid, behavior_score=0.0, was_verified=False,
                                        round_num=server_round)

        verified_updates = {cid: all_updates[cid] for cid in inliers_for_agg if cid in all_updates}
        promoted_updates = {cid: all_updates[cid] for cid in P_ids if cid in all_updates}
        
        # Aggregation weight = trust factor x dataset size.
        #
        # The dataset-size term is standard FedAvg (w_i proportional to n_i) and
        # was missing: weighting came from trust alone, so a client holding 811
        # samples had exactly the same vote as one holding 5261. Under a Dirichlet
        # split at alpha=0.3 that spread is real -- measured 6.5x between the
        # smallest and largest client -- and it systematically over-weights the
        # small, label-skewed clients whose local optimum sits furthest from the
        # global one.
        #
        # The trust factor is retained as a multiplier, so TAVS still discounts
        # unverified clients; it now discounts them relative to a correct base
        # weight instead of replacing it.
        # Ablation switch. When on, every included client gets weight
        # num_examples only (pure FedAvg) -- no trust, no behaviour score.
        # Turning this on and leaving everything else intact tests whether
        # TAVS's win over random-skip comes from trust-weighted aggregation
        # or from something else (staleness caps, skip schedule, tier
        # structure). Cosine gate and magnitude clip still run because they
        # act on updates before this line and are inherited by random-skip.
        disable_trust_agg = getattr(self.config,
                                    "disable_trust_weighted_aggregation", False)

        if disable_trust_agg:
            promoted_weights = {cid: num_examples.get(cid, 1) for cid in P_ids}
        else:
            promoted_weights = {
                cid: self.scheduler.bayesian_posterior_weight(
                    self.scheduler.get_effective_trust(cid, server_round)
                ) * num_examples.get(cid, 1)
                for cid in P_ids
            }

        # The 0.05 floor guarantees a client the detector accepted is never
        # silenced by a marginal score. It is deliberately NOT applied to a
        # flagged client: an update far enough out must be able to reach zero
        # weight, or soft weighting would leave every attacker a residual voice.
        soft = getattr(self.config, "soft_outlier_weighting", True)
        if disable_trust_agg:
            verified_weights = {
                cid: num_examples.get(cid, 1) for cid in verified_updates.keys()
            }
        else:
            verified_weights = {
                cid: (float(behavior_scores.get(cid, 0.0))
                      if (soft and cid in outliers)
                      else max(0.05, float(behavior_scores.get(cid, 0.0)))) * num_examples.get(cid, 1)
                for cid in verified_updates.keys()
            }
        clip_stats: Dict[str, object] = {}
        cosine_stats: Dict[str, object] = {}
        aggregated_blocks = UnifiedBayesianAggregator.aggregate(
            verified_updates, verified_weights,
            promoted_updates, promoted_weights,
            clip_promoted=getattr(self.config, "clip_promoted_updates", True),
            clip_factor=getattr(self.config, "promoted_clip_factor", 1.0),
            clip_stats_out=clip_stats,
            cosine_filter=getattr(self.config, "cosine_filter_promoted", True),
            cosine_min=getattr(self.config, "promoted_cosine_min", 0.0),
            previous_global=self._previous_global,
            cosine_stats_out=cosine_stats,
        )
        if cosine_stats.get("num_rejected"):
            logger.info(
                f"Round {server_round} Cosine gate: rejected "
                f"{cosine_stats['num_rejected']} promoted update(s) pointing against "
                f"the verified direction (min cosine {cosine_stats['min_cosine_seen']:.3f})"
            )
        if clip_stats.get("num_clipped"):
            logger.info(
                f"Round {server_round} Clipping: {clip_stats['num_clipped']} promoted "
                f"update(s) exceeded radius {clip_stats['clip_radius']:.4g} "
                f"(max deviation {clip_stats['max_deviation_ratio']:.1f}x the radius)"
            )
        elif clip_stats.get("skipped_reason"):
            logger.debug(f"Round {server_round} Clipping skipped: {clip_stats['skipped_reason']}")

        execution_time_ms = (time.time() - start_time) * 1000

        if not aggregated_blocks:
            return None, {}

        tiers = {cid: self.scheduler.get_tier(cid, server_round)
                 for cid in self.scheduler.trust_scores}
        analytics = LegacyAnalyticsBridge(server_round, outliers, self.scheduler.trust_scores,
                                          P_ids, execution_time_ms, tiers=tiers)
        self.round_analytics.append(analytics)

        # Actual per-round scheduling counts, measured rather than assumed.
        # The comparison experiment used to hardcode these as clients_per_round
        # for TAVS and num_clients for the baseline, which produced a fixed
        # "2.5x fewer verifications" regardless of what the scheduler really did
        # -- and what it really did was verify everyone, because promotion was
        # unreachable. Recording the true counts makes that visible.
        self.scheduling_history.append({
            "round": server_round,
            "cohort_size": len(V_ids) + len(P_ids),
            "num_verified": len(V_ids),
            "num_promoted": len(P_ids),
            "num_inliers": len(inliers),
            "num_outliers": len(outliers),
            "num_clipped": int(clip_stats.get("num_clipped") or 0),
            "num_cosine_rejected": int(cosine_stats.get("num_rejected") or 0),
            # Decoys: Tier 3 clients verified server-side while being told they
            # were promoted. This path sits inside the Tier 3 branch, and Tier 3
            # never fired until the trust split, so it has never executed in any
            # experiment. Recorded so "it ran" is a measurement, not a assumption.
            "num_decoys": len((scheduled or {}).get("decoy", ())),
            # Clients forced back to verification by the staleness cap.
            #
            # Captured in configure_fit, BEFORE update_trust resets the staleness
            # clocks. Evaluating it here always returned 0: by this point every
            # verified client has had appearances_since_verified zeroed and
            # last_verified_round set to the current round, so is_stale() is
            # false for all of them by construction.
            "num_forced_stale": self._forced_stale.get(server_round, 0),
            # Detector diagnostics: the statistic the threshold acts on, and the
            # denominator that sets its scale. Empty when detection is disabled.
            **{f"det_{k}": v for k, v in
               (getattr(self.detector, "last_stats", None) or {}).items()},
            "clip_radius": clip_stats.get("clip_radius"),
            # Raw cosines, so a threshold can be calibrated post hoc from logged
            # runs rather than by re-running a sweep per candidate value. Sorted
            # lists rather than per-client dicts: client ids add no analysis value
            # here and would bloat the results file every round.
            "promoted_cosines": sorted(cosine_stats.get("promoted_cosines", {}).values()),
            "verified_cosines": sorted(cosine_stats.get("verified_cosines", {}).values()),
        })

        aggregated_ndarrays = []
        for name in self.model_blocks.keys():
            # CORNER CASE 3 FIXED: Safely detach from GPU/MPS before NumPy conversion
            flat_agg = aggregated_blocks[name].detach().cpu().numpy()
            target_shape = self.block_shapes.get(name)
            
            if target_shape and np.prod(target_shape) == flat_agg.size:
                reshaped_array = flat_agg.reshape(target_shape)
                aggregated_ndarrays.append(reshaped_array)
            else:
                aggregated_ndarrays.append(flat_agg)

        return ndarrays_to_parameters(aggregated_ndarrays), {"inliers": len(inliers), "outliers": len(outliers)}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}
    
    def evaluate(self, server_round, parameters):
        evaluate_fn = getattr(self.config, 'evaluate_fn', None)
        if evaluate_fn is None:
            return None

        result = evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
        if result is None:
            return None

        # Record before returning: the server drops these into a History that
        # run_simulation() never hands back to us (see __init__).
        loss, metrics = result
        self.evaluation_history.append({
            "round": server_round,
            "loss": float(loss),
            "accuracy": float(metrics.get("accuracy", 0.0)) if isinstance(metrics, dict) else 0.0,
            "metrics": metrics,
        })
        return result

    def export_complete_state(self):
        # `cid_to_client_config_id` bridges Flower's opaque proxy.cid (used as
        # the trust-dict key) to the human-readable "honest_XX" from the client
        # configs. Without this, downstream analysis cannot tell which pool
        # entry a particular trust score belongs to -- the pilot's diagnostic
        # ("did TAVS put low trust on the actually-noisy clients?") is
        # answerable only via this map. Populated defensively (empty when the
        # strategy never saw a FitRes with the client_id metric).
        #
        # `round_assignments` is the ground-truth V/P/D split per round --
        # necessary for forensics because tier_evolution defaults absent
        # clients to Tier 1 (Verified), which makes RandomSkip and
        # FullVerification arms look like they never promoted anyone. This
        # is the fix: dump the strategy's own record.
        return {
            "trust_state": self.scheduler.trust_scores,
            "cid_to_client_config_id": dict(getattr(self, "cid_to_client_config_id", {})),
            "round_assignments": {
                r: {k: sorted(v) for k, v in assn.items()}
                for r, assn in getattr(self, "_round_assignments", {}).items()
            },
        }

class FullVerificationStrategy(TavsEspStrategy):
    """
    Traditional Byzantine-robust baseline: verify EVERY client EVERY round.

    This is the control arm for the TAVS comparison. It runs the identical
    defence pipeline (ESP projection -> BVD outlier detection -> aggregation) but
    performs no trust-adaptive scheduling: no tiers, no promotion, no decoys, no
    budget constraint. Every sampled client is verified, always.

    Why this exists as a class rather than a TavsEspConfig preset
    -------------------------------------------------------------
    The comparison experiment previously built its baseline by disabling TAVS
    through configuration: theta_low=0.0, theta_high=1.0, gamma_budget=1.0.
    Tracing that through TavsScheduler.schedule_verifications shows it produces
    the exact opposite of the intent:

        t_eff < theta_low   -> `t < 0.0` is never true -> no client is verified
        t_eff >= theta_high -> `t >= 1.0` is never true -> no client is promoted
                                                          via the Tier 3 path
        everything else     -> falls to the Tier 2 branch -> tentatively promoted
        gamma_budget = 1.0  -> the demotion loop never triggers

    So the "verify everything" baseline verified nothing and trusted everyone,
    which is why it ran ~30x faster than TAVS and reported the efficiency
    comparison backwards. Encoding the baseline as an explicit override makes it
    impossible to misconfigure it into a different algorithm by accident.
    """

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: fl.server.client_manager.ClientManager):
        self._current_round = server_round
        self._previous_global = self._parameters_to_blocks(parameters)
        sampled = list(self._sample_cohort(client_manager).values())
        if not sampled:
            return []

        # Record cohort composition so cross-arm diagnostics (cohort-fix era)
        # can audit who participated each round for the full-verify arm too.
        # V = everyone, P = empty, by construction.
        sampled_cids = {proxy.cid for proxy in sampled}
        self._round_assignments[server_round] = {
            "verified": set(sampled_cids), "promoted": set(), "decoy": set(),
            "sampled": set(sampled_cids),
        }

        # Every sampled client is verified. aggregate_fit splits verified from
        # promoted on the is_verified flag the client echoes back, so setting it
        # True here routes all of them down the verified path.
        config_dict = {"server_round": server_round, "is_verified": True}
        return [(proxy, FitIns(parameters, config_dict.copy())) for proxy in sampled]


class RandomSkipStrategy(TavsEspStrategy):
    """
    Fair baseline for TAVS: same skip rate, uniform-random selection.

    Runs the identical pipeline (ESP projection -> BVD on verified -> cosine
    gate + magnitude clip on promoted -> aggregate) but replaces the
    trust-adaptive scheduler with a coin flip. Each sampled client is placed in
    the verified set with probability (1 - skip_rate); the rest go to promoted.

    This isolates the value of the trust signal itself: TAVS and RandomSkip
    save the same BVD compute per round in expectation, so any accuracy gap
    between them is attributable to whom each policy chose to skip, not to how
    many. If the two are indistinguishable, the trust EMA is not doing work
    that a fair coin could not.

    Trust EMA still runs (so aggregate_fit's code path is unchanged), but its
    output is ignored for scheduling.
    """

    def __init__(self, config, model_structure=None, skip_rate: float = 0.4,
                 skip_seed: int = 0):
        super().__init__(config, model_structure=model_structure)
        # Guard the rate: 0.0 degenerates to full-verify; 1.0 verifies no one
        # and starves the cosine gate / clip of a reference cohort, so aggregate_fit
        # falls back to fabricating behaviour and this experiment stops measuring
        # anything meaningful. Reject the endpoints early.
        if not (0.0 <= skip_rate < 1.0):
            raise ValueError(f"skip_rate must be in [0, 1); got {skip_rate}")
        self.skip_rate = skip_rate
        # Own generator so seeds are reproducible across policy runs without
        # depending on Flower's or numpy's global state.
        self._skip_rng = random.Random(skip_seed)

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: fl.server.client_manager.ClientManager):
        self._current_round = server_round
        self._previous_global = self._parameters_to_blocks(parameters)
        available_clients = self._sample_cohort(client_manager)
        client_ids = sorted(available_clients.keys())  # deterministic ordering
        if not client_ids:
            return []

        # Draw a Bernoulli per client rather than fixing an exact skip count.
        # The exact-count alternative correlates the assignments (once we've
        # promoted the target number the rest must be verified), which biases
        # the trust EMA updates the pipeline still runs behind the scenes. A
        # per-client coin keeps the two policies' aggregation math identical in
        # expectation.
        V, P = set(), set()
        for cid in client_ids:
            if self._skip_rng.random() < self.skip_rate:
                P.add(cid)
            else:
                V.add(cid)

        # Safety: if the draws happen to promote everyone, force one client to
        # verified so the cohort has a reference for cosine gate + clipping.
        # Without this the gates skip with skipped_reason="no_verified_clients_*"
        # and every promoted update passes through unbounded, which is not what
        # a "same-pipeline, different selection" baseline should measure.
        if not V:
            demoted = self._skip_rng.choice(client_ids)
            P.discard(demoted); V.add(demoted)

        self._round_assignments[server_round] = {
            "verified": set(V), "promoted": set(P), "decoy": set(),
            "sampled": set(V) | set(P),
        }
        # Staleness accounting is meaningless without a trust-based promotion
        # policy; record zero so downstream summarisation does not choke.
        self._forced_stale[server_round] = 0
        logger.info(
            f"Round {server_round} RandomSkip: {len(V)} Verified, "
            f"{len(P)} Promoted (rate={self.skip_rate})"
        )

        fit_configurations = []
        for cid in client_ids:
            proxy = available_clients[cid]
            told_verified = cid in V
            config_dict = {
                "server_round": server_round,
                "is_verified": told_verified,
                "tavs_assignment": "verified" if told_verified else "promoted",
                # Present trust as 0.5 so honest clients trained end-to-end do
                # not read "trust=0.0" as a signal from the server. Random-skip
                # advertises no trust judgement.
                "trust_score": 0.5,
            }
            fit_configurations.append((proxy, FitIns(parameters, config_dict)))
        return fit_configurations
