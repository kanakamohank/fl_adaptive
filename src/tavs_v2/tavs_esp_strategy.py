import logging
import random
import time
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
import torch

import flwr as fl
from flwr.common import (
    FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
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
    tau_ramp: float = 30.0
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

    # Re-draw the per-round cohort with our own seeded RNG instead of relying on
    # Flower's module-level one, which the run seed does not reach. Without this
    # the same seed produced different cohorts across runs.
    deterministic_sampling: bool = True
    sampling_seed: int = 0

    # Reject unverified updates pointing against the verified cohort's proposed
    # direction. Complements clipping, which bounds distance but not direction:
    # an update inside the clip ball can still point the opposite way.
    cosine_filter_promoted: bool = True
    # Minimum cosine to the verified movement. 0.0 rejects only updates actively
    # pulling backwards, which is the unambiguous case; raising it also rejects
    # merely-orthogonal updates and will start catching honest heterogeneity.
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
    def __init__(self, round_num, outliers, trust_scores, p_ids, execution_time_ms):
        self.round_number = round_num
        self.byzantine_detected = list(outliers)
        self.consensus_achieved = True
        self.projection_time_ms = 0.0   
        self.detection_time_ms = 0.0
        self.aggregation_time_ms = execution_time_ms
        self.promoted_count = len(p_ids)
        
        class MockSchedulingDecision:
            def __init__(self, scores, p_ids):
                self.trust_scores = scores.copy()
                self.tier_assignments = {cid: (3 if cid in p_ids else 1) for cid in scores.keys()}
                
        self.scheduling_decision = MockSchedulingDecision(trust_scores, p_ids)

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
            tau_ramp=getattr(self.config, 'tau_ramp', 30.0),
            k_trust=getattr(self.config, 'k_trust', 10),
            p_decoy=getattr(self.config, 'p_decoy', getattr(self.config, 'decoy_probability', 0.15)),
            master_key=getattr(self.config, 'master_key', b'default_key')
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

        # Server-side record of each round's verified/promoted/decoy sets, so
        # aggregate_fit never has to trust a client's self-report.
        self._round_assignments: Dict[int, Dict[str, set]] = {}

        # Round index, set by configure_fit so cohort sampling can be keyed on it.
        self._current_round = 0

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

    def _sample_cohort(self, client_manager) -> Dict[str, ClientProxy]:
        """
        Select this round's participating clients, keyed by client id.

        Falls back to full participation only when the client manager cannot
        sample, so older mocks and single-cohort setups keep working.
        """
        if not hasattr(client_manager, "sample") or not hasattr(client_manager, "num_available"):
            return dict(client_manager.all())

        num_available = client_manager.num_available()
        requested = getattr(self.config, "min_fit_clients", num_available)
        sample_size = max(1, min(requested, num_available))
        min_num = min(getattr(self.config, "min_available_clients", sample_size), num_available)

        sampled = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)

        # Flower samples with its own module-level RNG, which our seed never
        # touches. That is why re-running the same seed changed verification
        # counts (107 -> 112) and moved late accuracy by up to 0.092. Re-select
        # deterministically from the returned pool instead: sort by client id so
        # the order does not depend on Flower's internal state, then draw with a
        # generator keyed on (round, seed).
        if getattr(self.config, "deterministic_sampling", True) and hasattr(client_manager, "all"):
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

        V, P, D = self.scheduler.schedule_verifications(client_ids, server_round)
        logger.info(
            f"Round {server_round} Scheduling: {len(V)} Verified "
            f"({len(D)} of them decoys), {len(P)} Promoted"
        )

        self._round_assignments[server_round] = {
            "verified": set(V), "promoted": set(P), "decoy": set(D),
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

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
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
        
        inliers, outliers, behavior_scores = self.detector.detect_outliers(projected_updates, V_ids)
        logger.info(f"Round {server_round} Detection: {len(inliers)} Inliers, {len(outliers)} Outliers")

        min_frac = getattr(self.config, "min_inlier_fraction_for_agg", 0.25)
        inliers_for_agg = set(inliers)
        if V_ids and len(inliers) < max(1, int(min_frac * len(V_ids))):
            inliers_for_agg = set(V_ids)

        for cid in V_ids:
            self.scheduler.update_trust(cid, behavior_score=behavior_scores.get(cid, 0.0), was_verified=True)
        for cid in P_ids:
            self.scheduler.update_trust(cid, behavior_score=0.0, was_verified=False)

        verified_updates = {cid: all_updates[cid] for cid in inliers_for_agg if cid in all_updates}
        promoted_updates = {cid: all_updates[cid] for cid in P_ids if cid in all_updates}
        
        promoted_weights = {
            cid: self.scheduler.bayesian_posterior_weight(
                self.scheduler.get_effective_trust(cid, server_round)
            ) for cid in P_ids
        }
        
        verified_weights = {
            cid: max(0.05, float(behavior_scores.get(cid, 0.0)))
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

        analytics = LegacyAnalyticsBridge(server_round, outliers, self.scheduler.trust_scores, P_ids, execution_time_ms)
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
            "clip_radius": clip_stats.get("clip_radius"),
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
        return {"trust_state": self.scheduler.trust_scores}

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

        # Every sampled client is verified. aggregate_fit splits verified from
        # promoted on the is_verified flag the client echoes back, so setting it
        # True here routes all of them down the verified path.
        config_dict = {"server_round": server_round, "is_verified": True}
        return [(proxy, FitIns(parameters, config_dict.copy())) for proxy in sampled]
