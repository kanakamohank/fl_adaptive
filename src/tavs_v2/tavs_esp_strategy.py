import logging
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

    def initialize_parameters(self, client_manager):
        from src.core.models import get_model
        
        # Dynamically instantiate the correct model type (safe for both Phase 4 and Phase 5)
        model_type = getattr(self.config, "model_type", "cifar_cnn")
        model = get_model(model_type, num_classes=10)
        
        # ---> THE FIX: Force copy=True and float32 for clean Ray serialization
        return ndarrays_to_parameters(
            [np.array(p.detach().cpu().numpy(), dtype=np.float32, copy=True) for p in model.parameters()]
        ) 

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
        return {proxy.cid: proxy for proxy in sampled}

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager):
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
        logger.info(f"Round {server_round} Scheduling: {len(V)} Verified, {len(P)} Promoted, {len(D)} Dropped")
        
        fit_configurations = []
        config_dict = {"server_round": server_round}
        
        for cid, client_proxy in available_clients.items():
            if cid in V:
                config_dict["is_verified"] = True
                fit_configurations.append((client_proxy, FitIns(parameters, config_dict.copy())))
            elif cid in P:
                config_dict["is_verified"] = False
                fit_configurations.append((client_proxy, FitIns(parameters, config_dict.copy())))
                
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
        
        V_ids = {proxy.cid for proxy, res in results if res.metrics.get("is_verified", True)}
        P_ids = {proxy.cid for proxy, res in results if not res.metrics.get("is_verified", True)}
        
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
        
        try:
            verified_weights = {cid: max(0.05, float(behavior_scores.get(cid, 0.0))) for cid in verified_updates.keys()}
            aggregated_blocks = UnifiedBayesianAggregator.aggregate(
                verified_updates, verified_weights, 
                promoted_updates, promoted_weights
            )
        except TypeError:
            aggregated_blocks = UnifiedBayesianAggregator.aggregate(
                verified_updates, promoted_updates, promoted_weights
            )

        execution_time_ms = (time.time() - start_time) * 1000

        if not aggregated_blocks:
            return None, {}

        analytics = LegacyAnalyticsBridge(server_round, outliers, self.scheduler.trust_scores, P_ids, execution_time_ms)
        self.round_analytics.append(analytics)

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
        if hasattr(self.config, 'evaluate_fn') and self.config.evaluate_fn is not None:
            return self.config.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
        return None

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
        sampled = list(self._sample_cohort(client_manager).values())
        if not sampled:
            return []

        # Every sampled client is verified. aggregate_fit splits verified from
        # promoted on the is_verified flag the client echoes back, so setting it
        # True here routes all of them down the verified path.
        config_dict = {"server_round": server_round, "is_verified": True}
        return [(proxy, FitIns(parameters, config_dict.copy())) for proxy in sampled]
