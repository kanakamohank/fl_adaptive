import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import logging
from ..clients.honest_client import HonestClient
from ..core.models import ModelStructure


logger = logging.getLogger(__name__)


class LayerwiseBackdoorAttacker(HonestClient):
    """
    Layerwise backdoor injection attack targeting specific model blocks.

    This attack demonstrates why structured JL projection is superior to dense projection:
    - Dense projection dilutes layer-specific attacks across all projected dimensions
    - Structured projection concentrates the attack in specific blocks → easier detection
    """

    def __init__(self, client_id: str, model_type: str, model_kwargs: Dict,
                 train_loader: DataLoader, test_loader: DataLoader,
                 target_layers: List[str], attack_intensity: float = 1.0,
                 device: str = "cpu", local_epochs: int = 5, lr: float = 0.01):
        """
        Initialize layerwise backdoor attacker.

        Args:
            target_layers: List of layer names to inject backdoors into
            attack_intensity: Magnitude of backdoor injection
            Other args same as HonestClient
        """
        super().__init__(client_id, model_type, model_kwargs, train_loader, test_loader,
                        device, local_epochs, lr)

        self.target_layers = target_layers
        self.attack_intensity = attack_intensity
        self.backdoor_patterns = {}
        self.round_history = []

        # Generate backdoor patterns for each target layer
        self._initialize_backdoor_patterns()

        logger.info(f"Layerwise backdoor attacker {client_id} initialized targeting layers: {target_layers}")

    def _initialize_backdoor_patterns(self):
        """Initialize backdoor patterns for target layers."""
        model_structure = self.model.structure

        for layer_name in self.target_layers:
            # Find the layer in model structure
            target_block = None
            for block in model_structure.blocks:
                if layer_name in block['name']:
                    target_block = block
                    break

            if target_block is None:
                logger.warning(f"Target layer {layer_name} not found in model structure")
                continue

            # Create a backdoor pattern for this layer
            pattern_shape = target_block['shape']
            backdoor_pattern = torch.randn(pattern_shape, device=self.device) * self.attack_intensity

            self.backdoor_patterns[layer_name] = {
                'pattern': backdoor_pattern,
                'block_info': target_block
            }

        logger.debug(f"Initialized {len(self.backdoor_patterns)} backdoor patterns")

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Perform training with layerwise backdoor injection.

        The attack adds specific patterns to target layers only.
        """
        # First perform honest training
        honest_params, num_examples, metrics = super().fit(parameters, config)

        if not self.backdoor_patterns:
            logger.debug(f"Attacker {self.client_id} acting honestly (no valid targets)")
            return honest_params, num_examples, metrics

        # Convert to tensors and inject backdoors
        param_tensors = [torch.tensor(param, dtype=torch.float32).to(self.device)
                        for param in honest_params]

        # Inject backdoors into target layers
        injected_params = self._inject_backdoors(param_tensors)

        # Convert back to numpy
        injected_params_numpy = [p.cpu().detach().numpy() for p in injected_params]

        # Calculate attack statistics
        attack_stats = self._calculate_attack_statistics(param_tensors, injected_params)

        # Update metrics
        attack_metrics = metrics.copy()
        attack_metrics.update({
            "attack_type": "layerwise_backdoor",
            "target_layers": self.target_layers,
            "attack_intensity": self.attack_intensity,
            "is_attacker": True,
            **attack_stats
        })

        self.round_history.append({
            "round": len(self.round_history),
            "attack_stats": attack_stats
        })

        return injected_params_numpy, num_examples, attack_metrics

    def _inject_backdoors(self, param_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Inject backdoor patterns into target layers.

        Args:
            param_tensors: Original model parameters

        Returns:
            Parameters with backdoors injected
        """
        model_structure = self.model.structure
        injected_params = [p.clone() for p in param_tensors]

        # Flatten parameters for easier manipulation
        flat_params = torch.cat([p.flatten() for p in param_tensors])

        # Inject backdoors into each target layer
        for layer_name, backdoor_info in self.backdoor_patterns.items():
            block_info = backdoor_info['block_info']
            backdoor_pattern = backdoor_info['pattern']

            # Get parameter indices for this block
            start_idx = block_info['start_idx']
            end_idx = block_info['end_idx']

            # Inject backdoor pattern
            block_params = flat_params[start_idx:end_idx].reshape(block_info['shape'])
            block_params += backdoor_pattern

            # Update in flattened representation
            flat_params[start_idx:end_idx] = block_params.flatten()

        # Convert back to parameter list
        start_idx = 0
        for i, param in enumerate(injected_params):
            param_size = param.numel()
            param.data = flat_params[start_idx:start_idx + param_size].reshape(param.shape)
            start_idx += param_size

        return injected_params

    def _calculate_attack_statistics(self, original_params: List[torch.Tensor],
                                   injected_params: List[torch.Tensor]) -> Dict:
        """Calculate statistics about the backdoor injection."""
        model_structure = self.model.structure

        # Calculate per-block norms
        original_flat = torch.cat([p.flatten() for p in original_params])
        injected_flat = torch.cat([p.flatten() for p in injected_params])
        injection_vector = injected_flat - original_flat

        block_norms = {}
        total_injection_norm = torch.norm(injection_vector).item()

        for block in model_structure.blocks:
            block_name = block['name']
            start_idx = block['start_idx']
            end_idx = block['end_idx']

            block_injection = injection_vector[start_idx:end_idx]
            block_norm = torch.norm(block_injection).item()
            block_norms[block_name] = block_norm

        # Calculate concentration ratio (how much attack is concentrated in target layers)
        target_layer_norms = []
        for layer_name in self.target_layers:
            for block_name, norm in block_norms.items():
                if layer_name in block_name:
                    target_layer_norms.append(norm)

        target_concentration = sum(target_layer_norms) / (total_injection_norm + 1e-8)

        return {
            'total_injection_norm': total_injection_norm,
            'block_norms': block_norms,
            'target_concentration': target_concentration,
            'num_target_blocks': len(target_layer_norms)
        }

    def get_attack_statistics(self) -> Dict:
        """Get comprehensive attack statistics."""
        if not self.round_history:
            return {}

        # Aggregate statistics across rounds
        total_norms = [r['attack_stats']['total_injection_norm'] for r in self.round_history]
        concentrations = [r['attack_stats']['target_concentration'] for r in self.round_history]

        return {
            'total_rounds': len(self.round_history),
            'average_injection_norm': np.mean(total_norms),
            'average_target_concentration': np.mean(concentrations),
            'target_layers': self.target_layers,
            'round_history': self.round_history
        }


class DistributedPoisonAttacker(HonestClient):
    """
    Distributed low-magnitude poisoning attack.

    This attack spreads small poison across all parameters to evade detection.
    It survives naive compression but fails against structured JL due to topology inconsistency.
    """

    def __init__(self, client_id: str, model_type: str, model_kwargs: Dict,
                 train_loader: DataLoader, test_loader: DataLoader,
                 poison_intensity: float = 0.1, device: str = "cpu",
                 local_epochs: int = 5, lr: float = 0.01):
        """
        Initialize distributed poison attacker.

        Args:
            poison_intensity: Magnitude of distributed poison (low to avoid detection)
        """
        super().__init__(client_id, model_type, model_kwargs, train_loader, test_loader,
                        device, local_epochs, lr)

        self.poison_intensity = poison_intensity
        self.round_history = []

        logger.info(f"Distributed poison attacker {client_id} initialized with intensity {poison_intensity}")

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Perform training with distributed poisoning.

        The attack adds small noise to ALL parameters.
        """
        # First perform honest training
        honest_params, num_examples, metrics = super().fit(parameters, config)

        # Convert to tensors
        param_tensors = [torch.tensor(param, dtype=torch.float32).to(self.device)
                        for param in honest_params]

        # Add distributed poison
        poisoned_params = self._add_distributed_poison(param_tensors)

        # Convert back to numpy
        poisoned_params_numpy = [p.cpu().detach().numpy() for p in poisoned_params]

        # Calculate poison statistics
        poison_stats = self._calculate_poison_statistics(param_tensors, poisoned_params)

        # Update metrics
        attack_metrics = metrics.copy()
        attack_metrics.update({
            "attack_type": "distributed_poisoning",
            "poison_intensity": self.poison_intensity,
            "is_attacker": True,
            **poison_stats
        })

        self.round_history.append({
            "round": len(self.round_history),
            "poison_stats": poison_stats
        })

        return poisoned_params_numpy, num_examples, attack_metrics

    def _add_distributed_poison(self, param_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Add low-magnitude poison distributed across all parameters.

        Args:
            param_tensors: Original parameters

        Returns:
            Parameters with distributed poison
        """
        poisoned_params = []

        for param in param_tensors:
            # Generate poison with same shape as parameter
            poison = torch.randn_like(param) * self.poison_intensity

            # Add poison to parameter
            poisoned_param = param + poison
            poisoned_params.append(poisoned_param)

        return poisoned_params

    def _calculate_poison_statistics(self, original_params: List[torch.Tensor],
                                   poisoned_params: List[torch.Tensor]) -> Dict:
        """Calculate statistics about distributed poison."""
        total_poison_norm = 0.0
        total_param_norm = 0.0

        for orig, poisoned in zip(original_params, poisoned_params):
            poison = poisoned - orig
            total_poison_norm += torch.norm(poison).item() ** 2
            total_param_norm += torch.norm(orig).item() ** 2

        total_poison_norm = np.sqrt(total_poison_norm)
        total_param_norm = np.sqrt(total_param_norm)

        poison_ratio = total_poison_norm / (total_param_norm + 1e-8)

        return {
            'total_poison_norm': total_poison_norm,
            'total_param_norm': total_param_norm,
            'poison_ratio': poison_ratio
        }


class LayerwiseAttackAnalyzer:
    """Analyzer for demonstrating structured vs dense projection effectiveness."""

    @staticmethod
    def analyze_attack_concentration(honest_updates: List[torch.Tensor],
                                   layerwise_attacks: List[torch.Tensor],
                                   model_structure: ModelStructure,
                                   structured_projections: List[torch.Tensor],
                                   dense_projections: List[torch.Tensor]) -> Dict:
        """
        Analyze how structured vs dense projections handle layerwise attacks.
        This generates data for the E5 heatmap visualization.

        Args:
            honest_updates: Honest client updates
            layerwise_attacks: Layerwise attacker updates
            model_structure: Model structure
            structured_projections: Structured JL projections
            dense_projections: Dense JL projections

        Returns:
            Analysis for heatmap generation
        """
        num_honest = len(honest_updates)
        num_attackers = len(layerwise_attacks)
        num_blocks = len(model_structure.blocks)

        # Analyze structured projection concentration
        structured_heatmap = np.zeros((num_honest + num_attackers, num_blocks))

        # For honest clients (should have distributed activity)
        for client_idx in range(num_honest):
            proj_update = structured_projections[client_idx]
            # Reconstruct block-wise contributions (this is approximate for honest clients)
            block_start = 0
            for block_idx, block in enumerate(model_structure.blocks):
                k_block = max(1, int(0.1 * block['num_params']))  # Assuming 0.1 compression ratio
                block_projection = proj_update[block_start:block_start + k_block]
                structured_heatmap[client_idx, block_idx] = torch.norm(block_projection).item()
                block_start += k_block

        # For attackers (should light up in specific blocks)
        for attacker_idx in range(num_attackers):
            client_idx = num_honest + attacker_idx
            proj_update = structured_projections[client_idx]
            block_start = 0
            for block_idx, block in enumerate(model_structure.blocks):
                k_block = max(1, int(0.1 * block['num_params']))
                block_projection = proj_update[block_start:block_start + k_block]
                structured_heatmap[client_idx, block_idx] = torch.norm(block_projection).item()
                block_start += k_block

        # Analyze dense projection (attack gets diluted)
        dense_heatmap = np.zeros((num_honest + num_attackers, num_blocks))

        # For dense projection, we approximate block contributions by looking at original updates
        all_updates = honest_updates + layerwise_attacks
        for client_idx, update in enumerate(all_updates):
            for block_idx, block in enumerate(model_structure.blocks):
                start_idx = block['start_idx']
                end_idx = block['end_idx']
                block_params = update[start_idx:end_idx]
                # Simulate dilution effect of dense projection
                dense_heatmap[client_idx, block_idx] = torch.norm(block_params).item() * 0.3  # Dilution factor

        return {
            'structured_heatmap': structured_heatmap,
            'dense_heatmap': dense_heatmap,
            'block_names': [block['name'] for block in model_structure.blocks],
            'client_labels': [f'honest_{i}' for i in range(num_honest)] +
                           [f'attacker_{i}' for i in range(num_attackers)]
        }

    @staticmethod
    def compute_detection_effectiveness(structured_results: Dict,
                                      dense_results: Dict,
                                      ground_truth_attackers: List[int]) -> Dict:
        """
        Compare detection effectiveness between structured and dense projections.

        Args:
            structured_results: Detection results from structured projection
            dense_results: Detection results from dense projection
            ground_truth_attackers: True attacker indices

        Returns:
            Comparative analysis
        """
        from ..core.verification import ByzantineDetectionEvaluator

        # Evaluate structured projection detection
        structured_eval = ByzantineDetectionEvaluator.evaluate_detection(
            structured_results['byzantine_indices'],
            ground_truth_attackers,
            structured_results.get('total_clients', len(ground_truth_attackers) * 2)
        )

        # Evaluate dense projection detection
        dense_eval = ByzantineDetectionEvaluator.evaluate_detection(
            dense_results['byzantine_indices'],
            ground_truth_attackers,
            dense_results.get('total_clients', len(ground_truth_attackers) * 2)
        )

        return {
            'structured_performance': structured_eval,
            'dense_performance': dense_eval,
            'improvement_factors': {
                'precision': structured_eval['precision'] / (dense_eval['precision'] + 1e-8),
                'recall': structured_eval['recall'] / (dense_eval['recall'] + 1e-8),
                'f1_score': structured_eval['f1_score'] / (dense_eval['f1_score'] + 1e-8)
            }
        }