import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import logging
from ..clients.honest_client import HonestClient
from ..core.projection import DenseJLProjection


logger = logging.getLogger(__name__)


class NullSpaceAttacker(HonestClient):
    """
    Null-space poisoning attack that exploits static projection matrices.

    This is the "Royal Flush" attack that demonstrates why ephemeral projections are necessary.

    Attack mechanism:
    1. If the projection matrix R is static/known, attacker can compute its null space
    2. Attacker adds a vector z to their update where Rz ≈ 0
    3. The poisoned update appears normal in projected space but is malicious in full space
    4. This breaks KETS-style defenses that use static compression

    Defense: Ephemeral R_t changes every round → null space changes → attack impossible
    """

    def __init__(self, client_id: str, model_type: str, model_kwargs: Dict,
                 train_loader: DataLoader, test_loader: DataLoader,
                 attack_intensity: float = 1.0, device: str = "cpu",
                 local_epochs: int = 5, lr: float = 0.01):
        """
        Initialize null-space attacker.

        Args:
            attack_intensity: Magnitude of null-space poisoning
            Other args same as HonestClient
        """
        super().__init__(client_id, model_type, model_kwargs, train_loader, test_loader,
                        device, local_epochs, lr)

        self.attack_intensity = attack_intensity
        self.static_projection_matrix = None
        self.null_space_vectors = None
        self.round_history = []

        logger.info(f"Null-space attacker {client_id} initialized with intensity {attack_intensity}")

    def learn_static_projection(self, projection_matrix: torch.Tensor):
        """
        Learn the static projection matrix (simulates attack on KETS-style defenses).

        In reality, an attacker might:
        1. Observe projected updates over multiple rounds
        2. Use matrix factorization to recover projection matrix
        3. Or exploit known compression methods like PCA

        Args:
            projection_matrix: The static projection matrix R
        """
        self.static_projection_matrix = projection_matrix.clone()

        # Compute null space of the projection matrix
        self.null_space_vectors = self._compute_null_space(projection_matrix)

        logger.info(f"Attacker {self.client_id} learned static projection matrix "
                   f"and computed {len(self.null_space_vectors)} null space vectors")

    def _compute_null_space(self, matrix: torch.Tensor, tolerance: float = 1e-6) -> List[torch.Tensor]:
        """
        Compute null space of projection matrix using SVD.

        Args:
            matrix: Projection matrix R (k x d)
            tolerance: Numerical tolerance for null space

        Returns:
            List of null space basis vectors
        """
        # Use CPU for SVD (more stable)
        matrix_cpu = matrix.cpu()

        # SVD: R = U Σ V^T
        U, S, Vt = torch.linalg.svd(matrix_cpu, full_matrices=True)

        # Find indices where singular values are effectively zero
        null_indices = torch.where(S < tolerance)[0]

        if len(null_indices) == 0:
            logger.warning("No null space found - projection matrix is full rank")
            return []

        # Null space vectors are columns of V corresponding to zero singular values
        null_vectors = Vt[len(S):, :].T  # Shape: (d, null_dim)

        # Convert back to device and split into list
        null_space_vectors = []
        for i in range(null_vectors.shape[1]):
            null_vec = null_vectors[:, i].to(self.device)
            null_space_vectors.append(null_vec)

        logger.debug(f"Computed null space: {len(null_space_vectors)} vectors from matrix "
                    f"with {len(null_indices)} zero singular values")

        return null_space_vectors

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Perform training with null-space poisoning attack.

        This method adds null-space poison to the honest update.
        """
        # First perform honest training
        honest_params, num_examples, metrics = super().fit(parameters, config)

        # If we haven't learned the projection matrix, act honestly
        if self.static_projection_matrix is None or not self.null_space_vectors:
            logger.debug(f"Attacker {self.client_id} acting honestly (no projection learned)")
            return honest_params, num_examples, metrics

        # Convert parameters to tensors for manipulation
        param_tensors = [torch.tensor(param, dtype=torch.float32).to(self.device)
                        for param in honest_params]

        # Flatten parameters
        honest_flat = torch.cat([p.flatten() for p in param_tensors])

        # Generate null-space poison
        poison_vector = self._generate_null_space_poison()

        # Add poison to honest update
        poisoned_flat = honest_flat + poison_vector

        # Verify poison is in null space (for logging)
        if self.static_projection_matrix is not None:
            projected_poison = self.static_projection_matrix @ poison_vector
            poison_norm_projected = torch.norm(projected_poison).item()
            poison_norm_original = torch.norm(poison_vector).item()

            logger.debug(f"Attacker {self.client_id}: poison norm in original space = {poison_norm_original:.4f}, "
                        f"in projected space = {poison_norm_projected:.6f}")

        # Convert back to parameter format
        poisoned_params = self._unflatten_parameters(poisoned_flat, param_tensors)
        poisoned_params_numpy = [p.cpu().detach().numpy() for p in poisoned_params]

        # Update metrics to include attack info
        attack_metrics = metrics.copy()
        attack_metrics.update({
            "attack_type": "null_space_poisoning",
            "attack_intensity": self.attack_intensity,
            "poison_norm": torch.norm(poison_vector).item(),
            "is_attacker": True
        })

        self.round_history.append({
            "round": len(self.round_history),
            "honest_norm": torch.norm(honest_flat).item(),
            "poison_norm": torch.norm(poison_vector).item(),
            "poisoned_norm": torch.norm(poisoned_flat).item()
        })

        return poisoned_params_numpy, num_examples, attack_metrics

    def _generate_null_space_poison(self) -> torch.Tensor:
        """
        Generate poison vector in null space of static projection matrix.

        Returns:
            Poison vector that is invisible in projected space
        """
        if not self.null_space_vectors:
            return torch.zeros(self.model.get_weights_flat().shape, device=self.device)

        # Randomly combine null space vectors
        poison = torch.zeros_like(self.model.get_weights_flat())

        for null_vec in self.null_space_vectors:
            # Random coefficient for this null space vector
            coeff = torch.randn(1, device=self.device) * self.attack_intensity
            poison += coeff * null_vec

        return poison

    def _unflatten_parameters(self, flat_params: torch.Tensor,
                            template_params: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Unflatten parameter vector back to original parameter shapes.

        Args:
            flat_params: Flattened parameter vector
            template_params: Template parameters for shapes

        Returns:
            List of parameters in original shapes
        """
        unflattened = []
        start_idx = 0

        for template_param in template_params:
            param_size = template_param.numel()
            param_data = flat_params[start_idx:start_idx + param_size]
            param_reshaped = param_data.reshape(template_param.shape)
            unflattened.append(param_reshaped)
            start_idx += param_size

        return unflattened

    def get_attack_statistics(self) -> Dict:
        """Get statistics about the attack over rounds."""
        if not self.round_history:
            return {}

        poison_norms = [round_data["poison_norm"] for round_data in self.round_history]
        honest_norms = [round_data["honest_norm"] for round_data in self.round_history]

        return {
            "total_rounds": len(self.round_history),
            "average_poison_norm": np.mean(poison_norms),
            "max_poison_norm": np.max(poison_norms),
            "poison_to_honest_ratio": np.mean([p/h for p, h in zip(poison_norms, honest_norms)]),
            "round_history": self.round_history
        }


class NullSpaceAttackAnalyzer:
    """Analyzer for null-space attacks to demonstrate ephemeral defense effectiveness."""

    @staticmethod
    def demonstrate_attack_effectiveness(honest_clients: List[HonestClient],
                                       null_space_attackers: List[NullSpaceAttacker],
                                       static_projection: DenseJLProjection,
                                       num_rounds: int = 10) -> Dict:
        """
        Demonstrate null-space attack effectiveness against static projections.

        This generates the "Royal Flush" evidence showing:
        1. Static defense: Attack invisible in projected space
        2. Ephemeral defense: Attack detected due to changing null space

        Args:
            honest_clients: List of honest clients
            null_space_attackers: List of null-space attackers
            static_projection: Static projection matrix (vulnerable)
            num_rounds: Number of rounds to simulate

        Returns:
            Analysis results for visualization
        """
        results = {
            "static_defense_results": [],
            "ephemeral_defense_results": [],
            "attack_visibility": []
        }

        # Simulate attack against static defense
        logger.info("Simulating null-space attack against static defense...")

        for round_num in range(num_rounds):
            # Generate static projection matrix (same every round)
            static_R = static_projection.generate_projection_matrix(round_number=0)  # Fixed seed

            # Teach attackers the static projection
            for attacker in null_space_attackers:
                if attacker.static_projection_matrix is None:
                    attacker.learn_static_projection(static_R)

            # Collect updates (simulate training)
            honest_updates = []
            attack_updates = []

            # Honest client updates (simulated)
            for client in honest_clients:
                update = torch.randn(static_projection.original_dim, device=static_R.device)
                honest_updates.append(update)

            # Attacker updates with null-space poison
            for attacker in null_space_attackers:
                honest_update = torch.randn(static_projection.original_dim, device=static_R.device)
                poison = attacker._generate_null_space_poison()
                poisoned_update = honest_update + poison
                attack_updates.append(poisoned_update)

            # Project all updates
            all_updates = honest_updates + attack_updates
            projected_updates = [static_R @ update for update in all_updates]

            # Measure attack visibility
            honest_projected_norms = [torch.norm(proj).item() for proj in projected_updates[:len(honest_updates)]]
            attack_projected_norms = [torch.norm(proj).item() for proj in projected_updates[len(honest_updates):]]

            # Attack is invisible if projected norms are similar to honest clients
            visibility_score = np.mean(attack_projected_norms) / (np.mean(honest_projected_norms) + 1e-8)

            results["static_defense_results"].append({
                "round": round_num,
                "honest_projected_norms": honest_projected_norms,
                "attack_projected_norms": attack_projected_norms,
                "visibility_score": visibility_score
            })

        # Simulate defense with ephemeral projections
        logger.info("Simulating null-space attack against ephemeral defense...")

        for round_num in range(num_rounds):
            # Generate ephemeral projection matrix (different every round)
            ephemeral_R = static_projection.generate_projection_matrix(round_number=round_num)

            # Attackers cannot adapt to new null space in time
            # They use poison based on old null space (now invalid)

            # Collect updates
            honest_updates = []
            attack_updates = []

            for client in honest_clients:
                update = torch.randn(static_projection.original_dim, device=ephemeral_R.device)
                honest_updates.append(update)

            for attacker in null_space_attackers:
                honest_update = torch.randn(static_projection.original_dim, device=ephemeral_R.device)
                # Attacker uses poison based on OLD null space (ineffective)
                old_poison = attacker._generate_null_space_poison()
                poisoned_update = honest_update + old_poison
                attack_updates.append(poisoned_update)

            # Project with NEW projection matrix
            all_updates = honest_updates + attack_updates
            projected_updates = [ephemeral_R @ update for update in all_updates]

            # Measure attack visibility (should be high now)
            honest_projected_norms = [torch.norm(proj).item() for proj in projected_updates[:len(honest_updates)]]
            attack_projected_norms = [torch.norm(proj).item() for proj in projected_updates[len(honest_updates):]]

            visibility_score = np.mean(attack_projected_norms) / (np.mean(honest_projected_norms) + 1e-8)

            results["ephemeral_defense_results"].append({
                "round": round_num,
                "honest_projected_norms": honest_projected_norms,
                "attack_projected_norms": attack_projected_norms,
                "visibility_score": visibility_score
            })

        # Compute overall statistics
        static_visibility_scores = [r["visibility_score"] for r in results["static_defense_results"]]
        ephemeral_visibility_scores = [r["visibility_score"] for r in results["ephemeral_defense_results"]]

        results["summary"] = {
            "static_defense_avg_visibility": np.mean(static_visibility_scores),
            "ephemeral_defense_avg_visibility": np.mean(ephemeral_visibility_scores),
            "defense_improvement_factor": np.mean(ephemeral_visibility_scores) / (np.mean(static_visibility_scores) + 1e-8)
        }

        logger.info(f"Attack visibility - Static: {np.mean(static_visibility_scores):.3f}, "
                   f"Ephemeral: {np.mean(ephemeral_visibility_scores):.3f}")

        return results