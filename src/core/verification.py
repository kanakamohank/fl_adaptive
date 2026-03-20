import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import time
import logging


logger = logging.getLogger(__name__)


class GeometricMedian:
    """
    Geometric median computation for robust aggregation.
    This is the core of isomorphic verification.
    """

    @staticmethod
    def compute(vectors: List[torch.Tensor], max_iterations: int = 1000,
               tolerance: float = 1e-6, device: str = "cpu") -> torch.Tensor:
        """
        Compute geometric median using Weiszfeld's algorithm.

        Args:
            vectors: List of vectors to find median of
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
            device: Device for computation

        Returns:
            Geometric median vector
        """
        if not vectors:
            raise ValueError("Cannot compute geometric median of empty list")

        if len(vectors) == 1:
            return vectors[0].clone()

        # Convert to tensor stack
        X = torch.stack(vectors).to(device)  # Shape: (n_vectors, dim)
        n, dim = X.shape

        # Initialize with arithmetic mean
        median = X.mean(dim=0)

        for iteration in range(max_iterations):
            old_median = median.clone()

            # Compute distances from current median
            distances = torch.norm(X - median.unsqueeze(0), dim=1, p=2)

            # Avoid division by zero
            eps = 1e-8
            weights = 1.0 / (distances + eps)

            # Update median using weighted average
            median = torch.sum(weights.unsqueeze(1) * X, dim=0) / torch.sum(weights)

            # Check convergence
            change = torch.norm(median - old_median)
            if change < tolerance:
                logger.debug(f"Geometric median converged in {iteration + 1} iterations")
                break

        return median

    @staticmethod
    def compute_distances_to_median(vectors: List[torch.Tensor],
                                  median: torch.Tensor) -> List[float]:
        """Compute distances from each vector to the geometric median."""
        distances = []
        for vector in vectors:
            dist = torch.norm(vector - median, p=2).item()
            distances.append(dist)
        return distances


# class IsomorphicVerification:
#     """
#     Isomorphic verification for Byzantine detection using geometric topology.
#
#     Key insight: We detect Byzantine clients based on topological inconsistency
#     in the projected space, not just distance-based outliers.
#     """
#
#     def __init__(self, detection_threshold: float = 1.5, min_consensus: float = 0.6):
#         """
#         Initialize isomorphic verification.
#
#         Args:
#             detection_threshold: Multiplier for median distance threshold
#             min_consensus: Minimum fraction of clients needed for consensus
#         """
#         self.detection_threshold = detection_threshold
#         self.min_consensus = min_consensus
#
#     def detect_byzantine_clients(self, projected_updates: List[torch.Tensor],
#                                client_ids: Optional[List[str]] = None) -> Dict:
#         """
#         Detect Byzantine clients using isomorphic verification.
#
#         Args:
#             projected_updates: List of projected parameter updates
#             client_ids: Optional client identifiers
#
#         Returns:
#             Detection results with trust scores and Byzantine indices
#         """
#         start_time = time.time()
#
#         if client_ids is None:
#             client_ids = [f"client_{i}" for i in range(len(projected_updates))]
#
#         n_clients = len(projected_updates)
#         if n_clients < 2:
#             logger.warning("Need at least 2 clients for Byzantine detection")
#             return {
#                 'byzantine_indices': [],
#                 'trust_scores': [1.0] * n_clients,
#                 'geometric_median': projected_updates[0] if projected_updates else None,
#                 'detection_time': time.time() - start_time
#             }
#
#         # Step 1: Compute geometric median
#         geometric_median = GeometricMedian.compute(projected_updates)
#
#         # Step 2: Compute distances to geometric median
#         distances = GeometricMedian.compute_distances_to_median(projected_updates, geometric_median)
#
#         # Step 3: Determine outlier threshold
#         median_distance = np.median(distances)
#         threshold = median_distance * self.detection_threshold
#
#         # Step 4: Identify Byzantine clients
#         byzantine_indices = []
#         trust_scores = []
#
#         for i, distance in enumerate(distances):
#             if distance > threshold:
#                 byzantine_indices.append(i)
#                 # Trust score inversely related to distance (normalized)
#                 trust_score = max(0.0, 1.0 - (distance - median_distance) / median_distance)
#             else:
#                 trust_score = 1.0 - (distance / (median_distance + 1e-8))
#
#             trust_scores.append(trust_score)
#
#         # Step 5: Consensus check
#         honest_fraction = (n_clients - len(byzantine_indices)) / n_clients
#         consensus_achieved = honest_fraction >= self.min_consensus
#
#         detection_time = time.time() - start_time
#
#         results = {
#             'byzantine_indices': byzantine_indices,
#             'byzantine_client_ids': [client_ids[i] for i in byzantine_indices],
#             'trust_scores': trust_scores,
#             'geometric_median': geometric_median,
#             'distances_to_median': distances,
#             'detection_threshold': threshold,
#             'median_distance': median_distance,
#             'consensus_achieved': consensus_achieved,
#             'honest_fraction': honest_fraction,
#             'detection_time': detection_time
#         }
#
#         logger.info(f"Byzantine detection completed: {len(byzantine_indices)}/{n_clients} "
#                    f"Byzantine clients detected in {detection_time:.4f}s")
#
#         return results
#
#     def compute_topology_consistency(self, projected_updates: List[torch.Tensor]) -> Dict:
#         """
#         Compute topological consistency metrics.
#         This helps distinguish between honest edge-case clients and Byzantine attackers.
#
#         Args:
#             projected_updates: List of projected parameter updates
#
#         Returns:
#             Topology consistency analysis
#         """
#         if len(projected_updates) < 3:
#             return {'consistency_score': 1.0, 'analysis': 'insufficient_data'}
#
#         # Convert to numpy for distance computations
#         vectors = [update.cpu().numpy() for update in projected_updates]
#         X = np.array(vectors)
#
#         # Compute pairwise distances
#         distances = squareform(pdist(X, metric='euclidean'))
#
#         # Analyze local neighborhoods
#         n_neighbors = min(3, len(projected_updates) - 1)
#         nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
#         nbrs.fit(X)
#
#         # For each point, check if its local neighborhood is consistent
#         consistency_scores = []
#         for i in range(len(X)):
#             neighbors_distances, neighbors_indices = nbrs.kneighbors([X[i]])
#             neighbors_distances = neighbors_distances[0]
#
#             # Consistency based on smoothness of local distances
#             if len(neighbors_distances) > 1:
#                 distance_variance = np.var(neighbors_distances[1:])  # Exclude self (distance 0)
#                 consistency = 1.0 / (1.0 + distance_variance)
#             else:
#                 consistency = 1.0
#
#             consistency_scores.append(consistency)
#
#         return {
#             'consistency_scores': consistency_scores,
#             'mean_consistency': np.mean(consistency_scores),
#             'pairwise_distances': distances,
#             'analysis': 'topology_computed'
#         }

class IsomorphicVerification:
    """
    Isomorphic verification for Byzantine detection using geometric topology.

    Upgraded to support Block-Variance-Normalized Detection (BVD) as defined
    in TAVS-ESP Paper 2.
    """

    def __init__(self,
                 detection_threshold: float = 1.5,
                 min_consensus: float = 0.6,
                 use_bvd: bool = True,
                 bvd_threshold: float = 5.0,
                 variance_ema_alpha: float = 0.9):
        """
        Initialize isomorphic verification.

        Args:
            detection_threshold: Multiplier for median distance threshold
            min_consensus: Minimum fraction of clients needed for consensus
            use_bvd: Whether to enable Block-Variance-Normalized Detection (Paper 2)
            bvd_threshold: Z-score threshold for BVD
            variance_ema_alpha: EMA decay factor for block variance tracking
        """
        self.detection_threshold = detection_threshold
        self.min_consensus = min_consensus

        # New BVD parameters
        self.use_bvd = use_bvd
        self.bvd_threshold = bvd_threshold
        self.variance_ema_alpha = variance_ema_alpha
        self.block_variances: Dict[int, float] = {}

    def detect_byzantine_clients(self, projected_updates: List[torch.Tensor],
                                 client_ids: Optional[List[str]] = None,
                                 block_sizes: Optional[List[int]] = None) -> Dict:
        """
        Detect Byzantine clients using isomorphic verification.

        Args:
            projected_updates: List of projected parameter updates
            client_ids: Optional client identifiers
            block_sizes: Sizes of semantic blocks (triggers BVD if provided)

        Returns:
            Detection results with trust scores and Byzantine indices
        """
        start_time = time.time()

        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(len(projected_updates))]

        n_clients = len(projected_updates)
        if n_clients < 2:
            logger.warning("Need at least 2 clients for Byzantine detection")
            return {
                'byzantine_indices': [],
                'trust_scores': [1.0] * n_clients,
                'geometric_median': projected_updates[0] if projected_updates else None,
                'detection_time': time.time() - start_time
            }

        # Step 1: Compute geometric median
        geometric_median = GeometricMedian.compute(projected_updates)

        byzantine_indices = []
        trust_scores = []

        # Step 2: Choose Detection Path (BVD vs STD)
        if self.use_bvd and block_sizes is not None:
            # --- NEW PATH: Block-Variance-Normalized Detection (Paper 2) ---
            num_blocks = len(block_sizes)

            # Stack and split
            X = torch.stack(projected_updates)
            median_blocks = torch.split(geometric_median, block_sizes)
            client_blocks = torch.split(X, block_sizes, dim=1)

            A_scores = np.zeros(n_clients)

            for m in range(num_blocks):
                # Distance per block
                diff = client_blocks[m] - median_blocks[m].unsqueeze(0)
                d_im = torch.norm(diff, dim=1, p=2).pow(2).cpu().numpy()

                # Current block variance estimate
                current_var_estimate = np.median(d_im) + 1e-6

                if m not in self.block_variances:
                    self.block_variances[m] = current_var_estimate
                else:
                    self.block_variances[m] = (self.variance_ema_alpha * self.block_variances[m] +
                                               (1 - self.variance_ema_alpha) * current_var_estimate)

                # Z-scores
                Z_m = d_im / self.block_variances[m]
                A_scores = np.maximum(A_scores, Z_m)

            distances = A_scores.tolist()
            median_distance = np.median(distances)
            threshold = self.bvd_threshold

            for i, a_score in enumerate(A_scores):
                if a_score > threshold:
                    byzantine_indices.append(i)
                    trust_score = max(0.0, 1.0 - (a_score / threshold))
                else:
                    trust_score = 1.0 - (a_score / (threshold * 2))
                trust_scores.append(trust_score)

        else:
            # --- ORIGINAL PATH: Scalar-Threshold Detection (Preserves Tests) ---
            # Step 2: Compute distances to geometric median
            distances = GeometricMedian.compute_distances_to_median(projected_updates, geometric_median)

            # Step 3: Determine outlier threshold
            median_distance = np.median(distances)
            threshold = median_distance * self.detection_threshold

            # Step 4: Identify Byzantine clients
            for i, distance in enumerate(distances):
                if distance > threshold:
                    byzantine_indices.append(i)
                    # Trust score inversely related to distance (normalized)
                    trust_score = max(0.0, 1.0 - (distance - median_distance) / median_distance)
                else:
                    trust_score = 1.0 - (distance / (median_distance + 1e-8))

                trust_scores.append(trust_score)

        # Step 5: Consensus check
        honest_fraction = (n_clients - len(byzantine_indices)) / n_clients
        consensus_achieved = honest_fraction >= self.min_consensus

        detection_time = time.time() - start_time

        results = {
            'byzantine_indices': byzantine_indices,
            'byzantine_client_ids': [client_ids[i] for i in byzantine_indices],
            'trust_scores': trust_scores,
            'geometric_median': geometric_median,
            'distances_to_median': distances,
            'detection_threshold': threshold,
            'median_distance': median_distance,
            'consensus_achieved': consensus_achieved,
            'honest_fraction': honest_fraction,
            'detection_time': detection_time,
            'method': 'BVD' if (self.use_bvd and block_sizes is not None) else 'STD'
        }

        logger.info(f"Byzantine detection ({results['method']}) completed: {len(byzantine_indices)}/{n_clients} "
                    f"Byzantine clients detected in {detection_time:.4f}s")

        return results

class AdaptiveThreshold:
    """Adaptive threshold for Byzantine detection that evolves over rounds."""

    def __init__(self, initial_threshold: float = 1.5, learning_rate: float = 0.1):
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.history = []

    def update_threshold(self, detection_results: Dict, true_byzantines: Optional[List[int]] = None):
        """
        Update detection threshold based on previous round results.

        Args:
            detection_results: Results from Byzantine detection
            true_byzantines: Ground truth Byzantine indices (for simulation)
        """
        if true_byzantines is not None:
            # Supervised learning: adjust based on detection accuracy
            detected = set(detection_results['byzantine_indices'])
            actual = set(true_byzantines)

            false_positives = len(detected - actual)
            false_negatives = len(actual - detected)

            # Increase threshold if too many false positives
            # Decrease threshold if too many false negatives
            adjustment = (false_positives - false_negatives) * self.learning_rate * 0.1
            self.threshold = max(1.0, self.threshold + adjustment)

        else:
            # Unsupervised: adjust based on consistency metrics
            if 'honest_fraction' in detection_results:
                honest_fraction = detection_results['honest_fraction']
                if honest_fraction < 0.5:  # Too many detected as Byzantine
                    self.threshold *= 1.05  # Be more lenient
                elif honest_fraction > 0.9:  # Too few detected as Byzantine
                    self.threshold *= 0.95  # Be more strict

        self.history.append(self.threshold)


class ByzantineDetectionEvaluator:
    """Evaluator for Byzantine detection performance."""

    @staticmethod
    def evaluate_detection(detected_indices: List[int], true_byzantine_indices: List[int],
                         total_clients: int) -> Dict:
        """
        Evaluate Byzantine detection performance.

        Args:
            detected_indices: Detected Byzantine client indices
            true_byzantine_indices: Ground truth Byzantine client indices
            total_clients: Total number of clients

        Returns:
            Evaluation metrics
        """
        detected_set = set(detected_indices)
        true_set = set(true_byzantine_indices)

        true_positives = len(detected_set & true_set)
        false_positives = len(detected_set - true_set)
        false_negatives = len(true_set - detected_set)
        true_negatives = total_clients - len(detected_set | true_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        accuracy = (true_positives + true_negatives) / total_clients

        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }

    @staticmethod
    def analyze_honest_preservation(detected_indices: List[int], client_data_distributions: List[Dict],
                                  rare_class_threshold: int = 10) -> Dict:
        """
        Analyze how well the detection preserves honest clients with rare/tail class data.
        This is critical for E1 experiment.

        Args:
            detected_indices: Detected Byzantine client indices
            client_data_distributions: Class distributions for each client
            rare_class_threshold: Threshold for considering a class "rare"

        Returns:
            Analysis of honest client preservation
        """
        honest_indices = [i for i in range(len(client_data_distributions)) if i not in detected_indices]

        # Identify clients with rare class data
        rare_class_clients = []
        for i, distribution in enumerate(client_data_distributions):
            has_rare_class = any(count <= rare_class_threshold for count in distribution.values())
            if has_rare_class:
                rare_class_clients.append(i)

        # Calculate preservation metrics
        rare_class_clients_set = set(rare_class_clients)
        honest_set = set(honest_indices)

        preserved_rare_clients = len(rare_class_clients_set & honest_set)
        total_rare_clients = len(rare_class_clients_set)

        preservation_rate = preserved_rare_clients / total_rare_clients if total_rare_clients > 0 else 1.0

        return {
            'total_rare_class_clients': total_rare_clients,
            'preserved_rare_class_clients': preserved_rare_clients,
            'preservation_rate': preservation_rate,
            'rare_class_client_indices': rare_class_clients,
            'falsely_detected_rare_clients': list(rare_class_clients_set - honest_set)
        }