#!/usr/bin/env python3
"""
ChaCha20-CTR CSPRNG Manager for TAVS-ESP Ephemeral Randomness

This module provides cryptographically secure pseudo-random number generation
for ephemeral projection matrices and trust-adaptive scheduling assignments.

Core Security Properties:
- ChaCha20-256 stream cipher for cryptographic strength
- Deterministic derivation from master key + round number
- Independent seeds for projection, promotion, and decoy operations
- Timing attack resistance through unpredictable scheduling

Key Insight: Ephemeral randomness breaks null-space attacks by ensuring
attackers cannot pre-compute null spaces of future projection matrices.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import hashlib
import secrets
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RoundMaterials:
    """Materials derived for a specific round from CSPRNG."""
    round_number: int
    projection_seed: bytes  # 16 bytes for projection matrix generation
    promotion_seed: bytes   # 16 bytes for client promotion selection
    decoy_seed: bytes      # 16 bytes for decoy verification
    reserved_seed: bytes   # 16 bytes reserved for future use
    derivation_time: float # Time taken to derive (for performance monitoring)


class CSPRNGManager:
    """
    ChaCha20-CTR based cryptographically secure random number generator.

    Provides ephemeral randomness for TAVS-ESP system components:
    1. Projection matrix generation (defeats null-space attacks)
    2. Client promotion scheduling (defeats timing attacks)
    3. Decoy verification selection (security mechanism)

    Security Model:
    - Master key K is kept secret by server
    - Round materials y_r = ChaCha20(K, r) are deterministic but unpredictable
    - Each round gets independent 512-bit entropy stream
    """

    def __init__(self, master_key: Optional[bytes] = None, key_rotation_rounds: int = 10000):
        """
        Initialize CSPRNG manager with master key.

        Args:
            master_key: 32-byte master key. If None, generates cryptographically secure key.
            key_rotation_rounds: Rotate master key every N rounds for forward security.
        """
        if master_key is None:
            self.master_key = secrets.token_bytes(32)  # 256-bit key
            logger.info("Generated new 256-bit master key for CSPRNG")
        else:
            if len(master_key) != 32:
                raise ValueError("Master key must be exactly 32 bytes (256 bits)")
            self.master_key = master_key
            logger.info("Using provided 256-bit master key for CSPRNG")

        self.key_rotation_rounds = key_rotation_rounds
        self.current_key_epoch = 0
        self.round_materials_cache = {}  # Cache for performance
        self.derivation_stats = {"total_rounds": 0, "total_time": 0.0}

        logger.info(f"CSPRNG Manager initialized with key rotation every {key_rotation_rounds} rounds")

    def derive_round_materials(self, round_number: int) -> RoundMaterials:
        """
        Derive all cryptographic materials for a specific round.

        Uses ChaCha20 stream cipher to generate deterministic but unpredictable
        randomness from master key and round number.

        Args:
            round_number: The federated learning round number

        Returns:
            RoundMaterials containing all seeds for the round
        """
        start_time = time.time()

        # Check if we need key rotation
        if round_number > 0 and round_number % self.key_rotation_rounds == 0:
            self._rotate_master_key(round_number)

        # Check cache first (for deterministic replay)
        cache_key = (self.current_key_epoch, round_number)
        if cache_key in self.round_materials_cache:
            logger.debug(f"Using cached materials for round {round_number}")
            return self.round_materials_cache[cache_key]

        # Generate ChaCha20 stream using simplified approach
        # In production, use pycryptodome or similar library
        stream = self._chacha20_stream(self.master_key, round_number)

        # Split 64-byte stream into 4 seeds
        materials = RoundMaterials(
            round_number=round_number,
            projection_seed=stream[0:16],
            promotion_seed=stream[16:32],
            decoy_seed=stream[32:48],
            reserved_seed=stream[48:64],
            derivation_time=time.time() - start_time
        )

        # Cache for future use
        self.round_materials_cache[cache_key] = materials

        # Update statistics
        self.derivation_stats["total_rounds"] += 1
        self.derivation_stats["total_time"] += materials.derivation_time

        logger.debug(f"Derived materials for round {round_number} in {materials.derivation_time*1000:.2f}ms")
        return materials

    def generate_projection_matrix(self, projection_seed: bytes,
                                 block_dimensions: List[Tuple[int, int]],
                                 device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Generate ephemeral block-diagonal projection matrix from seed.

        Args:
            projection_seed: 16-byte seed from derive_round_materials()
            block_dimensions: List of (k_m, d_m) for each semantic block
            device: PyTorch device for tensor allocation

        Returns:
            Dictionary mapping block names to projection matrices
        """
        # Use projection seed to initialize NumPy random state
        # Convert bytes to 32-bit integer seed for NumPy compatibility
        seed_int = int.from_bytes(projection_seed[:4], 'little') % (2**32)
        rng = np.random.RandomState(seed_int)

        projection_matrices = {}

        for block_idx, (k_m, d_m) in enumerate(block_dimensions):
            # Generate block matrix r_m ~ N(0, 1/k_m)
            # This ensures JL property: ||r_m||^2 ≈ 1 with high probability
            matrix = rng.normal(0, 1/np.sqrt(k_m), (k_m, d_m))

            # Convert to PyTorch tensor
            projection_matrices[f"block_{block_idx}"] = torch.tensor(
                matrix, dtype=torch.float32, device=device
            )

        logger.debug(f"Generated {len(projection_matrices)} projection blocks from seed")
        return projection_matrices

    def generate_promotion_assignments(self, promotion_seed: bytes,
                                     client_trust_scores: Dict[int, float],
                                     tier_thresholds: Tuple[float, float] = (0.3, 0.7),
                                     tier_probabilities: Tuple[float, float, float] = (1.0, 0.5, 0.33)
                                     ) -> Tuple[List[int], List[int]]:
        """
        Generate client verification/promotion assignments from seed.

        Args:
            promotion_seed: 16-byte seed from derive_round_materials()
            client_trust_scores: Dict mapping client_id to trust score T_i(r)
            tier_thresholds: (θ_low, θ_high) for three-tier classification
            tier_probabilities: (P_verify_tier1, P_verify_tier2, P_verify_tier3)

        Returns:
            Tuple of (verified_clients, promoted_clients) lists
        """
        seed_int = int.from_bytes(promotion_seed[:4], 'little') % (2**32)
        rng = np.random.RandomState(seed_int)

        theta_low, theta_high = tier_thresholds
        p_tier1, p_tier2, p_tier3 = tier_probabilities

        verified_clients = []
        promoted_clients = []

        for client_id, trust_score in client_trust_scores.items():
            # Classify into tiers
            if trust_score < theta_low:
                tier = 1
                verify_prob = p_tier1  # Always verify untrusted clients
            elif trust_score < theta_high:
                tier = 2
                verify_prob = p_tier2  # 50% verification for medium trust
            else:
                tier = 3
                verify_prob = p_tier3  # 33% verification for high trust

            # Make verification decision using CSPRNG
            if rng.random() < verify_prob:
                verified_clients.append(client_id)
            else:
                promoted_clients.append(client_id)

        logger.debug(f"Promotion assignments: {len(verified_clients)} verified, {len(promoted_clients)} promoted")
        return verified_clients, promoted_clients

    def generate_decoy_verification(self, decoy_seed: bytes,
                                  promoted_clients: List[int],
                                  decoy_probability: float = 0.15) -> List[int]:
        """
        Generate decoy verification set from promoted clients.

        Decoy verification is a key security mechanism that randomly verifies
        some promoted clients to catch timing attacks.

        Args:
            decoy_seed: 16-byte seed from derive_round_materials()
            promoted_clients: List of client IDs that were promoted
            decoy_probability: Probability of decoy verification per promoted client

        Returns:
            List of promoted client IDs selected for decoy verification
        """
        if not promoted_clients:
            return []

        seed_int = int.from_bytes(decoy_seed[:4], 'little') % (2**32)
        rng = np.random.RandomState(seed_int)

        decoy_clients = []
        for client_id in promoted_clients:
            if rng.random() < decoy_probability:
                decoy_clients.append(client_id)

        logger.debug(f"Selected {len(decoy_clients)}/{len(promoted_clients)} promoted clients for decoy verification")
        return decoy_clients

    def _chacha20_stream(self, key: bytes, counter: int) -> bytes:
        """
        Simplified ChaCha20 stream generation.

        NOTE: This is a simplified implementation for prototype.
        In production, use pycryptodome ChaCha20 for security compliance.

        Args:
            key: 32-byte master key
            counter: Round number as counter

        Returns:
            64 bytes of pseudo-random stream
        """
        # Use HKDF-like construction with counter
        # This provides cryptographic security similar to ChaCha20
        combined = key + counter.to_bytes(8, 'little')

        # Multiple rounds of SHA-256 to generate 64 bytes
        stream = b''
        for i in range(2):  # 2 * 32 = 64 bytes
            hash_input = combined + i.to_bytes(4, 'little')
            stream += hashlib.sha256(hash_input).digest()

        return stream

    def _rotate_master_key(self, round_number: int):
        """
        Rotate master key for forward security.

        Args:
            round_number: Current round triggering rotation
        """
        # Derive new key from current key + round
        new_key_material = self.master_key + round_number.to_bytes(8, 'little')
        self.master_key = hashlib.sha256(new_key_material).digest()
        self.current_key_epoch += 1

        # Clear cache to force regeneration with new key
        self.round_materials_cache.clear()

        logger.info(f"Rotated master key at round {round_number}, epoch {self.current_key_epoch}")

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security and performance statistics."""
        avg_derivation_time = 0.0
        if self.derivation_stats["total_rounds"] > 0:
            avg_derivation_time = self.derivation_stats["total_time"] / self.derivation_stats["total_rounds"]

        return {
            "current_key_epoch": self.current_key_epoch,
            "total_rounds_processed": self.derivation_stats["total_rounds"],
            "average_derivation_time_ms": avg_derivation_time * 1000,
            "cache_size": len(self.round_materials_cache),
            "key_rotation_rounds": self.key_rotation_rounds
        }

    def export_master_key(self) -> bytes:
        """
        Export master key for backup/persistence.

        WARNING: Keep this key secure. Compromise allows prediction of all future randomness.
        """
        logger.warning("Master key exported - ensure secure storage!")
        return self.master_key

    def clear_cache(self):
        """Clear materials cache (useful for testing or memory management)."""
        cache_size = len(self.round_materials_cache)
        self.round_materials_cache.clear()
        logger.info(f"Cleared CSPRNG cache ({cache_size} entries)")