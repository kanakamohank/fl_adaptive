"""
TAVS (Trust-Adaptive Verification Scheduling) System

This module implements the Layer 1 components of the TAVS-ESP framework:
- Trust-adaptive three-tier client scheduling
- ChaCha20-CTR cryptographically secure randomness
- Bayesian posterior weight computation
- Budget constraints and Sybil resistance mechanisms

The TAVS system governs WHO gets verified and WHEN, providing timing attack
resistance and efficient trust-based client management.
"""

from .csprng_manager import CSPRNGManager, RoundMaterials

__all__ = [
    'CSPRNGManager',
    'RoundMaterials'
]