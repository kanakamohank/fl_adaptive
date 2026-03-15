#!/usr/bin/env python3
"""
GPT-2 Small Model Integration for TAVS-ESP Federated Learning

This module provides GPT-2 small model integration with semantic block boundaries
for Ephemeral Structured Projections (ESP) in federated learning scenarios.

Key Innovations:
- Semantic block boundary identification for GPT-2 architecture
- GPT-2 specific model structure for ESP projections
- HuggingFace transformers integration with TAVS-ESP
- Memory-efficient federated learning for autoregressive language models

Core Components:
1. GPT2ModelStructure: Semantic block identification for ESP
2. GPT2FederatedModel: GPT-2 wrapper for FL with projection support
3. Transformer decoder-specific projection alignment
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from dataclasses import dataclass

from .models import ModelStructure, FederatedModel

logger = logging.getLogger(__name__)


@dataclass
class GPT2BlockInfo:
    """Information about GPT-2 semantic blocks for ESP projections."""
    name: str
    layer_idx: int
    parameter_count: int
    block_type: str  # "embedding", "transformer_block", "ln_f", "lm_head"
    attention_heads: Optional[int] = None
    hidden_size: Optional[int] = None


class GPT2ModelStructure(ModelStructure):
    """
    GPT-2 specific model structure with semantic block boundaries.

    This class identifies semantic blocks in GPT-2 architecture for
    structured projections that preserve autoregressive transformer semantics.
    """

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 model structure with semantic blocks."""
        super().__init__()
        self.config = config
        self.gpt2_blocks: List[GPT2BlockInfo] = []

        self._identify_semantic_blocks()
        logger.info(f"GPT-2 structure: {len(self.gpt2_blocks)} semantic blocks identified")

    def _identify_semantic_blocks(self):
        """Identify semantic block boundaries in GPT-2 architecture."""

        # 1. Token Embedding Block
        token_embedding_params = self.config.vocab_size * self.config.n_embd

        token_emb_block = GPT2BlockInfo(
            name="token_embeddings",
            layer_idx=0,
            parameter_count=token_embedding_params,
            block_type="embedding",
            hidden_size=self.config.n_embd
        )
        self.gpt2_blocks.append(token_emb_block)

        # 2. Position Embedding Block
        position_embedding_params = self.config.n_positions * self.config.n_embd

        pos_emb_block = GPT2BlockInfo(
            name="position_embeddings",
            layer_idx=1,
            parameter_count=position_embedding_params,
            block_type="embedding",
            hidden_size=self.config.n_embd
        )
        self.gpt2_blocks.append(pos_emb_block)

        # 3. Transformer Blocks (each layer is semantically independent)
        for layer_idx in range(self.config.n_layer):
            # Each transformer block contains:
            # - Multi-head self-attention
            # - Feed-forward network (MLP)
            # - Layer normalizations

            # Attention parameters
            # c_attn: combined Q, K, V projection (3 * n_embd * n_embd)
            # c_proj: output projection (n_embd * n_embd)
            attention_params = (
                3 * self.config.n_embd * self.config.n_embd +  # c_attn weight
                3 * self.config.n_embd +                       # c_attn bias
                self.config.n_embd * self.config.n_embd +      # c_proj weight
                self.config.n_embd                             # c_proj bias
            )

            # MLP parameters
            # c_fc: input projection (n_embd * 4 * n_embd)
            # c_proj: output projection (4 * n_embd * n_embd)
            mlp_params = (
                self.config.n_embd * (4 * self.config.n_embd) +  # c_fc weight
                4 * self.config.n_embd +                          # c_fc bias
                (4 * self.config.n_embd) * self.config.n_embd +  # c_proj weight
                self.config.n_embd                                # c_proj bias
            )

            # Layer norm parameters (2 layer norms per block)
            ln_params = 2 * (
                self.config.n_embd +  # weight
                self.config.n_embd    # bias
            )

            total_block_params = attention_params + mlp_params + ln_params

            transformer_block = GPT2BlockInfo(
                name=f"transformer_block_{layer_idx}",
                layer_idx=layer_idx + 2,  # After embeddings
                parameter_count=total_block_params,
                block_type="transformer_block",
                attention_heads=self.config.n_head,
                hidden_size=self.config.n_embd
            )
            self.gpt2_blocks.append(transformer_block)

        # 4. Final Layer Norm
        ln_f_params = 2 * self.config.n_embd  # weight + bias

        ln_f_block = GPT2BlockInfo(
            name="ln_f",
            layer_idx=2 + self.config.n_layer,
            parameter_count=ln_f_params,
            block_type="ln_f",
            hidden_size=self.config.n_embd
        )
        self.gpt2_blocks.append(ln_f_block)

        # 5. Language Model Head (if present)
        lm_head_params = self.config.vocab_size * self.config.n_embd

        lm_head_block = GPT2BlockInfo(
            name="lm_head",
            layer_idx=3 + self.config.n_layer,
            parameter_count=lm_head_params,
            block_type="lm_head",
            hidden_size=self.config.n_embd
        )
        self.gpt2_blocks.append(lm_head_block)

        # Add blocks to base ModelStructure
        for block in self.gpt2_blocks:
            self.add_block(
                name=block.name,
                shape=(block.parameter_count,),
                num_params=block.parameter_count
            )

        logger.info(f"GPT-2 semantic blocks: {[b.name for b in self.gpt2_blocks]}")
        logger.info(f"Total GPT-2 parameters: {self.total_params:,}")

    def get_projection_groups(self) -> List[List[str]]:
        """Get semantic projection groups for ESP structured projections."""
        projection_groups = []

        # Group 1: Embeddings (semantically cohesive input representations)
        projection_groups.append(["token_embeddings", "position_embeddings"])

        # Group 2: Early transformer layers (lower-level language features)
        early_layers = [f"transformer_block_{i}" for i in range(self.config.n_layer // 3)]
        if early_layers:
            projection_groups.append(early_layers)

        # Group 3: Middle transformer layers (intermediate language features)
        mid_start = self.config.n_layer // 3
        mid_end = 2 * self.config.n_layer // 3
        mid_layers = [f"transformer_block_{i}" for i in range(mid_start, mid_end)]
        if mid_layers:
            projection_groups.append(mid_layers)

        # Group 4: Late transformer layers (high-level language features)
        late_start = 2 * self.config.n_layer // 3
        late_layers = [f"transformer_block_{i}" for i in range(late_start, self.config.n_layer)]
        if late_layers:
            projection_groups.append(late_layers)

        # Group 5: Output components (final processing)
        projection_groups.append(["ln_f", "lm_head"])

        logger.info(f"GPT-2 projection groups: {len(projection_groups)} groups")
        return projection_groups

    def get_attention_boundaries(self) -> Dict[str, List[str]]:
        """Get attention mechanism boundaries for projection alignment."""
        attention_boundaries = {}

        for block in self.gpt2_blocks:
            if block.block_type == "transformer_block":
                # Each transformer block has self-attention that should be preserved
                attention_boundaries[block.name] = [block.name]

        return attention_boundaries


class GPT2FederatedModel(FederatedModel):
    """
    GPT-2 model wrapper for federated learning with TAVS-ESP support.

    Provides GPT-2 small model with:
    - Semantic block structure for ESP projections
    - Memory-efficient parameter handling
    - HuggingFace transformers compatibility
    - Language modeling and generation capabilities
    """

    def __init__(self,
                 model_name: str = "gpt2",  # gpt2 = small (124M params)
                 use_lm_head: bool = True,
                 device: str = "cpu"):
        """
        Initialize GPT-2 federated model.

        Args:
            model_name: HuggingFace model name ("gpt2" for small)
            use_lm_head: Whether to include language modeling head
            device: Device for model computation
        """

        # Load GPT-2 configuration
        self.config = GPT2Config.from_pretrained(model_name)

        # Load model (with or without LM head)
        if use_lm_head:
            self.model = GPT2LMHeadModel.from_pretrained(model_name, config=self.config)
            self.gpt2 = self.model.transformer
            self.lm_head = self.model.lm_head
        else:
            self.gpt2 = GPT2Model.from_pretrained(model_name, config=self.config)
            self.model = self.gpt2
            self.lm_head = None

        # Initialize as FederatedModel
        super().__init__()

        # Create semantic model structure
        self.structure = GPT2ModelStructure(self.config)

        self.use_lm_head = use_lm_head
        self.device = torch.device(device)
        self.model.to(self.device)

        logger.info(f"GPT-2 Federated Model initialized: {model_name} "
                   f"({self.config.n_layer} layers, {self.structure.total_params:,} params)")

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        """Forward pass through GPT-2."""
        if self.use_lm_head:
            # Language modeling forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values
            )
            return outputs.logits
        else:
            # Base model forward pass
            outputs = self.gpt2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values
            )
            return outputs.last_hidden_state

    def get_weights_flat(self) -> torch.Tensor:
        """Get model weights as flat tensor for federated learning."""
        weights = []

        # Get all model parameters
        for param in self.model.parameters():
            weights.append(param.flatten())

        return torch.cat(weights)

    def set_weights_flat(self, weights_flat: torch.Tensor):
        """Set model weights from flat tensor."""
        offset = 0

        # Set all model parameters
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = weights_flat[offset:offset + param_size].reshape(param.shape)
            offset += param_size

        if offset != len(weights_flat):
            logger.warning(f"Weight size mismatch: expected {offset}, got {len(weights_flat)}")

    def get_structured_weights(self) -> Dict[str, torch.Tensor]:
        """Get weights organized by semantic blocks."""
        structured_weights = {}

        # Token embeddings
        structured_weights["token_embeddings"] = self.gpt2.wte.weight.flatten()

        # Position embeddings
        structured_weights["position_embeddings"] = self.gpt2.wpe.weight.flatten()

        # Transformer blocks
        for i, block in enumerate(self.gpt2.h):
            block_weights = []

            # Attention weights (c_attn and c_proj)
            block_weights.append(block.attn.c_attn.weight.flatten())
            block_weights.append(block.attn.c_attn.bias.flatten())
            block_weights.append(block.attn.c_proj.weight.flatten())
            block_weights.append(block.attn.c_proj.bias.flatten())

            # MLP weights (c_fc and c_proj)
            block_weights.append(block.mlp.c_fc.weight.flatten())
            block_weights.append(block.mlp.c_fc.bias.flatten())
            block_weights.append(block.mlp.c_proj.weight.flatten())
            block_weights.append(block.mlp.c_proj.bias.flatten())

            # Layer norms
            block_weights.append(block.ln_1.weight.flatten())
            block_weights.append(block.ln_1.bias.flatten())
            block_weights.append(block.ln_2.weight.flatten())
            block_weights.append(block.ln_2.bias.flatten())

            structured_weights[f"transformer_block_{i}"] = torch.cat(block_weights)

        # Final layer norm
        ln_f_weights = []
        ln_f_weights.append(self.gpt2.ln_f.weight.flatten())
        ln_f_weights.append(self.gpt2.ln_f.bias.flatten())
        structured_weights["ln_f"] = torch.cat(ln_f_weights)

        # Language model head (if present)
        if self.use_lm_head and self.lm_head is not None:
            structured_weights["lm_head"] = self.lm_head.weight.flatten()

        return structured_weights

    def set_structured_weights(self, structured_weights: Dict[str, torch.Tensor]):
        """Set weights from structured format."""

        # Set token embeddings
        if "token_embeddings" in structured_weights:
            self.gpt2.wte.weight.data = structured_weights["token_embeddings"].reshape(
                self.gpt2.wte.weight.shape)

        # Set position embeddings
        if "position_embeddings" in structured_weights:
            self.gpt2.wpe.weight.data = structured_weights["position_embeddings"].reshape(
                self.gpt2.wpe.weight.shape)

        # Set transformer block weights
        for i, block in enumerate(self.gpt2.h):
            block_key = f"transformer_block_{i}"
            if block_key in structured_weights:
                block_weights = structured_weights[block_key]
                offset = 0

                # Set attention weights
                c_attn_w_size = block.attn.c_attn.weight.numel()
                block.attn.c_attn.weight.data = block_weights[offset:offset + c_attn_w_size].reshape(
                    block.attn.c_attn.weight.shape)
                offset += c_attn_w_size

                c_attn_b_size = block.attn.c_attn.bias.numel()
                block.attn.c_attn.bias.data = block_weights[offset:offset + c_attn_b_size].reshape(
                    block.attn.c_attn.bias.shape)
                offset += c_attn_b_size

                c_proj_w_size = block.attn.c_proj.weight.numel()
                block.attn.c_proj.weight.data = block_weights[offset:offset + c_proj_w_size].reshape(
                    block.attn.c_proj.weight.shape)
                offset += c_proj_w_size

                c_proj_b_size = block.attn.c_proj.bias.numel()
                block.attn.c_proj.bias.data = block_weights[offset:offset + c_proj_b_size].reshape(
                    block.attn.c_proj.bias.shape)
                offset += c_proj_b_size

                # Set MLP weights
                mlp_fc_w_size = block.mlp.c_fc.weight.numel()
                block.mlp.c_fc.weight.data = block_weights[offset:offset + mlp_fc_w_size].reshape(
                    block.mlp.c_fc.weight.shape)
                offset += mlp_fc_w_size

                mlp_fc_b_size = block.mlp.c_fc.bias.numel()
                block.mlp.c_fc.bias.data = block_weights[offset:offset + mlp_fc_b_size].reshape(
                    block.mlp.c_fc.bias.shape)
                offset += mlp_fc_b_size

                mlp_proj_w_size = block.mlp.c_proj.weight.numel()
                block.mlp.c_proj.weight.data = block_weights[offset:offset + mlp_proj_w_size].reshape(
                    block.mlp.c_proj.weight.shape)
                offset += mlp_proj_w_size

                mlp_proj_b_size = block.mlp.c_proj.bias.numel()
                block.mlp.c_proj.bias.data = block_weights[offset:offset + mlp_proj_b_size].reshape(
                    block.mlp.c_proj.bias.shape)
                offset += mlp_proj_b_size

                # Set layer norms
                ln1_w_size = block.ln_1.weight.numel()
                block.ln_1.weight.data = block_weights[offset:offset + ln1_w_size].reshape(
                    block.ln_1.weight.shape)
                offset += ln1_w_size

                ln1_b_size = block.ln_1.bias.numel()
                block.ln_1.bias.data = block_weights[offset:offset + ln1_b_size].reshape(
                    block.ln_1.bias.shape)
                offset += ln1_b_size

                ln2_w_size = block.ln_2.weight.numel()
                block.ln_2.weight.data = block_weights[offset:offset + ln2_w_size].reshape(
                    block.ln_2.weight.shape)
                offset += ln2_w_size

                ln2_b_size = block.ln_2.bias.numel()
                block.ln_2.bias.data = block_weights[offset:offset + ln2_b_size].reshape(
                    block.ln_2.bias.shape)

        # Set final layer norm
        if "ln_f" in structured_weights:
            ln_f_weights = structured_weights["ln_f"]
            offset = 0

            ln_f_w_size = self.gpt2.ln_f.weight.numel()
            self.gpt2.ln_f.weight.data = ln_f_weights[offset:offset + ln_f_w_size].reshape(
                self.gpt2.ln_f.weight.shape)
            offset += ln_f_w_size

            ln_f_b_size = self.gpt2.ln_f.bias.numel()
            self.gpt2.ln_f.bias.data = ln_f_weights[offset:offset + ln_f_b_size].reshape(
                self.gpt2.ln_f.bias.shape)

        # Set language model head
        if "lm_head" in structured_weights and self.lm_head is not None:
            self.lm_head.weight.data = structured_weights["lm_head"].reshape(
                self.lm_head.weight.shape)

        logger.debug("GPT-2 structured weights updated")

    def generate(self, input_ids, max_length=50, temperature=1.0, pad_token_id=None):
        """Generate text using the GPT-2 model."""
        if not self.use_lm_head:
            raise ValueError("Generation requires language model head")

        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=pad_token_id,
                do_sample=True
            )

        return generated


def get_gpt2_model(model_name: str = "gpt2",
                   use_lm_head: bool = True,
                   device: str = "cpu") -> GPT2FederatedModel:
    """
    Factory function to create GPT-2 federated model.

    Args:
        model_name: HuggingFace model identifier ("gpt2" for small)
        use_lm_head: Whether to include language modeling head
        device: Device for computation

    Returns:
        Configured GPT2FederatedModel instance
    """
    return GPT2FederatedModel(
        model_name=model_name,
        use_lm_head=use_lm_head,
        device=device
    )


def create_gpt2_tokenizer(model_name: str = "gpt2") -> GPT2Tokenizer:
    """Create GPT-2 tokenizer for preprocessing."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Add padding token (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer