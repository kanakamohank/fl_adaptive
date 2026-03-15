#!/usr/bin/env python3
"""
GPT-2 TAVS Client for Federated Language Model Learning

This module provides a specialized TAVS client for GPT-2 models in federated
learning scenarios, supporting language modeling tasks with TAVS-ESP security.

Key Features:
- GPT-2 specific training and evaluation
- Language modeling loss computation
- Text generation capabilities
- TAVS-ESP integration for Byzantine robustness
- HuggingFace datasets compatibility

Core Innovation: Federated language model learning with Byzantine-robust
TAVS-ESP framework for secure distributed LLM training.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer

from .honest_client import HonestClient
from ..core.gpt2_model import GPT2FederatedModel, create_gpt2_tokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for GPT-2 training."""

    def __init__(self, texts: List[str], tokenizer: GPT2Tokenizer, max_length: int = 128):
        """
        Initialize text dataset.

        Args:
            texts: List of text strings
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize all texts
        self.tokenized_texts = []
        for text in texts:
            tokens = self.tokenizer.encode(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            self.tokenized_texts.append(tokens.squeeze())

        logger.info(f"TextDataset initialized: {len(self.texts)} texts, max_length={max_length}")

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        # For language modeling: input = tokens[:-1], target = tokens[1:]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return input_ids, target_ids


class GPT2TAVSClient(HonestClient):
    """
    GPT-2 specialized TAVS client for federated language modeling.

    Extends HonestClient with GPT-2 specific functionality:
    - Language modeling loss computation
    - Text generation evaluation
    - GPT-2 parameter handling
    """

    def __init__(self,
                 client_id: str,
                 model_name: str = "gpt2",
                 train_texts: List[str] = None,
                 test_texts: List[str] = None,
                 device: str = "cpu",
                 local_epochs: int = 1,
                 lr: float = 5e-5,
                 batch_size: int = 4,
                 max_length: int = 128):
        """
        Initialize GPT-2 TAVS client.

        Args:
            client_id: Unique client identifier
            model_name: GPT-2 model name ("gpt2" for small)
            train_texts: Training text data
            test_texts: Test text data
            device: Computation device
            local_epochs: Local training epochs
            lr: Learning rate
            batch_size: Batch size
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device)

        # Create tokenizer
        self.tokenizer = create_gpt2_tokenizer(model_name)

        # Create GPT-2 model
        self.gpt2_model = GPT2FederatedModel(model_name=model_name, device=device)

        # Create datasets
        train_dataset = None
        test_dataset = None

        if train_texts:
            train_dataset = TextDataset(train_texts, self.tokenizer, max_length)

        if test_texts:
            test_dataset = TextDataset(test_texts, self.tokenizer, max_length)

        # Create data loaders
        train_loader = None
        test_loader = None

        if train_dataset:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self._collate_fn
            )

        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate_fn
            )

        # Initialize parent HonestClient with dummy parameters
        # We'll override the model and training logic
        super().__init__(
            client_id=client_id,
            model_type="gpt2",  # This won't be used since we override the model
            model_kwargs={},
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            local_epochs=local_epochs,
            lr=lr
        )

        # Replace the model and optimizer
        self.model = self.gpt2_model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        logger.info(f"GPT-2 TAVS Client initialized: {client_id} ({model_name})")

    def _collate_fn(self, batch):
        """Collate function for text batch processing."""
        input_ids = []
        target_ids = []

        for input_seq, target_seq in batch:
            input_ids.append(input_seq)
            target_ids.append(target_seq)

        # Stack tensors
        input_ids = torch.stack(input_ids)
        target_ids = torch.stack(target_ids)

        return input_ids, target_ids

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the GPT-2 model locally with language modeling objective."""

        # Set parameters from server
        self.set_parameters(parameters)

        # Local training
        self.model.train()
        epoch_losses = []
        total_tokens = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (input_ids, target_ids) in enumerate(self.train_loader):
                # Move to device
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                # Get logits from GPT-2
                logits = self.model(input_ids)

                # Compute language modeling loss
                # Flatten for cross-entropy: (batch_size * seq_len, vocab_size)
                shift_logits = logits.view(-1, logits.size(-1))
                shift_labels = target_ids.view(-1)

                loss = self.criterion(shift_logits, shift_labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Count non-padding tokens
                non_pad_mask = (target_ids != self.tokenizer.pad_token_id)
                total_tokens += non_pad_mask.sum().item()

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)

        # Calculate perplexity
        final_loss = epoch_losses[-1] if epoch_losses else 0.0
        perplexity = torch.exp(torch.tensor(final_loss)).item() if final_loss > 0 else 1.0

        # Store training metrics
        training_metrics = {
            "client_id": self.client_id,
            "epoch_losses": epoch_losses,
            "final_loss": final_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "num_examples": len(self.train_loader.dataset) if self.train_loader else 0,
        }
        self.training_history.append(training_metrics)

        logger.debug(f"Client {self.client_id}: Training complete - "
                    f"loss={final_loss:.4f}, perplexity={perplexity:.2f}")

        return self.get_parameters({}), total_tokens, training_metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the GPT-2 model with language modeling metrics."""

        if not self.test_loader:
            return 0.0, 0, {"accuracy": 0.0, "perplexity": 1.0}

        # Set parameters for evaluation
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0

        with torch.no_grad():
            for input_ids, target_ids in self.test_loader:
                # Move to device
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                # Forward pass
                logits = self.model(input_ids)

                # Compute loss
                shift_logits = logits.view(-1, logits.size(-1))
                shift_labels = target_ids.view(-1)

                loss = self.criterion(shift_logits, shift_labels)
                total_loss += loss.item()

                # Compute accuracy (next token prediction)
                predictions = torch.argmax(shift_logits, dim=-1)

                # Only count non-padding tokens
                non_pad_mask = (shift_labels != self.tokenizer.pad_token_id)
                correct_predictions += (predictions == shift_labels)[non_pad_mask].sum().item()
                total_tokens += non_pad_mask.sum().item()

        # Calculate metrics
        avg_loss = total_loss / len(self.test_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0

        metrics = {
            "client_id": self.client_id,
            "accuracy": accuracy,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
        }

        logger.debug(f"Client {self.client_id}: Evaluation - "
                    f"loss={avg_loss:.4f}, perplexity={perplexity:.2f}, accuracy={accuracy:.3f}")

        return avg_loss, total_tokens, metrics

    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
        """Generate text using the current model."""
        self.model.eval()

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            # Generate text
            generated_ids = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )

            # Decode generated text
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return generated_text

    def get_model_weights_flat(self) -> np.ndarray:
        """Get GPT-2 model weights as flat numpy array."""
        return self.model.get_weights_flat().cpu().detach().numpy()

    def set_model_weights_flat(self, weights_flat: np.ndarray):
        """Set GPT-2 model weights from flat numpy array."""
        weights_tensor = torch.tensor(weights_flat, dtype=torch.float32).to(self.device)
        self.model.set_weights_flat(weights_tensor)


def create_gpt2_tavs_client(client_id: str,
                           model_name: str = "gpt2",
                           train_texts: List[str] = None,
                           test_texts: List[str] = None,
                           device: str = "cpu",
                           **kwargs) -> GPT2TAVSClient:
    """
    Factory function to create GPT-2 TAVS clients.

    Args:
        client_id: Unique client identifier
        model_name: GPT-2 model name
        train_texts: Training text data
        test_texts: Test text data
        device: Computation device
        **kwargs: Additional client parameters

    Returns:
        Configured GPT2TAVSClient instance
    """
    return GPT2TAVSClient(
        client_id=client_id,
        model_name=model_name,
        train_texts=train_texts,
        test_texts=test_texts,
        device=device,
        **kwargs
    )


def create_sample_text_data() -> Tuple[List[str], List[str]]:
    """Create sample text data for testing GPT-2 federated learning."""

    # Sample training texts (simple sentences for demonstration)
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Federated learning enables distributed machine learning.",
        "GPT-2 is a powerful language model for text generation.",
        "TAVS-ESP provides Byzantine-robust federated learning.",
        "Natural language processing has many applications.",
        "Machine learning models require careful training.",
        "Text generation can be useful for many tasks.",
        "Deep learning has revolutionized artificial intelligence.",
    ]

    # Sample test texts
    test_texts = [
        "The cat sat on the mat and looked around.",
        "Artificial intelligence is changing the world.",
        "Language models can generate human-like text.",
        "Distributed systems face unique challenges.",
    ]

    return train_texts, test_texts