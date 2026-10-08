#!/usr/bin/env python3
"""
TAVS Flower Client Integration

This module provides a Flower-compatible client wrapper that integrates with
the TAVS-ESP system while maintaining compatibility with existing honest
and attacker client implementations.

Core Integration:
- Wraps existing HonestClient and attacker implementations
- Handles Flower serialization and communication protocols
- Processes TAVS assignment messages (verified vs promoted)
- Maintains attack coordination capabilities for research validation

Key Innovation: Seamless integration between TAVS trust-adaptive system
and existing FL client implementations without breaking compatibility.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch
from dataclasses import dataclass

# Flower imports
import flwr as fl
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

# TAVS-ESP imports
from .honest_client import HonestClient
from ..attacks.null_space_attack import NullSpaceAttacker
from ..attacks.layerwise_attacks import LayerwiseBackdoorAttacker, DistributedPoisonAttacker

logger = logging.getLogger(__name__)


@dataclass
class TAVSClientConfig:
    """Configuration for TAVS Flower client."""
    client_id: str
    client_type: str  # "honest", "null_space", "layerwise", "distributed"
    model_type: str = "cifar_cnn"
    model_kwargs: Dict[str, Any] = None
    device: str = "cpu"

    # Attack-specific parameters
    attack_intensity: float = 1.0
    target_fraction: float = 0.001  # For layerwise attacks
    # Adaptive adversary: attack only when the client believes it is unobserved.
    # This is the threat model CSPRNG decoy verification is designed against; a
    # non-adaptive attacker that poisons even under inspection is caught by any
    # verification and makes decoys pointless. Set False for a naive adversary.
    adaptive_evasion: bool = True

    # Training parameters
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01


class TAVSFlowerClient(NumPyClient):
    """
    TAVS-compatible Flower client wrapper.

    This client acts as a bridge between the Flower federated learning framework
    and our existing client implementations (honest clients and attackers).

    Key Features:
    - Delegates training to underlying client implementation
    - Processes TAVS assignment messages from server
    - Maintains attack coordination for Byzantine clients
    - Handles parameter serialization for Flower communication
    """

    def __init__(self,
                 config: TAVSClientConfig,
                 train_loader = None,
                 test_loader = None,
                 partition_id: int = None,
                 val_loader = None):
        """
        Initialize TAVS Flower client.

        Args:
            config: Client configuration including type and parameters
            train_loader: Training data loader
            test_loader: Test data loader (optional)
            partition_id: The client's partition index (0..num_clients-1) as
                assigned by Flower's node_config. Reported via get_properties
                so the strategy can build a stable cid <-> partition-id map
                for cross-arm cohort alignment.
            val_loader: Local held-out validation DataLoader. When present,
                fit() evaluates the INCOMING global model on it before local
                training and reports the fraction of correct predictions
                ('small_loss_fraction') in the fit metrics. The strategy uses
                this as a drop-in replacement for BVD's behavior_score when
                config.trust_signal == 'small_loss_fraction'. None disables
                the signal; the server-side override simply falls back to
                BVD's score for that client.
        """
        super().__init__()

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.partition_id = partition_id

        # Initialize underlying client based on type
        self.underlying_client = self._create_underlying_client()

        # TAVS state tracking
        self.current_assignment = "verified"  # "verified" or "promoted"
        self.trust_score = 0.5  # Current trust score from server
        self.tier = 1  # Current tier assignment
        self.round_number = 0
        self.is_decoy = False  # Server-side only; never revealed to the client
        self.adaptive_evasion = getattr(config, "adaptive_evasion", True)

        # Performance tracking
        self.training_history = []
        self.assignment_history = []

        logger.info(f"TAVS Flower Client initialized: {config.client_id} ({config.client_type})")

    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        """Report the client's partition-id so the strategy can build a
        stable cid <-> partition-id map. The strategy uses this to sample
        cohorts in partition-id space, which is identical across arms with
        the same seed (cid is a per-simulation random 64-bit int and cannot
        be used for cross-arm alignment). See TavsEspStrategy._ensure_partition_map.

        Returns -1 when the client was constructed without partition_id --
        old-style tests / direct instantiations. Strategy code treats a
        missing partition-id as a bootstrap failure and raises loudly, so
        real pipelines cannot silently regress to divergent cohorts.
        """
        pid = self.partition_id if self.partition_id is not None else -1
        return {"partition-id": int(pid)}

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters as NumPy arrays."""
        try:
            # Delegate to underlying client
            if hasattr(self.underlying_client, 'get_parameters'):
                params = self.underlying_client.get_parameters(config)
            else:
                # Fallback: get parameters from model
                params = self.underlying_client.model.get_weights_flat()

            # Convert to list of NumPy arrays for Flower
            if isinstance(params, torch.Tensor):
                # Handle single flattened tensor
                param_arrays = [params.detach().cpu().numpy()]
            elif isinstance(params, list):
                # Handle list of tensors
                param_arrays = []
                for param in params:
                    if isinstance(param, torch.Tensor):
                        param_arrays.append(param.detach().cpu().numpy())
                    elif isinstance(param, np.ndarray):
                        param_arrays.append(param)
                    else:
                        param_arrays.append(np.array(param))
            else:
                # Handle numpy array
                param_arrays = [np.array(params)]

            logger.debug(f"Client {self.config.client_id}: Retrieved {len(param_arrays)} parameter arrays")
            return param_arrays

        except Exception as e:
            logger.error(f"Client {self.config.client_id}: Error getting parameters: {e}")
            # Return dummy parameters to prevent crash
            return [np.array([0.0])]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from NumPy arrays."""
        try:
            # Convert NumPy arrays back to appropriate format for underlying client
            if hasattr(self.underlying_client, 'set_parameters'):
                # HonestClient expects list of numpy arrays
                self.underlying_client.set_parameters(parameters)
            else:
                # Fallback: set parameters directly on model
                if len(parameters) == 1:
                    param_tensor = torch.tensor(parameters[0], dtype=torch.float32)
                    self.underlying_client.model.set_weights_flat(param_tensor)
                else:
                    # Concatenate multiple arrays if needed
                    all_params = np.concatenate([p.flatten() for p in parameters])
                    param_tensor = torch.tensor(all_params, dtype=torch.float32)
                    self.underlying_client.model.set_weights_flat(param_tensor)

            logger.debug(f"Client {self.config.client_id}: Set {len(parameters)} parameter arrays")

        except Exception as e:
            logger.error(f"Client {self.config.client_id}: Error setting parameters: {e}")

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model and return updated parameters.

        This method processes TAVS assignments and delegates training to the
        underlying client implementation.
        """
        try:
            # DEBUG: Log incoming parameters from Flower
            logger.debug(f"Client {self.config.client_id}: FLOWER INPUT - "
                        f"parameters type: {type(parameters)}, length: {len(parameters) if hasattr(parameters, '__len__') else 'N/A'}")
            if hasattr(parameters, '__len__') and len(parameters) > 0:
                for i, param in enumerate(parameters):
                    logger.debug(f"  Param {i}: type={type(param)}, shape={getattr(param, 'shape', 'N/A')}, dtype={getattr(param, 'dtype', 'N/A')}")

            # Update TAVS state from server config
            self._process_tavs_config(config)

            # Set initial parameters
            self.set_parameters(parameters)

            # BEFORE any local training, measure how well the INCOMING global
            # model fits this client's own held-out val labels. See
            # TAVSFlowerClient.__init__ docstring on val_loader. We compute
            # this now (pre-training) because Co-teaching's small-loss insight
            # is about the model's current fit to the LABELS, not the model's
            # fit after it has been trained on them -- the latter would let
            # noisy clients fit their wrong labels by memorisation and erase
            # the signal.
            small_loss_fraction = self._small_loss_fraction_on_val()

            # Pre-training oracle diagnostics (opt-in; off by default).
            # Must happen BEFORE training so noisy clients haven't memorised
            # their wrong labels yet. See reviewer note on signal collapse.
            oracle_pretrain = None
            if getattr(self, "_log_oracle_signals", False):
                num_classes = (self.config.model_kwargs or {}).get("num_classes", 10)
                oracle_pretrain = self._oracle_signals_on_train(num_classes=num_classes)

            # Execute training based on client type and assignment
            num_examples = self._execute_training(config)

            # Get updated parameters
            updated_parameters = self.get_parameters(config)

            # Prepare metrics for server
            metrics = self._prepare_fit_metrics(num_examples)
            if small_loss_fraction is not None:
                metrics["small_loss_fraction"] = float(small_loss_fraction)

            if getattr(self, "_log_oracle_signals", False):
                # Update-norm from incoming vs outgoing params (not grad-norm;
                # the two diverge over 5 local epochs with momentum, so this
                # is a trajectory-length proxy, not a per-step gradient size).
                try:
                    import numpy as _np
                    diff_sq = 0.0
                    for p_in, p_out in zip(parameters, updated_parameters):
                        d = _np.asarray(p_out).ravel() - _np.asarray(p_in).ravel()
                        diff_sq += float((d * d).sum())
                    metrics["update_norm"] = float(diff_sq ** 0.5)
                except Exception:
                    pass
                # Memorization gap from honest_client's epoch_losses, if present.
                try:
                    uh = getattr(self.underlying_client, "training_history", None)
                    if uh:
                        last = uh[-1]
                        el = last.get("epoch_losses") or []
                        if len(el) >= 1:
                            metrics["loss_epoch_first"] = float(el[0])
                            metrics["loss_epoch_last"] = float(el[-1])
                            metrics["memorization_gap"] = float(el[0] - el[-1])
                        if "first_batch_grad_norm" in last:
                            metrics["first_batch_grad_norm"] = float(last["first_batch_grad_norm"])
                except Exception:
                    pass
                if oracle_pretrain is not None:
                    # Flower's protobuf serde for fit metrics accepts only
                    # Scalar (bool/bytes/float/int/str) -- a list raises
                    # ValueError at serde time, AFTER fit() returns, which
                    # Flower reports as a client-side failure. The whole
                    # fit result is then dropped and the model never
                    # updates. Encode list-valued fields as JSON strings;
                    # aggregate_fit decodes them before writing to
                    # oracle_signal_history.
                    import json as _json
                    for k, v in oracle_pretrain.items():
                        if isinstance(v, (list, tuple)):
                            metrics[k] = _json.dumps(list(v))
                        else:
                            metrics[k] = v

            logger.info(f"Client {self.config.client_id}: Training complete "
                       f"(round {self.round_number}, {self.current_assignment}, "
                       f"trust={self.trust_score:.3f})")

            return updated_parameters, num_examples, metrics

        except Exception as e:
            logger.error(f"Client {self.config.client_id}: Training failed: {e}")
            # Return original parameters to prevent crash
            return parameters, 0, {"error": str(e)}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model and return metrics.

        Optional method for client-side evaluation.
        """
        try:
            if self.test_loader is None:
                return 0.0, 0, {"accuracy": 0.0}

            # Set parameters for evaluation
            self.set_parameters(parameters)

            # Delegate evaluation to underlying client
            if hasattr(self.underlying_client, 'evaluate'):
                loss, num_examples, eval_metrics = self.underlying_client.evaluate(parameters, config)
                accuracy = eval_metrics.get('accuracy', 0.0)
            else:
                # Simple evaluation fallback
                loss = 0.0
                accuracy = 0.0
                num_examples = len(self.test_loader.dataset) if hasattr(self.test_loader, 'dataset') else 100

            metrics = {
                "accuracy": accuracy,
                "client_id": self.config.client_id,
                "client_type": self.config.client_type
            }

            logger.debug(f"Client {self.config.client_id}: Evaluation - "
                        f"loss={loss:.4f}, accuracy={accuracy:.3f}")

            return float(loss), int(num_examples), metrics

        except Exception as e:
            logger.error(f"Client {self.config.client_id}: Evaluation failed: {e}")
            return 0.0, 0, {"error": str(e)}

    def _create_underlying_client(self):
        """Create the underlying client implementation based on configuration."""
        model_kwargs = self.config.model_kwargs or {"num_classes": 10}

        if self.config.client_type == "honest":
            # For honest clients, create directly using HonestClient constructor
            from .honest_client import HonestClient
            return HonestClient(
                client_id=self.config.client_id,
                model_type=self.config.model_type,
                model_kwargs=model_kwargs,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
                device=self.config.device
            )

        elif self.config.client_type == "null_space":
            return NullSpaceAttacker(
                client_id=self.config.client_id,
                model_type=self.config.model_type,
                model_kwargs=model_kwargs,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
                attack_intensity=self.config.attack_intensity,
                device=self.config.device
            )

        elif self.config.client_type == "layerwise":
            # Use target layers based on target_fraction
            target_layers = ["fc1"] if self.config.target_fraction > 0 else []
            return LayerwiseBackdoorAttacker(
                client_id=self.config.client_id,
                model_type=self.config.model_type,
                model_kwargs=model_kwargs,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
                target_layers=target_layers,
                attack_intensity=self.config.attack_intensity,
                device=self.config.device
            )

        elif self.config.client_type == "distributed":
            return DistributedPoisonAttacker(
                client_id=self.config.client_id,
                model_type=self.config.model_type,
                model_kwargs=model_kwargs,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
                poison_intensity=self.config.attack_intensity,
                device=self.config.device
            )

        else:
            raise ValueError(f"Unknown client type: {self.config.client_type}")

    def _process_tavs_config(self, config: Dict[str, Scalar]):
        """Process TAVS-specific configuration from server."""
        if "round" in config:
            self.round_number = int(config["round"])
        if "server_round" in config:
            self.round_number = int(config["server_round"])

        # Server strategy puts verified vs promoted in FitIns; must echo for aggregate_fit
        if "is_verified" in config:
            v = config["is_verified"]
            if isinstance(v, str):
                self._last_is_verified = v.lower() in ("true", "1", "yes")
            else:
                self._last_is_verified = bool(v)
        else:
            self._last_is_verified = True

        if "tavs_assignment" in config:
            self.current_assignment = str(config["tavs_assignment"])

        if "trust_score" in config:
            self.trust_score = float(config["trust_score"])

        if "tier" in config:
            self.tier = int(config["tier"])

        if "is_decoy" in config:
            self.is_decoy = bool(config["is_decoy"])

        if "log_oracle_signals" in config:
            v = config["log_oracle_signals"]
            if isinstance(v, str):
                self._log_oracle_signals = v.lower() in ("true", "1", "yes")
            else:
                self._log_oracle_signals = bool(v)
        else:
            self._log_oracle_signals = getattr(self, "_log_oracle_signals", False)

        # Store assignment history
        self.assignment_history.append({
            "round": self.round_number,
            "assignment": self.current_assignment,
            "trust_score": self.trust_score,
            "tier": self.tier,
            "is_decoy": self.is_decoy
        })

        logger.debug(f"Client {self.config.client_id}: TAVS assignment - "
                    f"{self.current_assignment} (tier {self.tier}, trust={self.trust_score:.3f})")

    def _execute_training(self, config: Dict[str, Scalar]) -> int:
        """Execute training based on client type and TAVS assignment."""
        # Convert config to format expected by underlying client
        training_config = {
            "round": self.round_number,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate
        }

        # Get initial parameters in numpy format
        initial_params = self.get_parameters(config)

        # DEBUG: Log parameters being passed to underlying client
        logger.debug(f"Client {self.config.client_id}: UNDERLYING CLIENT INPUT - "
                    f"initial_params type: {type(initial_params)}, length: {len(initial_params) if hasattr(initial_params, '__len__') else 'N/A'}")
        if hasattr(initial_params, '__len__') and len(initial_params) > 0:
            for i, param in enumerate(initial_params):
                logger.debug(f"  Underlying param {i}: type={type(param)}, shape={getattr(param, 'shape', 'N/A')}, dtype={getattr(param, 'dtype', 'N/A')}")

        # Execute training through underlying client
        if self.config.client_type == "honest":
            # Honest clients train normally regardless of assignment
            trained_params, num_examples, metrics = self.underlying_client.fit(
                initial_params, training_config
            )

        elif self.adaptive_evasion and self.current_assignment == "verified":
            # Adaptive adversary: behaves honestly whenever it BELIEVES it is
            # under inspection. `current_assignment` is what the server told this
            # client, which for a decoy is deliberately "promoted" even though
            # the server verifies it -- that asymmetry is what lets a decoy catch
            # an evading attacker.
            #
            # Note this reads current_assignment only, not is_decoy: a client
            # that could see its own decoy flag would evade every hidden check,
            # which is precisely the leak the mechanism is designed to avoid.
            trained_params, num_examples, metrics = self.underlying_client.behave_honestly(
                initial_params, training_config
            )
            metrics = dict(metrics)
            metrics["evaded_this_round"] = True

        else:
            # Believes it is unobserved (or evasion disabled): full attack.
            trained_params, num_examples, metrics = self.underlying_client.fit(
                initial_params, training_config
            )
            metrics = dict(metrics)
            metrics.setdefault("evaded_this_round", False)

        # Update our parameters
        self.set_parameters(trained_params)

        # Store training history
        self.training_history.append({
            "round": self.round_number,
            "assignment": self.current_assignment,
            "num_examples": num_examples,
            "metrics": metrics
        })

        return num_examples

    def _oracle_signals_on_train(self, num_classes: int = 10):
        """Pre-training diagnostic signals for the oracle noise-detection
        experiment. Runs ONE eval pass over this client's train_loader with
        the fresh global weights (no backward, no grad). Measures what the
        INCOMING global model sees in each sample's label — before any local
        step can memorise noise.

        Returns a dict with:
          per_class_pretrain_loss: list[num_classes]  (NaN if class absent)
          per_class_pretrain_count: list[num_classes]
          pretrain_loss_mean: float
          pretrain_loss_var: float   (variance across samples)
        Or None if train_loader is unavailable / empty.
        """
        if self.train_loader is None:
            return None
        model = getattr(self.underlying_client, "model", None)
        if model is None:
            return None
        import torch as _torch
        try:
            device = next(model.parameters()).device
        except StopIteration:
            return None
        model.eval()
        loss_fn = _torch.nn.CrossEntropyLoss(reduction="none")
        per_class_sum = [0.0] * num_classes
        per_class_cnt = [0] * num_classes
        all_losses = []
        with _torch.no_grad():
            for batch in self.train_loader:
                if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
                    continue
                x, y = batch
                try:
                    x = x.to(device)
                    y = y.to(device, dtype=_torch.long)
                    out = model(x)
                    per_sample = loss_fn(out, y)
                except Exception:
                    return None
                for i in range(y.shape[0]):
                    c = int(y[i].item())
                    if 0 <= c < num_classes:
                        per_class_sum[c] += float(per_sample[i].item())
                        per_class_cnt[c] += 1
                all_losses.extend(float(v) for v in per_sample.detach().cpu().tolist())
        if not all_losses:
            return None
        per_class_loss = []
        for c in range(num_classes):
            per_class_loss.append(
                (per_class_sum[c] / per_class_cnt[c]) if per_class_cnt[c] > 0
                else float("nan")
            )
        n = len(all_losses)
        mean = sum(all_losses) / n
        var = sum((v - mean) ** 2 for v in all_losses) / n if n > 1 else 0.0
        return {
            "per_class_pretrain_loss": per_class_loss,
            "per_class_pretrain_count": per_class_cnt,
            "pretrain_loss_mean": float(mean),
            "pretrain_loss_var": float(var),
        }

    def _small_loss_fraction_on_val(self):
        """Return the fraction of val samples whose label the current model
        predicts correctly on top-1.

        What we actually compute: top-1 accuracy of the just-set model
        against the client's own (possibly-noisy) val labels, over the
        local held-out val loader.

        How we frame it: as a cheap proxy for the "small-loss fraction"
        intuition in Co-teaching (Han et al. 2018). Co-teaching's own
        small-loss selection operates per-sample inside a batch; what we
        send to the server is an aggregate SCALAR per client. The equivalence
        between "accuracy" and "fraction with cross-entropy below some tau"
        only holds once the model is above random chance -- in round 1 it
        does not. The signal becomes useful once the global model has
        learned the easy classes. Reviewer: closer precedent is
        FedCorr-style client-reliability signals. We retain the
        `small_loss_fraction` key to match the config-flag name, but it
        is top-1 accuracy on val, not a loss-threshold fraction.

        Returns None when:
          - no val_loader attached (server-side override falls back to BVD)
          - the underlying client exposes no .model handle (future variants)
          - the val loader produces no (x, y) pairs
        Device handling: moves (x, y) to the model's device before each
        forward so GPU/MPS runs do not silently throw inside the forward
        and get swallowed by the broad except.
        """
        if self.val_loader is None:
            return None
        # Underlying client (honest / attacker variants) exposes its model
        # directly on .model. Give up silently if a future variant lacks
        # that handle so the pipeline keeps running with BVD fallback.
        model = getattr(self.underlying_client, "model", None)
        if model is None:
            return None
        model.eval()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            # Model with no parameters -- degenerate case; fall back silently.
            return None
        correct = 0
        total = 0
        import torch as _torch
        with _torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                else:
                    continue
                try:
                    x = x.to(device)
                    y = y.to(device)
                    out = model(x)
                except Exception:
                    return None
                pred = out.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.shape[0])
        if total == 0:
            return None
        return correct / total

    def _prepare_fit_metrics(self, num_examples: int) -> Dict[str, Scalar]:
        """Prepare metrics to send back to server."""
        metrics = {
            "client_id": self.config.client_id,
            "client_type": self.config.client_type,
            "tavs_assignment": self.current_assignment,
            "trust_score": self.trust_score,
            "tier": self.tier,
            "round": self.round_number,
            "num_examples": num_examples,
            "is_verified": getattr(self, "_last_is_verified", True),
        }

        # Add attack-specific metrics if applicable
        if self.config.client_type != "honest":
            metrics["attack_intensity"] = self.config.attack_intensity

            if self.config.client_type == "layerwise":
                metrics["target_fraction"] = self.config.target_fraction

        return metrics

    def get_tavs_statistics(self) -> Dict[str, Any]:
        """Get comprehensive TAVS client statistics."""
        return {
            "client_id": self.config.client_id,
            "client_type": self.config.client_type,
            "current_trust_score": self.trust_score,
            "current_tier": self.tier,
            "current_assignment": self.current_assignment,
            "total_rounds": len(self.assignment_history),
            "verification_count": sum(1 for h in self.assignment_history if h["assignment"] == "verified"),
            "promotion_count": sum(1 for h in self.assignment_history if h["assignment"] == "promoted"),
            "decoy_count": sum(1 for h in self.assignment_history if h.get("is_decoy", False)),
            "assignment_history": self.assignment_history[-10:],  # Last 10 rounds
            "training_history": self.training_history[-5:]  # Last 5 rounds
        }


def create_tavs_flower_client(config: TAVSClientConfig,
                             train_loader = None,
                             test_loader = None,
                             partition_id: int = None,
                             val_loader = None) -> TAVSFlowerClient:
    """
    Factory function to create TAVS Flower clients.

    Args:
        config: Client configuration
        train_loader: Training data loader
        test_loader: Test data loader (optional)
        partition_id: Partition index for cross-arm cohort alignment
            (see TAVSFlowerClient.__init__).
        val_loader: Held-out local val loader for small_loss_fraction
            (see TAVSFlowerClient.__init__).

    Returns:
        Configured TAVSFlowerClient instance
    """
    return TAVSFlowerClient(
        config=config,
        train_loader=train_loader,
        test_loader=test_loader,
        partition_id=partition_id,
        val_loader=val_loader,
    )