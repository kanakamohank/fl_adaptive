import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from ..core.models import get_model
from ..utils.data_utils import create_dataloaders

logger = logging.getLogger(__name__)


class HonestClient(NumPyClient):
    """Honest client implementation for federated learning."""

    def __init__(self, client_id: str, model_type: str, model_kwargs: Dict,
                 train_loader: DataLoader, test_loader: DataLoader,
                 device: str = "cpu", local_epochs: int = 5, lr: float = 0.01):
        self.client_id = client_id
        self.device = torch.device(device)
        self.local_epochs = local_epochs
        self.lr = lr

        # Initialize model
        self.model = get_model(model_type, **model_kwargs).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

        # Data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Track training metrics
        self.training_history = []

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get current model parameters."""
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters."""
        if not parameters:
            return

        params_dict = zip(self.model.parameters(), parameters)
        for idx, (param, new_param) in enumerate(params_dict):
            try:
                logger.debug(f"Setting parameter {idx}: model_param_shape={param.shape}, new_param_type={type(new_param)}, new_param_shape={getattr(new_param, 'shape', 'N/A')}")

                # Ensure new_param is a numpy array
                if not isinstance(new_param, np.ndarray):
                    logger.debug(f"Parameter {idx} is not numpy array, converting from {type(new_param)}")
                    if hasattr(new_param, 'numpy'):
                        new_param = new_param.numpy()
                    else:
                        new_param = np.array(new_param)
                    logger.debug(f"Converted parameter {idx} to numpy: shape={new_param.shape}, dtype={new_param.dtype}")

                # Convert to tensor with proper type and device
                logger.debug(f"Converting parameter {idx} to tensor: target_dtype={param.dtype}, device={self.device}")
                tensor_param = torch.tensor(new_param, dtype=param.dtype)
                logger.debug(f"Tensor created successfully for parameter {idx}")
                param.data = tensor_param.to(self.device)
                logger.debug(f"Parameter {idx} set successfully")
            except Exception as e:
                logger.error(f"Error setting parameter {idx}: {e}")
                logger.error(f"  Model param shape: {param.shape}")
                logger.error(f"  New param type: {type(new_param)}")
                logger.error(f"  New param content: {new_param}")
                raise

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model locally."""
        # DEBUG: Log parameters received in HonestClient
        logger.debug(f"HonestClient {self.client_id}: RECEIVED PARAMETERS - "
                    f"type: {type(parameters)}, length: {len(parameters) if hasattr(parameters, '__len__') else 'N/A'}")
        if hasattr(parameters, '__len__') and len(parameters) > 0:
            for i, param in enumerate(parameters):
                logger.debug(f"  HonestClient param {i}: type={type(param)}, shape={getattr(param, 'shape', 'N/A')}, dtype={getattr(param, 'dtype', 'N/A')}")
                if hasattr(param, '__len__') and len(param) > 0:
                    logger.debug(f"    First element type: {type(param.flat[0]) if hasattr(param, 'flat') else 'N/A'}")

        # Set parameters from server
        self.set_parameters(parameters)

        # Store initial parameters to compute update
        initial_params = [param.clone() for param in self.model.parameters()]

        # Local training
        self.model.train()
        epoch_losses = []

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Convert target to tensor if it's not already, preserving batch dimension
                if not isinstance(target, torch.Tensor):
                    if hasattr(target, '__len__'):  # List or array
                        target = torch.tensor(target, dtype=torch.long)
                    else:  # Single scalar
                        target = torch.tensor([target], dtype=torch.long)
                elif target.dim() == 0:  # 0-dimensional tensor (scalar)
                    target = target.unsqueeze(0)

                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)

        # Compute parameter update (for analysis)
        final_params = list(self.model.parameters())
        param_updates = []
        for initial, final in zip(initial_params, final_params):
            update = final - initial
            param_updates.append(update.cpu().detach().numpy())

        # Store training metrics
        training_metrics = {
            "client_id": self.client_id,
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "num_examples": len(self.train_loader.dataset),
        }
        self.training_history.append(training_metrics)

        return self.get_parameters({}), len(self.train_loader.dataset), training_metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model."""
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                # Convert target to tensor if it's not already, preserving batch dimension
                if not isinstance(target, torch.Tensor):
                    if hasattr(target, '__len__'):  # List or array
                        target = torch.tensor(target, dtype=torch.long)
                    else:  # Single scalar
                        target = torch.tensor([target], dtype=torch.long)
                elif target.dim() == 0:  # 0-dimensional tensor (scalar)
                    target = target.unsqueeze(0)

                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0

        metrics = {
            "client_id": self.client_id,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        return avg_loss, total, metrics

    def get_model_update(self, initial_params: List[np.ndarray]) -> List[np.ndarray]:
        """Compute model update (difference from initial parameters)."""
        current_params = self.get_parameters({})
        updates = []
        for current, initial in zip(current_params, initial_params):
            updates.append(current - initial)
        return updates

    def get_model_weights_flat(self) -> np.ndarray:
        """Get model weights as a flat numpy array."""
        return self.model.get_weights_flat().cpu().detach().numpy()

    def set_model_weights_flat(self, weights_flat: np.ndarray):
        """Set model weights from a flat numpy array."""
        weights_tensor = torch.tensor(weights_flat, dtype=torch.float32).to(self.device)
        self.model.set_weights_flat(weights_tensor)


def create_honest_client(client_id: str, model_type: str, model_kwargs: Dict,
                        train_data, test_data, device: str = "cpu",
                        batch_size: int = 32, local_epochs: int = 5,
                        lr: float = 0.01) -> HonestClient:
    """Factory function to create an honest client."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return HonestClient(
        client_id=client_id,
        model_type=model_type,
        model_kwargs=model_kwargs,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        local_epochs=local_epochs,
        lr=lr
    )