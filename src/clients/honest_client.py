import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from typing import Dict, List, Tuple, Optional
import numpy as np
from ..core.models import get_model
from ..utils.data_utils import create_dataloaders


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
        params_dict = zip(self.model.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param, dtype=param.dtype).to(self.device)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model locally."""
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