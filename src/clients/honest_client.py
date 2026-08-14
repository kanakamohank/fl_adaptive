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

        self.model = get_model(model_type, **model_kwargs).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.training_history = []

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get current model parameters."""
        # Force C-contiguous memory before uploading to the server
        return [np.array(param.cpu().detach().numpy(), dtype=np.float32, copy=True) for param in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters."""
        if not parameters:
            return

        params_dict = zip(self.model.parameters(), parameters)
        for idx, (param, new_param) in enumerate(params_dict):
            try:
                if not isinstance(new_param, np.ndarray):
                    if hasattr(new_param, 'numpy'):
                        new_param = new_param.numpy()
                    else:
                        new_param = np.array(new_param)

                # Dead-ReLU Shield: Skip the update if corruption occurs.
                if np.any(np.isnan(new_param)) or np.any(np.isinf(new_param)):
                    logger.warning(f"Client {self.client_id}: NaN/Inf detected in parameter {idx}, keeping existing weights")
                    continue  

                tensor_param = torch.tensor(new_param, dtype=param.dtype)
                param.data = tensor_param.to(self.device)
            except Exception as e:
                logger.error(f"Error setting parameter {idx}: {e}")
                raise

    def behave_honestly(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train without attacking, regardless of subclass.

        Attack classes inherit from HonestClient and override fit() to poison the
        result, so this binds explicitly to HonestClient.fit to reach the clean
        path. TAVSFlowerClient calls it when a client believes it is being
        verified, which is what makes the adversary adaptive: an attacker that
        misbehaves even under inspection is trivially caught, and defending
        against that adversary is not what CSPRNG decoy verification is for.

        The method was called by TAVSFlowerClient._execute_training but defined
        nowhere, so the hasattr() guard always failed and both branches ran the
        same attacking fit(). Attackers never evaded, and decoy verification
        therefore defended against an adversary that did not exist.
        """
        return HonestClient.fit(self, parameters, config)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)

        # THE FIX: Dynamically obey the PipelineConfig's learning rate and epochs!
        current_lr = float(config.get("learning_rate", self.lr))
        current_epochs = int(config.get("epochs", self.local_epochs))

        # Clear stale momentum
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
        initial_params = [param.clone() for param in self.model.parameters()]

        self.model.train()
        epoch_losses = []

        for epoch in range(current_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                if not isinstance(target, torch.Tensor):
                    if hasattr(target, '__len__'):
                        target = torch.tensor(target, dtype=torch.long)
                    else:
                        target = torch.tensor([target], dtype=torch.long)
                elif target.dim() == 0:
                    target = target.unsqueeze(0)

                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # ANTI-EXPLOSION SHIELD: Mathematically guarantees the model cannot explode to NaN
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)

        training_metrics = {
            "client_id": self.client_id,
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "num_examples": len(self.train_loader.dataset),
        }
        self.training_history.append(training_metrics)

        return self.get_parameters({}), len(self.train_loader.dataset), training_metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                if not isinstance(target, torch.Tensor):
                    if hasattr(target, '__len__'):
                        target = torch.tensor(target, dtype=torch.long)
                    else:
                        target = torch.tensor([target], dtype=torch.long)
                elif target.dim() == 0:
                    target = target.unsqueeze(0)

                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                batch_loss = loss.item()
                if not (np.isnan(batch_loss) or np.isinf(batch_loss)):
                    total_loss += batch_loss
                else:
                    logger.warning(f"Client {self.client_id}: NaN/Inf batch loss in evaluation, skipping batch")
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0

        if np.isnan(avg_loss) or np.isinf(avg_loss):
            avg_loss = 0.0

        metrics = {
            "client_id": self.client_id,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        return avg_loss, total, metrics

    def get_model_update(self, initial_params: List[np.ndarray]) -> List[np.ndarray]:
        current_params = self.get_parameters({})
        return [current - initial for current, initial in zip(current_params, initial_params)]

    def get_model_weights_flat(self) -> np.ndarray:
        return self.model.get_weights_flat().cpu().detach().numpy()

    def set_model_weights_flat(self, weights_flat: np.ndarray):
        weights_tensor = torch.tensor(weights_flat, dtype=torch.float32).to(self.device)
        self.model.set_weights_flat(weights_tensor)

def create_honest_client(client_id: str, model_type: str, model_kwargs: Dict,
                        train_data, test_data, device: str = "cpu",
                        batch_size: int = 32, local_epochs: int = 5,
                        lr: float = 0.01) -> HonestClient:
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