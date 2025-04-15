import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing import Literal


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=7, min_delta=0):
        """
        patience (int): How many epochs to wait before stopping when loss is not improving
        min_delta (float): Minimum change in monitored value to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model = None

    def __call__(self, current_value: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = current_value
            self.best_model = self._get_model_copy(model)
            return False

        if current_value < self.best_loss - self.min_delta:
            self.best_loss = current_value
            self.counter = 0
            self.best_model = self._get_model_copy(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False

    def _get_model_copy(self, model: nn.Module) -> dict:
        """Returns a deep copy of the model state"""
        return {
            key: val.cpu().clone().detach()
            for key, val in model.state_dict().items()
        }


class TrainingManager:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        early_stopping: EarlyStopping,
        logdir: str,
        monitor: Literal["val_loss"] | Literal["val_acc"],
        device: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(logdir)
        self.model.to(device)

        # Early stopping setup
        self.monitor = monitor
        self.early_stopping = early_stopping

    def train(
        self, num_epochs: int, train_loader: DataLoader, val_loader: DataLoader
    ):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader, epoch)

            # Check early stopping
            monitor_value = val_loss if self.monitor == "val_loss" else val_acc
            if self.early_stopping(monitor_value, self.model):
                print(f"\nEarly stopping triggered after epoch {epoch}")
                break

    def train_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            pbar.set_postfix(
                {
                    "loss": running_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                }
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        # Log metrics to tensorboard
        self.writer.add_scalar("Training Loss", epoch_loss, epoch)
        self.writer.add_scalar("Training Accuracy", epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def validate(
        self, val_loader: DataLoader, epoch: int
    ) -> tuple[float, float]:
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        # Log validation metrics to tensorboard
        self.writer.add_scalar("Validation Loss", val_loss, epoch)
        self.writer.add_scalar("Validation Accuracy", val_acc, epoch)

        return val_loss, val_acc

    def get_best_model(self) -> dict:
        if self.early_stopping.best_model is None:
            raise Exception("Model not trained!")

        return self.early_stopping.best_model
