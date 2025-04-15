import torch
import torch.nn as nn
import torch.optim as optim
import os

from model import LogisticMLP
from dataloader import get_train_loader, get_val_loader
from training_manager import TrainingManager, EarlyStopping
from vis_utils import visualize_model_performance

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Hyperparameters
    input_dim = 5
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32

    # Initialize model, criterion, and optimizer
    model = LogisticMLP(input_dim=input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(7, 0)

    # Get data loaders
    train_loader = get_train_loader(batch_size=batch_size)
    val_loader = get_val_loader(batch_size=batch_size)

    # Initialize training manager
    trainer = TrainingManager(
        model,
        optimizer,
        criterion,
        early_stopping,
        logdir="runs/binary_classification_experiment",
        monitor="val_loss",
        device=device,
    )
    trainer.train(num_epochs, train_loader, val_loader)
    model.load_state_dict(trainer.get_best_model())

    # Graph confusion matrix + AUC
    cm_fig, roc_fig = visualize_model_performance(model, val_loader, device)
    os.makedirs("./figs", exist_ok=True)
    cm_fig.savefig("./figs/confusion_matrix.png")
    roc_fig.savefig("./figs/ROC_curve.png")


if __name__ == "__main__":
    main()
