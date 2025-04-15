import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: str
) -> tuple[list[float], list[float], list[float]]:
    """
    Evaluate the model and return true labels, predicted probabilities and predicted labels
    """
    model.eval()
    true_labels = []
    pred_probs = []
    pred_labels = []

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            # Convert outputs to numpy for sklearn metrics
            pred_probs.extend(outputs.cpu().numpy().flatten())
            pred_labels.extend((outputs > 0.5).float().cpu().numpy().flatten())
            true_labels.extend(targets.cpu().numpy().flatten())

    return true_labels, pred_probs, pred_labels


def plot_confusion_matrix(
    true_labels: list[float],
    pred_labels: list[float],
    class_names=["Class 0", "Class 1"],
) -> matplotlib.figure.Figure:
    """
    Plot confusion matrix with metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 16},
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add metrics text box
    plt.figtext(
        0.5,
        0.03,
        f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}",
        ha="center",
        fontsize=10,
        bbox=dict(
            facecolor="white",
            boxstyle="round,pad=0.3",
            edgecolor="black",
            linewidth=1,
        ),
    )

    plt.subplots_adjust(bottom=0.13)
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))

    return plt.gcf()


def plot_roc_curve(
    true_labels: list[float], pred_probs: list[float]
) -> matplotlib.figure.Figure:
    """
    Plot ROC curve with AUC score
    """
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    auc = roc_auc_score(true_labels, pred_probs)

    plt.figure(figsize=(8, 6))

    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=2)  # reference line

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()

    return plt.gcf()


def visualize_model_performance(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    class_names=["Class 0", "Class 1"],
) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """
    Comprehensive model evaluation and visualization
    """
    true_labels, pred_probs, pred_labels = evaluate_model(
        model, data_loader, device
    )
    cm_fig = plot_confusion_matrix(true_labels, pred_labels, class_names)
    roc_fig = plot_roc_curve(true_labels, pred_probs)

    return cm_fig, roc_fig
