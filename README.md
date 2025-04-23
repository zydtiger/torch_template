# Neural Network Training Framework

This project implements a modular neural network training framework with comprehensive training management, visualization capabilities, and custom data handling. The framework is designed to be flexible and easy to use for various machine learning tasks.

## Project Structure

1. `create_sample_data.py`

- Generates sample data for testing and demonstration
- Creates training and validation datasets
- Configurable number of samples
- Outputs data files required for training

2. `model.py`

- Implements the `LogisticMLP` neural network model
- Includes:
  - Configurable input dimensions
  - Simple MLP architecture for classification tasks
  - Forward pass implementation

3. `dataloader.py`

- Implements custom dataset handling functionality
- Includes:
  - `CustomDataset` class for data management
  - Utility functions for creating train and validation data loaders
  - Configurable batch size (default: 32)

4. `training_manager.py`

- Comprehensive training management system
- Key components:
  - `TrainingManager` class for orchestrating the training process
  - `EarlyStopping` implementation for preventing overfitting
  - TensorBoard logging integration
  - Support for different monitoring metrics (validation loss/accuracy)

5. `vis_utils.py`

- Visualization utilities for model evaluation
- Includes:
  - Confusion matrix plotting
  - ROC curve visualization
  - Model performance evaluation functions
  - Support for binary classification visualization

6. `main.py`

- Entry point for the training pipeline
- Orchestrates the entire training process

## Setup and Usage

1. First, generate the sample data:

```bash
python create_sample_data.py
```

2. Then train the model:

```bash
python main.py
```

This will:

- Initialize the model and training components
- Execute the training loop
- Generate performance visualizations

## Output Files

The training process generates:

- Confusion matrix plot (`figs/confusion_matrix.png`)
- ROC curve plot (`figs/ROC_curve.png`)
- TensorBoard logs in `runs/binary_classification_experiment/`

## Requirements

- PyTorch
- NumPy
- Matplotlib
- TensorBoard
- tqdm

## Data Format

The framework expects data in a simple matrix format where:

- Each row represents one sample
- The last column contains the labels (0 or 1 for binary classification)
- All other columns contain the feature values
