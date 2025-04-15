import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)


def generate_sample_data(num_samples):
    # Generate random features
    features = np.random.randn(num_samples, 5)

    # Generate labels based on a simple rule
    # If sum of features is positive, label is 1, else 0
    labels = (np.sum(features, axis=1) > 0).astype(float)

    # Combine features and labels
    data = np.column_stack((features, labels))

    # Create DataFrame
    columns = [f"feature_{i+1}" for i in range(5)] + ["label"]
    df = pd.DataFrame(data, columns=columns)

    return df


os.makedirs("./data", exist_ok=True)

# Generate training data
train_df = generate_sample_data(1000)
train_df.to_csv("./data/train.csv", index=False)

# Generate validation data
val_df = generate_sample_data(200)
val_df.to_csv("./data/validate.csv", index=False)
