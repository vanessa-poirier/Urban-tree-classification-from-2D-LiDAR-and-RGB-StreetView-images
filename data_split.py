# Import necessary libraries
import os
import random

# Define split into training, validation, and testing sets
def split_dataset(image_paths, split_ratio=(0.6, 0.20, 0.20), seed=42): # Seed for reproducibility and training/validation/test split
    random.seed(seed)
    random.shuffle(image_paths)
    n = len(image_paths)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    return image_paths[:n_train], image_paths[n_train:n_train+n_val], image_paths[n_train+n_val:]
