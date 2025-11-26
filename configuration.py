# TRAINING AND VALIDATION PARAMETERS
# Import necessary libraries
import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter
import sys

# === General parameters ===
# This includes root directory path, batch size, device to be used, and mean + standard deviation of pixel values calculated in calculate_pixel_mean_std.py 
## EXAMPLE USE:
root_dir = "/media/vanessa/ZEFI_neuroa/PhD/data/point_clouds/2D-lidar_streetview_images/streetview_images_model" 
batch_size = 16 # Adjust as needed according to your GPU memory
image_size = (150,100) # height and width in pixels, respectively
mean_streetview=[0.4719216525554657, 0.47749724984169006, 0.446537047624588]
std_streetview=[0.28776127099990845, 0.28991416096687317, 0.31630629301071167]
mean_lidar=[0.870621919631958, 0.8944705128669739, 0.8733227849006653]
std_lidar=[0.30000773072242737, 0.18829095363616943, 0.2912716269493103]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === TRANSFORMATIONS ===
# Here specified the transformations to be done on the images (streetview regular, streetview for data augmentation purposes, and 2D-LIDAR, respectively)
# See data_loader.py for details on when/how these transformations are implemented
## EXAMPLE USE:
transform_streetview = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean_streetview, std_streetview)
])

transform_augment = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(p=1), # For the augmentated data, the probability of being flipped is 100%
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean_streetview, std_streetview)
])

transform_lidar = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean_lidar, std_lidar)
])

image_paths, class_to_idx = get_image_paths(root_dir) # get's the season specific folder and lidar folder that the images are in for each tree
train_list, val_list, test_list = split_dataset(image_paths)

print("Train classes:", Counter([lbl for _, _, lbl in train_list]))
print("Val classes:", Counter([lbl for _, _, lbl in val_list]))
print("Test classes:", Counter([lbl for _, _, lbl in test_list]))

train_dataset = TreeDataset(train_list, class_to_idx, data_source="both", transform_streetview=transform_streetview, transform_augment=transform_augment, transform_lidar=transform_lidar)
val_dataset = TreeDataset(val_list, class_to_idx, data_source="both", transform_streetview=transform_streetview, transform_augment=transform_augment, transform_lidar=transform_lidar)
test_dataset = TreeDataset(test_list, class_to_idx, data_source="both", transform_streetview=transform_streetview, transform_augment=transform_augment, transform_lidar=transform_lidar)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Class_weights handles class imbalance
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique([lbl for _, _, lbl in train_list]),
    y=[lbl for _, _, lbl in train_list]
)

weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

model = DualResNetClassifier(num_classes=len(class_to_idx)).to(device)
#model = ResNetTreeClassifier(num_classes=len(class_to_idx)).to(device) # one would use this model if data source is singular (i.e. either 2D-LiDAR or Streetview only)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
