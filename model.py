# Import necessary libraries
import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet34



#### For the ResNet tree classification model which incorporates a single data type (either 2D-LiDAR or Streetview images) ####
class ResNetTreeClassifier(nn.Module): # nn.Module is the base class for all neural network modules
    def __init__(self, num_classes):
        super().__init__()
        base_model = resnet34(weights="DEFAULT")
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Adds all layers of the base model besides the final FC
        self.classifier = nn.Linear(512, num_classes) # replace last FC of ResNet with a linear transformation

    def forward(self, x):
        # x : (B, N, C, H, W)
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        feats = self.feature_extractor(x)  # (B*N, 512, 1, 1)
        feats = feats.view(B, N, -1)	   # (B, N, 512)
        feats = feats.mean(dim=1)

        return self.classifier(feats)



#### For the combined ResNet tree classification model which incorporates the 2D-LiDAR data and the Streetview data ####
class DualResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Linear(512*2, num_classes)
 
        # --- Branch 1 ---
        base1 = resnet34(weights="DEFAULT")
        base1.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False) # change the first Conv layer so it accepts 6 channels
        self.feat1 = nn.Sequential(*list(base1.children())[:-1])  # -> (B,512,1,1)
 
        # --- Branch 2 ---
        base2 = resnet34(weights="DEFAULT")
        base2.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.feat2 = nn.Sequential(*list(base2.children())[:-1])
 
    @staticmethod # this is a function that is not beholden to the class object
    def _encode_branch(x, feat):
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        z = feat(x)                # (B*N, 512, 1, 1)
        z = z.view(B, N, -1)       # (B, N, 512)
        z = z.mean(dim=1)          # agrégation multi-vues -> (B, 512)
        return z
 
    def forward(self, x1, x2):
        z1 = self._encode_branch(x1, self.feat1)
        z2 = self._encode_branch(x2, self.feat2)
        z = torch.cat([z1, z2], dim=1)   # (B, 1024)
        predictions = self.classifier(z)  # (B, num_classes)
        return predictions 
