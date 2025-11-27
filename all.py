## This script combines all py files to run the ResNet34 tree classification model from computing pixel averages to model evaluation
# Import libraries
import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision.models import resnet34
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from collections import Counter
import sys

image_size = (150, 100)

# Define function to compute means and standard deviations of pixel values RGB across the whole dataset (i.e. all trees in all classes)
def compute_mean_std(root_dir, resize=(150,100), data_source="both", max_images=5000):
    """
    Calculate means and standard deviations of pixel values RGB across the whole dataset (i.e. all trees in all classes).
    
    Args:
        root_dir: String. Path containing the data (i.e. images) organized by class
        resize: Tuple. Desired size of the images (H, W).
        data_source: Character specifying whether the 2D-LiDAR images (lidar), Streetview images (street), or both are used. Accepted values are "lidar", "street", and "both"
        max_images (int): maximum number of images to use for the caluclations.

    Returns:
        tuple: (mean, std), each as a list of 3 values (R, G, B)
    """
    transform = transforms.Compose([
        transforms.Resize(resize), # resize is set to the actually size of the image (150 height and 100 width)
        transforms.ToTensor()
    ])

    all_pixels = []
    images_per_tree = []

    count = 0
    for class_dir in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        if data_source == "lidar":
            for tree_id in os.listdir(class_path):
                lidar_dir = os.path.join(class_path, tree_id, "lidar")
                if os.path.isdir(lidar_dir) and (len([name for name in os.listdir(lidar_dir)])>0):
                    images_per_tree.append(len([name for name in os.listdir(lidar_dir)]))
                    for img_file in os.listdir(lidar_dir):
                        img_path = os.path.join(lidar_dir, img_file)
                        try:
                            img = default_loader(img_path)
                            tensor = transform(img)  # (3, H, W)
                            all_pixels.append(tensor.view(3, -1))  # flatten H×W
                            count += 1
                            if max_images and count >= max_images:
                                raise StopIteration
                        except:
                            continue
            
            # concat (N, 3, P) → (3, total_pixels)
            max_images_per_tree = max(images_per_tree)
            all_pixels = torch.cat(all_pixels, dim=1)
            mean_lidar = all_pixels.mean(dim=1).tolist()
            std_lidar = all_pixels.std(dim=1).tolist()
            return mean_lidar, std_lidar, max_images_per_tree


        if data_source == "street":
            for tree_id in os.listdir(class_path):
                primary_image_dir = os.path.join(class_path, tree_id, "scan_date")
                secondary_image_dir = os.path.join(class_path, tree_id, "automn")
                tertiary_image_dir = os.path.join(class_path, tree_id, "summer")
                if os.path.isdir(primary_image_dir) and (len([name for name in os.listdir(primary_image_dir)])>0):
                    images_per_tree.append(len([name for name in os.listdir(primary_image_dir)]))
                    for img_file in os.listdir(primary_image_dir):
                        img_path = os.path.join(primary_image_dir, img_file)
                        try:
                            img = default_loader(img_path)
                            tensor = transform(img)  # (3, H, W)
                            all_pixels.append(tensor.view(3, -1))  # flatten H×W
                            count += 1
                            if max_images and count >= max_images:
                                raise StopIteration
                        except:
                            continue
                elif os.path.isdir(secondary_image_dir)and (len([name for name in os.listdir(secondary_image_dir)])>0):
                    images_per_tree.append(len([name for name in os.listdir(secondary_image_dir)]))
                    for img_file in os.listdir(secondary_image_dir):
                        img_path = os.path.join(secondary_image_dir, img_file)
                        try:
                            img = default_loader(img_path)
                            tensor = transform(img)  # (3, H, W)
                            all_pixels.append(tensor.view(3, -1))  # flatten H×W
                            count += 1
                            if max_images and count >= max_images:
                                raise StopIteration
                        except:
                            continue 
                elif os.path.isdir(tertiary_image_dir) and (len([name for name in os.listdir(tertiary_image_dir)])>0):
                    images_per_tree.append(len([name for name in os.listdir(tertiary_image_dir)]))
                    for img_file in os.listdir(tertiary_image_dir):
                        img_path = os.path.join(tertiary_image_dir, img_file)
                        try:
                            img = default_loader(img_path)
                            tensor = transform(img)  # (3, H, W)
                            all_pixels.append(tensor.view(3, -1))  # flatten H×W
                            count += 1
                            if max_images and count >= max_images:
                                raise StopIteration
                        except:
                            continue
                else:
                    print("No photos available for this tree")
                    continue
            
            # concat (N, 3, P) → (3, total_pixels)
            max_images_per_tree = max(images_per_tree)
            all_pixels = torch.cat(all_pixels, dim=1)
            mean_streetview = all_pixels.mean(dim=1).tolist()
            std_streetview = all_pixels.std(dim=1).tolist()
            return mean_streetview, std_streetview, max_images_per_tree

        
        if data_source == "both":
            all_pixels_lidar = []
            all_pixels_street = []
            images_per_tree_lidar = []
            images_per_tree_street = []

            for tree_id in os.listdir(class_path):
                lidar_dir = os.path.join(class_path, tree_id, "lidar")
                if os.path.isdir(lidar_dir) and (len([name for name in os.listdir(lidar_dir)])>0):
                    images_per_tree_lidar.append(len([name for name in os.listdir(lidar_dir)]))
                    for img_file in os.listdir(lidar_dir):
                        img_path = os.path.join(lidar_dir, img_file)
                        try:
                            img = default_loader(img_path)
                            tensor = transform(img)  # (3, H, W)
                            all_pixels_lidar.append(tensor.view(3, -1))  # flatten H×W
                            count += 1
                            if max_images and count >= max_images:
                                raise StopIteration
                        except:
                            continue
                            
            for tree_id in os.listdir(class_path):
                primary_image_dir = os.path.join(class_path, tree_id, "scan_date")
                secondary_image_dir = os.path.join(class_path, tree_id, "automn")
                tertiary_image_dir = os.path.join(class_path, tree_id, "summer")
                if os.path.isdir(primary_image_dir) and (len([name for name in os.listdir(primary_image_dir)])>0):
                    images_per_tree_street.append(len([name for name in os.listdir(primary_image_dir)]))
                    for img_file in os.listdir(primary_image_dir):
                        img_path = os.path.join(primary_image_dir, img_file)
                        try:
                            img = default_loader(img_path)
                            tensor = transform(img)  # (3, H, W)
                            all_pixels_street.append(tensor.view(3, -1))  # flatten H×W
                            count += 1
                            if max_images and count >= max_images:
                                raise StopIteration
                        except:
                            continue
                elif os.path.isdir(secondary_image_dir)and (len([name for name in os.listdir(secondary_image_dir)])>0):
                    images_per_tree_street.append(len([name for name in os.listdir(secondary_image_dir)]))
                    for img_file in os.listdir(secondary_image_dir):
                        img_path = os.path.join(secondary_image_dir, img_file)
                        try:
                            img = default_loader(img_path)
                            tensor = transform(img)  # (3, H, W)
                            all_pixels_street.append(tensor.view(3, -1))  # flatten H×W
                            count += 1
                            if max_images and count >= max_images:
                                raise StopIteration
                        except:
                            continue 
                elif os.path.isdir(tertiary_image_dir) and (len([name for name in os.listdir(tertiary_image_dir)])>0):
                    images_per_tree_street.append(len([name for name in os.listdir(tertiary_image_dir)]))
                    for img_file in os.listdir(tertiary_image_dir):
                        img_path = os.path.join(tertiary_image_dir, img_file)
                        try:
                            img = default_loader(img_path)
                            tensor = transform(img)  # (3, H, W)
                            all_pixels_street.append(tensor.view(3, -1))  # flatten H×W
                            count += 1
                            if max_images and count >= max_images:
                                raise StopIteration
                        except:
                            continue
                else:
                    #print("No photos available for this tree")
                    continue
            
            # concat (N, 3, P) → (3, total_pixels)
            max_images_per_tree_lidar = max(images_per_tree_lidar)
            all_pixels_lidar = torch.cat(all_pixels_lidar, dim=1)
            mean_lidar = all_pixels_lidar.mean(dim=1).tolist()
            std_lidar = all_pixels_lidar.std(dim=1).tolist()
            
            max_images_per_tree_street = max(images_per_tree_street)
            all_pixels_street = torch.cat(all_pixels_street, dim=1)
            mean_streetview = all_pixels_street.mean(dim=1).tolist()
            std_streetview = all_pixels_street.std(dim=1).tolist()
            return mean_lidar, std_lidar, max_images_per_tree_lidar, mean_streetview, std_streetview, max_images_per_tree_street




# Define function to get image paths
def get_image_paths(root_dir, data_source): # Gets all streetview and lidar image paths and their corresponding class names
    """
        Args:
            root_dir: Character showing the directory which contains all class directories
            data_source: Character specifying whether the 2D-LiDAR images (lidar), Streetview images (street), or both should be loaded
                         Accepted values are "lidar", "street", and "both"
    """
    image_paths = [] # will be a list
    class_to_idx = {} # will be a dictionary
    for i, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_to_idx[class_name] = i

        if data_source == "lidar":
            for tree_id in os.listdir(class_dir):
               lidar_image_dir = os.path.join(class_dir, tree_id, "lidar")
               if os.path.isdir(lidar_image_dir):
                  image_paths.append(lidar_image_dir, class_name)

        elif data_source == "street":
            for tree_id in os.listdir(class_dir):
               primary_image_dir = os.path.join(class_dir, tree_id, "scan_date")
               secondary_image_dir = os.path.join(class_dir, tree_id, "automn")
               tertiary_image_dir = os.path.join(class_dir, tree_id, "summer")

               if os.path.isdir(primary_image_dir) and (len([name for name in os.listdir(primary_image_dir)])>0):
                   image_path = primary_image_dir
                   image_paths.append((image_path, class_name))
               elif os.path.isdir(secondary_image_dir) and (len([name for name in os.listdir(secondary_image_dir)])>0):
                   image_path = secondary_image_dir
                   image_paths.append((image_path, class_name))
               elif os.path.isdir(tertiary_image_dir) and (len([name for name in os.listdir(tertiary_image_dir)])>0):
                   image_path = tertiary_image_dir
                   image_paths.append((image_path, class_name))
               else:
                   #print("No photos available for this tree")
                   continue
                   
        elif data_source == "both":
            for tree_id in os.listdir(class_dir):
               lidar_image_dir = os.path.join(class_dir, tree_id, "lidar")
               primary_image_dir = os.path.join(class_dir, tree_id, "scan_date") # preferentially get images from scan date, otherwise from automn, otherwise from summer
               secondary_image_dir = os.path.join(class_dir, tree_id, "automn")
               tertiary_image_dir = os.path.join(class_dir, tree_id, "summer")
           
               if os.path.isdir(primary_image_dir) and (len([name for name in os.listdir(primary_image_dir)])>0):
                   image_paths.append((primary_image_dir, lidar_image_dir, class_name))
               elif os.path.isdir(secondary_image_dir) and (len([name for name in os.listdir(secondary_image_dir)])>0):
                   image_paths.append((secondary_image_dir, lidar_image_dir, class_name))
               elif os.path.isdir(tertiary_image_dir) and (len([name for name in os.listdir(tertiary_image_dir)])>0):
                   image_paths.append((tertiary_image_dir, lidar_image_dir, class_name))
               else:
                   #print("No photos available for this tree")
                   continue
    return image_paths, class_to_idx



# Split dataset into training, validation, and testing datasets. Note that each tree belongs exclusively to one dataset type
def split_dataset(image_paths, split_ratio=(0.6, 0.20, 0.20), seed=42): # Seed for reproducibility and training/validation/test split
    random.seed(seed)
    random.shuffle(image_paths)
    n = len(image_paths)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    return image_paths[:n_train], image_paths[n_train:n_train+n_val], image_paths[n_train+n_val:] # returns train_dataset, val_dataset, test_dataset



# Define the tree dataset to be used
class TreeDataset(Dataset):
    def __init__(self, tree_list, class_to_idx, data_source, transform_streetview=None, transform_augment=None, transform_lidar=None, max_images=10):
        """
        Args:
            tree_list (list): List of tuples (streetview_image_path, lidar_image_path, class_name)
            class_to_idx (dict): Dictionary {class_name: index_integer}
            data_source: Character specifying whether the 2D-LiDAR images (lidar), Streetview images (street), or both are used. Accepted values are "lidar", "street", and "both"
            transform_streetview: Transformation to apply to the streetview images
            transform_augment: Transformation to apply to the streetview images for data augmentation (until the total number of images per tree == max_images)
            transform_lidar: Transformation to apply to the 2D-LiDAR images
            max_images: Number of streetview images per tree. Data will be augmented (following transform_augment) until all trees have this number of streetview images
        """
        self.tree_list = tree_list
        self.class_to_idx = class_to_idx
        self.data_source = data_source
        self.transform_streetview = transform_streetview # because there are different means and stds to be used in transform for streetview
        self.transform_augment = transform_augment
        self.transform_lidar = transform_lidar           # and LiDAR
        self.max_images = max_images # the number of streetview images that I would like to have (total) per tree. Should correspond to streetview_max

    def __len__(self):
        return len(self.tree_list) # specifies that the length of a TreeDataset object will be = len(tree_list)

    def __getitem__(self, idx): # For Multi-view learning

        if self.data_source == "lidar":
            lidar_image_path, class_name = self.tree_list[idx] # gets the image folders and class name of each tree in tree_list
            label = self.class_to_idx[class_name] # label = dictionary item including class name and its index

            lidar_images = []
            for img_name in sorted(os.listdir(lidar_image_path)): # for every image in the lidar folder
                img_path = os.path.join(lidar_image_path, img_name) # get image path
                img = Image.open(img_path).convert("RGB") # open image (as an RGB image)
                if self.transform_lidar:
                    img = self.transform_lidar(img)  # transformation to perform according to the transform argument of the class
                if img.numel() > 0:   # keep only non-empty tensors
                    lidar_images.append(img) # add the image to the total list of images
            
            if len(lidar_images) == 0:
                return None
                
            lidar_images = torch.stack(lidar_images) # (N, C, H, W) stack all the images (i.e. views) from the same tree

            return lidar_images, label

        
        elif self.data_source == "street": 
            streetview_image_path, class_name = self.tree_list[idx] # gets the image folders and class name of each tree in tree_list
            label = self.class_to_idx[class_name] # label = dictionary item including class name and its index
            
            # for all trees that have < max_images, randomly augment until the number of streetview images = max_images
            streetview_images_augmented = []
            if len(os.listdir(streetview_image_path)) < self.max_images:
               num_to_augment = self.max_images - len(os.listdir(streetview_image_path))
               while len(streetview_images_augmented) < num_to_augment:
                  img_name = random.choice(os.listdir(streetview_image_path))  # pick random image
                  img_path = os.path.join(streetview_image_path, img_name) # get the image path
                  img = Image.open(img_path).convert("RGB") # open image (as an RGB image)
                  aug_img = self.transform_augment(img) # augment the image
                  streetview_images_augmented.append(aug_img) # save the augmented image

            # load all the original streetview images
            streetview_images_reg = []
            for img_name in sorted(os.listdir(streetview_image_path)): 
                img_path = os.path.join(streetview_image_path, img_name) # get image path
                img = Image.open(img_path).convert("RGB") # open image (as an RGB image)
                if self.transform_streetview:
                    img = self.transform_streetview(img)
                if img.numel() > 0:   # keep only non-empty tensors
                    streetview_images_reg.append(img) # add the image to the total list of images

            streetview_images = streetview_images_reg + streetview_images_augmented # combine the original and augmented images
            
            if len(streetview_images) == 0:
                return None
      
            streetview_images = torch.stack(streetview_images)  # (N, C, H, W) stack all the images (i.e. views) from the same tree

            return streetview_images, label

        
        elif self.data_source == "both": 
            streetview_image_path, lidar_image_path, class_name = self.tree_list[idx] # gets the image folders and class name of each tree in tree_list
            label = self.class_to_idx[class_name] # label = dictionary item including class name and its index

            # for all trees that have < max_images, randomly augment until the number of streetview images = max_images
            streetview_images_augmented = []
            if len(os.listdir(streetview_image_path)) < self.max_images:
               num_to_augment = self.max_images - len(os.listdir(streetview_image_path))
               while len(streetview_images_augmented) < num_to_augment:
                  img_name = random.choice(os.listdir(streetview_image_path))  # pick random image
                  img_path = os.path.join(streetview_image_path, img_name) # get the image path
                  img = Image.open(img_path).convert("RGB") # open image (as an RGB image)
                  aug_img = self.transform_augment(img) # augment the image
                  streetview_images_augmented.append(aug_img) # save the augmented image

            # load all the original streetview images
            streetview_images_reg = []
            for img_name in sorted(os.listdir(streetview_image_path)): 
                img_path = os.path.join(streetview_image_path, img_name) # get image path
                img = Image.open(img_path).convert("RGB") # open image (as an RGB image)
                if self.transform_streetview:
                    img = self.transform_streetview(img)
                if img.numel() > 0:   # keep only non-empty tensors
                    streetview_images_reg.append(img) # add the image to the total list of images

            streetview_images = streetview_images_reg + streetview_images_augmented # combine the original and augmented images
        
            lidar_images = []
            for img_name in sorted(os.listdir(lidar_image_path)): # for every image in the lidar folder
                img_path = os.path.join(lidar_image_path, img_name) # get image path
                img = Image.open(img_path).convert("RGB") # open image (as an RGB image)
                if self.transform_lidar:
                    img = self.transform_lidar(img)  # transformation to perform according to the transform argument of the class
                if img.numel() > 0:   # keep only non-empty tensors
                    lidar_images.append(img) # add the image to the total list of images

            # if there are no streetview or lidar images, return None
            if len(streetview_images) == 0:
                return None
        
            if len(lidar_images) == 0:
                return None

            streetview_images = torch.stack(streetview_images)  # (N, C, H, W) stack all the images (i.e. views) from the same tree
            lidar_images = torch.stack(lidar_images)

            return streetview_images, lidar_images, label # returns all images of 1 tree stacked on top of each other, along with the class (i.e. genus) info



## For the ResNet tree classification model which incorporates a single data type (either 2D-LiDAR or Streetview images) ##
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



## Training pipeline
def train_model(model, train_loader, val_loader, criterion, optimizer, device, type="dual",
                num_epochs=20, lr_decay=0.75, decay_every=5, patience=5):
    """
        Args:
            model: Trained pytorch model to be evaluated
            train_loader: Data Loader object containing the training data
            val_loader: Data Loader object containing the validation data
            criterion: Criterion used for model training
            optimizer: Optimizer used for model training
            device: CPU or GPU used for model training
            type: Character specifying whether the the model was trained with a single data source or with two data sources. Accepted values are "single" and "dual"
            num_epochs: Number of training epochs
            lr_decay: Learning rate decay
            decay_every: Number of epochs to run before reducing the learning rate
            patience: Number of epochs to wait before observing a minimal improvement in model accuracy
    """
                  
    print(device)
    best_model_wts = None
    best_val_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        if epoch % decay_every == 0 and epoch > 0:
            for g in optimizer.param_groups:
                g["lr"] *= lr_decay
            print(f"[Epoch {epoch}] ➤ Learning rate decayed.")

        # TRAIN
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if type == "single":
            for inputs, labels in train_loader:
               #print("loading training data")
                inputs = inputs.to(device)  # (B, N, C, H, W)
                labels = labels.to(device)

                optimizer.zero_grad()
                #print("about to train model")
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

        elif type == "dual":
            for inputs1, inputs2, labels in train_loader:
               #print("loading training data")
                inputs1 = inputs1.to(device)  # (B, N, C, H, W)
                inputs2 = inputs2.to(device)  
                labels = labels.to(device)

                optimizer.zero_grad()
                #print("about to train model")
                outputs = model(inputs1, inputs2)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        if type == "single":
            with torch.no_grad():
                for inputs, labels in val_loader:
                    #print("loading validation data")
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    #print("about to put validation data into model")
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

        elif type == "dual":
            with torch.no_grad():
                for inputs1, inputs2, labels in val_loader:
                    inputs1 = inputs1.to(device)
                    inputs2 = inputs2.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs1, inputs2)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs1.size(0)
                    preds = outputs.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} ➤ Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")

        # Early stopping
        # Not needed if you decrease the learning rate each N epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping.")
                break

    if best_model_wts:
        model.load_state_dict(best_model_wts)

    return model
                  


#### RUNNING FUNCTIONS ####
# === Calculate pixel means and stds ===
print("calculating pixel means and stds")
root_dir = "E:/PhD/data/point_clouds/lidar_streetview_images/streetview_images_model" 
mean_lidar, std_lidar, max_images_per_tree_lidar, mean_streetview, std_streetview, max_images_per_tree_street = compute_mean_std(root_dir, resize=image_size, data_source="both", max_images=None)

print(f"Mean LiDAR: {mean_lidar}, Mean Streetview: {mean_steetview}")
print(f"Std LiDAR:  {std_lidar}, Std Streetview: {std_steetview}")
print(f"Max LiDAR images per tree:  {max_images_per_tree_lidar}, Max Streetview images per tree:  {max_images_per_tree_street}")


# === General parameters ===
# This includes root directory path, batch size, device to be used, and mean + standard deviation of pixel values calculated in calculate_pixel_mean_std.py 
batch_size = 12 # Adjust as needed according to your GPU memory
image_size = (150,100) # height and width in pixels, respectively
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === TRANSFORMATIONS ===
# Here specified the transformations to be done on the images (streetview regular, streetview for data augmentation purposes, and 2D-LIDAR, respectively)
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

image_paths, class_to_idx = get_image_paths(root_dir, data_source) # get's the season specific folder and lidar folder that the images are in for each tree
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

# Train model
print("Model training starting now")
model = train_model(model, train_loader, val_loader, criterion, optimizer, device, type="dual") # function train model is defined in training_pipeline.py

## EXAMPLE USE FOR SAVING
dir = 'E:/PhD'
model_save_path = os.path.join(dir, "ResNet_results", "final_dualresnetmodel_2025-11-27.pth")
torch.save(model.state_dict(), model_save_path) 
print("Model saved")


# Define function to evaluate model
def evaluate_model_metrics(model, test_loader, class_names, device, type="dual"):
    """
        Args:
            model: trained pytorch model to be evaluated
            type: Character specifying whether the the model was trained with a single data source or with two data sources. Accepted values are "single" and "dual"
    """
  
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
      if type == "single":
        for inputs, labels in test_loader:
            inputs = inputs.to(device)  # (B, N, C, H, W)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

      elif type == "dual":
        for inputs1, inputs2, labels in test_loader:
            inputs1 = inputs1.to(device)  # (B, N, C, H, W)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)
            outputs = model(inputs1, inputs2)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    label_indices = list(range(len(class_names)))  # 0, 1, ..., n_classes-1

    precision, recall, fscore, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=label_indices,
        zero_division=0  # avoid division by zero warnings
        )
    
    cm = confusion_matrix(all_labels, all_preds, labels=label_indices) # confusion matrix
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    kappa = cohen_kappa_score(all_labels,
        all_preds,
        labels=label_indices
        )
    
    return(precision, recall, fscore, support, per_class_acc, acc, kappa)


# Run evaluation 100 times
print('evaluations starting')
evaluations = []

for _ in range(100):
    result = evaluate_model_metrics(model, type="dual", test_loader, class_names=list(class_to_idx.keys()), device=device)
    evaluations.append(result)

print('evaluations done')


# Calculate sums and standard deviations for each metric (per class)
class_names = ['Acer','Betula','Celtis','Fraxinus','Gleditsia','Malus','Other','Picea','Pinus','Populus','Quercus','Syringa','Thuja','Tilia','Ulmus']
num_classes = 15

precisions, recalls, fscores, supports, per_class_accs, accuracies, kappas = zip(*evaluations)

precisions_mat = np.stack(precisions) # STACK METRIC VECTORS INTO A MATRIX
recalls_mat = np.stack(recalls)
fscores_mat = np.stack(fscores)

precision_sum = []
recall_sum = []
fscore_sum = []

for i in range(num_classes):  # Iterate through all tests to get means and standard deviations
    mean = np.mean(precisions_mat[:,i])
    sd = np.std(precisions_mat[:,i])
    index = class_names[i]
    precision_sum.append((mean, sd, index))

    mean = np.mean(recalls_mat[:,i])
    sd = np.std(recalls_mat[:,i])
    index = class_names[i]
    recall_sum.append((mean, sd, index))

    mean = np.mean(fscores_mat[:,i])
    sd = np.std(fscores_mat[:,i])
    index = class_names[i]
    fscore_sum.append((mean, sd, index))

# mean and sd of per class accuracies sums
per_class_accs_sum = (np.mean(per_class_accs), np.std(per_class_accs))

# Take mean and standard deviation of global accuracies and kappas
accuracy_sum = (np.mean(accuracies), np.std(accuracies))

kappa_sum = (np.mean(kappas), np.std(kappas))

# save evaluations to csv files
precision_mean, precision_sd, class_name = zip(*precision_sum)
recall_mean, recall_sd, class_name = zip(*recall_sum)
fscore_mean, fscore_sd, class_name = zip(*fscore_sum)

data_sum = { # data with means and sds
        'class' : class_name,
        'precision_mean' : precision_mean,
        'precision_sd' : precision_sd,
        'recall_mean' : recall_mean,
        'recall_sd' : recall_sd,
        'fscore_mean' : fscore_mean,
        'fscore_sd' : fscore_sd,
        'support' : supports[0]
    }


## EXAMPLE USE TO SAVE DATAFRAMES TO CSV FILES
# Export model performance info as csv
df_summary = pd.DataFrame(data_sum)
file_name = 'E:/PhD/ResNet_results/model_performance_100x_final_dualresnetmodel_2025-11-27.csv'
df_summary.to_csv(file_name)
print(f"Model performance metrics saved to {file_name}")

df_fscore = pd.DataFrame(fscores_mat, columns=f"f_score_{class_name}")
df_precision = pd.DataFrame(precisions_mat, columns=f"precision_{class_name}")
df_recall = pd.DataFrame(recalls_mat, columns=f"recall_{class_name}")
df_all = pd.concat([df_precision, df_recall, df_fscore], axis=1) # cbind the dataframes
file_name2 = 'E:/PhD/ResNet_results/model_performance_fscores_100x_final_dualresnetmodel_2025-11-27.csv'
df2.to_csv(file_name2)
