# Import necessary libraries
import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Specify pixel means, standard deviations, and number of streetview images desired per tree
# These values should be calculated following the calculate_pixel_mean_std.py code
streetview_mean = [0.39968493580818176, 0.423340380191803, 0.3796834349632263] # mean pixel values, RGB
streetview_std = [0.2591862976551056, 0.25725704431533813, 0.27917301654815674] # standard deviation of pixel values, RGB
streetview_max = 6 # maximum number of streetview images per tree

lidar_mean = [0.8706936836242676, 0.8945264220237732, 0.8733906149864197] # mean pixel values, RGB
lidar_std = [0.29993873834609985, 0.18825730681419373, 0.29120898246765137] # standard deviation of pixel values, RGB


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
