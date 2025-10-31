# Import libraries
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
import numpy as np

# Define function to compute means and standard deviations of pixel values RGB across the whole dataset (i.e. all trees in all classes)
def compute_mean_std(root_dir, resize=(150, 100), max_images=5000):
    """
    Calculate means and standard deviations of pixel values RGB across the whole dataset (i.e. all trees in all classes).
    
    Args:
        root_dir: String. Path containing the data (i.e. images) organized by class
        resize: Tuple. Desired size of the images (H, W).
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
        for tree_id in os.listdir(class_path):
            primary_image_dir = os.path.join(class_path, tree_id, "scan_date")
            secondary_image_dir = os.path.join(class_path, tree_id, "automn")
            tertiary_image_dir = os.path.join(class_path, tree_id, "summer")
            if os.path.isdir(primary_image_dir) and (len([name for name in os.listdir(primary_image_dir)])>0):
                #print(len([name for name in os.listdir(primary_image_dir)]))
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
    mean = all_pixels.mean(dim=1).tolist()
    std = all_pixels.std(dim=1).tolist()
    return mean, std, max_images_per_tree


# === Example of use ===
root_dir = "./data"
mean, std, max_images_per_tree = compute_mean_std(root_dir, resize=(150, 100), max_images=None)

print(f"Mean: {mean}")
print(f"Std:  {std}")
print(f"Max images per tree:  {max_images_per_tree}")
