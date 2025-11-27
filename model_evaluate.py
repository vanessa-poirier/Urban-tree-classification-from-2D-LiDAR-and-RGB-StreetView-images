import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
import numpy as np
import sys

# Load model
## EXAMPLE USE
saved_model = os.path.join("dir", "ResNet_results", "ResNet34_streetview_images_augmented_earlystop12_09-18-2025.pth") # path to saved model
torch.load(saved_model, weights_only=False)


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
file_name = 'E:/PhD/ResNet_results/model_performance_100x_resnet34_streetview_2025-11-18.csv'
df_summary.to_csv(file_name)
print(f"Model performance metrics saved to {file_name}")

df_fscore = pd.DataFrame(fscores_mat, columns=f"f_score_{class_name}")
df_precision = pd.DataFrame(precisions_mat, columns=f"precision_{class_name}")
df_recall = pd.DataFrame(recalls_mat, columns=f"recall_{class_name}")
df_all = pd.concat([df_precision, df_recall, df_fscore], axis=1) # cbind the dataframes
file_name2 = 'E:/PhD/ResNet_results/model_performance_fscores_100x_resnet34_streetview_2025-11-18.csv'
df2.to_csv(file_name2)
