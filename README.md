# Urban-tree-classification-from-2D-LiDAR-and-RGB-StreetView-images
This repository hosts all the code associated with building three ResNet34-based Deep Learning models which aim to classify 15 urban tree genera from LiDAR and RGB StreetView images.

The context for this repository and the methods to follow are detailed in **Poirier, V., Kneeshaw, D., Herrault, P.A., Wenger, R., Charbonneau, Z., Paquette, A. IN PREP. Urban tree genera classification using mobile terrestrial LiDAR and Streetview images**

All scripts which are part of the Deep Learning model construction are in Python (version 3.10.13). Some pre-processing of the LiDAR data is required (to transform the 3D point clouds into 2D RGB images) and can be completed using the 3D_to_2D_LIDAR_transformation.R script. Original authors used R version 4.4.0.

**Data pre-processing (all in R):**
LiDAR_transformation
  arguments_glost_2D.R # create arguments to be fed to GLOST for parallel processing of the pt_to_2D_image_rgb.R script
  pt_to_2D_image_rgb.R # transform tree point cloud (laz file) into 10 2D images, each showcasing a different angle of view

data_assembly
  extract_streetview_images_by_date.R # sorts streetview images by their date of capture (by season and LiDAR scan dates) 
  move_lidar_images_to_streetview_folder.R # moves 2D-LiDAR images into the same folders as the Streetview images

**Tree classification model (all in Python):**
Scripts should be processed in the following order:
  1. calculate_pixel_mean_std.py # for normalizatioion purposes: calculates pixel means and standard deviations of 2D-LiDAR and Streetview images
  2. data_loader.py # data loader for model training, validation, and testing
  3. model_define.py # defining of model class
  4. training_pipeline.py # defining of model training and validation pipeline
  5. configuration.py # data transformation specifics and model training optimization criterions
  6. model_train.py # train the model
  7. model_evaluate.py # calculate accuracies, kappa, F1-scores, precisions, and recalls
