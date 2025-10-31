# Urban-tree-classification-from-2D-LiDAR-and-RGB-StreetView-images
This repository hosts all the code associated with building three ResNet34-based Deep Learning models which aim to classify 15 urban tree genera from LiDAR and RGB StreetView images.

The context for this repository and the methods to follow are detailed in **Poirier et al. 2026** **cite article**

All scripts which are part of the Deep Learning model construction are in Python (version 3.10.13). Some pre-processing of the LiDAR data is required (to transform the 3D point clouds into 2D RGB images) and can be completed using the 3D_to_2D_LIDAR_transformation.R script. Original authors used R version 4.4.0.

**Data pre-processing (all in R):**
3D_to_2D_LIDAR_transformation.R

**Tree classification model (all in Python):**

