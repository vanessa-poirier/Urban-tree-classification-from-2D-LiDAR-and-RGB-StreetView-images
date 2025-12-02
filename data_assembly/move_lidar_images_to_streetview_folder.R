## This script is to move the treated lidar images into the same directory as the treated/sorted streetview images
# the resulting directory which combines lidar images and streetview images for each tree can be used directly for model training

# load necessary libraries
library(here)
library(fs)

#### Get 2D-LiDAR image files ####
root_dir = here("data","point_clouds")
current_paths <- list.files(paste(root_dir, "best_2D_format", "rgb_intensity_callibrated", sep="/"), pattern="*.jpg", recursive=TRUE, full.names=TRUE)
head(current_paths)


#### Define new file paths ####
# new file paths will correspond to where the 2D-LiDAR images will be copied to (i.e. in the same folders as the streetview images)

# define split path function
split_path <- function(path) {
    rev(setdiff(strsplit(path,"/|\\\\")[[1]], ""))
} 

# define function to get new paths
get_new_path <- function(current_path) {
    file <- as.vector(split_path(current_path)[1])
    subfolder <- as.vector(split_path(current_path)[2])
    uuid <- gsub("^.*?_","", subfolder)
    genus <- as.vector(split_path(current_path)[3])

    new_path <- paste(root_dir, "streetview_images_model", genus, uuid, "lidar", file, sep="/") # parent directory of new path must match the Streetview image path
    return(new_path)
}

new_paths <- lapply(current_paths, get_new_path)
new_paths <- unlist(new_paths)

# check that there are the same number of current paths as new paths
length(current_paths)
length(new_paths)


#### Copy files from current paths to new paths ####
# make a lidar folder in all the tree paths (if it doesn' already exist)
create_lidar_dir <- function(path) {
    dir <- dirname(path)
    dir.create(dir, showWarnings = TRUE)
}

## Move files from current path to new path
mapply(function(x, y) file.copy(x, y, overwrite = TRUE), current_paths, new_paths)
