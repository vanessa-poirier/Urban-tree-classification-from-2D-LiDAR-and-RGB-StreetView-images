# This script is used to transform point clouds of trees into 2D images (i.e. create 'screenshots' of the point clouds from 10 different angles)
# RGB values represent each point's r (distance to tree's central axis), intensity value, and Z (i.e. height) value
# This script is meant to be used on one file (i.e. point cloud) at a time. We present its use to be coupled with GLOST with the goal of processing many files in parallel

# Load necessary libraries
library(lidR)
library(here)
library(ggplot2)
library(dplyr)
library(tools)
library(data.table)
library(FNN)


#### Load .laz file (i.e. point cloud) ####

## OPTION 1: Read file from glost argument
# Get the arguments (i.e. input file) from the bash command line (arguments saved in a text file, see arguments_glost_2D.R for instructions on how to create this text file)
args <- commandArgs(trailingOnly = TRUE) # trailingOnly means it only returns t>
file <- gsub("[\r\n]","",args[1]) # gets first argument

# exit out silently (with warning) if the file doesn't exist
if (!file.exists(file)) {
  message(sprintf("Unable to find the file %s",file))
}

print("was able to load file")

## OPTION 2: Read file from directory (does not permit parallel processing)
file <- here("data","point_clouds","best_laz_format","best_laz_clean","Betula","2021-10-19_817bd9f6-615c-4f81-9fff-dcbc171830b4.laz") # example file path

if (!file.exists(file)) {
  message(sprintf("Unable to find the file %s",file))
}

print("was able to load file")


#### Get genus and uuid info from file name ####
split_path <- function(path) {
    rev(setdiff(strsplit(path,"/|\\\\")[[1]], ""))
} 

genus_status <-  as.vector(split_path(file)[2])

uuid <- file_path_sans_ext(basename(file)) # keep uuid info (with date)



#### Load xyz (+ intensity) information from laz file ####

laz <- readTLSLAS(file, select = "xyzi") # only loading XYZ coords and intensity values
xyz <- laz@data

# subset to keep only 10000 points from point cloud
if (nrow(xyz) >= 10000) { 
        xyz <- xyz[sample(nrow(xyz), 10000),] # subset to keep only 10000 points
    } else {
        xyz <- xyz
    }

# calculate xy plane centroid (i.e. center access of the tree)
knn <- get.knn(subset(xyz, select=c("X","Y","Z")), k = 10) # calculate knn distance for each point
knn_means <- rowMeans(knn$nn.dist) # average the knn distance for each point (which I will use as weights to find the centroid)
weights <- 1 - (knn_means - min(knn_means)) / (max(knn_means)-min(knn_means)) # inverse these weights (so that less distance carries a higher weight) and proportionalize them

centroid_X <- sum(xyz$X * weights) / sum(weights)
centroid_Y <- sum(xyz$Y * weights) / sum(weights)

# shift point cloud coordinates around the center axis
xyz <- xyz %>% 
       mutate(X = centroid_X - X,
              Y = centroid_Y - Y,
              Z = Z,
              Intensity = Intensity)

# calculate cylindrical coordinates, particularly theta and r (of which r will be used as the color blue in the RGB images)
xyz$theta <- atan(xyz$Y/xyz$X) # atan returns value in radians
xyz$r <- sqrt((xyz$X^2)+(xyz$Y^2)) # r is the distance (in xy plane) from the point to the Z axis of origin


#### Rotate trees around z-axis at 10 different angles ####
# angles of rotation: 0, 36, 72, 108, 144, 180, 216, 252, 288, 324
angles <- (c(0, 36, 72, 108, 144, 180, 216, 252, 288, 324)) # angles are in degrees (they are converted to radians in the below for loop)

rotations <- list() # create rotations list that the subsequent for loop will be saved to

    for (t in 1:length(angles)) {
    df <- xyz %>%
                mutate(X = (cos(angles[t]*(pi/180)) * X) + (-sin(angles[t]*(pi/180)) * Y),
                        Y = (sin(angles[t]*(pi/180)) * X) + (cos(angles[t]*(pi/180)) * Y))
    rotations[[t]] <- df
    names(rotations)[t] <- paste(file_path_sans_ext(basename(file)), angles[t], sep="_") # name each dataframe 'filename_angle'
    }


#### Plot and export 2D images ####

# create destination folder to put the images in
dir.create(file.path(here("data","point_clouds","best_2D_format", "rgb_intensity_callibrated", genus_status, uuid)), recursive = TRUE) # create subfolders

    for (r in 1:length(rotations)) { # for each of the rotated point clouds
        df <- rotations[[r]] %>% mutate(ic = (Intensity-0)/(65535-0), # i value for the purpose of coloring where max intensity possible is 65535 and the minimum is zero
                                zc = (Z-min(Z))/(max(Z)-min(Z)), # Z value for the purpose of coloring
                                rc = (r-min(r))/(max(r)-min(r)), # r value for the purpose of coloring
                                rgb.val=rgb(ic,zc,rc)) # RGB needs the colors to range from 0 to 1
        plot <- ggplot(df, aes(X, Z, color=rgb.val)) + # plot 2D RGB image
            geom_point(shape=".") +
            #scale_color_gradient(low="grey", high="black") + # for grey scale images
            theme_void() + # remove all background
            theme(legend.position="none")
        # export plot
        png(filename = here("data","point_clouds","best_2D_format", "rgb_intensity_callibrated", genus_status, uuid, paste0(names(rotations)[r], ".jpg")), width = 100, height = 150)
        print(plot)
        dev.off()
    }
