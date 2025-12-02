## This script creates a text file (.txt) where each line is an bash line argument to be employed in parallel using GLOST

# Load necessary libraries
library(here)


#### Get info on files to be treated ####
data <- list.files(here("best_laz_clean"), pattern="*.laz", full.names=TRUE, recursive=TRUE) # path to laz files to be transformed into 2D images

split_path <- function(path) {
    rev(setdiff(strsplit(path,"/|\\\\")[[1]], ""))
} 

file_name <- lapply(data, function(x) as.vector(split_path(x)[1])) # create list of filenames
subfolder <- lapply(data, function(x) as.vector(split_path(x)[2])) # create list of subfolders
file_path <- lapply(subfolder, function(x) paste0("best_laz_clean/", x)) # create list of file paths. "besst_laz_clean" is the parent directory
file_path <- as.list(mapply(function(x, y) paste(x, y, sep="/"), file_path, file_name))


#### Create arguments txt file for GLOST ####
arguments <- file(here("arguments-laz-rgb.txt"))

lines <- lapply(file_path, function(k) {paste("Rscript", "pt_to_2Dimage_rgb.R", k, sep=" ")}) # Rscript is the bash command to run an R script. "pt_to_2D_image_rgb.R" is the script we would like to run

cat(paste(unlist(lines), collapse = "\n"), "\n", file=arguments)

close(arguments)
