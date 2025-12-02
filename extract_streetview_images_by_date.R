# This script is to extract uuid and date info from streeview photos. 
  # The png files are expected to be kept in a parent directory and grouped into subfolders where each subfolder is a tree. File names should include uuid and date information.
  # Example of Streetview file name: "00a06940-467e-443f-aed8-316345aa2a89#jak1_20211029_72588s005ms#2021.png" uuid, date, time, and year

# Load necessary libraries
library(here)
library(stringr)
library(mark)
library(lubridate)


#### Load Streetview files ####

dir <- here("data","point_clouds","streetview_images_non-extended")
files <- list.files(dir, pattern='*.png', full.names=TRUE, recursive=TRUE)

# split function to split the file path
split_path <- function(path) {
    rev(setdiff(strsplit(path,"/|\\\\")[[1]], ""))
} 

subfolders <- lapply(files, function(x) as.vector(split_path(x)[2])) # subfolder is the uuid
uuids <- unlist(subfolders)
#uuids <- uuids[!duplicated(uuids)] # remove any duplicates

# extract dates and time from file names
file_names <- lapply(files, function(x) as.vector(split_path(x)[1]))
dates <- lapply(file_names, function(x) str_extract_date(x, format="%Y%m%d"))
times <- lapply(file_names, function(x) str_extract(x, "[0-9]{5}[s]+[0-9]{3}[m][s]"))


#### Get date of LiDAR scans ####
# with the goal of finding any Streetview images that match that scan date

# determine date of scan:
scans <- list.files(here("data","point_clouds","best_laz_format","best_laz_clean"), pattern="*.laz", recursive=TRUE, full.names=TRUE) # path to laz point cloud files

scan_names <- lapply(scans, function(x) as.vector(split_path(x)[1]))
genus <- unlist(lapply(scans, function(x) as.vector(split_path(x)[2])))
uuid <- gsub("\\.laz", "", gsub("^.*?_","", scan_names))
date <- str_extract_date(scan_names, format = "%Y-%m-%d")
month <- month(as.POSIXlt(date, format="%Y-%m-%d"))

data <- data.frame(genus=genus, uuid=uuid, date=date, month=month)


#### Extract photos from specific dates ####
## Specify season months
summer <- "202[0-9]06|202[0-9]07|202[0-9]08" 
automn <- "202[0-9]09|202[0-9]10|202[0-9]11"
winter <- "202[0-9]01|202[0-9]02|202[0-9]03|202[0-9]04|202[0-9]05|202[0-9]12"



extract_files <- function(subfolder) {
    ## DATA PREP
    # get subfolder path
    x <- subfolder
    print(x)

    # get uuid of subfolder
    uuid1 <- as.vector(split_path(x)[1])
    print(uuid1)

    # get list of photo paths
    photo_paths <- list.files(x, full.names=TRUE)

    # get dates of photos
    dates <- lapply(photo_paths, function(x) str_extract_date(x, format="%Y%m%d"))
    pattern_dates <- unlist(lapply(dates, function(x) gsub("-", "", x)))
    print("Available dates are: ")
    print(as.vector(pattern_dates[!duplicated(pattern_dates)]))

    # get date of scan
    scan_date <- subset(data, uuid == uuid1)$date
    print(scan_date)

    # get genus of scan
    scan_genus <- subset(data, uuid == uuid1)$genus

    # group scans based on dates
    try(grouped_files <- split(photo_paths, pattern_dates))



    ## STEP 1 : extract photos that match the date of the scan
    scan_photos <- grep(paste(gsub("-", "", scan_date), collapse="|"), photo_paths, value=TRUE)

    if (length(scan_photos) == 0) {
        scan_photos <- NA
        print(paste0("No photos on scan date for ", uuid1))
    } else {
        print(paste0(length(scan_photos), " photo(s) from scan date"))
    }



    ## STEP 2 : extract photos that match summer time
    summer_photos <- NULL
    try(summer_photos <- grouped_files[[grep(summer, names(grouped_files))]])

    if (is.null(summer_photos) == TRUE) {
        summer_photos <- NA
        print(paste0("No photos from summer for ", uuid1))
    } else {
        print(paste0(length(summer_photos), " photo(s) from summer"))
    }

    # how many different dates do I have from summer?
    dates <- lapply(summer_photos, function(x) str_extract_date(x, format="%Y%m%d"))
    pattern_dates <- unlist(lapply(dates, function(x) gsub("-", "", x)))

    try(summer_diff_dates <- split(summer_photos, pattern_dates))
    print(paste0("There are ", length(names(summer_diff_dates)), " different summer dates"))

    # extract photos from date with the most photos
    if (length(names(summer_diff_dates)) == 0|1) {
        print("No summer photos OR summer photos only includes one date")
    } else {
        photo_counts <- lapply(summer_diff_dates, length)
        index_most_photos <- which(as.vector(unlist(photo_counts)) == max(as.vector(unlist(photo_counts))))
        summer_photos <- summer_diff_dates[index_most_photos]
    }


    ## STEP 3 : extract photos that match automn time
    automn_photos <- NULL
    try(automn_photos <- grouped_files[[grep(automn, names(grouped_files))]])

    if (is.null(automn_photos) == TRUE) {
        automn_photos <- NA
        print(paste0("No photos from automn for ", uuid1))
    } else {
        print(paste0(length(automn_photos), " photo(s) from automn"))
    }

    # how many different dates do I have from automn?
    dates <- lapply(automn_photos, function(x) str_extract_date(x, format="%Y%m%d"))
    pattern_dates <- unlist(lapply(dates, function(x) gsub("-", "", x)))

    try(automn_diff_dates <- split(automn_photos, pattern_dates))
    print(paste0("There are ", length(names(automn_diff_dates)), " different automn dates"))

    # extract photos from date with the most photos
    if (length(names(automn_diff_dates)) == 0|1) {
        print("No automn photos OR automn photos only includes one date")
    } else {
        photo_counts <- lapply(automn_diff_dates, length)
        index_most_photos <- which(as.vector(unlist(photo_counts)) == max(as.vector(unlist(photo_counts))))
        automn_photos <- automn_diff_dates[index_most_photos]
    }



    ## STEP 4 : extract photos that match winter time
    winter_photos <- NULL
    try(winter_photos <- grouped_files[[grep(winter, names(grouped_files))]])

    if (is.null(winter_photos) == TRUE) {
        winter_photos <- NA
        print(paste0("No photos from winter for ", uuid1))
    } else {
        print(paste0(length(winter_photos), " photo(s) from winter"))
    }

    # how many different dates do I have from winter?
    dates <- lapply(winter_photos, function(x) str_extract_date(x, format="%Y%m%d"))
    pattern_dates <- unlist(lapply(dates, function(x) gsub("-", "", x)))

    try(winter_diff_dates <- split(winter_photos, pattern_dates))
    print(paste0("There are ", length(names(winter_diff_dates)), " different winter dates"))

    # extract photos from date with the most photos
    if (length(names(winter_diff_dates)) == 0|1) {
        print("No winter photos OR winter photos only includes one date")
    } else {
        photo_counts <- lapply(winter_diff_dates, length)
        index_most_photos <- which(as.vector(unlist(photo_counts)) == max(as.vector(unlist(photo_counts))))
        winter_photos <- winter_diff_dates[index_most_photos]
    }


    #### Copy photo files that I want to use for the model into another folder ####

    try(if (is.na(scan_photos)[1]==TRUE) {
        print("no scan photos")
    } else {
        files <- lapply(scan_photos, basename)
        copy_path_scan <- lapply(files, function(x) paste("/media/vanessa/ZEFI_neuroa/PhD/streetview_images_model", scan_genus, uuid1, "scan_date", x, sep="/"))
    })

    try(if (is.na(summer_photos)[1]==TRUE) {
        print("no summer photos")
    } else {
        files <- lapply(summer_photos, basename)
        copy_path_summer <- lapply(files, function(x) paste("/media/vanessa/ZEFI_neuroa/PhD/streetview_images_model", scan_genus, uuid1, "summer", x, sep="/"))
    })

    try(if (is.na(automn_photos)[1]==TRUE) {
        print("no automn photos")
    } else {
        files <- lapply(automn_photos, basename)
        copy_path_automn <- lapply(files, function(x) paste("/media/vanessa/ZEFI_neuroa/PhD/streetview_images_model", scan_genus, uuid1, "automn", x, sep="/"))
    })

    try(if (is.na(winter_photos)[1]==TRUE) {
        print("no winter photos")
    } else {
        files <- lapply(winter_photos, basename)
        copy_path_winter <- lapply(files, function(x) paste("/media/vanessa/ZEFI_neuroa/PhD/streetview_images_model", scan_genus, uuid1, "winter", x, sep="/"))
    })

    # create directories to put files if they don't already exist
    main_dir <- "/media/vanessa/ZEFI_neuroa/PhD/streetview_images_model"
    genus_dir <- scan_genus
    sub_dir1 <- uuid1
    try(sub_dir_scan <- as.vector(split_path(copy_path_scan[[1]])[2]))
    try(sub_dir_summer <- as.vector(split_path(copy_path_summer[[1]])[2]))
    try(sub_dir_automn <- as.vector(split_path(copy_path_automn[[1]])[2]))
    try(sub_dir_winter <- as.vector(split_path(copy_path_winter[[1]])[2]))

    dir.create(file.path(main_dir, genus_dir), showWarnings = TRUE)
    dir.create(file.path(paste(main_dir, genus_dir, sep="/"), sub_dir1), showWarnings = TRUE)
    try(dir.create(file.path(paste(main_dir, genus_dir, sub_dir1, sep="/"), sub_dir_scan), showWarnings = TRUE))
    try(dir.create(file.path(paste(main_dir, genus_dir, sub_dir1, sep="/"), sub_dir_summer), showWarnings = TRUE))
    try(dir.create(file.path(paste(main_dir, genus_dir, sub_dir1, sep="/"), sub_dir_automn), showWarnings = TRUE))
    try(dir.create(file.path(paste(main_dir, genus_dir, sub_dir1, sep="/"), sub_dir_winter), showWarnings = TRUE))


    # Write copy files command to text file for glost
    try(lines_scan <- as.list(mapply(function(x,y) paste("cp", x, y, sep=" "), scan_photos, copy_path_scan)))
    try(names(lines_scan) <- NULL)
    try(lines_summer <- as.list(mapply(function(x,y) paste("cp", x, y, sep=" "), summer_photos, copy_path_summer)))
    try(names(lines_summer) <- NULL)
    try(lines_automn <- as.list(mapply(function(x,y) paste("cp", x, y, sep=" "), automn_photos, copy_path_automn)))
    try(names(lines_automn) <- NULL)
    try(lines_winter <- as.list(mapply(function(x,y) paste("cp", x, y, sep=" "), winter_photos, copy_path_winter)))
    try(names(lines_winter) <- NULL)


    # add lines together (if they exist)
    items_to_check <- c("lines_scan", "lines_summer", "lines_automn", "lines_winter")

    # Initialize an empty list to store valid items
    lines <- list()

    # Iterate through the items and add existing ones to the list
    for (item_name in items_to_check) {
    if (exists(item_name)) {
        lines[[item_name]] <- get(item_name)
    }
    }

    names(lines) <- NULL
    lines <- as.list(lines)

    return(lines)
}

## get photo paths
subfolder_paths <- list.files("/media/vanessa/ZEFI_neuroa/PhD/streetview_images_non-extended", include.dirs=TRUE, full.names=TRUE)
head(subfolder_paths)

all_lines <- lapply(subfolder_paths, extract_files)
head(unlist(all_lines))
