
# Set url

basic_infor_url <- "https://zenodo.org/record/4905618/files/exams.csv?download=1"
ecg_data_url <- "https://zenodo.org/record/4905618/files/exams.zip?download=1"

# Set filename

basic_infor_file_path <- "data/raw_data/Sami-Trop.csv"

ecg_data_zip_path <- "data/raw_data/Sami-Trop.zip"
ecg_data_file_dir <- "data/raw_data/"

# Download and zip

download.file(url = basic_infor_url, destfile = basic_infor_file_path)

download.file(url = ecg_data_url, destfile = ecg_data_zip_path, mode = "wb")
unzip(zipfile = ecg_data_zip_path, exdir = ecg_data_file_dir)
file.rename(from = paste0(ecg_data_file_dir, "exams.hdf5"), to = paste0(ecg_data_file_dir, "Sami-Trop.hdf5"))




