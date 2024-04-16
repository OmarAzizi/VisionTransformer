import os
import zipfile

# Define function to extract zip file
def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed.")

# Path to your CIBS-DDSM dataset zip file
zip_file_path = './dataset.zip'

# Define directory to extract data
extract_path = './content/cibs_ddsm_dataset/'

# Extract the dataset
extract_zip(zip_file_path, extract_path)

# List files in the extracted directory
extracted_files = os.listdir(extract_path)
print("Extracted files:", extracted_files)
