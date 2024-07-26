import os
import zipfile
import shutil

# Provided the ZIP folders are located in C:\Users\sophi\repos\repos_thesis\Buildings\measurements
# Define the paths
base_dir = os.path.join('Buildings', 'measurements')
seeweg_data_dir = os.path.join(base_dir, 'seeweg_data')
mendel_data_dir = os.path.join(base_dir, 'mendel_data')

# Create directories if they don't exist
os.makedirs(seeweg_data_dir, exist_ok=True)
os.makedirs(mendel_data_dir, exist_ok=True)

# Function to extract and copy measurement folders
def extract_measurement_folders(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extract_to_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(zip_path))[0])
        zip_ref.extractall(extract_to_dir)
        
        # Navigate to the 'data' folder in the extracted content
        data_dir = os.path.join(extract_to_dir, 'data')
        
        if os.path.exists(data_dir):
            for location in ['seeweg', 'mendel']:
                location_dir = os.path.join(data_dir, location)
                if os.path.exists(location_dir):
                    for date_folder in os.listdir(location_dir):
                        src = os.path.join(location_dir, date_folder)
                        if os.path.isdir(src):
                            if location == 'seeweg':
                                dst = os.path.join(seeweg_data_dir, date_folder)
                            elif location == 'mendel':
                                dst = os.path.join(mendel_data_dir, date_folder)
                            shutil.copytree(src, dst, dirs_exist_ok=True)
        
        # Clean up the extracted folder after processing
        shutil.rmtree(extract_to_dir)

# Iterate over all zip files in the measurements folder
for item in os.listdir(base_dir):
    if item.endswith('.zip'):
        zip_path = os.path.join(base_dir, item)
        extract_measurement_folders(zip_path)

print("Folders have been organized successfully.")
