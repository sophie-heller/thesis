import os
import pandas as pd

# Define the paths
base_dir = os.path.join('Buildings', 'measurements')
seeweg_data_dir = os.path.join(base_dir, 'seeweg_data')
seeweg_sensor_data_dir = os.path.join(base_dir, 'seeweg_sensor_data')

# Create the seeweg_sensor_data directory if it doesn't exist
os.makedirs(seeweg_sensor_data_dir, exist_ok=True)

# Dictionary to store data for each sensor ID
sensor_data = {}

# Iterate over all date folders in the seeweg_data directory
for date_folder in os.listdir(seeweg_data_dir):
    date_folder_path = os.path.join(seeweg_data_dir, date_folder)
    if os.path.isdir(date_folder_path):
        # Iterate over all CSV files in the date folder
        for csv_file in os.listdir(date_folder_path):
            if csv_file.endswith('.csv'):
                sensor_id = os.path.splitext(csv_file)[0]
                csv_file_path = os.path.join(date_folder_path, csv_file)
                
                # Read the CSV file
                df = pd.read_csv(csv_file_path)
                
                # If the sensor ID is not in the dictionary, initialize an empty DataFrame
                if sensor_id not in sensor_data:
                    sensor_data[sensor_id] = pd.DataFrame()
                
                # Append the data to the corresponding sensor ID DataFrame
                sensor_data[sensor_id] = pd.concat([sensor_data[sensor_id], df], ignore_index=True)

# Save the combined data for each sensor ID to a new CSV file in seeweg_sensor_data
for sensor_id, data in sensor_data.items():
    output_file_path = os.path.join(seeweg_sensor_data_dir, f"{sensor_id}_total.csv")
    data.to_csv(output_file_path, index=False)

print("Sensor data has been aggregated successfully.")
