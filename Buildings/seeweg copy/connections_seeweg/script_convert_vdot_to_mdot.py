import pandas as pd
import glob
import os

# Define the path where your CSV files are located
# If the script is in the same directory, use '.'
path = './Buildings/seeweg copy/connections_seeweg'
print(path)

# Find all CSV files in the specified path that start with 'c'
csv_files = glob.glob(os.path.join(path, 'c*.csv'))

# Loop through each file
for file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Extract the filename without extension (e.g., c1, c2, ...)
    filename = os.path.basename(file).split('.')[0]
    
    # Identify the column name for v_cx based on the filename
    v_cx_column = f'v_{filename}'
    mt_cx_column = f'mt_{filename}'
    
    # Create the new mdot_cx column based on the v_cx column
    if v_cx_column in df.columns:
        df[f'mdot_{filename}'] = df[v_cx_column] * 1000 / 3600000 # vdot original in [l/h] * 1000 [kg/m³] / (1000 [l/m³] * 3600 [s/h]) = mdot in [kg/s]
        
        # Drop the old v_cx and mt_cx columns
        df.drop(columns=[v_cx_column, mt_cx_column], inplace=True, errors='ignore')
        
        # Save the modified DataFrame back to the CSV file
        df.to_csv(file, index=False)
    else:
        print(f"Column {v_cx_column} not found in {file}. Skipping this file.")

print("All files have been processed.")