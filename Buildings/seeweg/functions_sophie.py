# import  libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from apps.utils import graphql_requests
from datetime import datetime, timedelta
from apps.utils.data_utils import remove_outliers
#from experiments.sophie.functions_sophie import read_csv_sensor
#import experiments.sophie.functions_sophie as fs
from apps.utils.data_utils import remove_outliers
import os

# define functions

def read_csv_sensor(folder, file):
    """
    Read a CSV file from the specified folder.

    Parameters:
    - folder (str): The name of the folder containing the CSV file.
    - file (str): The name of the CSV file (without the '.csv' extension).

    Returns:
    - DataFrame: A pandas DataFrame containing the data from the CSV file. Only one value per minute (resample)
    """
    try:
        file_path = f"data_csv/{folder}/{file}.csv"
        df = pd.read_csv(file_path)
        
        # Convert 'datetime' column to datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Set 'datetime' column as index
        df.set_index('datetime', inplace=True)
        
        # Return resampled dataframe with one mean value per minute
        return df.resample('T').mean()
    
    except FileNotFoundError:
        print(f"Error: File '{file}.csv' not found in folder '{folder}'.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def drop_rows_with_zero_value(df):
    """
    Drop rows from DataFrame where the value in the first column is zero.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.

    Returns:
    - DataFrame: A pandas DataFrame with rows where the value in the first column is zero removed.
    """
    zero_indices = df.index[df.iloc[:, 0] == 0].tolist()
    print("Cleaning of zero values: " + str(len(zero_indices)) + " rows with 0°C as value")
    return df.drop(zero_indices)

def resample_minutes(df):
    """
    Resamples a dataframe with 'datetime' minutewise. The calculated value corresponds to the mean of all datapoints in the minute's interval.

    Parameters:
    - df (DataFrame): The pandas dataframe with time stamp as 'datetime'ArithmeticError
    
    Returns:
    - DataFrame: a pandas Dataframe with only one value per minute
    """

    return df.resample('T').mean()


def plot_comparison(df1, df2, label1='DataFrame 1', label2='DataFrame 2', xlabel='Date', ylabel='value', title= 'Comparison'):
    """
    Plot the 'temperature' column from two DataFrames on the same plot with customizable labels.

    Parameters:
    - df1 (DataFrame): The first pandas DataFrame containing temperature data.
    - df2 (DataFrame): The second pandas DataFrame containing temperature data.
    - label1 (str): Label for the first DataFrame (default is 'DataFrame 1').
    - label2 (str): Label for the second DataFrame (default is 'DataFrame 2').
    - xlabel (str): Label for the x-axis (default is 'Date').
    - ylabel (str): Label for the y-axis (default is 'Temperature').
    - title (str): Title for the plot (default is 'Temperature Comparison').
    """
    # Plot 'temperature' column from df1
    plt.plot(df1.index, df1['temperature'], label=label1)

    # Plot 'temperature' column from df2
    plt.plot(df2.index, df2['temperature'], label=label2)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Add legend
    plt.legend()

    # Show plot
    plt.show()



folder = "2024.01.14 - 2024.01.13"

#= read_csv_sensor(folder, "")

TVV1 = read_csv_sensor(folder, "TVV1") # TWW feedflow                            3359
VZHK1 = read_csv_sensor(folder, "VZHK1") # TWW feedflow / return after valve     2577
TVHGWP1 = read_csv_sensor(folder, "TVHGWP1")
TRHGWP1 = read_csv_sensor(folder, "TRHGWP1")
VZHGWP1 = read_csv_sensor(folder, "VZHGWP1")
PEZHS = read_csv_sensor(folder, "PEZHS")
EEZHS = read_csv_sensor(folder, "EEZHS")
TRH1 = read_csv_sensor(folder, "TRH1") # TWW return before the valve             2350
VenK1 = read_csv_sensor(folder, "VenK1")
TVH1 = read_csv_sensor(folder, "TVH1") # TWW flow after the valve                2469
TSPO = read_csv_sensor(folder, "TSPO")
TSPM = read_csv_sensor(folder, "TSPM")
TSPU = read_csv_sensor(folder, "TSPU")



#Tank Sensors



startt = datetime(2024, 1, 15)
endd = datetime(2024, 1, 20)


#df1 = c1_t.loc[startt:endd]

s1 = TSPO # t tank top
s2 = TSPM # t tank middle
s3 = TSPU # t tank bottom


# create figure
plt.figure(figsize=(10, 6))

# define plots
plt.plot(s1.index, s1, label='t top')
plt.plot(s2.index, s2, label='t middle')
plt.plot(s3.index, s3, label = 't bottom')


# Add legend
plt.legend()
plt.title("Tank temperatures")

# show plot
plt.show()
