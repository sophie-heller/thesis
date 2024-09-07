import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import expit
import importlib.util
import os




###########################################
###########################################
##### definition of the logistic function
def step(T1, T2):
    m = 1 # miu in paper 4_1D.. is set to 1
    # s = 1/(1 + np.exp(-m*(T1-T2))) # UNSTABLE
    #s = expit(m * (T1 - T2))
    #return s
    T1 = np.asarray(T1)
    T2 = np.asarray(T2)
    
    return np.where(T1 >= T2, 1, 0)


###########################################
###########################################
##### resample original data
def resample_and_interpolate(df, freq='60S', method='linear'):
    """
    Resample the input DataFrame to the specified frequency and interpolate missing values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with datetime index.
    - freq (str): The target frequency for resampling (e.g., 'S' for seconds).
    - method (str): The interpolation method to use (default is 'linear').

    Returns:
    - pd.DataFrame: The resampled and interpolated DataFrame.
    """
    # Resample to the desired frequency
    df_resampled_seconds = df.resample("5S").asfreq()
    df_resampled_seconds = df_resampled_seconds.interpolate(method=method)

    df_resampled = df_resampled_seconds.resample(freq).asfreq()

    # Interpolate the missing values
    df_interpolated = df_resampled.interpolate(method=method)

    return df_interpolated

def resample_and_interpolate_in_list(sensor_data_list, freq='60S', method='linear'):
    """
    Resample and interpolate the DataFrame in each list within sensor_data_list.

    Parameters:
    - sensor_data_list (list of lists): Each list contains two elements, where the second element is the DataFrame to be resampled.
    - freq (str): The target frequency for resampling (default is '120S' for 120-second intervals).
    - method (str): The interpolation method to use (default is 'linear').
    """
    for sensor_data in sensor_data_list:
        # Resample and interpolate the DataFrame at index [1] of each sublist
        sensor_data[1] = resample_and_interpolate(sensor_data[1], freq=freq, method=method)

    return sensor_data_list  # If you want to return the updated list, though it's modified in-place



###########################################
###########################################
def process_mdot_normal(mdot_normal, layers, schichtlader_status, layernumber_c, layernumber_l):
    """
    Processes the mdot_normal list to generate a list of dataframes for each layer in the tank.

    Parameters:
    mdot_normal (list): A list where each entry is a list containing the layer number and a DataFrame [layer_number, DataFrame].
    The DataFrame has the temperatures on the first column and the mass rates on the second column. Index is date-time.
    layers (int): The number of layers in the tank.
    layernumber_c (list): A list of layer numbers where normal connections are present.
    layernumber_l (list): A list of layer numbers where loader connections (Schichtlader) are present.

    Returns:
    tuple: Two lists of DataFrames (tm_list_df, mdot_list_df) containing temperature and mass flow rate data.
    """
    
    # Create a vector that contains the max amount of streams that are possible per layer (excluding Schichtlader)
    layer_c_counts_vector = np.zeros(layers, dtype=int)
    counts_c_normal = np.bincount(layernumber_c, minlength=layers)  # count how many connections appear on each layer
    layer_c_counts_vector = counts_c_normal.copy()

    # Increment the count layer_c_counts_vector all layers specified in layernumber_l, where a loader connection is present
    if schichtlader_status == True :  # Check if layernumber_l size is exactly 1 -> no exactly [0]
        np.add.at(layer_c_counts_vector, layernumber_l, 1)

    # Create an integer that represents the max amount of streams that appear in any layer of the tank
    layer_c_counts_max = max(layer_c_counts_vector).astype(int)

    # Initialize mdot_list as a list of lists. It will contain the cX appearing in the layer appended in one df per layer
    mdot_list = [[] for _ in range(layers)]
    
    # Copy an existing cX DataFrame and fill all columns with 0
    df_c_zero_filled = mdot_normal[1][1].copy()
    df_c_zero_filled.loc[:, :] = 0

    # Rename the first column to "t" and the second column to "mdot"
    df_c_zero_filled.columns = ["t", "mdot"]

    # Iterate through each layer to create a list mdot_list with all connected stream data (excluding Schichtlader)
    for n in range(layers):
        count = 0  # Initialize count of dataframes added to mdot_list[n]

        # Iterate through each normal mdot connection
        for cX in mdot_normal:
            if cX[0] == n:  # If cX is located in the current layer n:
                mdot_list[n].append(cX[1])  # Store t, mdot, and mt of cX (cX[1]) in mdot_list[n]
                count += 1

        # Fill mdot_list[n] with df_c_zero_filled until layer_c_counts_max is achieved
        while count < layer_c_counts_max:
            mdot_list[n].append(df_c_zero_filled.copy())
            count += 1

    # Create a list with layer_c_counts_max DataFrames containing the layers data (t_layer and mdot_layer as columns)
    tm = [pd.DataFrame() for _ in range(layer_c_counts_max)]
    mdot = [pd.DataFrame() for _ in range(layer_c_counts_max)]

    for m in range(layer_c_counts_max):
        for n in range(layers):
            tm[m] = pd.concat([tm[m], mdot_list[n][m].iloc[:, 0]], axis=1)
            mdot[m] = pd.concat([mdot[m], mdot_list[n][m].iloc[:, 1]], axis=1)  # mdot in [kg/s]

    return tm, mdot

# fucntion for tank sensor layer assignation and interpolation
def assign_and_interpolate_layers(layers, t_sp):
    """
    Assign temperature sensors to layers and perform linear interpolation if necessary.
    
    Parameters:
    layers (int): Number of layers in the tank.
    t_sp (list): List of sensor data in the form of [layer_number, temperature_dataframe].

    Returns:
    pd.DataFrame: DataFrame with interpolated temperature data for each layer.
    """
    
    # Initialize lists to hold sensor assignments and interpolated data
    t_init = []  # contains list with layer and sensor info [layer, df]
    t_init_sensordf = []  # contains only the sensor data [df]

    # Iterate through each layer index from 0 to layers-1
    for n in range(layers):
        # Initialize variables to track the closest sensor
        closest_diff = np.inf
        closest_t_sp = [np.nan, None]  # Initialize with NaN values
        
        # Iterate through each t_spX in t_sp
        for t_spX in t_sp:
            # Get the layer number associated with t_spX and its difference from n + 0.5
            layernum = t_spX[0]
            if layernum // 1 == n:
                diff = np.abs(layernum - (n + 0.5))
                # Check if this t_spX can be used for the current layer and has a smaller diff
                if diff <= 0.5 and diff < closest_diff:
                    closest_diff = diff
                    closest_t_sp = t_spX
        
        # Append the closest found sensor to t_init
        t_init.append(closest_t_sp)
        t_init_sensordf.append(closest_t_sp[1])

    # Now t_init contains the selected temperature sensor data for each layer

    # Initialize list to hold the final interpolated data
    t_init_interpolated = []

    # Iterate through each layer index from 0 to layers-1
    for n in range(layers):
        # Check if t_init has a valid entry or needs interpolation
        if t_init[n][1] is not None:
            # Use the existing temperature data
            t_init_interpolated.append(t_init[n][1])
        else:
            # Find the nearest layers with valid temperature data
            layer_below = None
            layer_above = None
            
            # Search backwards for layer_below
            for layer in t_init[:n][::-1]:
                if layer[1] is not None:
                    layer_below = layer
                    break
            
            # Search forwards for layer_above
            for layer in t_init[n+1:]:
                if layer[1] is not None:
                    layer_above = layer
                    break
            
            # Perform interpolation or duplicate closest valid value
            if n == 0 and layer_above is not None:
                # Duplicate closest value from layer_above
                duplicate = layer_above[1].rename("duplicated")
                t_init_interpolated.append(duplicate)
            elif n == layers - 1 and layer_below is not None:
                # Duplicate closest value from layer_below
                duplicate = layer_below[1].rename("duplicated")
                t_init_interpolated.append(duplicate)
            elif layer_below is not None and layer_above is not None:
                # Perform linear interpolation
                layernum_below = layer_below[0]
                layernum_above = layer_above[0]
                temp_below = layer_below[1]
                temp_above = layer_above[1]
                
                # Calculate interpolated temperature
                interpolated_temp = temp_below + ((n + 0.5 - layernum_below) / (layernum_above - layernum_below)) * (temp_above - temp_below)
                interpolated_temp = interpolated_temp.rename("interpolated")
                t_init_interpolated.append(interpolated_temp)
            else:
                # If no valid layers found for interpolation or duplication, append NaN values
                print(f"Sensor data for layer {n} is NaN after interpolation. Check!")
                t_init_interpolated.append(pd.Series([np.nan], name="interpolated", index=t_init[0][1].index))

    # Concatenate the interpolated data into a DataFrame
    t_init_interpolated_df = pd.concat(t_init_interpolated, axis=1)
    
    return t_init_interpolated_df

# function for processing the e_X data 
def process_qdot_normal(qdot_normal, layers, layernumber_e):
    """
    Processes the qdot_normal list to generate a list of DataFrames for each layer in the tank.

    Parameters:
    qdot_normal (list): A list where each entry is a list containing the layer number and a DataFrame [layer_number, DataFrame: date-time, pel].
    
    layers (int): The number of layers in the tank.
    layernumber_e (list): A list of layer numbers where exchangers are present.

    Returns:
    qdot_list_df (list): A list of DataFrames, each containing exchanger data (qdot) for each stream across layers.
    """

    # Create a vector that contains the max number of exchangers present per layer
    layer_e_counts_vector = np.zeros(layers, dtype=int)
    counts_e = np.bincount(layernumber_e, minlength=layers)
    layer_e_counts_vector[:len(counts_e)] = counts_e

    # Determine the max amount of exchangers that appear in any layer of the tank
    layer_e_counts_max = max(layer_e_counts_vector).astype(int)

    # Initialize qdot_list as a list of lists to store data for each layer
    qdot_list = [[] for _ in range(layers)]

    # Copy the DataFrame and fill all columns with 0
    df_e_zero_filled = qdot_normal[0][1].copy() # take first e_X, should be always defined. It is fille dwith 0 if no exchanger present.
    df_e_zero_filled.loc[:] = 0  # Zero out the values in the copied DataFrame
    df_e_zero_filled.columns = ['pel']  # Rename column to "pel"

    # Iterate through each layer to populate qdot_list
    for n in range(layers):
        count = 0  # Initialize count of DataFrames added to qdot_list[n]

        # Iterate through each normal qdot connection
        for eX in qdot_normal:
            if eX[0] == n:  # If eX is located in the current layer n
                qdot_list[n].append(eX[1])  # Store the exchanger data (eX[1]) in qdot_list[n]
                count += 1

        # !!! i think this part can be gone, no need for "extra" space
        # Fill qdot_list[n] with df_e_zero_filled until layer_e_counts_max is achieved
        while count < layer_e_counts_max:
            qdot_list[n].append(df_e_zero_filled.copy())
            count += 1

    # Create a list with count_e_max DataFrames containing layers data for integration into the model
    qdot = [pd.DataFrame() for _ in range(layer_e_counts_max)]

    for m in range(layer_e_counts_max):
        for n in range(layers):
            qdot[m] = pd.concat([qdot[m], qdot_list[n][m].iloc[:]], axis=1)

    # Return the final list of DataFrames
    return qdot



###########################################
###########################################
#### definitions for plotting
def plot_results_time(results, dt, timestamp_df, graph_title):
    # Create traces for each layer
    traces = []
    for i, temp_array in enumerate(np.array(results).T):
        trace = go.Scatter(
            x=timestamp_df.index,
            y=temp_array,
            mode='lines',
            name=f'Layer {i}'
        )
        traces.append(trace)

    # Create the plot layout
    layout = go.Layout(
        title=graph_title,
        xaxis=dict(title='Date Time'),  # Update x-axis title
        yaxis=dict(title='Temperature °C'),
    )

    # Create the figure and plot it
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_tanklayers_time(temperatures, dt, numsteps, timestamp_df, graph_title):
    # Create traces for each layer
    traces = []
    for i, temp_array in enumerate(np.array(temperatures).T):
        #time_passed = dt * np.arange(len(temp_array))  # Calculate the time passed for each time step
        trace = go.Scatter(
            x= timestamp_df.index,  # Use time_passed as x-axis values
            y=temp_array,
            mode='lines',
            name=f'Layer {i}, {temperatures.columns[i]}'
        )
        traces.append(trace)

    # Create the plot layout
    layout = go.Layout(
        title=graph_title,
        xaxis=dict(title='Date Time'),  # Update x-axis title
        yaxis=dict(title='Temperature °C'),
    )

    # Create the figure and plot it
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_results_height(results, heights, dt, z, dz, graph_title):
    # Create traces for each time step
    traces = []
    for i, temp_array in enumerate(results):
        time_passed = dt * i
        trace = go.Scatter(
            x=heights,
            y=temp_array,
            mode='lines',
            name=f'{time_passed} seconds'
        )
        traces.append(trace)
        
    # Create shapes for vertical dotted lines at intervals of dz
    vertical_lines_x = list(np.arange(dz, z, dz))  # Adjusted range
    shapes = [{
        'type': 'line',
        'x0': x,
        'x1': x,
        'y0': 0,
        'y1': 1,
        'xref': 'x',
        'yref': 'paper',
        'line': {
            'color': 'grey',
            'width': 1,
            'dash': 'dot'
        }
    } for x in vertical_lines_x]
    
    # Create the plot layout
    layout = go.Layout(
        title=graph_title,
        xaxis=dict(title='Height', range=[0, z]),  # Adjust x-axis range from 0 to z
        yaxis=dict(title='Temperature'), #, range=[0, 100]),
        legend=dict(title='Time Passed'),
        shapes=shapes
    )

    # Create the figure and plot it
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_cX_time(connection_list, column, dt, freq, numsteps, timestamp_df, graph_title):
    # Create a list to hold all traces
    naming = ["Temperature °C", "Mass rate kg/s", "Energy rate W"]
    traces = []
    count = 0
    # Iterate through the list of lists
    for connection_data in connection_list:
        count +=1
        layer_num, df = connection_data  # Extract layer number and DataFrame
        cX_array = np.array(df.iloc[::freq, column].iloc[:numsteps+1])  # Extract the desired column (0=t, 1= mdot, 2 = mt)
        trace = go.Scatter(
            x=timestamp_df.index,  # Use time_passed as x-axis values
            y=cX_array,
            mode='lines',
            name=f'c{count}:Layer {layer_num}'
        )
        traces.append(trace)

    # Create the plot layout
    layout = go.Layout(
        title=graph_title,
        xaxis=dict(title='Date Time'),  # Update x-axis title
        yaxis=dict(title=naming[column]),
    )

    # Create the figure and plot it
    fig = go.Figure(data=traces, layout=layout)
    fig.show()



###########################################
###########################################
# Error metric calculations
def calculate_errors(results_df, t_init_interpolated_df):
    # Initialize lists to store the errors for each layer
    rmse_list = []
    mae_list = []
    maxae_list = []
    
    # Ensure both DataFrames have the same number of columns
    assert results_df.shape[1] == t_init_interpolated_df.shape[1], "DataFrames must have the same number of columns"
    
    # Iterate through the columns by index
    for i in range(results_df.shape[1]):
        # Extract the corresponding columns from each DataFrame
        simulated = results_df.iloc[:, i]
        measured = t_init_interpolated_df.iloc[:, i]
        
        # Calculate the errors
        rmse = np.sqrt(np.mean((simulated - measured) ** 2))
        mae = np.mean(np.abs(simulated - measured))
        maxae = np.max(np.abs(simulated - measured))
        
        # Store the errors in the respective lists
        rmse_list.append(rmse)
        mae_list.append(mae)
        maxae_list.append(maxae)
    
    # Convert lists to DataFrames for better readability
    error_df = pd.DataFrame({
        'RMSE': rmse_list,
        'MAE': mae_list,
        'MaxAE': maxae_list
    }, index=results_df.columns)
    
    # Calculate overall errors
    overall_rmse = np.sqrt(np.mean((results_df.values - t_init_interpolated_df.values) ** 2))
    overall_mae = np.mean(np.abs(results_df.values - t_init_interpolated_df.values))
    overall_maxae = np.max(np.abs(results_df.values - t_init_interpolated_df.values))
    
    # Append the overall errors to the DataFrame
    overall_errors = pd.DataFrame({
        'RMSE': [overall_rmse],
        'MAE': [overall_mae],
        'MaxAE': [overall_maxae]
    }, index=['Overall'])
    
    # Combine individual layer errors with overall errors
    error_df = pd.concat([error_df, overall_errors])
    
    return error_df



