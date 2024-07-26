import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import expit
import importlib.util
import os

##### definition of the logistic function
def step(T1, T2):
    m = 1 # miu in paper 4_1D.. is set to 1
    # s = 1/(1 + np.exp(-m*(T1-T2))) # UNSTABLE
    #s = expit(m * (T1 - T2))
    #return s
    T1 = np.asarray(T1)
    T2 = np.asarray(T2)
    
    return np.where(T1 >= T2, 1, 0)

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

def plot_tanklayers_time(temperatures, dt, freq, numsteps, timestamp_df, graph_title):
    # Create traces for each layer
    traces = []
    for i, temp_array in enumerate(np.array(temperatures.iloc[::freq, :].iloc[:numsteps+1, :]).T):
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


def import_variables(file_name):
    try:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print("Current directory:", current_dir)

        # Construct the full path to the file
        file_path = os.path.join(current_dir, f"{file_name}.py")
        print("File path:", file_path)

        # Use importlib to create a module from the file
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Import variables directly into the global namespace
        for name in dir(module):
            if not name.startswith('__'):
                globals()[name] = getattr(module, name)

        print("Variables imported successfully:", globals().keys())

    except FileNotFoundError:
        print(f"File '{file_name}.py' not found.")
    except Exception as e:
        print(f"Error occurred while importing '{file_name}.py': {e}")


def create_zero_tuple(n, T_zero):
  """
  Creates a tuple filled with n zero arrays.

  Args:
      n: The number of zero arrays to include in the tuple.

  Returns:
      A tuple containing n zero arrays.
  """
  zero_tuple = ()
  for _ in range(n):
    zero_tuple += (np.zeros(len(T_zero)),)  # Replace (...) with desired array shape

  return zero_tuple




def create_input_mat(num_entries, vector):
  """
  Creates a NumPy array by stacking the given Qdot0 array vertically num_entries times.

  Args:
      num_entries: The number of times to stack the Qdot0 array vertically -> number of time stamps
      vector: np.array containing the vector to be stacked

  Returns:
      A NumPy array with the stacked Qdot0 arrays Qdot0[n,m], with n being num_entries and m the lenght of the given vector
  """
  input_mat = np.vstack([vector for _ in range(num_entries)])
  return input_mat



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



