import numpy as np
import pandas as pd
from functions_seeweg import create_zero_tuple, create_input_mat
import os

############### DEFINE TANK
# Solvis Stratos 917l

# tank description
z = 1.873                               # height of the tank [m]
layers = 10
dz = z / layers                # height of the section (layer)
d = 0.79                             # diameter of the cross section [m]
P_i = np.pi * d                     # cross-sectional perimeter [m]
A_i = np.pi * (d/2)**2              # cross-sectional area [m²]
# material properties
alpha = 0.000000146                 # heat diffusivity of the fluid [m²/s]
rho = 998                           # density of the fluid [kg/m³]
cp =  4186                          # heat capacity of the fluid [J/kgK]
k_i = 0.5                           # thermal condunctance of the wall [W/m²K]
# losses: Beta calculations
beta_i = (P_i*k_i)/(rho * cp * A_i) # coefficient of heat losses through the wall of the tank [1/s]
beta_top = (k_i/(rho * cp * dz))    # coefficient of heat losses through the ceiling of the tank [1/s]
beta_bottom = (k_i/(rho * cp * dz)) # coefficient of heat losses through the ground of the tank [1/s]
# indirect heat 
lambda_i = (1/(A_i*rho*cp))         # coefficient of the indirect heat (heat echanger)
# direct heat
phi_i = (1/(A_i*rho))               # coefficient of the direct heat (mass stream)


# Heights of entries
heights_positions_tank = [1.843, 1.443, 1.343, 1.243, 1.143, 1.043, 0.728, 0.628, 0.278, 0.036, 0.036]  # all positions (1-12) except position 10 (schichtlader)
heights_schichtlader = [1.419, 1.112, 0.825, 0.528]                                                     # posiiton of the openings of the Schichtlader
heights_sensors = [1.643, 1.243, 1.043, 0.728, 0.278, 0.153]                                            # ** position og the 6 installed sensors in Seeweg

# assign the height of each connection c (9 connections)
c_z = [heights_positions_tank[0], heights_positions_tank[1], heights_positions_tank[3], heights_positions_tank[5], 
       heights_positions_tank[6], heights_positions_tank[7], heights_positions_tank[8], heights_positions_tank[9], heights_positions_tank[10]]
l_z = [heights_schichtlader] # (1 Schichtlader)
# assign the height of the heat exchangers (1 exchanger)
e_z = [heights_positions_tank[2]]

# Assign the fictive layers to the physical entries
layernumber_c =  (np.array(c_z) / dz).astype(int)   # connections (streams)
layernumber_l = (np.array(heights_schichtlader) / dz).astype(int)    # layerloader (Schichtlader)
layernumber_e = (np.array(e_z) / dz).astype(int)    # heat exchanger
layernumber_sp = (np.array(heights_sensors) / dz)    # temperature sensors as float to be able to determine which sensor to use for layers with multiple sensors

##################################### IMPORTING THE DATA
# Each connector data is saved into a separate .csv file with the name cX (X=#streams). Columns = timestamp, t_cX, v_cX, mt_cX. mt_cX is not relevant.


###################################### TANK TEMPERATURE SENSORS
############### IMPORT DATA
#os.chdir(r'C:\Users\sophi\repos\repos_thesis\Buildings\seeweg\connections_seeweg')
path = r'C:\Users\sophi\repos\repos_thesis'
#### TEMPERATURE SENSORS
t_sp_tot = pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\sp.csv",    
                 parse_dates=['timestamp'],
                 index_col='timestamp') 

t_sp1 = [layernumber_sp[0], t_sp_tot["t_sp1"]] #TSPO # pos1

t_sp2 = [layernumber_sp[1], t_sp_tot["t_sp2"]] # pos4

t_sp3 = [layernumber_sp[2], t_sp_tot["t_sp3"]] # pos6

t_sp4 = [layernumber_sp[3], t_sp_tot["t_sp4"]] # pos7

t_sp5 = [layernumber_sp[4], t_sp_tot["t_sp5"]] # pos9

t_sp6 = [layernumber_sp[5], t_sp_tot["t_sp6"]] #TSPU # pos11

t_sp = [t_sp1, t_sp2, t_sp3, t_sp4, t_sp5, t_sp6]




################# TANK SENSORS ASSIGNATION TO LAYER: 
##### Assing t_init by checking all sensor positions and the defined layers. Select the sensor closest to the middle of the layer or NaN if no sensor is in the layer
# Initialize t_init list
t_init = []   # contains list with layer and sensor info [layer, df]
t_init_sensordf = [] # contains only the sensor data [df]


# Iterate through each layer index from 0 to layers-1
for n in range(layers):
    # Initialize variables to track the closest sensor
    closest_diff = np.inf
    closest_t_sp = [np.nan, None]  # Initialize with NaN values
    
    # Iterate through each t_spX in t_sp
    for t_spX in t_sp:
        # Get the layer number associated with t_spX and its difference from n + 0.5
        layernum = t_spX[0]
        diff = np.abs(layernum - (n + 0.5))
        
        # Check if this t_spX can be used for the current layer and has a smaller diff
        if diff <= 0.5 and diff < closest_diff:
            closest_diff = diff
            closest_t_sp = t_spX
    
    # Append the closest found sensor to t_init
    t_init.append(closest_t_sp)
    t_init_sensordf.append(closest_t_sp[1])

# Now t_init contains the selected temperature sensor data (closest to n + 0.5) for each layer
# Each entry in t_init will be a list like [layer number, temperature data] or [np.nan, None] if no sensor was found



###### calculate linear interpolation if layer has no sensor
# Initialize t_init list
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
        
        # Find the nearest layers with valid temperature data
        for layer in t_init[:n][::-1]:  # Search backwards for layer_below
            if layer[1] is not None:
                layer_below = layer
                break
        
        for layer in t_init[n+1:]:  # Search forwards for layer_above
            if layer[1] is not None:
                layer_above = layer
                break
        
        # Perform interpolation or duplicate closest valid value
        if n == 0 and layer_above is not None:
            # Duplicate closest value from layer_above
            t_init_interpolated.append(layer_above[1])
        elif n == layers - 1 and layer_below is not None:
            # Duplicate closest value from layer_below
            t_init_interpolated.append(layer_below[1])
        elif layer_below is not None and layer_above is not None:
            # Perform linear interpolation
            layernum_below = layer_below[0]
            layernum_above = layer_above[0]
            temp_below = layer_below[1]
            temp_above = layer_above[1]
            
            # Calculate interpolated temperature
            interpolated_temp = temp_below + ((n + 0.5 - layernum_below) / (layernum_above - layernum_below)) * (temp_above - temp_below)
            
            # Append the interpolated temperature to t_init_interpolated
            #t_init_interpolated.append([n + 0.5, interpolated_temp])
            t_init_interpolated.append(interpolated_temp)
        else:
            # If no valid layers found for interpolation or duplication, append NaN values
            print(f"Sensor data for layer {n} is NaN after interpolation. Check!")
            #t_init_interpolated.append([np.nan, None])  # Replace None with np.nan if needed for temperature data
            t_init_interpolated.append([None])  # Replace None with np.nan if needed for temperature data


t_init_interpolated_df = pd.concat(t_init_interpolated, axis=1)
#t_init_interpolated_arr = t_init_interpolated_df.to_numpy()

# Now t_init_interpolated contains the interpolated or duplicated temperature data for each layer
# Each entry in t_init_interpolated will be a list like [layer number, interpolated/duplicated temperature] (or [np.nan, None] if no interpolation/duplication was possible)

#t_init_interpolated1 = t_init_interpolated[][1]








###################################### CONNECTION DATA (MDOT; QDOT) ASSIGNATION TO LAYER
############### IMPORT DATA
#### CONNECTION DATA cX as [position of sensor, df with sensor data]
c1 = [layernumber_c[0], pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\c1.csv",    #WW VL
                 parse_dates=['timestamp'],
                 index_col='timestamp')]
#c1[1].iloc[:,1] = c1[1].iloc[:,1]*10
c2 = [layernumber_c[1], pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\c2.csv",    
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c3 = [layernumber_c[2], pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\c3.csv",    
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c4 = [layernumber_c[3], pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\c4.csv",    # WW RL1
                 parse_dates=['timestamp'],
                 index_col='timestamp')]
#c4[1].iloc[:,1] = c4[1].iloc[:,1]*10
c5 = [layernumber_c[4], pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\c5.csv",    
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c6 = [layernumber_c[5], pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\c6.csv",    
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c7 = [layernumber_c[6], pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\c7.csv",    
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c8 = [layernumber_l, pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\c8.csv",    # schichtlader!
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c9 = [layernumber_c[7], pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\c9.csv",    # WW RL2
                 parse_dates=['timestamp'],
                 index_col='timestamp')]
#c9[1].iloc[:,1] = c9[1].iloc[:,1]*10

c10 = [layernumber_c[8], pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\c10.csv",    
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

e1 = [layernumber_e, pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\e1.csv",    
                 parse_dates=['timestamp'],
                 index_col='timestamp')]


#### STREAM CONNECTIONS MATRIX
# Create a vector that contains the max amount of streams that are possible per layer (including Schichtlader)
layer_c_counts_vector = np.zeros(layers, dtype=int)
counts_c_normal = np.bincount(layernumber_c, minlength=layers)
counts_schichtlader = np.bincount(layernumber_l, minlength=layers)
layer_c_counts_vector[:len(counts_c_normal)] = counts_c_normal + counts_schichtlader
# Create an integer that represents the max amount of streams that appear in any layer of the tank

layer_c_counts_max = max(layer_c_counts_vector).astype(int)

# list of all "normal" connections (stream flows where height is defined)
mdot_normal = [c1, c2, c3, c4, c5, c6, c7, c9, c10]

# Initialize mdot_list as a list of lists
mdot_list = [[] for _ in range(layers)]
# Copy the DataFrame and fill all columns with 0
df_c_zero_filled = c1[1].copy()
df_c_zero_filled.loc[:, :] = 0
df_c_zero_filled.rename(columns={"t_c1": "t", "v_c1": "v", "mt_c1": "mt"}, inplace=True)

# Iterate through each layer to create a list mdot_list with all connected stream data (!! Schichtlader not included, this has to be evaluated on each time step in the simulation)
#        For the Schichtlader, overwrite the last entry of mdot_list (should be empty) for the corresponding layer where stream flows out
for n in range(layers):
    count = 0  # Initialize count of dataframes added to mdot_list[n]
    
    # Iterate through each normal mdot connection
    for cX in mdot_normal:
        if cX[0] == n:                      # cX[0] is layer number where cX is located. If cX is located in current layer n:
            mdot_list[n].append(cX[1])      # store t, mdot and mt of cX (cX[1])
            count += 1
    
    # Fill mdot_list[n] with df_zero_filled until layer_counts_max is achieved
    while count < layer_c_counts_max:
        mdot_list[n].append(df_c_zero_filled.copy())
        count += 1
        


# Create a list with count_c_max dataframes which contain the layers data (t_layer and vdot_layer as columns) for the intergation in the model
# Initialize tm and vdot as empty lists of lists
tm = [pd.DataFrame() for _ in range(layer_c_counts_max)]
mdot = [pd.DataFrame() for _ in range(layer_c_counts_max)]
mt = [pd.DataFrame() for _ in range(layer_c_counts_max)]

for m in range(layer_c_counts_max):
    for n in range(layers):
        tm[m] = pd.concat([tm[m], mdot_list[n][m].iloc[:,0]], axis=1)
        mdot[m] = pd.concat([mdot[m], mdot_list[n][m].iloc[:,1]*(rho/3600000)], axis=1)   # vdot*0.001*rho=mdot   convert vdot [l/h]*0,001 to [m³/h] and [m³/h]/ to [m³/s] ->/3600000 and multiply it by rho [kg/m³] to get mdot [kg/s]
        mt[m] = pd.concat([mt[m], mdot_list[n][m].iloc[:,2]*(rho/3600000)], axis=1) 

tm_list_df = tm
mdot_list_df = mdot
mt_list_df = mt










#### EXCHANGERS CONNECTIONS MATRIX
# Create a vector that contains the max amount of exchangers that are possible per layer
layer_e_counts_vector = np.zeros(layers, dtype=int)
counts_e = np.bincount(layernumber_e, minlength=layers)
layer_e_counts_vector[:len(counts_e)] = counts_e
# Create an integer that represents the max amount of streams that appear in any layer of the tank

layer_e_counts_max = max(layer_e_counts_vector).astype(int)

# list of all "normal" connections (stream flows where height is defined)
qdot_normal = [e1]

# Initialize qdot_list as a list of lists
qdot_list = [[] for _ in range(layers)]
# Copy the DataFrame and fill all columns with 0
df_e_zero_filled = e1[1].copy()
df_e_zero_filled.loc[:, :] = 0
df_e_zero_filled.rename(columns={"pel_hr": "pel"}, inplace=True)

# Iterate through each layer to create a list mdot_list with all connected stream data (!! Schichtlader not included, this has to be evaluated on each time step in the simulation)
#        For the Schichtlader, overwrite the last entry of mdot_list (should be empty) for the corresponding layer where stream flows out
for n in range(layers):
    count = 0  # Initialize count of dataframes added to mdot_list[n]
    
    # Iterate through each normal mdot connection
    for eX in qdot_normal:
        if eX[0] == n:
            qdot_list[n].append(eX[1])
            count += 1
    
    # Fill mdot_list[n] with df_zero_filled until layer_counts_max is achieved
    while count < layer_e_counts_max:
        qdot_list[n].append(df_e_zero_filled.copy())
        count += 1

# Create a list with count_e_max dataframes which contain the layers data (pel_layer as column) for the intergation in the model
# Initialize tm and vdot as empty lists of lists
qdot = [pd.DataFrame() for _ in range(layer_e_counts_max)]

for m in range(layer_e_counts_max):
    for n in range(layers):
        qdot[m] = pd.concat([qdot[m], qdot_list[n][m].iloc[:,0]], axis=1)


qdot_list_df = qdot

##############################################
dt = 1*60                             # length of the time steps [s]
#freq = int(dt/60)                         # frequency of the stored values to match the time step of the simulation dt
num_steps = 10000                   # number of time steps to be made
#select rows of the data
start = 1200 
end = start + 25000

#timestamp_df = pd.DataFrame(index=.index[::freq5][:num_steps+1]))

T_a = 20                       # ambient temperature

############## pass only filtered dataframes with start and end
t_init_interpolated_df = t_init_interpolated_df.iloc[start:end]     # layer temperatures (sensor or interpolated) over time
mdot_list_df = [dff.iloc[start:end] for dff in mdot_list_df ]       # list of mdots with max one stream per layer

tm_list_df = [dff.iloc[start:end] for dff in tm_list_df ]           # list of temperatures of the streams with max one stream per layer
mt_list_df = [dff.iloc[start:end] for dff in mt_list_df ]           # list of mcpT with max one stream per layer
#mt_all_streams = pd.concat([df.drop(columns=['mt']) for df in mt_list_df], axis=1)
qdot_list_df = [dff.iloc[start:end] for dff in qdot_list_df ]       # list of heat connection with max one connection per layer

c1_df = c1.copy()                       #c1
c1_df[1] = c1_df[1].iloc[start:end]
#c1_df[1].iloc[:,1] = c1_df[1].iloc[:,1]*10
c2_df = c2.copy()                       #c2
c2_df[1] = c2_df[1].iloc[start:end]
c3_df = c3.copy()                       #c3
c3_df[1] = c3_df[1].iloc[start:end]
c4_df = c4.copy()                       #c4
c4_df[1] = c4_df[1].iloc[start:end]
#c4_df[1].iloc[:,1] = c4_df[1].iloc[:,1]*10
c5_df = c5.copy()                       #c5
c5_df[1] = c5_df[1].iloc[start:end]
c6_df = c6.copy()                       #c6
c6_df[1] = c6_df[1].iloc[start:end]
c7_df = c7.copy()                       #c7
c7_df[1] = c7_df[1].iloc[start:end]
c8_df = c8.copy()                       #c8
c8_df[1] = c8_df[1].iloc[start:end]
c9_df = c9.copy()                       #c9
c9_df[1] = c9_df[1].iloc[start:end]
#c9_df[1].iloc[:,1] = c9_df[1].iloc[:,1]*10
c10_df = c10.copy()                     #c10
c10_df[1] = c10_df[1].iloc[start:end]
e1_df = e1.copy()                       #e1
e1_df[1] = e1_df[1].iloc[start:end]

connection_list = [c1_df, c2_df, c3_df, c4_df, c5_df, c6_df, c7_df, c8_df, c9_df, c10_df]
#cX_t_df = pd.concat([df.iloc[:, 0] for layer_number, df in connection_list], axis = 1)
#cX_mdot_df = pd.concat([df.iloc[:, 1] for layer_number, df in connection_list], axis = 1)
#cX_mt_df = pd.concat([df.iloc[:, 2] for layer_number, df in connection_list], axis = 1)








