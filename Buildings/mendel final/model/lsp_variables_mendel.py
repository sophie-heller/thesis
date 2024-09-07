import numpy as np
import pandas as pd
from functions_mendel import assign_and_interpolate_layers, process_mdot_normal, process_qdot_normal, resample_and_interpolate, resample_and_interpolate_in_list
#from Buildings.mendel.model.functions_mendel import create_zero_tuple, 
import os
y = os.getcwd()

##############################################
##############################################
##############################################
##############################################
# SIMULATION

dt = 8                             # length of the time steps [s]
start_datetime = pd.to_datetime("2024-05-21 13:40:00+00:00")  # Start of the simulation
num_steps = 500                   # number of time steps to be made



# frequency for the resampling of the data to fit the desired dt
freq_resample = f"{dt}S"




##############################################
##############################################
##############################################
##############################################
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

# Heights of entries (streams and heat exchangers)
heights_positions_tank = [1.843, 
                          1.443, 
                          1.343, 
                          1.243, 
                          1.143, 
                          1.043, 
                          0.728, 
                          0.628, 
                          0.278, 
                          0.036, 
                          0.036]  # all positions (1-12) except position 10 (schichtlader)
heights_schichtlader = [1.419, 
                        1.112, 
                        0.825, 
                        0.528]                                                     # position of the openings of the Schichtlader
heights_sensors = [1.643, 
                   1.343,
                   1.043, 
                   0.278, 
                   0.153]                                            # ** position of the 5 installed sensors in LSP Mendel. 

# assign the height of each connection c (9 connections)
c_z = [heights_positions_tank[0],   #c1 pos1
       heights_positions_tank[1],   #c2 pos2
       heights_positions_tank[2],   #c3 pos3
       heights_positions_tank[3],   #c4 pos4
       heights_positions_tank[5],   #c5 pos6
       heights_positions_tank[6],   #c6 pos7
       heights_positions_tank[8],   #c7 pos9
       heights_positions_tank[9],   #c9 pos11
       heights_positions_tank[10]]  #c10 pos12

schichtlader_status = True
l_z = [heights_schichtlader] # #c8 (1 Schichtlader)
# assign the height of the heat exchangers (1 exchanger)
e_z = [0] # [heights_positions_tank[n]] or  [0] -> no heat exhanger


# Assign the fictive layers to the physical entries
layernumber_c =  (np.array(c_z) / dz).astype(int)   # connections (streams)
layernumber_l = (np.array(heights_schichtlader) / dz).astype(int)    # layerloader (Schichtlader)
layernumber_e = (np.array(e_z) / dz).astype(int)    # heat exchanger
layernumber_sp = (np.array(heights_sensors) / dz)    # temperature sensors as float to be able to determine which sensor to use for layers with multiple sensors

##################################### IMPORTING THE DATA
# ALL DATA IS STORED IN "path\path_csv" folder, where all .csv files are contained
path = r'C:\Users\sophi\repos\repos_thesis'
path_csv = "\\Buildings\\mendel copy\\connections_mendel_lsp"

# MASS STREAM: Each connector data is saved into a separate .csv file with the name cX (X=#streams). Columns = date-time, t_cX [°C], mdot_cX [kg/s]
# HEAT EXCHANGER: Each heat exchanger data is as a separate .csv file with the name eX (X=#exchangers). Columns = date-time, pel_eX [W]
# TANK SENSORS: Each sensor is stored in a separate .csv file named t_tank_X (X=#sensor). Columns = date-time, t_tank_X [°C]
#               alt: the complete sensor information is stored in one .csv file with sensors as columns.

########################################################
########################################################
####### TANK TEMPERATURE SENSORS
#### TEMPERATURE SENSORS have to be given as SERIES, name of series is the sensor name. date-time as index

# import data from csv as series:
t_sp_tot = pd.read_csv(f"{path}{path_csv}\\sp.csv",    
                 parse_dates=['timestamp'],
                 index_col='timestamp') 

t_sp1 = [layernumber_sp[0], t_sp_tot["t_lsp1"]] # pos1 TLSPos
#t_sp1[1] = resample_and_interpolate(t_sp1[1], freq_resample)

t_sp2 = [layernumber_sp[1], t_sp_tot["t_lsp2"]] # pos4 TLSPo
#t_sp2[1] = resample_and_interpolate(t_sp2[1], freq_resample)

t_sp3 = [layernumber_sp[2], t_sp_tot["t_lsp3"]] # pos6 TLSPm
#t_sp3[1] = resample_and_interpolate(t_sp3[1], freq_resample)

#t_sp4 = [layernumber_sp[3], t_sp_tot["t_lsp4"]] # pos9         After-Installed manually, left out because temps are higher than top layer (and its on the bottom) 
#t_sp4[1] = resample_and_interpolate(t_sp1[1], freq_resample)

t_sp5 = [layernumber_sp[4], t_sp_tot["t_lsp5"]] # pos11 TLSPu
#t_sp5[1] = resample_and_interpolate(t_sp5[1], freq_resample)

########## TANK SENSOR MATRIX
# store all sensor data in a list
t_sp = [t_sp1, t_sp2, t_sp3, t_sp5] # t_sp4, t_sp5]
resample_and_interpolate_in_list(t_sp, freq_resample)


# create the matrix
t_init_interpolated_df = assign_and_interpolate_layers(layers, t_sp)


########################################################
########################################################
###### CONNECTION DATA (MDOT; QDOT) ASSIGNATION TO LAYER
###### CONNECTION DATA cX as list [position of sensor, df with sensor data] -> df with sensor data = tm, mdot as columns and date-time as index

# import data as df and store it as list 
c1 = [layernumber_c[0], pd.read_csv(f"{path}{path_csv}\\c1.csv",    # HW VL
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c2 = [layernumber_c[1], pd.read_csv(f"{path}{path_csv}\\c2.csv",    # HPHG VL (=0)
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c3 = [layernumber_c[2], pd.read_csv(f"{path}{path_csv}\\c3.csv",    # B VL          c3 = layernumber_c[2],
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c4 = [layernumber_c[3], pd.read_csv(f"{path}{path_csv}\\c4.csv",    # HPHG RL (=0)
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c5 = [layernumber_c[4], pd.read_csv(f"{path}{path_csv}\\c5.csv",    # B RL         c5 = layernumber_c[4],
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c6 = [layernumber_c[5], pd.read_csv(f"{path}{path_csv}\\c6.csv",    # L VL
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c7 = [layernumber_c[6], pd.read_csv(f"{path}{path_csv}\\c7.csv",    # HW RL
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c8 = [layernumber_l, pd.read_csv(f"{path}{path_csv}\\c8.csv",       # schichtlader! HP VL (=0)
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c9 = [layernumber_c[7], pd.read_csv(f"{path}{path_csv}\\c9.csv",    # L RL
                 parse_dates=['timestamp'],
                 index_col='timestamp')]

c10 = [layernumber_c[8], pd.read_csv(f"{path}{path_csv}\\c10.csv",    # HP RL (=0)
                 parse_dates=['timestamp'],
                 index_col='timestamp')]


#### STREAM CONNECTIONS MATRIX
# list of all "normal" connections (stream flows in (+ value) or out (- value) where height is defined)
mdot_normal = [c1, c2, c3, c4, c5, c6, c7, c9, c10] # c8 schichtlader is considered for each time step in simulation
resample_and_interpolate_in_list(mdot_normal, freq_resample)

if schichtlader_status == True:
    resample_and_interpolate_in_list([c8], freq_resample)                  # manually enter connection data of Schichtlader cX

# Store the list of dataframes containing the stream information (t and mdot). One df per max number of streams in one layer.
tm_list_df, mdot_list_df = process_mdot_normal(mdot_normal, layers, schichtlader_status, layernumber_c, layernumber_l)


########################################################
########################################################
###### HEAT EXCHANGER (MDOT; QDOT) ASSIGNATION TO LAYER
######CONNECTION DATA cX as list [position of sensor, df with sensor data] -> df with sensor data = tm, mdot as columns and date-time as index

# import data as df and store it as list 
# Connect heat exchanger data eX [position of sensor, df with sensor data]
# check if there is a heat exchanger connected (through defined e_z)
if e_z[0] == 0: # no heat exchanger
    e1 = [0, t_sp1[1].copy()]
    e1[1].loc[:] = 0
    e1[1].rename("pel", inplace=True)
else:   # 1 heat exchanger
    e1 = [layernumber_e, pd.read_csv(f"{path}\\Buildings\\seeweg\\connections_seeweg\\e1.csv",    
                 parse_dates=['timestamp'],
                 index_col='timestamp')]


#### EXCHANGERS CONNECTIONS MATRIX
# list of all heat exchanger data ([e1] if no heat exchanger is present)
qdot_normal = [e1] # e1 is filled with 0 if no heat exchanger (e_z=[0])
resample_and_interpolate_in_list(qdot_normal, freq_resample)


# Store the list of df containing the heat per layer. One df per max heat exchangers present in one layer.
qdot_list_df = process_qdot_normal(qdot_normal, layers, layernumber_e)



T_a = 20                       # ambient temperature

################# START DATETIME GIVEN
# Calculate the index of the dataframes with timestamp as given in start_datetime
# return "start" variable with the index. This start will be the filter for all dfs to be passed to the simulation. Simulation will start at this datetime or closest value.

# Get all the available datetime indexes
datetime_index = t_init_interpolated_df.index

# Try to find the exact datetime
try:
    start = datetime_index.get_loc(start_datetime)
    print(f"Exact start datetime found at index: {start}")

except KeyError:
    # If exact datetime not found, get the closest available datetime
    closest_index = datetime_index.get_indexer([start_datetime], method="nearest")[0]
    start = closest_index  # Use closest index as the start
    closest_datetime = datetime_index[closest_index]
    start_datetime = closest_datetime  # Update start_datetime to the closest available one
    print(f"Exact start_datetime ", start_datetime, "not found. Continuing with closest available datetime: {start_datetime} at index {start} of the original interpolated dataframes.")


end = start + num_steps+1


############## pass filtered dataframes with data between start and end for the simulation
t_init_interpolated_df = t_init_interpolated_df.iloc[start:end]     # layer temperatures (sensor or interpolated) over time
mdot_list_df = [dff.iloc[start:end] for dff in mdot_list_df ]       # list of mdots with max one stream per layer, n dfs correspond to the max of streams n that can appear on any layer

tm_list_df = [dff.iloc[start:end] for dff in tm_list_df ]           # list of temperatures of the streams with max one stream per layer
#mt_list_df = [dff.iloc[start:end] for dff in mt_list_df ]           # list of mcpT with max one stream per layer
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



# Check for NaN values
# Initialize an empty set to collect unique indices with NaNs
nan_indices = set()

# Iterate over the t_sp list to collect all indices with NaNs (general data)
for item in t_sp:
    nan_indices.update(item[1][item[1].isna()].index)

for item in connection_list:
# Include NaN coming from the data cleaning of c1
    nan_indices.update(item[1][item[1].isna().any(axis=1)].index)

# Convert the set to a sorted list (optional)
nan_indices = sorted(nan_indices)





