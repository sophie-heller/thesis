import numpy as np
import pandas as pd
from functions_hri import assign_and_interpolate_layers, process_mdot_normal, process_qdot_normal
#from Buildings.HRI.functions_hri import create_zero_tuple
import os
y = os.getcwd()

############### DEFINE TANK
# HRI1
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
k_i = 0.5                           # thermal condunctance of the wall [W/m²K], with insulation 0.1m and spec heat conduct 0.05 W/mK
# losses: Beta calculations
beta_i = (P_i*k_i)/(rho * cp * A_i) # coefficient of heat losses through the wall of the tank [1/s]
beta_top = (k_i/(rho * cp * dz))    # coefficient of heat losses through the ceiling of the tank [1/s]
beta_bottom = (k_i/(rho * cp * dz)) # coefficient of heat losses through the ground of the tank [1/s]
# indirect heat 
lambda_i = (1/(A_i*rho*cp))         # coefficient of the indirect heat (heat echanger)
# direct heat
phi_i = (1/(A_i*rho))               # coefficient of the direct heat (mass stream)

# Heights of entries
heights_positions_tank = [1.843, 0.05]  # port_a on top and port_b at the bottom

heights_schichtlader = []                                                     # posiiton of the openings of the Schichtlader

heights_sensors = [z/(10)*9+z/20,       # sensor layer 9 in the middle of 9th section
                   z/(10)*8+z/20,       # sensor layer 8 in the middle of 8th section
                   z/(10)*7+z/20,       # sensor layer 7 in the middle of 7th section
                   z/(10)*6+z/20,       # sensor layer 6 in the middle of 6th section
                   z/(10)*5+z/20,       # sensor layer 5 in the middle of 5th section
                   z/(10)*4+z/20,       # sensor layer 4 in the middle of 4th section
                   z/(10)*3+z/20,       # sensor layer 3 in the middle of 3th section
                   z/(10)*2+z/20,       # sensor layer 2 in the middle of 2th section
                   z/(10)*1+z/20,       # sensor layer 1 in the middle of 1th section
                   z/20                 # sensor layer 0 in the middle of 0th section
                   ]         


# assign the height of the existing connections (streams) in the tank (direct mechanism)
# regular inlet/outlet
c_z = [heights_positions_tank[0], heights_positions_tank[1]]
# Layer loader (Schichtlader), enter the different possible exists of the lanze. Enter "[0]" if no Schichtlader is present in the tank
schichtlader_status = False
l_z = [0] # [heights_schichtlader] or [0]-> no Schichtlader
# assign the height where the heat exchangers  are connected to the tank. Enter "[0]" if no indirect mechanisms is present in the tankk
e_z = [0] #  -> no heat exhanger

# Assign the fictive layers to the physical entries
layernumber_c =  (np.array(c_z) / dz).astype(int)   # connections (streams)         (is the same as // dz)
layernumber_l = (np.array(l_z) / dz).astype(int)    # layerloader (Schichtlader)
layernumber_e = (np.array(e_z) / dz).astype(int)    # heat exchanger
layernumber_sp = np.array(heights_sensors)/dz       # temperature sensors as float to be able to determine which sensor to use for layers with multiple sensors


##################################### IMPORTING THE DATA
# ALL DATA IS STORED IN "path\path_csv" folder, where all .csv files are contained
path = r'C:\Users\sophi\repos\repos_thesis'
path_csv = "\\Buildings\\HRI\\connections_hri"

# MASS STREAM: Each connector data is saved into a separate .csv file with the name cX (X=#streams). Columns = date-time, t_cX [°C], mdot_cX [kg/s]
# HEAT EXCHANGER: Each heat exchanger data is as a separate .csv file with the name eX (X=#exchangers). Columns = date-time, pel_eX [W]
# TANK SENSORS: Each sensor is stored in a separate .csv file named t_tank_X (X=#sensor). Columns = date-time, t_tank_X [°C]
#               alt: the complete sensor information is stored in one .csv file with sensors as columns.

########################################################
########################################################
####### TANK TEMPERATURE SENSORS
#### TEMPERATURE SENSORS have to be given as SERIES, name of series is the sensor name. date-time as index

# import data from csv as series
t_sp9 = [layernumber_sp[0], pd.read_csv(f"{path}{path_csv}\\t_tank_9.csv",    
                 parse_dates=['date-time'],
                 index_col='date-time') ] # layer 9
t_sp9[1] = t_sp9[1].squeeze()               # convert to Series

t_sp8 = [layernumber_sp[1], pd.read_csv(f"{path}{path_csv}\\t_tank_8.csv",    
                 parse_dates=['date-time'],
                 index_col='date-time') ] # layer 8
t_sp8[1] = t_sp8[1].squeeze()               # convert to Series

t_sp7 = [layernumber_sp[2], pd.read_csv(f"{path}{path_csv}\\t_tank_7.csv",    
                 parse_dates=['date-time'],
                 index_col='date-time') ] # layer 7
t_sp7[1] = t_sp7[1].squeeze()               # convert to Series

t_sp6 = [layernumber_sp[3], pd.read_csv(f"{path}{path_csv}\\t_tank_6.csv",    
                 parse_dates=['date-time'],
                 index_col='date-time') ] # layer 6
t_sp6[1] = t_sp6[1].squeeze()               # convert to Series

t_sp5 = [layernumber_sp[4], pd.read_csv(f"{path}{path_csv}\\t_tank_5.csv",    
                 parse_dates=['date-time'],
                 index_col='date-time') ] # layer 5
t_sp5[1] = t_sp5[1].squeeze()               # convert to Series

t_sp4 = [layernumber_sp[5], pd.read_csv(f"{path}{path_csv}\\t_tank_4.csv",    
                 parse_dates=['date-time'],
                 index_col='date-time') ] # layer 4
t_sp4[1] = t_sp4[1].squeeze()               # convert to Series

t_sp3 = [layernumber_sp[6], pd.read_csv(f"{path}{path_csv}\\t_tank_3.csv",    
                 parse_dates=['date-time'],
                 index_col='date-time') ] # layer 3
t_sp3[1] = t_sp3[1].squeeze()               # convert to Series

t_sp2 = [layernumber_sp[7], pd.read_csv(f"{path}{path_csv}\\t_tank_2.csv",    
                 parse_dates=['date-time'],
                 index_col='date-time') ] # layer 2
t_sp2[1] = t_sp2[1].squeeze()               # convert to Series

t_sp1 = [layernumber_sp[8], pd.read_csv(f"{path}{path_csv}\\t_tank_1.csv",    
                 parse_dates=['date-time'],
                 index_col='date-time') ] # layer 1
t_sp1[1] = t_sp1[1].squeeze()               # convert to Series

t_sp0 = [layernumber_sp[9], pd.read_csv(f"{path}{path_csv}\\t_tank_0.csv",    
                 parse_dates=['date-time'],
                 index_col='date-time') ] # layer 0
t_sp0[1] = t_sp0[1].squeeze()               # convert to Series


########## TANK SENSOR MATRIX
# store all sensor data in a list
t_sp = [t_sp0, t_sp1, t_sp2, t_sp3, t_sp4, t_sp5, t_sp6, t_sp7, t_sp8, t_sp9]


# create the matrix
t_init_interpolated_df = assign_and_interpolate_layers(layers, t_sp)



########################################################
########################################################
###### CONNECTION DATA (MDOT; QDOT) ASSIGNATION TO LAYER
###### CONNECTION DATA cX as list [position of sensor, df with sensor data] -> df with sensor data = tm, mdot as columns and date-time as index

# import data as df and store it as list 
c1 = [layernumber_c[0], pd.read_csv(f"{path}{path_csv}\\porta.csv",    # Port A on top of the tank.
                 parse_dates=['date-time'],
                 index_col='date-time')]

c2 = [layernumber_c[1], pd.read_csv(f"{path}{path_csv}\\portb.csv",    # Port B at the bottom of the tank
                 parse_dates=['date-time'],
                 index_col='date-time')]


#### STREAM CONNECTIONS MATRIX
# list of all "normal" connections (stream flows in (+ value) or out (- value) where height is defined)
mdot_normal = [c1, c2] # exclude connection with Schichtlader, this will be handled in main code (each step must be evaluated to determine layer). The empty space for this assignation is created by process_mdot_normal :)

# Store the list of dataframes containing the stream information (t and mdot). One df per max number of streams in one layer.
tm_list_df, mdot_list_df = process_mdot_normal(mdot_normal, layers, layernumber_c, layernumber_l)



########################################################
########################################################
###### HEAT EXCHANGER (MDOT; QDOT) ASSIGNATION TO LAYER
######CONNECTION DATA cX as list [position of sensor, df with sensor data] -> df with sensor data = tm, mdot as columns and date-time as index

# import data as df and store it as list 
# Connect heat exchanger data eX [position of sensor, df with sensor data]
# check if there is a heat exchanger connected (through defined e_z)
if e_z[0] == 0: # no heat exchanger, create df filled with 0 for "e1"
    e1 = [0, t_sp1[1].copy()]
    #print(type(e1[1]))
    e1[1].loc[:] = 0
    e1[1].rename("pel")
# import data as df and store it as list 
else:   # 1 heat exchanger
    e1 = [layernumber_e, pd.read_csv(f"{path}\{path_csv}\\e1.csv",    
                 parse_dates=['date-time'],
                 index_col='date-time')]

#### EXCHANGERS CONNECTIONS MATRIX
# list of all heat exchanger data ([e1] if no heat exchanger is present)
qdot_normal = [e1] # e1 is filled with 0 if no heat exchanger (e_z=[0])



# Store the list of df containing the heat per layer. One df per max heat exchangers present in one layer.
qdot_list_df = process_qdot_normal(qdot_normal, layers, layernumber_e)


##############################################
##############################################
##############################################
##############################################
# SIMULATION
dt = 60                             # length of the time steps [s]
#freq = int(dt/60)                         # frequency of the stored values to match the time step of the simulation dt
num_steps = 15000                   # number of time steps to be made
#select rows of the data
start = 200
end = start + num_steps+1

T_a = 20                       # ambient temperature


############## pass only filtered dataframes with start and end
t_init_interpolated_df = t_init_interpolated_df.iloc[start:end]     # layer temperatures (sensor or interpolated) over time
mdot_list_df = [dff.iloc[start:end] for dff in mdot_list_df ]       # list of mdots with max one stream per layer, n dfs correspond to the max of streams n that can appear on any layer

tm_list_df = [dff.iloc[start:end] for dff in tm_list_df ]           # list of temperatures of the streams with max one stream per layer
#mt_list_df = [dff.iloc[start:end] for dff in mt_list_df ]           # list of mcpT with max one stream per layer
#mt_all_streams = pd.concat([df.drop(columns=['mt']) for df in mt_list_df], axis=1)
qdot_list_df = [dff.iloc[start:end] for dff in qdot_list_df ]       # list of heat connection with max one connection per layer

c1_df = c1.copy()                       #c1
c1_df[1] = c1_df[1].iloc[start:end]
c2_df = c2.copy()                       #c2
c2_df[1] = c2_df[1].iloc[start:end]
e1_df = e1.copy()                       #e1
e1_df[1] = e1_df[1].iloc[start:end]

connection_list = [c1_df, c2_df]



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





