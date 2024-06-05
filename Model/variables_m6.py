import numpy as np
from functions import create_zero_tuple, create_input_mat


############### VARIABLES
            
# initial temperature in the tank for each layer


# lenght of the selected T_zero defines the number of layers    
layers = 10
"""
T_1 = np.array([10, 10, 10, 10, 10, 90, 90, 90, 90, 90])
T_1a = np.array([90, 90, 90, 90, 90,10, 10, 10, 10, 10]) 
T_1b = np.array([90, 90, 10, 10, 10,10, 10, 10, 10, 10]) 

T_2 = np.array([10, 20, 30, 60, 50, 90, 30, 0, 15, 20])
T_3 = [100, 80, 50, 60, 20, 50, 10, 20, 50, 80]

T_4 = np.array([100, 90, 80 ,70, 60, 50, 40, 30, 20, 10])
T_4inv = T_4[::-1]

T_5 = [100, 10, 80 ,200, 60, 70, -10, 30, -10, 10]
T_6 = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]) 
T_7 = np.full(layers, 40) # standard initial temperature
"""
T_8 = np.full(layers, 40)
T_a = 20                       # ambient temperature

# tank description
T_zero = T_8                        # initial temperature vector
z = 2.099                               # height of the tank [m]
dz = z / len(T_zero)                # height of the section (layer)
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
lambda_i = (1/(A_i*rho*cp))         # coefficient of the input heat

# direct heat
phi_i = (1/(A_i*rho))

#mdot = np.zeros(len(T_zero))
#Tm = np.zeros(len(T_zero))

#mdot[2]=0.6
#Tm[2]=100


# create Qdot Matrix
#Qdot0 = np.zeros(len(T_zero))
#Qdot_an = np.copy(Qdot0)        # no charging
#Qdot_an[7] = 5000               # charging layer i=7

#Qdot_mat = np.vstack((Qdot0, Qdot0, Qdot0, Qdot_an, Qdot_an, Qdot_an, Qdot0,Qdot0,Qdot0,Qdot0,Qdot0)) # as a tupple
# Results looks like this for charging in layer i=7 in time steps 4,5,6
"""
    Qdot_mat=  [[   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
                [   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
                [   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
                [   0.    0.    0.    0.    0.    0.    0. 5000.    0.    0.]
                [   0.    0.    0.    0.    0.    0.    0. 5000.    0.    0.]
                [   0.    0.    0.    0.    0.    0.    0. 5000.    0.    0.]
                [   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
                [   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
                [   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
                [   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
                [   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]]
"""

# Display the original Qdot_mat
#print("Original Qdot_mat:\n", Qdot_mat)

# Step 3: Change the third number (i=2) from the second column (j=1)
#Qdot_mat[2, 1] = 1234  # Replace 1234 with the desired value

# Display the modified Qdot_mat
#print("Modified Qdot_mat:\n", Qdot_mat)



number_total_data = 10
v0 = np.zeros(len(T_zero))
mat0 = create_input_mat(number_total_data, v0)  # create matrix filled with 0

Qdot = np.copy(mat0)
Qdot[2:4,2] = 5000


mdot = np.copy(mat0)
mdot[5:7,7]=-1
mdot[5:7,6]=0.5
mdot[5:7,2] = 0.5


Tm = np.copy(mat0)
Tm[5:7,7]=60
Tm[5:7,6]=80
Tm[5:7,2] = 20 



#mdot_mat = np.vstack((mdot0, mdot_an, mdot_an, mdot0, mdot0, mdot0, mdot0, mdot_an, mdot_an, mdot0))


# model behaviour
incl_diffusivity=True,
incl_heat_loss=True,
incl_fast_buoyancy_qdot_charge=False,
incl_fast_buoyancy_qdot_discharge=True,
incl_fast_buoyancy_mdot_charge=False,
incl_fast_buoyancy_mdot_discharge=False,
incl_slow_buoyancy=False


dt = 60                             # length of the time steps [s]
num_steps = 10                    # number of time steps to be made




