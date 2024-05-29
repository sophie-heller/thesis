import numpy as np

############### VARIABLES
            
# initial temperature in the tank for each layer
# lenght of the selected T_zero defines the number of layers    
T_1 = np.array([10, 10, 10, 10, 10, 90, 90, 90, 90, 90])
T_1a = np.array([90, 90, 90, 90, 90,10, 10, 10, 10, 10]) 
T_1b = np.array([90, 90, 10, 10, 10,10, 10, 10, 10, 10]) 
T_2 = np.array([10, 20, 30, 60, 50, 90, 30, 0, 15, 20])
T_3 = [100, 80, 50, 60, 20, 50, 10, 20, 50, 80]
T_4 = np.array([100, 90, 80 ,70, 60, 50, 40, 30, 20, 10])
T_4inv = T_4[::-1]
T_5 = [100, 10, 80 ,200, 60, 70, -10, 30, -10, 10]
T_6 = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]) 
T_7 = np.array([65, 65, 65, 65, 65, 65, 65, 65, 65, 65]) # standard initial temperature
T_8 = np.full(10, 40)
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

# indirect heat 

lambda_i = (1/(A_i*rho*cp))         # coefficient of the input heat

#CHARGING
Qdot_char_aut = np.zeros(len(T_zero))
#Qdot_char_aut[2]=(5000)

Qdot_char = np.copy(Qdot_char_aut)

#DISCHARGING
Qdot_dischar_aut = np.zeros(len(T_zero))
Qdot_dischar_aut[7]=(-10000)

Qdot_dischar = np.copy(Qdot_dischar_aut)

#General
Qdot = Qdot_char + Qdot_dischar
#Qdot = Qdot_char

# direct heat
phi_i = (1/(A_i*rho))

mdot_0 = np.zeros(len(T_zero))
Tm_0 = np.zeros(len(T_zero))

mdot_aut= np.zeros(len(T_zero))
Tm_aut = np.zeros(len(T_zero))

#mdot_aut[2]=0.6
#Tm_aut[2]=100


mdot_char = mdot_aut
Tm_char = Tm_aut

mdot = mdot_aut
Tm = Tm_aut



incl_diffusivity=False,
incl_heat_loss=True,
incl_fast_buoyancy_qdot_charge=False,
incl_fast_buoyancy_qdot_discharge=True,
incl_fast_buoyancy_mdot_charge=False,
incl_fast_buoyancy_mdot_discharge=False,
incl_slow_buoyancy=False


dt = 60                             # length of the time steps [s]
num_steps = 20                    # number of time steps to be made


# losses: Beta calculations
beta_i = (P_i*k_i)/(rho * cp * A_i) # coefficient of heat losses through the wall of the tank [1/s]
beta_top = (k_i/(rho * cp * dz))    # coefficient of heat losses through the ceiling of the tank [1/s]
beta_bottom = (k_i/(rho * cp * dz)) # coefficient of heat losses through the ground of the tank [1/s]

