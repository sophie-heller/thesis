import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import expit


## GENERAL FUNCTIONS

##### definition of the logistic function
def step(T1, T2):
    m = 1 # miu in paper 4_1D.. is set to 1
    # s = 1/(1 + np.exp(-m*(T1-T2))) # UNSTABLE
    #s = expit(m * (T1 - T2))
    #return s
    T1 = np.asarray(T1)
    T2 = np.asarray(T2)
    
    return np.where(T1 >= T2, 1, 0)

def plot_results_time(results, dt, graph_title):
    # Create traces for each layer
    traces = []
    for i, temp_array in enumerate(np.array(results).T):
        time_passed = dt * np.arange(len(temp_array))  # Calculate the time passed for each time step
        trace = go.Scatter(
            x=time_passed,  # Use time_passed as x-axis values
            y=temp_array,
            mode='lines',
            name=f'Layer {i}'
        )
        traces.append(trace)

    # Create the plot layout
    layout = go.Layout(
        title=graph_title,
        xaxis=dict(title='Time Passed (s)'),  # Update x-axis title
        yaxis=dict(title='Temperature'),
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
T_a = 2                       # ambient temperature

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
Qdot_char_aut[2]=(5000)

Qdot_char = np.copy(Qdot_char_aut)

#DISCHARGING
Qdot_dischar_aut = np.zeros(len(T_zero))
Qdot_dischar_aut[7]=(-5000)

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


dt = 60                             # length of the time steps [s]
num_steps = 10                    # number of time steps to be made


# losses: Beta calculations
beta_i = (P_i*k_i)/(rho * cp * A_i) # coefficient of heat losses through the wall of the tank [1/s]
beta_top = (k_i/(rho * cp * dz))    # coefficient of heat losses through the ceiling of the tank [1/s]
beta_bottom = (k_i/(rho * cp * dz)) # coefficient of heat losses through the ground of the tank [1/s]



############### Definitons for the model 5 (integtrated -  direct)

class HeatDistributionVector_model5:
    def __init__(self, alpha, beta_i, beta_bottom, beta_top, lambda_i, phi_i, z, T_a, T_initial, dt, Qdot, mdot, Tm):
        
        self.alpha = alpha                           # heat diffusivity
        
        self.beta_i = beta_i                         # heat loss coefficient to the ambient in the inner layers
        self.beta_bottom = beta_bottom               # heat loss coefficient to the ambient at the bottom layer (i=0)
        self.beta_top = beta_top                     # heat loss coefficient to the ambient at the top layer (i=-1)
        
        self.lambda_i = lambda_i                     # coefficient of the input heat
        self.Qdot = Qdot                              # vector conaining the Q_i of each layer    
        
        self.phi_i = phi_i                           # coefficient of the input flow/stream
        self.mdot = mdot                             # vector conaining the streams flowing into /out of the tank mdot_i of each layer
        self.Tm = Tm                                 # vector containing the temperatures of the streams flowing in/out of the tank (each mdot_i has a Tm_i)

        self.z = z                                   # height of the tank
        self.num_layers = len(T_initial)             # number of layers (steps in space)
        self.dz = z / self.num_layers                # step size in space (delta z, height of a layer) is total height/number of layers, num_layers = how big the initial temperature vector is
        self.heights = [i * self.dz + self.dz/2 for i in range(len(T_initial))]     # list representing the height of the tank for plotting the temperatures in the middle of each layer

        self.dt = dt                                 # step size in time (time step)

        self.T_initial = np.array(T_initial)         # initial state of the temperatures along the tank [list] !!! Same lenght as num_layers
        self.T_a = T_a                               # ambient temperature outside of the tank

     
 # definition of the solver for the temperature vector in the next time step       
    def vector_solve(self, num_steps):
        T_old = np.copy(self.T_initial)
        results = [T_old.copy()]                    # Store initial temperature array

        for _ in range(num_steps):

            T_new = np.copy(T_old)

            T_old_next = np.roll(T_old, -1)         # roll every i by -1 so that the "next" i is selected
            T_old_prev = np.roll(T_old, 1)          # roll every i by 1 so that the "previous" i is selected
            
    
            ####### Separate Qdot into charging and discharging vectors
            Qdot_charging = np.where(self.Qdot > 0, self.Qdot, 0)
            Qdot_discharging = np.where(self.Qdot < 0, self.Qdot, 0)


            # INDIRECT CHARGING: Calculate Qdot_prime_char, the actual amount of heat transferred to layer i through heat exchange in and below layer i (due to buoyancy)
            Qdot_prime_char = np.zeros(self.num_layers)  # initiate vector with length like number of layers

            for i in range(self.num_layers):                    # iterate over all layers to calculate the actual heat of each layer Qdot_prime_char[0, 1, 2...]
                Qdot_sum_char = 0                                    # initiale the sum factor for calculating Qdot_prime_charge i

                for l in range(i+1):                            # iterate l from bottom layer (o) until current layer i (incl) to evaluate how heat inserted below in l affects layer i if buoyancy is present
                    nom=0                                           # initialize nominator for inspection of heat in layer l in relationship to layer i
                    den=0                                       # initialize denominator for inspection of heat in layer l in relationship to layer i

                    nom= step(T_old[l], T_old[i])                # evaluate if heat transfer is available (1) or not (0) with Step ----- Tl>=Ti -> 1, layer below is hoter than i

                    for j in range(l, self.num_layers):       # iterate over layers starting from l and above until top of tank to evaluate the share of heat in layer l (homogenous distribution)
                        den += step(T_old[l], T_old[j])          # sum up to calculate the denominator and thus the share
                    den = np.where(den == 0, 1, den)                  # prevent divisions by 0     
                    Qdot_sum_char += Qdot_charging[l] * nom / den    # amount of heat transfered to layer i is the sum of all heat below layer i and its share that have buoyancy effects    
                    
                Qdot_prime_char[i] = Qdot_sum_char

            # INDIRECT DISCHARGING: Calculate Qdot_prime_dischar, the actual amount of heat transferred to the layer i thorugh heat exchange in layers in and above i (due to buoyancy)
            Qdot_prime_dischar = np.zeros(self.num_layers)  # initiate vector with length like number of layers

            for i in range(self.num_layers):                    # iterate over all layers to calculate the actual heat of each layer Qdot_prime_char[0, 1, 2...]
                Qdot_sum_dischar = 0                                    # initiale the sum factor for calculating Qdot_prime_charge i

                for l in range(i, self.num_layers):                            # iterate l from bottom layer (o) until current layer i (incl) to evaluate how heat inserted below in l affects layer i if buoyancy is present
                    nom=0                                           # initialize nominator for inspection of heat in layer l in relationship to layer i
                    den=0                                       # initialize denominator for inspection of heat in layer l in relationship to layer i

                    nom= step(T_old[i], T_old[l])                # evaluate if heat transfer is available (1) or not (0) with Step ----- Tl>=Ti -> 1, layer below is hoter than i

                    for j in range(l+1):       # iterate over layers starting from l and above until top of tank to evaluate the share of heat in layer l (homogenous distribution)
                        den += step(T_old[j], T_old[l])          # sum up to calculate the denominator and thus the share
                    den = np.where(den == 0, 1, den)                  # prevent divisions by 0     
                    Qdot_sum_dischar += Qdot_discharging[l] * nom / den    # amount of heat transfered to layer i is the sum of all heat below layer i and its share that have buoyancy effects    
                    
                Qdot_prime_dischar[i] = Qdot_sum_dischar

            ####### Separate mdot, Tm into charging and discharging elements
            mdot_in = np.where(self.mdot > 0, self.mdot, 0)         # vector with only positive mdots
            Tm_in = np.where(self.mdot > 0, self.Tm, 0)             # vector with the temperature of the inflowing (+) streams mdot
            mdot_charging = np.where(Tm_in >= T_old, mdot_in, 0)     #
            mdot_discharging = np.where(Tm_in < T_old, mdot_in, 0)

            # DIRECT CHARGING OF LAYER i: entering hot stream
            mdot_prime_char = np.zeros(self.num_layers)  # initiate vector with length like number of layers

            for i in range(self.num_layers):                    # iterate over all layers to calculate the actual heat of each layer Qdot_prime_char[0, 1, 2...]
                mdot_sum_char = 0                                    # initiale the sum factor for calculating Qdot_prime_charge i

                for l in range(i+1):                            # iterate l from bottom layer (o) until current layer i (incl) to evaluate how heat inserted below in l affects layer i if buoyancy is present
                    nom=0                                           # initialize nominator for inspection of heat in layer l in relationship to layer i
                    den=0                                       # initialize denominator for inspection of heat in layer l in relationship to layer i

                    nom= step(Tm_in[l], T_old[i]) * (Tm_in[l] - T_old[i])                # evaluate if heat transfer is available (1) or not (0) with Step ----- Tl>=Ti -> 1, layer below is hoter than i

                    for j in range(l, self.num_layers):       # iterate over layers starting from l and above until top of tank to evaluate the share of heat in layer l (homogenous distribution)
                        den += step(Tm_in[l], T_old[j])          # sum up to calculate the denominator and thus the share
                    den = np.where(den == 0, 1, den)                  # prevent divisions by 0    
                    mdot_sum_char += mdot_charging[l] * nom / den    # amount of heat transfered to layer i is the sum of all heat below layer i and its share that have buoyancy effects    
                    
                mdot_prime_char[i] = mdot_sum_char

            # DIRECT DISCHARGING OF LAYER i: entering hot stream
            mdot_prime_dischar = np.zeros(self.num_layers)  # initiate vector with length like number of layers

            for i in range(self.num_layers):                    # iterate over all layers to calculate the actual heat of each layer Qdot_prime_char[0, 1, 2...]
                mdot_sum_dischar = 0                                    # initiale the sum factor for calculating Qdot_prime_charge i

                for l in range(i, self.num_layers):                            # iterate l from bottom layer (o) until current layer i (incl) to evaluate how heat inserted below in l affects layer i if buoyancy is present
                    nom=0                                           # initialize nominator for inspection of heat in layer l in relationship to layer i
                    den=0                                       # initialize denominator for inspection of heat in layer l in relationship to layer i

                    nom= step(T_old[i], Tm_in[l]) * (Tm_in[l] - T_old[i])                # evaluate if heat transfer is available (1) or not (0) with Step ----- Tl>=Ti -> 1, layer below is hoter than i

                    for j in range(l+1):       # iterate over layers starting from l and above until top of tank to evaluate the share of heat in layer l (homogenous distribution)
                        den += step(T_old[j], Tm_in[l])          # sum up to calculate the denominator and thus the share
                    den = np.where(den == 0, 1, den)                  # prevent divisions by 0    
                    mdot_sum_dischar += mdot_discharging[l] * nom / den    # amount of heat transfered to layer i is the sum of all heat below layer i and its share that have buoyancy effects    
                    
                mdot_prime_dischar[i] = mdot_sum_dischar


            # Apply heat transfer equation for model 0
            T_new = (T_old
                      + (((self.alpha) * (T_old_next - (2*T_old) + T_old_prev) / (self.dz**2))      # diffusion between layers
                      + (self.beta_i * (self.T_a - T_old))                                          # losses to the ambient
                      + ((self.lambda_i/self.dz) * Qdot_prime_char)                                 # indirect heat charging including fast buoyancy of other layers
                      + ((self.lambda_i/self.dz) * Qdot_prime_dischar)                              # indirect heat discharging including fast buoyancy of other layers
                      + ((self.phi_i/self.dz) * mdot_prime_char) #* (Tm_in - T_old))                                    # direct hot stream (charging)
                      + ((self.phi_i/self.dz) * mdot_prime_dischar)                                 # direct cold stream (discharging)
                      )* self.dt
                      + 0.5 * ((1/10) * np.logaddexp(0, 10 * (T_old_prev - T_old)))                 # slow buoyancy of the layer i-1 to i (temperature of i rises) [with np.log -> overflow encountered]
                      - 0.5 * ((1/10) * np.logaddexp(0, 10 * (T_old - T_old_next)))                 # slow buoyancy of the layer i to i+1 (temperature of i decreases) [with np.log -> overflow encountered]
                    )
            
            ### Boundary conditions
            # bottom of the tank
            T_new[0] = (T_old[0]                                              
                         + (((self.alpha) * (T_old[1] - (2*T_old[0]) + T_old[0]) / (self.dz**2))
                         + ((self.beta_i + self.beta_bottom) * (self.T_a - T_old[0]))               # heat loss through sides of the tank (beta_i) and through the floor (beta_bottom)
                         + ((self.lambda_i/self.dz) * Qdot_prime_char[0])
                         + ((self.lambda_i/self.dz) * Qdot_prime_dischar[0])
                         + ((self.phi_i/self.dz) * mdot_prime_char[0])                                    # direct hot stream (charging)
                         + ((self.phi_i/self.dz) * mdot_prime_dischar[0])                                 # direct cold stream (discharging)
                        
                         )* self.dt
                         - 0.5 * ((1/10) * np.logaddexp(0, 10 * (T_old[0] - T_old[1])))  # slow buoyancy from the bottom layer to the above layer
                         )        
            
            # top of the tank
            T_new[-1] = (T_old[-1]                                                              
                         + (((self.alpha) * (T_old[-1] - (2*T_old[-1]) + T_old[-2]) / (self.dz**2))
                         + ((self.beta_i + self.beta_top)* (self.T_a - T_old[-1]))                 # heat loss through sides of the tank (beta_i) and through ceiling (beta_top)
                         + ((self.lambda_i/self.dz) * Qdot_prime_char[-1])
                         + ((self.lambda_i/self.dz) * Qdot_prime_dischar[-1])
                         + ((self.phi_i/self.dz) * mdot_prime_char[-1])                                    # direct hot stream (charging)
                         + ((self.phi_i/self.dz) * mdot_prime_dischar[-1])                                 # direct cold stream (discharging)
                         )* self.dt
                         + 0.5 * ((1/10) * np.logaddexp(0, 10 * (T_old[-2] - T_old[-1])))  # slow buoyancy to the top layer from the below layer
                         )       

            T_old = np.copy(T_new) # return the new temperature as old temperature for the next iteration

            results.append(T_old.copy())            # Store the updated temperature array fo later plot

        return T_old, results
     
 # check the stability of the model with the selected dt
    def stability_check(self):
        # check if the time step dt is small enough with CFL condition: dt <= (dz^2) / (2 * alpha)
        cfl_dt_max = (self.dz ** 2) / (2 * self.alpha)
        if self.dt > cfl_dt_max:
            print(f"Warning: Time step size dt {self.dt} exceeds CFL stability limit ({cfl_dt_max}).")
            sc = 1
        else:
            sc = 0
        return sc
    



# MODEL 5 - fast buoy (mdot - integrated)
tank_vector5 = HeatDistributionVector_model5(alpha, beta_i, beta_bottom, beta_top, lambda_i, phi_i, z, T_a, T_zero, dt, Qdot, mdot, Tm)
stability5 = tank_vector5.stability_check()
if (stability5 == 0):
    # Solve for the temperatures
    final_temperature5, results5 = tank_vector5.vector_solve(num_steps)
    # Plot the results
    plot_results_height(results5, tank_vector5.heights, dt, z, dz, "Layer temperatures over tank height. M5: integrated direct charging")
    #plot_results_time(results4, dt, "M4: fast buoyancy (only indirect charging), Temperature development of each layer over time.)")

    

