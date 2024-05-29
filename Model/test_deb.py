import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import expit
from functions import step, plot_results_time, plot_results_height, import_variables
import os
import importlib.util

# Set the working directory
working_directory = r"C:\Users\sophi\repos\thesis\Model"
os.chdir(working_directory)

# Call the import_variables function to import variables from variables.py

file_name = "variables"
def import_variables(file_name):
    try:
        # Get the absolute path of the current working directory
        current_dir = os.getcwd()

        # Construct the full path to the file
        file_path = os.path.join(current_dir, f"{file_name}.py")

        # Use importlib to create a module from the file
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Import variables directly into the global namespace
        for name in dir(module):
            if not name.startswith('__'):
                globals()[name] = getattr(module, name)

    except FileNotFoundError:
        print(f"File '{file_name}.py' not found.")
    except Exception as e:
        print(f"Error occurred while importing '{file_name}.py': {e}")

# Call the function to import variables from the file
import_variables(file_name)
# Print the value of alpha immediately after importing
print("Value of alpha:", alpha)

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
    def vector_solve(self, num_steps, 
                     incl_diffusivity=True, 
                     incl_heat_loss=True, 
                     incl_fast_buoyancy_qdot_charge=True, 
                     incl_fast_buoyancy_qdot_discharge=True, 
                     incl_fast_buoyancy_mdot_charge=True,
                     incl_fast_buoyancy_mdot_discharge=True,
                     incl_slow_buoyancy=True):
        
        print("Heat diffusivity:", incl_diffusivity) 
        print("Heat losses:",incl_heat_loss)
        print("Fast Buoyancy, direct charging qdot:",incl_fast_buoyancy_qdot_charge)
        print("Fast Buoyancy, direct discharging qdot:",incl_fast_buoyancy_qdot_discharge)
        print("Fast Buoyancy, indirect charging mdot:",incl_fast_buoyancy_mdot_charge)
        print("Fast Buoyancy, indirect discharging mdot:",incl_fast_buoyancy_mdot_discharge)
        print("Slow Buoyancy:",incl_slow_buoyancy)


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

            ################ Calculate new temperatures after dt
            # separate the effects of the model to allow modularity
            # initiate arrays
            """
            diffusivity = np.zeros_like(T_old)
            heat_loss = np.zeros_like(T_old)
            fast_buoyancy_qdot_charge = np.zeros_like(T_old)
            fast_buoyancy_qdot_discharge = np.zeros_like(T_old)
            fast_buoyancy_mdot_charge = np.zeros_like(T_old)
            fast_buoyancy_mdot_discharge = np.zeros_like(T_old)
            slow_buoyancy  = np.zeros_like(T_old)"""

            # assing values
            diffusivity = ((self.alpha) * (T_old_next - (2*T_old) + T_old_prev) / (self.dz**2))     if incl_diffusivity else 0
            #print(incl_diffusivity)
            #print(diffusivity)
            heat_loss = (self.beta_i * (self.T_a - T_old))                                          if incl_heat_loss else 0
            fast_buoyancy_qdot_charge = ((self.lambda_i/self.dz) * Qdot_prime_char)                 if incl_fast_buoyancy_qdot_charge else ((self.lambda_i/self.dz) * Qdot_charging)
            fast_buoyancy_qdot_discharge = ((self.lambda_i/self.dz) * Qdot_prime_dischar)           if incl_fast_buoyancy_qdot_discharge else ((self.lambda_i/self.dz) * Qdot_discharging)
            fast_buoyancy_mdot_charge = ((self.phi_i/self.dz) * mdot_prime_char)                    if incl_fast_buoyancy_mdot_charge else ((self.phi_i/self.dz) * mdot_in * (Tm_in - T_old))
            fast_buoyancy_mdot_discharge = ((self.phi_i/self.dz) * mdot_prime_dischar)              if incl_fast_buoyancy_qdot_discharge else 0
            slow_buoyancy = (0.5 * ((1/10) * np.logaddexp(0, 10 * (T_old_prev - T_old)))                
                            - 0.5 * ((1/10) * np.logaddexp(0, 10 * (T_old - T_old_next))))          if incl_slow_buoyancy else 0
            
            

            T_new = (T_old
                     + (diffusivity
                        + heat_loss
                        + fast_buoyancy_qdot_charge
                        + fast_buoyancy_qdot_discharge
                        + fast_buoyancy_mdot_charge
                        + fast_buoyancy_mdot_discharge
                     )* self.dt
                     + slow_buoyancy
                    ) 
            
            #### Boundary conditions
            ### bottom of the tank (i=0)

            # separate the effects of the model to allow modularity
            diffusivity_bottom = ((self.alpha) * (T_old[1] - (2*T_old[0]) + T_old[0]) / (self.dz**2))   if incl_diffusivity else 0
            heat_loss_bottom = ((self.beta_i + self.beta_bottom) * (self.T_a - T_old[0]))               if incl_heat_loss else 0
            fast_buoyancy_qdot_charge_bottom = ((self.lambda_i/self.dz) * Qdot_prime_char[0])           if incl_fast_buoyancy_qdot_charge else 0
            fast_buoyancy_qdot_discharge_bottom = ((self.lambda_i/self.dz) * Qdot_prime_dischar[0])     if incl_fast_buoyancy_qdot_discharge else 0 
            fast_buoyancy_mdot_charge_bottom = ((self.phi_i/self.dz) * mdot_prime_char[0])              if incl_fast_buoyancy_mdot_charge else 0
            fast_buoyancy_mdot_discharge_bottom = ((self.phi_i/self.dz) * mdot_prime_dischar[0])        if incl_fast_buoyancy_mdot_discharge else 0
            slow_buoyancy_bottom = (- 0.5 * ((1/10) * np.logaddexp(0, 10 * (T_old[0] - T_old[1]))))     if incl_slow_buoyancy else 0
            
            T_new[0] = (T_old[0]
                     + (diffusivity_bottom
                        + heat_loss_bottom
                        + fast_buoyancy_qdot_charge_bottom
                        + fast_buoyancy_qdot_discharge_bottom
                        + fast_buoyancy_mdot_charge_bottom
                        + fast_buoyancy_mdot_discharge_bottom
                     )* self.dt
                     + slow_buoyancy_bottom
                    ) 
           
            ### top of the tank (i=-1)

            # separate the effects of the model to allow modularity
            diffusivity_top = ((self.alpha) * (T_old[-1] - (2*T_old[-1]) + T_old[-2]) / (self.dz**2))       if incl_diffusivity else 0
            heat_loss_top = ((self.beta_i + self.beta_top)* (self.T_a - T_old[-1]))                         if incl_heat_loss else 0
            fast_buoyancy_qdot_charge_top = ((self.lambda_i/self.dz) * Qdot_prime_char[-1])                 if incl_fast_buoyancy_qdot_charge else 0
            fast_buoyancy_qdot_discharge_top = ((self.lambda_i/self.dz) * Qdot_prime_dischar[-1])           if incl_fast_buoyancy_qdot_discharge else 0 
            fast_buoyancy_mdot_charge_top = ((self.phi_i/self.dz) * mdot_prime_char[-1])                    if incl_fast_buoyancy_mdot_charge else 0
            fast_buoyancy_mdot_discharge_top = ((self.phi_i/self.dz) * mdot_prime_dischar[-1])              if incl_fast_buoyancy_mdot_discharge else 0
            slow_buoyancy_top = (0.5 * ((1/10) * np.logaddexp(0, 10 * (T_old[-2] - T_old[-1]))))            if incl_slow_buoyancy else 0
            
            T_new[-1] = (T_old[-1]
                     + (diffusivity_top
                        + heat_loss_top
                        + fast_buoyancy_qdot_charge_top
                        + fast_buoyancy_qdot_discharge_top
                        + fast_buoyancy_mdot_charge_top
                        + fast_buoyancy_mdot_discharge_top
                     )* self.dt
                     + slow_buoyancy_top
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
    final_temperature5, results5 = tank_vector5.vector_solve(num_steps, 
                                                             incl_diffusivity=incl_diffusivity,
                                                             incl_heat_loss=incl_heat_loss,
                                                             incl_fast_buoyancy_qdot_charge=incl_fast_buoyancy_qdot_charge,
                                                             incl_fast_buoyancy_qdot_discharge=incl_fast_buoyancy_qdot_discharge,
                                                             incl_fast_buoyancy_mdot_charge=incl_fast_buoyancy_mdot_charge,
                                                             incl_fast_buoyancy_mdot_discharge=incl_fast_buoyancy_mdot_discharge,
                                                             incl_slow_buoyancy=incl_slow_buoyancy)
    # Plot the results
    plot_results_height(results5, tank_vector5.heights, dt, z, dz, "Layer temperatures over tank height. M5: integrated direct charging")
    #plot_results_time(results4, dt, "M4: fast buoyancy (only indirect charging), Temperature development of each layer over time.)")


tank_vector5 = HeatDistributionVector_model5(alpha, beta_i, beta_bottom, beta_top, lambda_i, phi_i, z, T_a, T_zero, dt, Qdot, mdot, Tm)
stability5 = tank_vector5.stability_check()
if (stability5 == 0):
    # Solve for the temperatures
    final_temperature5, results5 = tank_vector5.vector_solve(num_steps, 
                                                             incl_diffusivity=False,
                                                             incl_heat_loss=False,
                                                             incl_fast_buoyancy_qdot_charge=False,
                                                             incl_fast_buoyancy_qdot_discharge=False,
                                                             incl_fast_buoyancy_mdot_charge=False,
                                                             incl_fast_buoyancy_mdot_discharge=False,
                                                             incl_slow_buoyancy=False)
    # Plot the results
    plot_results_height(results5, tank_vector5.heights, dt, z, dz, "Layer temperatures over tank height. M5: integrated direct charging")
    #plot_results_time(results4, dt, "M4: fast buoyancy (only indirect charging), Temperature development of each layer over time.)")

    
