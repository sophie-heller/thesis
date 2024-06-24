import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import expit
from functions import step, plot_results_time, plot_results_height, import_variables
import os
import importlib.util

# Set the working directory
working_directory = r"C:\Users\sophi\repos\repos_thesis\Model"
os.chdir(working_directory)

# Call the import_variables function to import variables from variables.py



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
file_name = "variables_m6"
import_variables(file_name)
# Print the value of alpha immediately after importing
#print("Value of alpha:", alpha)

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
                     incl_fast_buoyancy=True,
                     incl_slow_buoyancy=True):
        
        # Check if the time step dt is a multiple of frequency
        storage_frequency = 60
        if self.dt % storage_frequency != 0:
            raise ValueError(f"The time step dt ({self.dt}s) must be a multiple of the data storage frequency ({freq}s).")

        freq = int(self.dt/storage_frequency)                         # frequency of the stored values to match the time step of the simulation dt

        # Check if there is enough stored data to run the desired simulation step
        num_data_available = int(len(self.Qdot) / freq)
        if num_steps > num_data_available:
            raise ValueError(f"Number of steps for the simulation ({num_steps}) must be smaller or equal to number of available stored data ({num_data_available}).")
        

        print("Heat diffusivity:", incl_diffusivity) 
        print("Heat losses:",incl_heat_loss)
        print("Fast Buoyancy, direct charging qdot:",incl_fast_buoyancy)
        print("Slow Buoyancy:",incl_slow_buoyancy)

        # Mass balance:
        # Calculate mdot balance
        mdot_balance = np.sum(self.mdot, axis=1)      # Compute the sum of vector mdot containing ONE input or output per layer


        # Check if mdot_balance is not equal to 0
        non_zero_indices = np.where(mdot_balance != 0)[0]  # Get indices where mdot_balance is not zero

        if non_zero_indices.size > 0:
            error_message = f"mdot_balance must be equal to 0 (mass conservation, stationary system for all time steps). Errors at time steps: {non_zero_indices}, values: {mdot_balance[non_zero_indices]}"
            raise ValueError(error_message)

        T_old = np.copy(self.T_initial)
        results = [T_old.copy()]                    # Store initial temperature array


        for k in range(num_steps):
            

            T_new = np.copy(T_old)

            T_old_next = np.roll(T_old, -1)         # roll every i by -1 so that the "next" i is selected
            T_old_prev = np.roll(T_old, 1)          # roll every i by 1 so that the "previous" i is selected

            
            ####### Separate Qdot into charging and discharging vectors
            Qdot_charging = np.where(self.Qdot[k*freq] > 0, self.Qdot[k*freq], 0)
            Qdot_discharging = np.where(self.Qdot[k*freq] < 0, self.Qdot[k*freq], 0)

            ####### Separate mdot into incoming and outgoing vectors, only incoming can charge/discharge
            mdot_in = np.where(self.mdot[k*freq] > 0, self.mdot[k*freq], 0)         # vector with only positive mdots
            mdot_out = np.where(self.mdot[k*freq] < 0, self.mdot[k*freq], 0)        # vector with only negative mdots
            Tm_in = np.where(self.mdot[k*freq] > 0, self.Tm[k*freq], 0)             # vector with the temperature of the inflowing (+) streams mdot

        
            # Initiate matrices with layer infos
            matrix_l = np.tile(T_old, (len(T_old), 1))      # create matrix with T as columns for len(T) rows (T_i const as row)
            matrix_i = matrix_l.T                           # create matrix with T as row for len(T) columns (T_l const as column)
            matrix_Tm = np.tile(Tm_in, (len(Tm_in), 1))     # create matrix with Tm as columns for len(T) rows (Tm_l const as row)



            # CALCULATE DISTRIBUTION OF INPUTS (Qdot and mdot,Tm) DUE TO FAST BUOYANCY EFFECTS
            ######## INDIRECT (Qdot)
            ##########################
            # Initiate vectors fo the buoyancy qdots
            Qdot_prime_char = np.zeros(self.num_layers)  # initiate vector with length like number of layers
            Qdot_prime_dischar = np.zeros(self.num_layers)  # initiate vector with length like number of layers

            ####
            # Indir CHARGING
            matrix_bool_qchar = np.where(matrix_l >= matrix_i, 1, 0) # check availability of heat exchange to the layer i from layer l

            # Selecting the bottom left half (diagonal) of the result_matrix (only layers below i matter)
            matrix_nom_qchar = np.tril(matrix_bool_qchar)
            # Sum the values in each column, this is the denominator for the factor by which Ql will be multiplied
            den_sums_qchar = np.sum(matrix_nom_qchar, axis=0)  # axis 0 are columns
            
            # calculate the facor (distirbution of Ql)
            factor_qchar = np.where(den_sums_qchar != 0, 1 / den_sums_qchar, 0)   # calculate 1/sum, prevent division by 0
            factor_qchar = np.where(den_sums_qchar != -0, 1 / den_sums_qchar, 0)   # calculate 1/sum, prevent division by 0
            # Nom * Den for each matrix element
            matrix_factor_qchar = matrix_nom_qchar * factor_qchar
            # Multiply the factor matrix with the Qdot vector
            matrix_qchar = matrix_factor_qchar * Qdot_charging
            # The total amount of Qdot_prime for each layer is the sum of the rows of matrix_char
            Qdot_prime_char = np.sum(matrix_qchar, axis=1)   # axis 1 are rows
            
            ####
            # Indir DISCHARGING
            matrix_bool_qdischar = np.where(matrix_i >= matrix_l, 1, 0) # check availability of heat exchange to the layer i from layer l
            # Selecting the top right half (diagonal) of the result_matrix (only layers above i matter)
            matrix_nom_qdischar = np.triu(matrix_bool_qdischar)

            # Sum the values in each column, this is the denominator for the factor by which Ql will be multiplied
            den_sums_qdischar = np.sum(matrix_nom_qdischar, axis=0)  # axis 0 are columns
            # calculate the facor (distirbution of Ql)
            factor_qdischar = np.where(den_sums_qdischar != 0, 1 / den_sums_qdischar, 0)   # calculate 1/sum, prevent division by 0

            # Nom * Den for each matrix element
            matrix_factor_qdischar = matrix_nom_qdischar * factor_qdischar
            # Multiply the factor matrix with the Qdot vector
            matrix_qdischar = matrix_factor_qdischar * Qdot_discharging
            # The total amount of Qdot_prime for each layer is the sum of the rows of matrix_char
            Qdot_prime_dischar = np.sum(matrix_qdischar, axis=1)   # axis 1 are rows
            



            ######## DIRECT (mdot), here the resulting mdot_prime is the total energy (enthalpy m*cp*dT) changing layer temperature due to incoming streams (fictive distribution of oringinal  mdots)
            #############################
            # differentiate between charging (Tm>Tl) and discharging (Tm<Tl) by checking the diagonal of the TmTl matrix
            mdot_in_charging = np.where(Tm_in > T_old, mdot_in, 0)
            mdot_in_discharging = np.where(Tm_in < T_old, mdot_in, 0)

            # Initiate vectors for charging and discharging
            mdot_prime_char = np.zeros(self.num_layers)  # initiate vector with length like number of layers
            mdot_prime_dischar = np.zeros(self.num_layers)  # initiate vector with length like number of layers
            mdot_in_char_tot = np.zeros(self.num_layers)    # initiate vector: the total amount of mass streaming into i through charging mdots in l will be saved here
            mdot_in_dischar_tot = np.zeros(self.num_layers)    # initiate vector: the total amount of mass streaming into i through discharging mdots in l will be saved here
            mdot_in_tot = np.zeros(self.num_layers)            # initiate vector: total amount of mdot flowing into layer i (charge+discharge incl buoyancy)

            ####
            # dir CHARGING
            matrix_diff_mchar = matrix_Tm - matrix_i
            #matrix_diff_mchar = np.where(matrix_diff_mchar == -0, 0, matrix_diff_mchar)
            matrix_bool_mchar = np.tril(np.where(matrix_diff_mchar > 0, 1, 0)) # check availability of heat exchange to the layer i from layer l
            # Selecting the bottom left half (diagonal) of the result_matrix (only layers below i matter)
            matrix_nom_mchar = matrix_bool_mchar * matrix_diff_mchar # Tm - Ti (diff) as nominator for bool=1, 0 for bool=0
            matrix_nom_mchar = np.where(matrix_nom_mchar == -0.0, 0.0, matrix_nom_mchar)
            # Distribution of mdot_l: Sum the values in each column, this is the denominator for the factor by which mdotl will be multiplied
            den_sums_mchar = np.sum(matrix_bool_mchar, axis=0)  # axis 0 are columns
            # calculate the factor (distirbution of mdot_l)
            # Calculate the distribution factor, handling division by zero
            #factor_mchar = np.divide(1, den_sums_mchar, where=den_sums_mchar != 0, out=np.zeros_like(den_sums_mchar))
            den_sums_mchar_one = np.where(den_sums_mchar == 0, 1, den_sums_mchar)
            factor_mchar = 1/den_sums_mchar_one#np.where(den_sums_mchar == 0, 0, (1 / den_sums_mchar))   # calculate 1/sum, prevent division by 0
            # Nom * Den for each matrix element
            matrix_bool_factor_mchar = matrix_bool_mchar * factor_mchar
            mdot_in_char_tot = np.sum(matrix_bool_factor_mchar*mdot_in_charging, axis=1)
            matrix_factor_mchar = matrix_nom_mchar * factor_mchar
            # Multiply the factor matrix with the mdot vector
            matrix_mchar = matrix_factor_mchar * mdot_in_charging
            # The total amount of mdot_prime for each layer is the sum of the rows of matrix_char

            mdot_prime_char = np.sum(matrix_mchar, axis=1)   # axis 1 are rows""""M4 Debugger.py"""

            ####
            # dir DISCHARGING
            matrix_diff_mdischar = matrix_Tm - matrix_i
            matrix_diff_mdischar = np.where(matrix_diff_mdischar != 0, matrix_diff_mdischar, 0)
            # Selecting the top right half (diagonal) of the result_matrix (only layers above i matter)
            matrix_bool_mdischar = np.triu(np.where(matrix_i - matrix_Tm > 0, 1, 0)) # check availability of heat exchange to the layer i from layer Tm l
            matrix_nom_mdischar = matrix_diff_mdischar * matrix_bool_mdischar # temperature difference * availability of heat exchange
            matrix_nom_mdischar = np.where(matrix_nom_mdischar == -0.0, 0.0, matrix_nom_mdischar)
            # Distribution of mdot_l: Sum the values in each column, this is the denominator for the factor by which mdotl will be multiplied
            den_sums_mdischar = np.sum(matrix_bool_mdischar, axis=0)  # axis 0 are columns
            # calculate the factor (distirbution of mdot_l)
            # Using np.errstate to suppress warnings
            with np.errstate(divide='ignore', invalid='ignore'):
                factor_mdischar = np.where(den_sums_mdischar != 0, 1 / den_sums_mdischar, 0)
            #factor_mdischar = np.where(den_sums_mdischar != 0, (1 / den_sums_mdischar), 0)   # calculate 1/sum, prevent division by 0
            # Nom * Den for each matrix element
            matrix_bool_factor_mdischar = matrix_bool_mdischar * factor_mdischar
            mdot_in_dischar_tot = np.sum(matrix_bool_factor_mdischar*mdot_in_discharging, axis=1)
            matrix_factor_mdischar = matrix_nom_mdischar * factor_mdischar
            # Multiply the factor matrix with the mdot vector
            matrix_mdischar = matrix_factor_mdischar * mdot_in_discharging
            # The total amount of mdot_prime for each layer is the sum of the rows of matrix_char

            mdot_prime_dischar = np.sum(matrix_mdischar, axis=1)   # axis 1 are rows

            mdot_in_tot = mdot_in_char_tot + mdot_in_dischar_tot # total amount of mdot flowing into layer i inlcuding buoyancy effects

            mdot_in_tot = np.where(T_old == Tm_in, mdot_in, mdot_in_tot)

            """# DIRECT DISCHARGING OF LAYER i: entering hot stream
            for i in range(self.num_layers):                    # iterate over all layers to calculate the actual heat of each layer Qdot_prime_char[0, 1, 2...]
                mdot_sum_dischar = 0                                    # initiale the sum factor for calculating Qdot_prime_charge i
                mdot_in_dischar = 0
                mdot_in_dischar_sum = 0

                for l in range(i, self.num_layers):                            # iterate l from bottom layer (o) until current layer i (incl) to evaluate how heat inserted below in l affects layer i if buoyancy is present
                    nom=0                                           # initialize nominator for inspection of heat in layer l in relationship to layer i
                    den=0                                       # initialize denominator for inspection of heat in layer l in relationship to layer i

                    nom= step(T_old[i], Tm_in[l])# * (Tm_in[l] - T_old[i])                # evaluate if heat transfer is available (1) or not (0) with Step ----- Tl>=Ti -> 1, layer below is hoter than i

                    for j in range(l+1):       # iterate over layers starting from l and above until top of tank to evaluate the share of heat in layer l (homogenous distribution)
                        den += step(T_old[j], Tm_in[l])          # sum up to calculate the denominator and thus the share
                    den = np.where(den == 0, 1, den)                  # prevent divisions by 0    
                    mdot_in_dischar = mdot_in[l] * nom / den
                    mdot_in_dischar_sum += mdot_in_dischar
                    mdot_sum_dischar += mdot_in_dischar * (Tm_in[l] - T_old[i])    # amount of heat transfered to layer i is the sum of all heat below layer i and its share that have buoyancy effects    
                
                mdot_in_dischar_tot[i] = mdot_in_dischar_sum  
                mdot_prime_dischar[i] = mdot_sum_dischar
            
            mdot_in_tot[i] = mdot_in_char_tot[i] + mdot_in_dischar_tot[i] # total amount of mdot flowing into layer i inlcuding buoyancy effects"""



            ##### FORCED MIXING
            ###################

            # Calculate mdot mix_in for layer i. Positive means from below to above (i-1 -> i). i=0 is 0 (outside of tank) and i=N+1= 0 (outside of tank)
            mdot_mix_in = np.zeros(self.num_layers+1)
            
            if incl_fast_buoyancy == False:
                mdot_in_tot = mdot_in

            # Calculate cumulative sum of the differences of the incoming - outgoing streams in each layer
            mdot_netto = mdot_in_tot + mdot_out                              # difference of incoming streams (also after considering buoyancy) and outgoing streams (negative in their value)
            cumulative_mdot_netto = np.cumsum(mdot_netto)                     # vector containing the cummulation of the sums of the net streams flowing out of the layers below

            mdot_mix_above = -cumulative_mdot_netto             # negative since it is "leaving" the layer
            mdot_mix_below = np.roll(cumulative_mdot_netto,+1)  # positive since it is "entering" the layer

            mdot_mix_prime_above = np.where(mdot_mix_above > 0, (mdot_mix_above * (T_old_next - T_old)), 0)
            mdot_mix_prime_below = np.where(mdot_mix_below > 0, (mdot_mix_below * (T_old_prev - T_old)), 0)

            mdot_prime_mix = mdot_mix_prime_above + mdot_mix_prime_below    # total amount of heat portion that will affect each layer by incoming and outflowing mass flows from forced streaming (direct charging/discharging)                        
            """
            # Assign cumulative sums to mdot_mix_in (shifted by one position)
            mdot_mix_in[1:self.num_layers] = cumulative_mdot_diff[:-1]      # create vector that contains mdot flowing from layer below to layer i. Begins with 0 (since no stream below i=0) and ends with 0

            # Initiate vector for resulting mdot_prime
            mdot_prime_mix = np.zeros(self.num_layers)

            # take only the inflowing streams for each layer into account
            for i in range(self.num_layers):
                mdot_prime_mix_in_above = 0
                mdot_prime_mix_below = 0
                
                if (-mdot_mix_in[i + 1]) > 0:                               # check if the value for -mdot_mix_in (above) is positive -> stream entering i. negative mdot since the defined direction in the general system is from l to l+1.
                    mdot_prime_mix_in_above = (-mdot_mix_in[i + 1]) * (T_old[i + 1] - T_old[i])
                    
                if (mdot_mix_in[i]) > 0:                                    # check if the value for mdot_mix_in (below) is positive -> stream entering i
                    mdot_prime_mix_below = (mdot_mix_in[i]) * (T_old[i - 1] - T_old[i])
                    
                mdot_prime_mix[i] += mdot_prime_mix_in_above + mdot_prime_mix_below    # total amount of heat portion that will affect each layer by incoming and outflowing mass flows from forced streaming (direct charging/discharging)                        
            """
            ################ Calculate new temperatures after dt
            # separate the effects of the model to allow modularity
            # initiate arrays
            
            diffusivity = np.zeros_like(T_old)
            heat_loss = np.zeros_like(T_old)
            fast_buoyancy_qdot_charge = np.zeros_like(T_old)
            fast_buoyancy_qdot_discharge = np.zeros_like(T_old)
            fast_buoyancy_mdot_charge = np.zeros_like(T_old)
            fast_buoyancy_mdot_discharge = np.zeros_like(T_old)
            slow_buoyancy  = np.zeros_like(T_old)

            # assing values
            diffusivity = ((self.alpha) * (T_old_next - (2*T_old) + T_old_prev) / (self.dz**2))     if incl_diffusivity else 0
            heat_loss = (self.beta_i * (self.T_a - T_old))                                          if incl_heat_loss else 0
            fast_buoyancy_qdot_charge = ((self.lambda_i/self.dz) * Qdot_prime_char)                 if incl_fast_buoyancy else ((self.lambda_i/self.dz) * self.Qdot[k*freq])
            fast_buoyancy_qdot_discharge = ((self.lambda_i/self.dz) * Qdot_prime_dischar)           if incl_fast_buoyancy else 0
            fast_buoyancy_mdot_charge = ((self.phi_i/self.dz) * mdot_prime_char)                    if incl_fast_buoyancy else ((self.phi_i/self.dz) * mdot_in * (Tm_in - T_old))
            fast_buoyancy_mdot_discharge = ((self.phi_i/self.dz) * mdot_prime_dischar)              if incl_fast_buoyancy else 0 
            slow_buoyancy = ((0.5 * np.maximum(0,(T_old_prev - T_old)))                
                            - (0.5 * np.maximum(0,(T_old - T_old_next))))                                if incl_slow_buoyancy else 0
            mix_mdot_internal = ((self.phi_i/self.dz) * mdot_prime_mix)
            
            

            T_new = (T_old
                     + (diffusivity
                        + heat_loss
                        + fast_buoyancy_qdot_charge
                        + fast_buoyancy_qdot_discharge
                        + fast_buoyancy_mdot_charge
                        + fast_buoyancy_mdot_discharge
                        + mix_mdot_internal
                     )* self.dt
                     + slow_buoyancy
                    ) 
            
            #### Boundary conditions
            ### bottom of the tank (i=0)

            # separate the effects of the model to allow modularity
            diffusivity_bottom = ((self.alpha) * (T_old[1] - (2*T_old[0]) + T_old[0]) / (self.dz**2))   if incl_diffusivity else 0
            heat_loss_bottom = ((self.beta_i + self.beta_bottom) * (self.T_a - T_old[0]))               if incl_heat_loss else 0
            fast_buoyancy_qdot_charge_bottom = ((self.lambda_i/self.dz) * Qdot_prime_char[0])           if incl_fast_buoyancy else ((self.lambda_i/self.dz) * self.Qdot[k*freq][0])
            fast_buoyancy_qdot_discharge_bottom = ((self.lambda_i/self.dz) * Qdot_prime_dischar[0])     if incl_fast_buoyancy else 0 
            fast_buoyancy_mdot_charge_bottom = ((self.phi_i/self.dz) * mdot_prime_char[0])              if incl_fast_buoyancy else ((self.phi_i/self.dz) * mdot_in[0] * (Tm_in[0] - T_old[0]))
            fast_buoyancy_mdot_discharge_bottom = ((self.phi_i/self.dz) * mdot_prime_dischar[0])        if incl_fast_buoyancy else 0
            mix_mdot_internal_bottom = ((self.phi_i/self.dz) * mdot_prime_mix[0])
            slow_buoyancy_bottom = (- 0.5 * np.maximum(0,(T_old[0] - T_old[1])))     if incl_slow_buoyancy else 0
            
            T_new[0] = (T_old[0]
                     + (diffusivity_bottom
                        + heat_loss_bottom
                        + fast_buoyancy_qdot_charge_bottom
                        + fast_buoyancy_qdot_discharge_bottom
                        + fast_buoyancy_mdot_charge_bottom
                        + fast_buoyancy_mdot_discharge_bottom
                        + mix_mdot_internal_bottom
                     )* self.dt
                     + slow_buoyancy_bottom
                    ) 
           
            ### top of the tank (i=-1)

            # separate the effects of the model to allow modularity
            diffusivity_top = ((self.alpha) * (T_old[-1] - (2*T_old[-1]) + T_old[-2]) / (self.dz**2))       if incl_diffusivity else 0
            heat_loss_top = ((self.beta_i + self.beta_top) * (self.T_a - T_old[-1]))                         if incl_heat_loss else 0
            fast_buoyancy_qdot_charge_top = ((self.lambda_i/self.dz) * Qdot_prime_char[-1])                 if incl_fast_buoyancy else ((self.lambda_i/self.dz) * self.Qdot[k*freq][-1])
            fast_buoyancy_qdot_discharge_top = ((self.lambda_i/self.dz) * Qdot_prime_dischar[-1])           if incl_fast_buoyancy else 0 
            fast_buoyancy_mdot_charge_top = ((self.phi_i/self.dz) * mdot_prime_char[-1])                    if incl_fast_buoyancy else ((self.phi_i/self.dz) * mdot_in[-1] * (Tm_in[-1] - T_old[1]))
            fast_buoyancy_mdot_discharge_top = ((self.phi_i/self.dz) * mdot_prime_dischar[-1])              if incl_fast_buoyancy else 0
            mix_mdot_internal_top = ((self.phi_i/self.dz) * mdot_prime_mix[-1])
            slow_buoyancy_top = (0.5 * np.maximum(0,((T_old[-2] - T_old[-1]))))            if incl_slow_buoyancy else 0
            
            T_new[-1] = (T_old[-1]
                     + (diffusivity_top
                        + heat_loss_top
                        + fast_buoyancy_qdot_charge_top
                        + fast_buoyancy_qdot_discharge_top
                        + fast_buoyancy_mdot_charge_top
                        + fast_buoyancy_mdot_discharge_top
                        + mix_mdot_internal_top
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
"""tank_vector5o = HeatDistributionVector_model5(alpha, beta_i, beta_bottom, beta_top, lambda_i, phi_i, z, T_a, T_zero, dt, Qdot, mdot, Tm)
stability5o = tank_vector5o.stability_check()
if (stability5o == 0):
    # Solve for the temperatures
    final_temperature5o, results5o = tank_vector5o.vector_solve(num_steps, 
                                                                incl_diffusivity=False,
                                                                incl_heat_loss=False,
                                                                incl_fast_buoyancy=False,
                                                                incl_slow_buoyancy=False)
    # Plot the results
    plot_results_height(results5o, tank_vector5o.heights, dt, z, dz, "Layer temperatures over tank height. M5o: integrated direct charging")"""




tank_vector5 = HeatDistributionVector_model5(alpha, beta_i, beta_bottom, beta_top, lambda_i, phi_i, z, T_a, T_zero, dt, Qdot, mdot, Tm)
stability5 = tank_vector5.stability_check()
if (stability5 == 0):
    # Solve for the temperatures
    final_temperature5, results5 = tank_vector5.vector_solve(num_steps, 
                                                             incl_diffusivity=True,
                                                             incl_heat_loss=True,
                                                             incl_fast_buoyancy=True,
                                                             incl_slow_buoyancy=True)
    # Plot the results
    plot_results_height(results5, tank_vector5.heights, dt, z, dz, "Layer temperatures over tank height. M5: integrated direct charging")
    plot_results_time(results5, dt, "M4: fast buoyancy (only indirect charging), Temperature development of each layer over time.)")


