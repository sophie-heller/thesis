
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.special import expit
from functions_seeweg import step, plot_results_height, import_variables, plot_tanklayers_time ,plot_results_time, plot_cX_time
import os
import importlib.util
import pandas as pd

from variables_seeweg import z, layers, dz, d, P_i, A_i, alpha, rho, cp, k_i, beta_i, beta_top, beta_bottom, lambda_i, phi_i, dt, num_steps, T_a
from variables_seeweg import t_init_interpolated_df, mdot_list_df, tm_list_df, mt_list_df, qdot_list_df
from variables_seeweg import c1_df, c2_df, c3_df, c4_df, c5_df, c6_df, c7_df, c8_df, c9_df, c10_df, e1_df, layernumber_l, connection_list # c8 is Schichtlader [layer, df], layernumber_l is different possible layers of outlets



# Set the working directory
working_directory = r"C:\Users\sophi\repos\repos_thesis\Model"
os.chdir(working_directory)

############### Definitons of the class for the model
class HeatDistributionVector_model5:
 # Definition of the parameters needed to describe the heat tank and the testing conditions
    def __init__(self, alpha, beta_i, beta_bottom, beta_top, lambda_i, phi_i, z, T_a, T_initial, dt, Qdot, mdot, Tm):
        
        self.alpha = alpha                           # heat diffusivity
        
        self.beta_i = beta_i                         # heat loss coefficient to the ambient in the inner layers
        self.beta_bottom = beta_bottom               # heat loss coefficient to the ambient at the bottom layer (i=0)
        self.beta_top = beta_top                     # heat loss coefficient to the ambient at the top layer (i=-1)
        
        self.lambda_i = lambda_i                     # coefficient of the input heat
        self.Qdot = Qdot                             # List containing as many dfs as the max amount of qdot connections per layer. Each df contains qdot (pel) for each layer    
        
        self.phi_i = phi_i                           # coefficient of the input flow/stream
        self.mdot = mdot                             # List containing as many dfs as the max amount of stream connections per layer. Each df contains the mass rate (mdot=vdot*rho) for each layer i
        self.Tm = Tm                                 # List containing as many dfs as the max amount of stream connections per layer. Each df contains the temperatures (Tm) for each layer i

        self.z = z                                   # height of the tank
        self.num_layers = T_initial.shape[1]             # number of layers (steps in space)
        self.dz = z / self.num_layers                # step size in space (delta z, height of a layer) is total height/number of layers, num_layers = how big the initial temperature vector is
        self.heights = [i * self.dz + self.dz/2 for i in range(len(T_initial))]     # list representing the height of the tank for plotting the temperatures in the middle of each layer

        self.dt = dt                                 # step size in time (time step)

        self.T_initial = T_initial                   # df containing the interpolated tank temperatures for each layer (column) over time (indexing, rows)
        self.T_a = T_a                               # ambient temperature outside of the tank

     
 # definition of the solver to calculate the temperature of the layers (vector) in the next time step       
    def vector_solve(self, num_steps, 
                     incl_diffusivity=True, 
                     incl_heat_loss=True, 
                     incl_fast_buoyancy=True,
                     incl_slow_buoyancy=True):          # all effects are by default turned on
        
        #### Model check:
        # Check if the time step dt is a multiple of frequency
        storage_frequency = 60
        freq = int(self.dt/storage_frequency)                         # frequency of the stored values to match the time step of the simulation dt
        if self.dt % storage_frequency != 0:
            raise ValueError(f"The time step dt ({self.dt}s) must be a multiple of the data storage frequency ({freq}s).")

        

        # Check if there is enough stored data to run the desired simulation step
        num_data_available = int(len(self.T_initial) / freq)
        if num_steps > num_data_available:
            raise ValueError(f"Number of steps for the simulation ({num_steps}) must be smaller or equal to number of available stored data ({num_data_available}).")
        

        print("Heat diffusivity:", incl_diffusivity) 
        print("Heat losses:",incl_heat_loss)
        print("Fast Buoyancy:",incl_fast_buoyancy)
        print("Slow Buoyancy:",incl_slow_buoyancy)


        """     # Model implementation
       # Mass balance: check for mass conservation in the system
        # Calculate mdot balance
        mdot_balance = np.sum(self.mdot, axis=1)      # Compute the sum of vector mdot containing ONE input or output per layer

        # Check if mdot_balance is not equal to 0
        non_zero_indices = np.where(mdot_balance != 0)[0]  # Get indices where mdot_balance is not zero

        if non_zero_indices.size > 0:
            error_message = f"mdot_balance must be equal to 0 (mass conservation, stationary system for all time steps). Errors at time steps: {non_zero_indices}, values: {mdot_balance[non_zero_indices]}"
            raise ValueError(error_message)"""
        
     # Initialize vector for the temperature of the layers
        T_initial_arr = self.T_initial.to_numpy()[0] # self.T_initial = t_init_interpolated_df -> select first entry of array (T_initial)
        T_old = np.copy(T_initial_arr)
        results = [T_old.copy()]                    # Store initial temperature array
     # Schichtlader df
        schichtl_df = c8_df[1]
        schichtl_arr = schichtl_df.to_numpy()
        schichtl_layer_active = np.zeros(num_steps)   # what layer is c8 entering at each time k

     # Iterate over k time steps
        for k in range(num_steps):
            
            # T_old is either the initial temperature (iteration 0) or the result of the new temperature of the last time step, which becomes the old temperature of the current time step (k loop)
            T_old_next = np.roll(T_old, -1)         # roll every i by -1 so that the "next" i is selected (T_k,i+1)
            T_old_prev = np.roll(T_old, 1)          # roll every i by 1 so that the "previous" i is selected (T_k,i-1)

          ###############################################################################  
          ##### CALCULATE THE INPUTS THAT WILL AFFECT THE TANK (Qdot and mdot,Tm) #######
          ###############################################################################
          # The effects caused by fast buoyancy are modeled by assuming that the physical position is replaced by fictive inputs homogenously distributed along the tank if buoyancy conditions are met.

            # Initiate matrices with layer infos (layer temperatures for time k)
            matrix_l = np.tile(T_old, (len(T_old), 1))      # create matrix with T as columns for len(T) rows (T_i const as row)
            matrix_i = matrix_l.T                           # create matrix with T as row for len(T) columns (T_l const as column)
            
            ######## INDIRECT (Qdot)
            # The amount of input in each layer is no longer the initial Qdot, but a Qdot prime that contains the collateral charge/discharge of each layer due to charging/discharging in other (and itself) layers and the temperature differences allowing buoyancy effect
            ##########################
            Qdot_prime_char_loop = np.zeros(self.num_layers)
            Qdot_prime_dischar_loop = np.zeros(self.num_layers)
            Qdot_nobuo = np.zeros(self.num_layers)

            
            # iterate over all dataframes in qdot. They allow multiple exchanger in one layer. One df contains 1 qdot per layer. Second df contains, in the layers where more than 1 qdot is connected, the data of the second qdot. Until u
            for u in range(len(self.Qdot)):
                Qdot_arr = self.Qdot[u].to_numpy()
                ####### Separate Qdot into charging and discharging vectors
                Qdot_charging = np.where(Qdot_arr[k*freq] > 0, Qdot_arr[k*freq], 0)
                Qdot_discharging = np.where(Qdot_arr[k*freq] < 0, Qdot_arr[k*freq], 0)
                Qdot_nobuo += Qdot_arr[k*freq]                                              # without buoyancy, the total amount of incoming mdot per layer is the sum of all mdots of each layer (sum over w)

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
                with np.errstate(divide='ignore', invalid='ignore'):
                    factor_qchar = np.where(den_sums_qchar != 0, 1 / den_sums_qchar, 0) # calculate 1/sum, prevent division by 0
                # Nom * Den for each matrix element
                matrix_factor_qchar = matrix_nom_qchar * factor_qchar
                # Multiply the factor matrix with the Qdot vector
                matrix_qchar = matrix_factor_qchar * Qdot_charging
                # The total amount of Qdot_prime for each layer is the sum of the rows of matrix_char
                Qdot_prime_char = np.sum(matrix_qchar, axis=1)   # axis 1 are rows


                Qdot_prime_char_loop += Qdot_prime_char  # Accumulate the results

                            
                ####
                # Indir DISCHARGING
                matrix_bool_qdischar = np.where(matrix_i >= matrix_l, 1, 0) # check availability of heat exchange to the layer i from layer l
                # Selecting the top right half (diagonal) of the result_matrix (only layers above i matter)
                matrix_nom_qdischar = np.triu(matrix_bool_qdischar)

                # Sum the values in each column, this is the denominator for the factor by which Ql will be multiplied
                den_sums_qdischar = np.sum(matrix_nom_qdischar, axis=0)  # axis 0 are columns
                # calculate the facor (distirbution of Ql)
                with np.errstate(divide='ignore', invalid='ignore'):
                    factor_qdischar = np.where(den_sums_qdischar != 0, 1 / den_sums_qdischar, 0)    # calculate 1/sum, prevent division by 0

                # Nom * Den for each matrix element
                matrix_factor_qdischar = matrix_nom_qdischar * factor_qdischar
                # Multiply the factor matrix with the Qdot vector
                matrix_qdischar = matrix_factor_qdischar * Qdot_discharging
                # The total amount of Qdot_prime for each layer is the sum of the rows of matrix_char
                Qdot_prime_dischar = np.sum(matrix_qdischar, axis=1)   # axis 1 are rows

                Qdot_prime_dischar_loop += Qdot_prime_dischar
                



            ######## DIRECT (mdot,Tm), ENERGY CHANGE IS GIVEN BY mdot*cp*dT
            # The fast buoyancy effect will cause the distribution of mdot,Tm along multiple layers of the tank, provided buoyancy conditions are met.
            # the resulting mdot_prime contains the sum of the energy changes in the layers through this new separation of the inputs. 
            #############################

            # Assign Schichtlader to the corresponding layer. Connection 8 has four possible outlets.
            schichtl_k = schichtl_arr[k*freq]                 # temp, vdot and mt of schichtlader c8 (schictl_arr) in time k
            schichtl_t_k = schichtl_k[0]                      # [0] corresponds to temperature of c8
            schichtl_mdot_k = schichtl_k[1]*(rho/3600000)     # [1] corresponds to vdot of c8 in l/h -> *(rho*3600000) to get m³/s
            schichtl_mt_k = schichtl_k[2]*(rho/3600000)       # [2] corresponds to mt = mcpT of c8 in l/h -> *(rho*3600000) to get m³/s*cp*T
            schichtlader_layers = layernumber_l[::-1]         # layers where outlets are located, organized from smallest (bottom) to biggest (top)


            temp_schichtlayers = T_old[schichtlader_layers]   # temperature of the layers where outlets are located
            schichtlader_layer_k = schichtlader_layers[np.argmin(np.abs(temp_schichtlayers - schichtl_t_k))]  # c8 will flow to the layer where the temp difference between stream and layer is the smallest. This returns the first layer where condition is met (bottom
            

            mdot_prime_char_loop = np.zeros(self.num_layers)
            mdot_in_char_tot_loop = np.zeros(self.num_layers)
            mdot_prime_dischar_loop = np.zeros(self.num_layers)
            mdot_in_dischar_tot_loop = np.zeros(self.num_layers)
            mdot_in_tot_loop = np.zeros(self.num_layers)
            mdot_in_nobuo = np.zeros(self.num_layers)
            mdot_prime_nobuo = np.zeros(self.num_layers)
            mdot_out = np.zeros(self.num_layers)

            test = self.mdot[1]
            # iterate over all dataframes stored in mdot. They allow multiple stream connections. One df contains 1 stream per layer. Second df contains, in the layers where more than 1 stream is connected, the data of the second stream. Until w
            for w in range(len(self.mdot)):
                #test= len(self.mdot)-1
                    

                mdot_arr = self.mdot[w].to_numpy()          # convert list element w into numpy array. This array contains the mass rate of one stream per layer.
                Tm_arr = self.Tm[w].to_numpy()              # convert list element w into numpy array. This array contains the temperature of one stream connected to each layer

                ####### Separate mdot into incoming and outgoing vectors, only incoming can charge/discharge
                mdot_in = np.where(mdot_arr[k*freq] > 0, mdot_arr[k*freq], 0)         # vector with only positive mdots
                mdot_out_w = np.where(mdot_arr[k*freq] < 0, mdot_arr[k*freq], 0)        # vector with only negative mdots
                mdot_out += mdot_out_w                                               # cummulative mdot_out vector over w dfs
                Tm_in = np.where(mdot_arr[k*freq] > 0, Tm_arr[k*freq], 0)             # vector with the temperature of the inflowing (+) streams mdot
                mdot_in_nobuo += mdot_in                                              # without buoyancy, the total amount of incoming mdot per layer is the sum of all mdots of each layer (sum over w)
                mdot_prime_nobuo += mdot_in_nobuo *(Tm_in - T_old)

                # Assign Schichtlader data to the corresponding layer k in the last df w
                if w == len(self.mdot)-1:
                    Tm_in[schichtlader_layer_k] = schichtl_t_k        # temp of schichtlader entering assigned layer 
                    mdot_in[schichtlader_layer_k] = schichtl_mdot_k       # mdot of schichtlader entering assigned layer, it can only flow in, no mdot_out 
                    schichtl_layer_active[k] = schichtlader_layer_k
                
                

                matrix_Tm = np.tile(Tm_in, (len(Tm_in), 1))      # create matrix with Tm as columns for len(T) rows (Tm_l const as row)

                # differentiate between charging (Tm>Tl) and discharging (Tm<Tl) by checking the diagonal of the TmTl matrix
                mdot_in_charging = np.where(Tm_in > T_old, mdot_in, 0)
                mdot_in_discharging = np.where(Tm_in < T_old, mdot_in, 0)
                mdot_in_neutral = np.where(Tm_in == T_old, mdot_in, 0)  # if Tm - Told = 0, there is no buoyancy and the mdot enters completely this layer (without changin its temp)

                # Initiate vectors for charging and discharging
                mdot_prime_char = np.zeros(self.num_layers)  # initiate vector with length like number of layers
                mdot_prime_dischar = np.zeros(self.num_layers)  # initiate vector with length like number of layers
                mdot_in_char_tot = np.zeros(self.num_layers)    # initiate vector: the total amount of mass streaming into i through charging mdots in l will be saved here
                mdot_in_dischar_tot = np.zeros(self.num_layers)    # initiate vector: the total amount of mass streaming into i through discharging mdots in l will be saved here
                #mdot_in_tot = np.zeros(self.num_layers)            # initiate vector: total amount of mdot flowing into layer i (charge+discharge incl buoyancy)

                ####
                # dir CHARGING
                matrix_diff_mchar = matrix_Tm - matrix_i
                # matrix_diff_mchar = np.where(matrix_diff_mchar == -0, 0, matrix_diff_mchar)
                matrix_bool_mchar = np.tril(np.where(matrix_diff_mchar > 0, 1, 0)) # check availability of heat exchange to the layer i from layer l
                # Selecting the bottom left half (diagonal) of the result_matrix (only layers below i matter)
                matrix_nom_mchar = matrix_bool_mchar * matrix_diff_mchar # Tm - Ti (diff) as nominator for bool=1, 0 for bool=0
                matrix_nom_mchar = np.where(matrix_nom_mchar == -0.0, 0.0, matrix_nom_mchar)
                # Distribution of mdot_l: Sum the values in each column, this is the denominator for the factor by which mdotl will be multiplied
                den_sums_mchar = np.sum(matrix_bool_mchar, axis=0)  # axis 0 are columns
                # calculate the factor (distirbution of mdot_l)
                with np.errstate(divide='ignore', invalid='ignore'):
                    factor_mchar = np.where(den_sums_mchar != 0, 1 / den_sums_mchar, 0)    # calculate 1/sum, prevent division by 0

                # Nom * Den for each matrix element
                matrix_bool_factor_mchar = matrix_bool_mchar * factor_mchar
                mdot_in_char_tot = np.sum(matrix_bool_factor_mchar*mdot_in_charging, axis=1)
                mdot_in_char_tot_loop += mdot_in_char_tot # not used, just check. The sum is made in mdot_in_tot_loop further below using mdot_in_char_tot
                matrix_factor_mchar = matrix_nom_mchar * factor_mchar
                # Multiply the factor matrix with the mdot vector
                matrix_mchar = matrix_factor_mchar * mdot_in_charging
                # The total amount of mdot_prime for each layer is the sum of the rows of matrix_char

                mdot_prime_char = np.sum(matrix_mchar, axis=1)   # axis 1 are rows
                mdot_prime_char_loop += mdot_prime_char

                ####
                # dir DISCHARGING

                matrix_diff_mdischar = matrix_Tm - matrix_i
                # Selecting the top right half (diagonal) of the result_matrix (only layers above i matter)
                matrix_bool_mdischar = np.triu(np.where(matrix_i - matrix_Tm > 0, 1, 0)) # check availability of heat exchange to the layer i from layer Tm l
                matrix_nom_mdischar = matrix_diff_mdischar * matrix_bool_mdischar # temperature difference * availability of heat exchange
                matrix_nom_mdischar = np.where(matrix_nom_mdischar == -0.0, 0.0, matrix_nom_mdischar)
                # Distribution of mdot_l: Sum the values in each column, this is the denominator for the factor by which mdotl will be multiplied
                den_sums_mdischar = np.sum(matrix_bool_mdischar, axis=0)  # axis 0 are columns
                # calculate the factor (distirbution of mdot_l)
                with np.errstate(divide='ignore', invalid='ignore'):
                    factor_mdischar = np.where(den_sums_mdischar != 0, 1 / den_sums_mdischar, 0)    # calculate 1/sum, prevent division by 0

                # Nom * Den for each matrix element
                matrix_bool_factor_mdischar = matrix_bool_mdischar * factor_mdischar
                mdot_in_dischar_tot = np.sum(matrix_bool_factor_mdischar*mdot_in_discharging, axis=1)
                mdot_in_dischar_tot_loop += mdot_in_dischar_tot # not used, just check
                matrix_factor_mdischar = matrix_nom_mdischar * factor_mdischar
                # Multiply the factor matrix with the mdot vector
                matrix_mdischar = matrix_factor_mdischar * mdot_in_discharging
                # The total amount of mdot_prime for each layer is the sum of the rows of matrix_char

                mdot_prime_dischar = np.sum(matrix_mdischar, axis=1)   # axis 1 are rows
                mdot_prime_dischar_loop += mdot_prime_dischar

                mdot_in_tot_loop += mdot_in_char_tot + mdot_in_dischar_tot + mdot_in_neutral # sums the inflowing mdot to each layer (through charging, discharging and neutral) inlcuding buoyancy effects

            #mdot_in_tot = mdot_in_char_tot_loop + mdot_in_dischar_tot_loop # total amount of mdot flowing into layer i inlcuding buoyancy effects

            #mdot_in_tot = np.where(T_old == Tm_in, mdot_in, mdot_in_tot)
            

            ##### FORCED MIXING
            ###################

            # Calculate mdot mix_in for layer i. Positive means from below to above (i-1 -> i). i=0 is 0 (outside of tank) and i=N+1= 0 (outside of tank)
            mdot_mix_in = np.zeros(self.num_layers+1)
            
            if incl_fast_buoyancy == False:
                mdot_in_tot = mdot_in_nobuo
            else:
                mdot_in_tot = mdot_in_tot_loop

            # Calculate cumulative sum of the differences of the incoming - outgoing streams in each layer
            mdot_netto = mdot_in_tot + mdot_out                              # difference of incoming streams (also after considering buoyancy) and outgoing streams (negative in their value)
            cumulative_mdot_netto = np.cumsum(mdot_netto)                     # vector containing the cummulation of the sums of the net streams flowing out of the layers below

            mdot_mix_above = -cumulative_mdot_netto             # negative since it is "leaving" the layer
            mdot_mix_below = np.roll(cumulative_mdot_netto,+1)  # positive since it is "entering" the layer

            mdot_mix_prime_above = np.where(mdot_mix_above > 0, (mdot_mix_above * (T_old_next - T_old)), 0)
            mdot_mix_prime_below = np.where(mdot_mix_below > 0, (mdot_mix_below * (T_old_prev - T_old)), 0)

            mdot_prime_mix = mdot_mix_prime_above + mdot_mix_prime_below    # total amount of heat portion that will affect each layer by incoming and outflowing mass flows from forced streaming (direct charging/discharging)                        

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

            # calculation of termns for the energy balance based equation
            diffusivity = ((self.alpha) * (T_old_next - (2*T_old) + T_old_prev) / (self.dz**2))     if incl_diffusivity else 0
            heat_loss = (self.beta_i * (self.T_a - T_old))                                          if incl_heat_loss else 0
            fast_buoyancy_qdot_charge = ((self.lambda_i/self.dz) * Qdot_prime_char)                 if incl_fast_buoyancy else ((self.lambda_i/self.dz) * Qdot_nobuo)
            fast_buoyancy_qdot_discharge = ((self.lambda_i/self.dz) * Qdot_prime_dischar)           if incl_fast_buoyancy else 0
            fast_buoyancy_mdot_charge = ((self.phi_i/self.dz) * mdot_prime_char)                    if incl_fast_buoyancy else ((self.phi_i/self.dz) * mdot_prime_nobuo)
            fast_buoyancy_mdot_discharge = ((self.phi_i/self.dz) * mdot_prime_dischar)              if incl_fast_buoyancy else 0 
            slow_buoyancy = ((0.5 * np.maximum(0,(T_old_prev - T_old)))                
                            - (0.5 * np.maximum(0,(T_old - T_old_next))))                                if incl_slow_buoyancy else 0
            mix_mdot_internal = ((self.phi_i/self.dz) * mdot_prime_mix)
            
            
            # Energy balance based equation
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
            fast_buoyancy_qdot_charge_bottom = ((self.lambda_i/self.dz) * Qdot_prime_char[0])           if incl_fast_buoyancy else ((self.lambda_i/self.dz) * Qdot_nobuo[0])
            fast_buoyancy_qdot_discharge_bottom = ((self.lambda_i/self.dz) * Qdot_prime_dischar[0])     if incl_fast_buoyancy else 0 
            fast_buoyancy_mdot_charge_bottom = ((self.phi_i/self.dz) * mdot_prime_char[0])              if incl_fast_buoyancy else ((self.phi_i/self.dz) * mdot_prime_nobuo[0])
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
            fast_buoyancy_qdot_charge_top = ((self.lambda_i/self.dz) * Qdot_prime_char[-1])                 if incl_fast_buoyancy else ((self.lambda_i/self.dz) * Qdot_nobuo[-1])
            fast_buoyancy_qdot_discharge_top = ((self.lambda_i/self.dz) * Qdot_prime_dischar[-1])           if incl_fast_buoyancy else 0 
            fast_buoyancy_mdot_charge_top = ((self.phi_i/self.dz) * mdot_prime_char[-1])                    if incl_fast_buoyancy else ((self.phi_i/self.dz) * mdot_prime_nobuo[-1])
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

        return T_old, results, freq
     
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


T_zero=t_init_interpolated_df
mdot = mdot_list_df
Tm = tm_list_df
Qdot = qdot_list_df
mt = mt_list_df



tank_vector5 = HeatDistributionVector_model5(alpha, beta_i, beta_bottom, beta_top, lambda_i, phi_i, z, T_a, T_zero, dt, Qdot, mdot, Tm)
stability5 = tank_vector5.stability_check()
if (stability5 == 0):
    # Solve for the temperatures
    final_temperature5, results5, freq5 = tank_vector5.vector_solve(num_steps, 
                                                             incl_diffusivity=True,
                                                             incl_heat_loss=True,
                                                             incl_fast_buoyancy=True,
                                                             incl_slow_buoyancy=True)
    
    # store the timestamps
    timestamp_df = pd.DataFrame(index=T_zero.index[::freq5][:num_steps+1])
    results_df = pd.DataFrame(results5, index=T_zero.index[::freq5][:num_steps+1])

    """# Plot the results
    #plot_results_height(results5, tank_vector5.heights, dt, z, dz, "Layer temperatures over tank height. M5: integrated direct charging")
    # simulated results over time
    plot_results_time(results5, dt, timestamp_df, "Simulated temperature development of each layer over time.")
    # measured tank temperatures over time
    plot_tanklayers_time(T_zero, dt, freq5, num_steps, timestamp_df, "Original temperature development of each layer over time.")
    # measured connections over time (temperature)
    plot_cX_time(connection_list, 2, dt, freq5, num_steps, timestamp_df, "Energy of the streams over time (mdot*cp*T)")
    plot_cX_time(connection_list, 0, dt, freq5, num_steps, timestamp_df, "Temperature of the streams over time")
    
    #plot_mts_time([c1_df, c2_df, c3_df, c4_df, c5_df, c6_df, c7_df, c8_df, c9_df, c10_df], 2, dt, freq5, num_steps, "mdot*cp*T")"""

    


    #plot_tanklayers_time(mt[2], dt, freq5, num_steps, "test1")
    #plot_tanklayers_time(Qdot[0], dt, freq5, num_steps, "qdot")

    #plot_mts_time([c1_df, c2_df, c3_df, c4_df, c5_df, c6_df, c7_df, c8_df, c9_df, c10_df], 1, dt, freq5, num_steps, "mdot")
    #plot_mts_time([c1_df, c2_df, c3_df, c4_df, c5_df, c6_df, c7_df, c8_df, c9_df, c10_df], 0, dt, freq5, num_steps, "T")
    


####### Energie im Tank

###  GROUP 1: WW -> 
#    c1 = c9 + c4
gr149_list = [c1_df[1], c9_df[1], c4_df[1]]

gr_149 = pd.concat((df for df in gr149_list), axis = 1)
gr_149["dvdot"] = gr_149_vdot["v_c9"] + gr_149_vdot["v_c4"] - gr_149_vdot["v_c1"]
gr_149["dT"] = gr_149["t_c9"] - gr_149["t_c1"]
gr_149["vdot"] = gr_149["v_c9"]

# Creating the plot
fig = go.Figure()

# Adding dT line
fig.add_trace(go.Scatter(
    x=gr_149.index,
    y=gr_149["dT"],
    name="dT",
    yaxis="y1",
    line=dict(color="blue")
))

# Adding vdot line
fig.add_trace(go.Scatter(
    x=gr_149.index,
    y=gr_149["vdot"],
    name="vdot",
    yaxis="y2",
    line=dict(color="red")
))

# Updating layout for the dual y-axis
fig.update_layout(
    title="dT and vdot over Time",
    xaxis_title="Time",
    yaxis=dict(
        title="dT",
        titlefont=dict(color="blue"),
        tickfont=dict(color="blue"),
        anchor="x",
        side="left"
    ),
    yaxis2=dict(
        title="vdot",
        titlefont=dict(color="red"),
        tickfont=dict(color="red"),
        overlaying="y",
        anchor="x",
        side="right"
    ),
    legend=dict(x=0.1, y=0.9)
)

# Show the plot
fig.show()







gr_149_T = pd.concat((df.iloc[:,0] for df in gr149), axis = 1)
#gr_149_T = pd.concat([c1_df[1].iloc[:,0], c4_df[1].iloc[:,0], c9_df[1].iloc[:,0]], axis = 1)
gr_149_T["dT"] = gr_149_T["t_c9"] - gr_149_T["t_c1"]
gr_149_vdot = pd.concat((df.iloc[:,1] for df in gr149), axis = 1)
gr_149_vdot["dvdot"] = gr_149_vdot["v_c9"] + gr_149_vdot["v_c4"] - gr_149_vdot["v_c1"]

###  GROUP 2: WP1_HG -> 
#    c2 = c3
gr23 = pd.concat([c2_df[1], c3_df[1]], axis = 1)

###  GROUP 3: HK and WP2 -> 
# c5 = - c6 - c7
gr567 = pd.concat([c5_df[1], c6_df[1], c7_df[1]], axis = 1)

### GROUP 4: WP1 + SOL
# c10 = c8
gr810 = pd.concat([c8_df[1], c10_df[1]], axis = 1)


def plot_gr_time(gr149, column, dt, freq, numsteps, timestamp_df, graph_title):
    # Create a list to hold all traces
    naming = ["Temperature °C", "Mass rate kg/s", "Energy rate W"]
    traces = []
    #count = 0
    # Iterate through the list of lists
    for connection_data in gr149:
        count +=1
        df = connection_data  # ExtractDataFrame
        gr_array = np.array(df.iloc[::freq, column].iloc[:numsteps+1])  # Extract the desired column (0=t, 1= mdot, 2 = mt)
        trace = go.Scatter(
            x=timestamp_df.index,  # Use time_passed as x-axis values
            y=gr_array,
            mode='lines',
            name=f'c{count}'
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




    