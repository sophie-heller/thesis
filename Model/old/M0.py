############### Definitons for the model 0 (heat diffusion)

class HeatDistributionVector_model0:
    def __init__(self, alpha, z, T_initial, dt):
        self.alpha = alpha                           # heat diffusivity
        self.z = z                                   # height of the tank
        self.num_layers = len(T_initial)             # number of layers (steps in space)
        self.dt = dt                                 # step size in time (time step)
        self.T_initial = np.array(T_initial)         # initial state of the temperatures along the tank [list] !!! Same lenght as num_layers
        
        self.dz = z / self.num_layers                # step size in space (delta z, height of a layer) is total height/number of layers, num_layers = how big the initial temperature vector is
        # Create a 
        self.heights = [i * self.dz + self.dz/2 for i in range(len(T_initial))]     # list representing the height of the tank for plotting the temperatures in the middle of each layer
  


 # definition of the solver for the temperature vector in the next time step       
    def vector_solve(self, num_steps):
        T_old = np.copy(self.T_initial)
        results = [T_old.copy()]                    # Store initial temperature array

        for _ in range(num_steps):

            T_new = np.copy(T_old)

            T_old_next = np.roll(T_old, -1)         # roll every i by -1 so that the "next" i is selected
            T_old_prev = np.roll(T_old, 1)          # roll every i by 1 so that the "previous" i is selected
            
            # Apply heat transfer equation for model 0
            T_new = (T_old 
                     + ((self.alpha) * (T_old_next - (2*T_old) + T_old_prev) / (self.dz**2))
                     * self.dt
                     )
            
            # Boundary conditions
            T_new[0] = (T_old[0]
                         + ((self.alpha) * (T_old[1] - (2*T_old[0]) + T_old[0]) / (self.dz**2))                 # assuming no heat exchange in the boundary, the tmeperature "outside" of the tank (T_old_prev of first entry) would be the same as inside of the tank (T[0])
                         * self.dt
                         )     
              
            T_new[-1] = (T_old[-1]
                          + ((self.alpha) * (T_old[-1] - (2*T_old[-1]) + T_old[-2]) / (self.dz**2))             # assuming no heat exchange in the boundary, the tmeperature "outside" of the tank (T_old_next of last entry) would be the same as inside of the tank T[-1]
                          * self.dt
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
    