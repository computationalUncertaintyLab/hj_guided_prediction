#mcandrew

class generate_data(object):
    def __init__( self
                  ,population: int                          = 12*10**6            #--population size 
                  ,I0:         float                        = 5./(12*10**6)       #--initial proportion of infectors
                  ,E0:         float                        = 5./(12*10**6)       #--initial proportion of exposed
                  ,H0:         float                        = 0                   #--initial proportion of hospitalized
                  ,total_window_of_observation: int         = 210                 #--total number of time steps to observe outbreak
                  ,ps:         float                        = 0.10                #--proportion of susceptible in the population
                  ,sigma:      float                        = 1./2                #--Duration of latent period
                  ,r0:         float                        = 1.75                #--Reproduction number 
                  ,gamma:      float                        = 1/3.                #--Duration of infectious period
                  ,kappa:      float                        = 1./7                #--Duration of hospitalization period
                  ,ph:         float                        = 0.025               #--Proportion of those who move from infected to hospitalized
                  ,noise:      float                        = 5.95                #--Variance (noise) to add to the true number of incident hospitalizations
                  ,continuous: int                          = True
                  ,rng_key                                  = None                #--Key for randomization
                 ):
        """
        population (int):  population size , default 12*10**6           
        I0         (float): initial proportion of infectors, default 5./(12*10**6)
        E0         (float): initial proportion of exposed, default 5./(12*10**6)
        H0         (float): initial proportion of hospitalized, default 0                   
        total_window_of_observation (int): total number of time steps to observe outbreak , default 210                 
        ps         (float): proportion of susceptible in the population, default 0.10                
        sigma      (float): Duration of latent period, default 1./2                
        r0         (float): Reproduction number , default 1.75                
        gamma      (float): Duration of infectious period, default 1/3.                
        kappa      (float): Duration of hospitalization period, default 1./7                
        ph         (float): Proportion of those who move from infected to hospitalized, default 0.025               
        noise      (float): Variance (noise) to add to the true number of incident hospitalizations, default 5.95
        rng_key    (PRNG key from Jax) , default None      
        """

        #--attached arguments to object
        self.population = population
        self.I0         = I0
        self.E0         = E0
        self.H0         = H0
        self.total_window_of_observation = total_window_of_observation
        self.ps         = ps
        self.sigma      = sigma
        self.r0         = r0
        self.gamma      = gamma
        self.kappa      = kappa
        self.ph         = ph
        self.noise      = noise
        self.continuous = continuous
        self.rng_key    = rng_key
        
    def simulate_surveillance_data(self):
        import jax
        import jax.numpy as jnp

        import numpy as np
        
        import numpyro
        import numpyro.distributions as dist
 
        from scipy.integrate import odeint

        def SEIHR(states,t, params):              #--System used to evolve states according to SEIR dynamics
            s,e,i,h,r,c = states                  #--States are SEIHR and then a sixth state C which records incidetn hospitalizations 
            sigma,beta,gamma,kappa,phi = params   #--Parameters needed are sigma, beta, gamma, kappa, and phi

            ds_dt = -s*i*beta
            de_dt = s*i*beta - sigma*e
            di_dt = sigma*e - gamma*i

            dh_dt = gamma*i*phi     - kappa*h
            dr_dt = gamma*i*(1-phi) + kappa*h

            dc_dt = gamma*i*phi

            return np.stack([ds_dt,de_dt,di_dt,dh_dt,dr_dt, dc_dt])

        #--Extract parameters from self
        E0, I0 = self.E0, self.I0
        ps, ph = self.ps, self.ph
        sigma, gamma, kappa = self.sigma, self.gamma, self.kappa
        r0 = self.r0
        beta = (1./ps)*gamma*r0

        H0 = self.H0
        
        #--set additional inital condition
        S0 = 1*ps - I0          #--Percent susectible at time 0
        R0 = 1 - S0 - I0 - E0   #--Percent removed/recovered at time 0

        #--The number of days from 0 to the total_window_of_observation
        times = jnp.arange(0,self.total_window_of_observation,1)

        #--Bundle together parameters to use for integration
        params = [sigma, beta, gamma, kappa, ph] 

        if self.continuous:
            #--integrate system
            states = odeint( lambda states,t: SEIHR(states,t,params), [S0,E0,I0,H0,R0, H0], times  )

            #--extract cumulative proportion of hospitalizations over time
            cum_hosps     = states[:,-1]

            #--append the initial cumualtive proportion
            cum_hosps = np.append(H0,cum_hosps)

            #--compute number of incident hospitalizations
            inc_hosps     = np.clip(np.diff(cum_hosps)*self.population, 0, np.inf)

        else:
            def evolve(carry,array, params):       #--Determinsitic system used to evolve states according to SEIR dynamics
                s,e,i,h,r,c = carry                #--States are SEIHR and then a sixth state C which records incidetn hospitalizations
                sigma,beta,gamma,kappa = params    #--Parameters needed are sigma, beta, gamma and kappa

                s2e = s*beta*i
                e2i = sigma*e
                i2h = gamma*i*ph
                i2r = gamma*i*(1-ph)
                h2r = kappa*h

                ns = s-s2e
                ne = e+s2e - e2i
                ni = i+e2i - (i2h+i2r)
                nh = h+i2h - h2r
                nr = r+i2r + h2r

                nc = i2h

                states = jnp.vstack( (ns,ne,ni,nh,nr, nc) )
                return states, states
            
            final, states = jax.lax.scan( lambda x,y: evolve(x,y, (sigma, beta, gamma,kappa) ), jnp.vstack( (S0,E0,I0,H0,R0, H0) ), times)

            #--Extract the proportion of incident hospitalizations
            inc_hosp_proportion = states[:,-1]

            #--The proportion of incident hospitalizations cannot be smaller than machine epsilon or greater than 1
            inc_hosp_proportion = jnp.clip(inc_hosp_proportion,a_min=0.,a_max= ps)

            #--Compute the number of incident hospitalizations as the proportion times the total number of individuals
            inc_hosps = (inc_hosp_proportion*self.population).reshape(-1,)
            
        #--add noise to inc_hosps according to an NB2
        noisy_hosps = np.asarray(dist.NegativeBinomial2( inc_hosps, self.noise ).sample(self.rng_key)) 

        #--compute the true peak
        time_at_peak = np.argmax(inc_hosps)

        #--compute the peak value
        peak_value   = inc_hosps[time_at_peak]

        #--attached to object and return
        self.inc_hosps    = inc_hosps
        self.noisy_hosps  = noisy_hosps
        self.time_at_peak = time_at_peak
        self.peak_value   = peak_value
        
        return inc_hosps, noisy_hosps, time_at_peak, peak_value

    def cut_data_based_on_week(self, week):
        keep_up_until_this_day = self.time_at_peak - 7*week
        keep_up_until_this_day = max(0,keep_up_until_this_day)
        return self.inc_hosps[:keep_up_until_this_day], self.noisy_hosps[:keep_up_until_this_day], keep_up_until_this_day

if __name__ == "__main__":
    import matplotlib.pyplot as plt #--only needed for plot

    
    from jax import random
    rng_key = random.PRNGKey(0)

    surv_data = generate_data(rng_key = rng_key)
    inc_hosps, noisy_hosps, time_at_peak, peak_value = surv_data.simulate_surveillance_data()

    plt.plot(inc_hosps)
    plt.show()
