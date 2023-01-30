#mcandrew

class generate_data(object):
    def __init__( self
                  ,population                   = 12*10**6            #--population size 
                  ,I0                           = 5./(12*10**6)       #--initial proportion of infectors
                  ,E0                           = 5./(12*10**6)       #--initial proportion of exposed
                  ,H0                           = 0                   #--initial proportion of hospitalized
                  ,total_window_of_observation  = 210                 #--total number of time steps to observe outbreak
                  ,ps                           = 0.10                #--proportion of susceptible in the population
                  ,sigma                        = 1./2                #--Duration of latent period
                  ,r0                           = 1.4                 #--Reproduction number 
                  ,gamma                        = 1/.2                #--Duration of infectious period
                  ,kappa                        = 1./7                #--Duration of hospitalization period
                  ,ph                           = 0.025               #--Proportion of those who move from infected to hospitalized
                  ,noise                        = 5.95                #--Variance (noise) to add to the true number of incident hospitalizations
                  ,rng_key                      = None                #--Key for randomization
                 ):

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

        #--integrate system
        states = odeint( lambda states,t: SEIHR(states,t,params), [S0,E0,I0,H0,R0, H0], times  )

        #--extract cumulative proportion of hospitalizations over time
        cum_hosps     = states[:,-1]

        #--compute number of incident hospitalizations
        inc_hosps     = np.clip(np.diff(cum_hosps)*self.population, 0, np.inf)

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

if __name__ == "__main__":

    from jax import random
    rng_key = random.PRNGKey(0)

    surv_data = generate_data(rng_key = rng_key)
    inc_hosps, noisy_hosps, time_at_peak, peak_value = surv_data.simulate_surveillance_data()

    plt.plot(inc_hosps)
    plt.show()
    
