#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.integrate import odeint

import numpyro.distributions as dist
import numpyro

from numpyro.infer import MCMC, NUTS, HMC, Predictive
from numpyro.distributions import constraints

import jax
from jax import random
import jax.numpy as jnp



def generate_data( N = 12*10**6        #--population size 
                  ,I0 = 5./(12*10**6)  #--initial proportion of infectors
                  ,E0 = 5./(12*10**6)  #--initial proportion of exposed
                  ,H0 = 0              #--initial proportion of hospitalized
                  ,T      = 500           #--total number of time steps to observe outbreak
                  ,ps     = 0.10         #--proportion of susceptible in the population
                  ,sigma  = 1./2
                  ,r0     = 1.4
                  ,gamma  = 1/.2
                  ,kappa  = 1./7
                  ,ph     = 0.025
                  ,rng_key=None):       

    def SEIHR(states,t, params):
        s,e,i,h,r,c = states
        sigma,b,gamma,kappa,phi = params
        
        ds_dt = -s*i*b
        de_dt = s*i*b - sigma*e
        di_dt = sigma*e - gamma*i

        dh_dt = gamma*i*phi     - kappa*h
        dr_dt = gamma*i*(1-phi) + kappa*h

        dc_dt = gamma*i*phi

        return np.stack([ds_dt,de_dt,di_dt,dh_dt,dr_dt, dc_dt])

    #--set additional parameters for integration
    S0 = 1*ps - I0
    R0 = 1 - S0 - I0 - E0

    times = jnp.arange(0,T,1)     
    params = [sigma, (1./ps)*gamma*r0, gamma, kappa, ph]

    #--integrate system
    states = odeint( lambda states,t: SEIHR(states,t,params), [S0,E0,I0,H0,R0, H0], times  )

    #--extract cumulative hospitalizations over time
    cum_hosps     = states[:,-1]

    #--compute incident hospitalizations
    inc_hosps     = np.clip(np.diff(cum_hosps)*N, 0, np.inf)

    #--add noise to inc_hosps
    noisy_hosps = np.asarray(dist.NegativeBinomial2( inc_hosps, 5.95 ).sample(rng_key))

    #--compute the true peak
    time_at_peak = np.argmax(inc_hosps)

    return inc_hosps, noisy_hosps, time_at_peak


def collect_training_data(inc_hosps,training_data,time_at_peak,pct_training_data, plus_peak=False):
    train_N       = int(time_at_peak*pct_training_data)
    training_data = noisy_hosps[:train_N]
    truth_data    = inc_hosps[:train_N]

    if plus_peak==False:
        return training_data, truth_data, None
    else:
        total_window_of_observation = len(inc_hosps)
        
        training_data = np.nan*np.ones((total_window_of_observation,)) #--all nans to start
        training_data[:train_N]     = noisy_hosps[:train_N]            #--data up to pct of peak
        training_data[time_at_peak] = noisy_hosps[time_at_peak]        #--data AT peak

        truth_data = np.nan*np.ones((total_window_of_observation,))    #--all nans to start
        truth_data[:train_N]     = inc_hosps[:train_N]                 #--data up to pct of peak
        truth_data[time_at_peak] = inc_hosps[time_at_peak]             #--data AT peak
        
        mask = ~np.isnan(training_data)

        return training_data, truth_data, mask

def SEIRH_Forecast(rng_key, training_data, mask, N, ps, total_window_of_observation ):
    def model(training_data,ttl, ps, times, mask):
        #--derive from data
        T = len(training_data)
        times = jnp.arange(0,total_window_of_observation)
        
        sigma = numpyro.sample("sigma", dist.Gamma( 1/2., 1. ) )

        gamma = 1./2  #--presets
        kappa = 1./7  #--presets
        phi   = 0.025 #--presets

        r0  = numpyro.sample("R0", dist.Uniform(0.75,4))

        beta = r0*gamma*(1./ps)

        def evolve(carry,array, params):
            s,e,i,h,r,c = carry
            sigma,b,gamma,kappa = params

            s2e = s*b*i
            e2i = sigma*e
            i2h = gamma*i*phi
            i2r = gamma*i*(1-phi)
            h2r = kappa*h

            ns = s-s2e
            ne = e+s2e - e2i
            ni = i+e2i - (i2h+i2r)
            nh = h+i2h - h2r
            nr = r+i2r + h2r

            nc = i2h

            states = jnp.vstack( (ns,ne,ni,nh,nr, nc) )
            return states, states

        E0 = numpyro.sample( "E0", dist.Uniform(1./ttl, 5./ttl) )
        I0 = numpyro.sample( "I0", dist.Uniform(1./ttl, 5./ttl) ) 

        S0 = ps*1. - I0
        H0 = 0
        R0 = 1. - S0 - I0 - E0
        
        final, states = jax.lax.scan( lambda x,y: evolve(x,y, (sigma, beta, gamma,kappa) ), jnp.vstack( (S0,E0,I0,H0,R0, H0) ), times)   

        #--sim
        states = numpyro.deterministic("states",states)
        
        inc_hosps = (states[:,-1]*ttl).reshape(-1,)
        inc_hosps = numpyro.deterministic("inc_hosps",inc_hosps)

        if mask is not None:
            with numpyro.handlers.mask(mask_array=mask):
                sim = numpyro.sample("sim_inc_hosps", dist.NegativeBinomial2(inc_hosps[:T], 10), obs = training_data)
        else:
            sim = numpyro.sample("sim_inc_hosps", dist.NegativeBinomial2(inc_hosps[:T], 10), obs = training_data)
                
    nuts_kernel = NUTS(model)
    mcmc        = MCMC( nuts_kernel , num_warmup=1500, num_samples=2000,progress_bar=True)

    mcmc.run(rng_key, extra_fields=('potential_energy',), training_data = training_data, ttl = N, ps = ps, times = times)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    return samples


if __name__ == "__main__":
    rng_key     = random.PRNGKey(0)
    
    inc_hosps, noisy_hosps, time_at_peak = generate_data(rng_key=rng_key, sigma = 1./7)
    training_data, truth_data, mask = collect_training_data(inc_hosps,noisy_hosps,time_at_peak,0.50, plus_peak=True)
    
    N     = 12*10**6
    ps    = 0.10
    total_window_of_observation = 500
    samples = SEIRH_Forecast(rng_key, training_data, mask, N, ps, total_window_of_observation ):
    
    #--plot
    plt.scatter(times[1:train_N], noisy_hosps[1:train_N], lw=1, color="blue", alpha=1,s=3)
    plt.scatter(times[train_N+1:] , noisy_hosps[train_N:], lw=1 , color="black", alpha=1,s=3)
    
    plt.plot(times[1:], inc_hosps, color="black")

    predicted_inc_hosps = samples["inc_hosps"].mean(0)

    plt.plot(predicted_inc_hosps, color= "red", ls='--')
    
    plt.show()


 
