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


if __name__ == "__main__":

    def SEIHR(states,t, params):
        s,e,i,h,r = states
        sigma,b,gamma,phi = params
        
        ds_dt = -s*i*b
        de_dt = s*i*b - sigma*e
        di_dt = sigma*e - gamma*i

        dh_dt = gamma*i*phi
        dr_dt = gamma*i*(1-phi)

        return np.stack([ds_dt,de_dt,di_dt,dh_dt,dr_dt])


    def model():
        sigma = numpyro.sample("sigma", dist.Gamma( 1/7., 1. ) )
        beta  = numpyro.sample("beta", dist.Gamma( 18/7., 1. ) )
        gamma = numpyro.sample("gamma", dist.Gamma( 1/7., 1. ) )

        phi   = numpyro.sample("phi", dist.Beta( 20*(0.025), 20*(1 - 0.025) ) )

        def evolve(carry,array, params):
            s,e,i,h,r,c = carry
            sigma,b,gamma = params

            s2e = -s*b*i
            e2i = sigma*e
            i2h = gamma*i*phi
            i2r = gamma*i*(1-phi)
            
            s+= s2e
            e+= s2e - e2i
            i+= e2i - (i2h+i2r)
            h+= i2h
            r+= i2r

            c = i2h

            states = jnp.vstack( (s,e,i,h,r, c) )
            return states, states

       
        final, states = jax.lax.scan( lambda x,y: evolve(x,y, (sigma, beta, gamma) ), jnp.vstack( (S0,E0,I0,H0,R0, H0) ), times)   

        #--sim
        inc_hosps = states[:,-1]

        sim = numpyro.sample("sim_inc_hosps", dist.NegativeBinomial2(inc_hosps, 0.5) )

    nuts_kernel = NUTS(model)
    mcmc        = MCMC( nuts_kernel , num_warmup=1500, num_samples=2000,progress_bar=True)
    rng_key     = random.PRNGKey(0)

    mcmc.run(rng_key, extra_fields=('potential_energy',))
    mcmc.print_summary()
    samples = mcmc.get_samples()


    N  = 12*10**6
    I0 = 5./N
    
    E0 = I0
    
    ps = 0.10
    S0 = 1*ps - I0 
    
    H0 = 0
    R0 = 1 - S0 - I0 - E0

    times = jnp.arange(0,500,1)     
    
    ps = 0.10
    S0 = 1*ps - I0

    params = [1./2, (1/ps)*(1./2)*1.4, (1./2), 0.025 ]
   
    states = odeint( lambda states,t: SEIHR(states,t,params), [S0,E0,I0,H0,R0], times  )

    hosps     = states[:,-2]
    inc_hosps = np.diff(hosps)*N

    noisy_hosps = dist.NegativeBinomial2( inc_hosps, 5.95 ).sample(rng_key).to_py()

    time_at_peak = np.argmax(inc_hosps)
    
    plt.scatter(times[1:], noisy_hosps, lw=1, color="blue", alpha=1,s=3)
    plt.plot(times[1:], inc_hosps, color="black")

    plt.show()


    
    
    


    
