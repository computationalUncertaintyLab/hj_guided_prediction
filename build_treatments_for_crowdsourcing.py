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

from joblib import Parallel, delayed

import scienceplots

import jax
from jax import random
import jax.numpy as jnp

import os
import pickle

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
                  ,noise  = 5.95  
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
    noisy_hosps = np.asarray(dist.NegativeBinomial2( inc_hosps, noise ).sample(rng_key)) 

    #--compute the true peak
    time_at_peak = np.argmax(inc_hosps)

    return inc_hosps, noisy_hosps, time_at_peak


def collect_training_data(inc_hosps,noisy_hosps,time_at_peak,pct_training_data, plus_peak=False):
    train_N       = int(time_at_peak*pct_training_data)
    
    if plus_peak==False:
        training_data = noisy_hosps[:train_N]
        truth_data    = inc_hosps[:train_N]
        return training_data, None, truth_data
    else:
        total_window_of_observation = len(inc_hosps)
        
        training_data = noisy_hosps[:train_N]            #--data up to pct of peak
        noisy_peak_value = noisy_hosps[time_at_peak]       #--data AT peak

        truth_data = np.nan*np.ones((total_window_of_observation,))    #--all nans to start
        truth_data[:train_N]     = inc_hosps[:train_N]                 #--data up to pct of peak
        truth_data[time_at_peak] = inc_hosps[time_at_peak]             #--data AT peak
        
        return training_data, noisy_peak_value, truth_data

def SEIRH_Forecast(rng_key, training_data, N, ps, total_window_of_observation, hj_peaks=None, hj_peak_intensities=None):
    def model(training_data,ttl, ps, times):
        #--derive from data
        T = len(training_data)
        times = jnp.arange(0,total_window_of_observation)
        
        sigma = numpyro.sample("sigma", dist.Gamma( 1/2., 1. ) )

        gamma = 1./1  #--presets
        kappa = 1./7  #--presets

        phi = numpyro.sample("phi", dist.Beta(0.025*10, (1-0.025)*10))
        r0  = numpyro.sample("R0" , dist.Uniform(0.75,4))

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

        E0 = numpyro.sample( "E0", dist.Uniform(1./ttl, 20./ttl) )
        I0 = numpyro.sample( "I0", dist.Uniform(1./ttl, 20./ttl) ) 

        S0 = ps*1. - I0
        H0 = 0
        R0 = 1. - S0 - I0 - E0
        
        final, states = jax.lax.scan( lambda x,y: evolve(x,y, (sigma, beta, gamma,kappa) ), jnp.vstack( (S0,E0,I0,H0,R0, H0) ), times)   

        #--sim
        states = numpyro.deterministic("states",states)

        #inc_hosp_proportion = jnp.clip(states[:,-1], a_min=1./ttl)
        inc_hosp_proportion = states[:,-1]
        
        inc_hosps = (inc_hosp_proportion*ttl).reshape(-1,)
        inc_hosps = numpyro.deterministic("inc_hosps",inc_hosps)

        #--clip inc hosps
        #inc_hosps = jnp.clip(inc_hosps,10**-10,N+1)
        #inc_hosps = jnp.nan_to_num(inc_hosps,1.)

        mask = ~jnp.isnan(inc_hosps)
        
        #--surveillance data
        with numpyro.handlers.mask(mask=mask[:T]):
            ll = numpyro.sample("ll_surveillance", dist.NegativeBinomial2(inc_hosps[:T], 1./3), obs = training_data)

        if hj_peaks is not None and hj_peak_intensities is not None:
            with numpyro.handlers.mask(mask=mask[hj_peaks]):
                ll2 = numpyro.sample("ll_hj", dist.NegativeBinomial2(inc_hosps[hj_peaks], 1./3), obs = hj_peak_intensities )
                   
    nuts_kernel = NUTS(model)
    mcmc        = MCMC( nuts_kernel , num_warmup=1500, num_samples=2000,progress_bar=True)

    mcmc.run(rng_key, extra_fields=('potential_energy',)
             , training_data = training_data
             , ttl = N
             , ps = ps
             , times = total_window_of_observation)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    return samples

def generate_multiple_noisy_measurements_of_peak_and_location(time_at_peak,inc_hosps, noisy_hosps, noise, rng_key, num_of_measurements, pct_training_data):
    train_N       = int(time_at_peak*pct_training_data)
    
    peak_value = inc_hosps[time_at_peak]
    noisy_peaks = np.asarray(dist.NegativeBinomial2( peak_value, noise ).sample(rng_key, sample_shape = (num_of_measurements,)  ))

    #--uniform distribution of time at peak
    noisy_time_at_peaks = time_at_peak + np.random.randint(low = -14, high=14, size = (num_of_measurements,))

    return noisy_hosps[:train_N], noisy_peaks, noisy_time_at_peaks
    
def mm2inch(x):
    return x/25.4

def stamp(ax,s):
    ax.text(0.0125,0.975,s=s,fontweight="bold",fontsize=10,ha="left",va="top",transform=ax.transAxes)

if __name__ == "__main__":
    rng_key     = random.PRNGKey(0)    #--seed for random number generation
    N     = 12*10**6                   #--Population of the state of PA
    ps    = 0.12                       #--Percent of the population that is considered to be susceptible
    total_window_of_observation = 210  #--total number of time units that we will observe

    if os.path.isdir("./for_crowdsourcing/"):
        pass
    else:
        os.makedirs("./for_crowdsourcing/")

    for pct in [0.20,0.40,0.60,0.80,1.20]:
        for model_included in [0,1]:
            #--generate simulated data
            inc_hosps, noisy_hosps, time_at_peak = generate_data(rng_key=rng_key, sigma = 1./2, gamma = 1./1,  noise = 100, T = total_window_of_observation)
            training_data, noisy_peak_value, truth_data = collect_training_data(inc_hosps,noisy_hosps
                                                                    ,time_at_peak
                                                                    ,pct
                                                                    ,plus_peak=True)

            if os.path.isdir("./for_crowdsourcing/data_collection__{:.2f}__{:d}/".format(pct, model_included)):
                pass
            else:
                os.makedirs("./for_crowdsourcing/data_collection__{:.2f}__{:d}/".format(pct, model_included))
            
            pd.DataFrame({"training_data":training_data}).to_csv("./for_crowdsourcing/data_collection__{:.2f}__{:d}/training_data__{:.2f}__{:d}.pdf".format(pct, model_included, pct, model_included))
            pd.DataFrame({"truth_data":truth_data}).to_csv("./for_crowdsourcing/data_collection__{:.2f}__{:d}/truth_data__{:.2f}__{:d}.pdf".format(pct, model_included, pct, model_included))

            #-plot
            plt.style.use('science')

            fig,ax = plt.subplots()
            times = np.arange(0,total_window_of_observation)

            #--plot without peak
            ax.scatter(times[:len(training_data)], training_data, lw=1, color="blue", alpha=1,s=3, label = "Surveillance data (up to day 16) ")

            if model_included:
                #--forecast with out peak data included
                samples__wopeak = SEIRH_Forecast(rng_key, training_data, N, ps, total_window_of_observation )

                pickle.dump( samples__wopeak, open("./for_crowdsourcing/data_collection__{:.2f}__{:d}/samples__{:.2f}__{:d}.pdf".format(pct, model_included, pct, model_included) ,"wb") )

                predicted_inc_hosps__wopeak = samples__wopeak["inc_hosps"].mean(0)
                lower_2p5__w0peak,lower25__w0peak, upper75__w0peak, upper97p5__w0peak = np.percentile( samples__wopeak["inc_hosps"], [2.5, 25, 75, 97.5], 0)

                quantiles = pd.DataFrame({"mean": predicted_inc_hosps__wopeak
                                          , "lower_2p5":lower_2p5__w0peak, "lower25__w0peak": lower25__w0peak
                                          , "upper75__w0peak":upper75__w0peak, "upper97p5__w0peak":upper97p5__w0peak  })
                quantiles.to_csv("./for_crowdsourcing/data_collection__{:.2f}__{:d}/quantiles__{:.2f}__{:d}.pdf".format(pct, model_included, pct, model_included))

                ax.plot(predicted_inc_hosps__wopeak, color= "red", ls='--', label = "Mean prediction")
                ax.fill_between(times, lower_2p5__w0peak,upper97p5__w0peak,  color= "red" ,ls='--', alpha = 0.25, label = "75 and 95 PI")
                ax.fill_between(times, lower25__w0peak,upper75__w0peak,  color= "red" ,ls='--', alpha = 0.25)

            ax.set_ylim(0,2500)
            ax.set_xlim(0,200)
            ax.set_ylabel("Incident hospitalizations", fontsize=10)

            stamp(ax,"A.")

            ax.legend(fontsize=10,frameon=False)

            fig.set_tight_layout(True)

            w = mm2inch(183)
            fig.set_size_inches( w, w/1.5 )

            plt.savefig("./for_crowdsourcing/data_collection__{:.2f}__{:d}/plot__{:.2f}__{:d}.pdf".format(pct, model_included, pct, model_included))
            plt.close()
