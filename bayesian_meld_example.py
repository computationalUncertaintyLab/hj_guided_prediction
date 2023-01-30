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

def SEIRH_Forecast(rng_key, training_data, N, ps, total_window_of_observation, peak_time_and_values, peak_times_only, peak_values_only, hj_peaks_and_intensities):
    def model(training_data,ttl, ps, times):
        #--derive from data
        
        times = jnp.arange(0,total_window_of_observation)
        
        sigma = numpyro.sample("sigma", dist.Beta( 1/2., (1-(1./2)) ) )

        gamma = 1./1  #--presets
        kappa = 1./7  #--presets

        phi = numpyro.sample("phi", dist.Beta(0.025*500, (1-0.025)*500))
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

        inc_hosp_proportion = states[:,-1]
        inc_hosp_proportion = jnp.clip(inc_hosp_proportion,a_min=0,a_max= ps)
        
        inc_hosps = (inc_hosp_proportion*ttl).reshape(-1,)
        inc_hosps = numpyro.deterministic("inc_hosps",inc_hosps)

        peak_time, peak_value = jnp.argmax(inc_hosps), jnp.max(inc_hosps)
        joint_peak_data = numpyro.deterministic("peak_time_and_value", jnp.array([peak_time,peak_value]))
        
        mask = ~jnp.isnan(inc_hosps)
        
        #--surveillance data
        if training_data is None:
            pass
        else:
            T = len(training_data)
            with numpyro.handlers.mask(mask=mask[:T]):
                ll_surveillance = numpyro.sample("ll_surveillance", dist.NegativeBinomial2(inc_hosps[:T], 1./3), obs = training_data)

        if hj_peaks_and_intensities is None:
            pass
        else:
            if peak_time_and_values:
                d = hj_peaks_and_intensities.shape[1]
                N = hj_peaks_and_intensities.shape[0]

                # Vector of variances for each of the d variables
                theta = numpyro.sample("theta", dist.HalfCauchy(1*jnp.ones(d)))

                concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
                corr_mat = numpyro.sample("corr_mat", dist.LKJ(d, concentration))
                std = jnp.sqrt(theta)

                # we can also use a faster formula `cov_mat = jnp.outer(theta, theta) * corr_mat`
                cov_mat = jnp.matmul(jnp.matmul(jnp.diag(std), corr_mat), jnp.diag(std))

                # Vector of expectations
                mu = jnp.array([peak_time, peak_value]).reshape(2,)

                with numpyro.plate("observations", N):
                    obs = numpyro.sample("obs", dist.MultivariateStudentT(loc=mu, scale_tril=cov_mat, df=1), obs=hj_peaks_and_intensities)
                    
            elif peak_times_only:
                s = numpyro.sample("s", dist.HalfCauchy(100*jnp.ones(1,)))
                ll_peak_times = numpyro.sample("ll_peak_times", dist.Normal( peak_time, s ), obs = hj_peaks_and_intensities[:,0].reshape(-1,) )
            elif peak_values_only:
                ll_peak_values = numpyro.sample("ll_peak", dist.NegativeBinomial2( peak_value, 0.3 ), obs = hj_peaks_and_intensities[:,-1].reshape(-1,) )
                
    nuts_kernel = NUTS(model)
    mcmc        = MCMC( nuts_kernel , num_warmup=3000, num_samples=2000,progress_bar=True)

    mcmc.run(rng_key, extra_fields=('potential_energy',)
             , training_data = training_data
             , ttl = N
             , ps = ps
             , times = total_window_of_observation)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    #--check for any trajetories that are nan and remove them
    non_nan_trajectories = np.where( np.isnan(samples["inc_hosps"]).sum(1)==0)[0]

    if len(non_nan_trajectories)==0:
        return samples
    
    stripped_samples = {}
    for k,v in samples.items():
        stripped_samples[k] = v[non_nan_trajectories]
    return stripped_samples

def generate_multiple_noisy_measurements_of_peak_and_location(time_at_peak,inc_hosps, noisy_hosps, noise, rng_key, num_of_measurements, pct_training_data):
    train_N       = int(time_at_peak*pct_training_data)
    
    peak_value = inc_hosps[time_at_peak]
    noisy_peaks = np.asarray(dist.NegativeBinomial2( peak_value, noise ).sample(rng_key, sample_shape = (num_of_measurements,)  ))

    #--uniform distribution of time at peak
    noisy_time_at_peaks = time_at_peak + np.random.randint(low = -28, high=28, size = (num_of_measurements,))

    return noisy_hosps[:train_N], noisy_peaks, noisy_time_at_peaks
    
def mm2inch(x):
    return x/25.4

def stamp(ax,s):
    ax.text(0.0125,0.975,s=s,fontweight="bold",fontsize=10,ha="left",va="top",transform=ax.transAxes)

if __name__ == "__main__":
    N     = 12*10**6                   #--Population of the state of PA
    rng_key     = random.PRNGKey(0)    #--seed for random number generation
    ps    = 0.12                       #--Percent of the population that is considered to be susceptible
    total_window_of_observation = 210  #--total number of time units that we will observe
    
    
    #--generate simulated data
    inc_hosps, noisy_hosps, time_at_peak = generate_data(rng_key=rng_key, sigma = 1./2, gamma = 1./1,  noise = 100, T = total_window_of_observation)

    truthdata = pd.DataFrame({"times": np.arange(1,total_window_of_observation), "hosps":inc_hosps, "noisy_hosps":noisy_hosps})
    truthdata["time_at_peak"] = time_at_peak
    truthdata["peakvalue"]    = inc_hosps[time_at_peak]

    truthdata.to_csv("./fig_predictions_toprow_and_densitiesbottomrow/truth.csv",index=False)


    
    training_data, noisy_peak_value, truth_data = collect_training_data(inc_hosps,noisy_hosps
                                                            ,time_at_peak
                                                            ,0.50
                                                            ,plus_peak=False)

    #--prior information only
    samples = SEIRH_Forecast(rng_key = rng_key
                             , training_data = None
                             , N  = N
                             , ps = ps
                             , total_window_of_observation = total_window_of_observation
                             , peak_time_and_values = False
                             , peak_values_only = False
                             , peak_times_only = False
                             , hj_peaks_and_intensities = None)

    
    peaks = samples["peak_time_and_value"][:,1]
    times = samples["peak_time_and_value"][:,0]

    peaks_and_times = pd.DataFrame({"peaks":peaks,"times":times})
    peaks_and_times.to_csv("./fig_predictions_toprow_and_densitiesbottomrow/prior__only__peaks_and_times.csv",index=False)

    predicted_inc_hosps= np.median(samples["inc_hosps"], 0)
    lower_2p5,lower25, upper75, upper97p5 = np.percentile( samples["inc_hosps"], [2.5, 25, 75, 97.5], 0)

    quantiles = pd.DataFrame({"t":np.arange(0, len(predicted_inc_hosps)), "median":predicted_inc_hosps, "lower2p5":lower_2p5, "lower25": lower25, "upper75":upper75, "upper97p5":upper97p5 })
    quantiles.to_csv("./fig_predictions_toprow_and_densitiesbottomrow/prior__only__quantiles.csv", index=False)
    
    pickle.dump( samples, open("prior__samples.pkl", "wb"))

    
    #--include surveillance data
    samples = SEIRH_Forecast(rng_key = rng_key
                             , training_data = training_data
                             , N  = N
                             , ps = ps
                             , total_window_of_observation = total_window_of_observation
                             , peak_time_and_values = False
                             , peak_values_only = False
                             , peak_times_only = False
                             , hj_peaks_and_intensities = None)
    peaks = samples["peak_time_and_value"][:,1]
    times = samples["peak_time_and_value"][:,0]

    peaks_and_times = pd.DataFrame({"peaks":peaks,"times":times})
    peaks_and_times.to_csv("./fig_predictions_toprow_and_densitiesbottomrow/survdata__peaks_and_times.csv",index=False)

    predicted_inc_hosps= np.median(samples["inc_hosps"], 0)
    lower_2p5,lower25, upper75, upper97p5 = np.percentile( samples["inc_hosps"], [2.5, 25, 75, 97.5], 0)

    quantiles = pd.DataFrame({"t":np.arange(0, len(predicted_inc_hosps)), "median":predicted_inc_hosps, "lower2p5":lower_2p5, "lower25": lower25, "upper75":upper75, "upper97p5":upper97p5 })
    quantiles.to_csv("./fig_predictions_toprow_and_densitiesbottomrow/survdata__quantiles.csv",index=False)
    
    pickle.dump( samples, open("survdata__samples.pkl", "wb"))
    
    #--include hj data
    training_data, noisy_peaks, noisy_time_at_peaks = generate_multiple_noisy_measurements_of_peak_and_location(time_at_peak
                                                                                                                , inc_hosps
                                                                                                                , noisy_hosps
                                                                                                                , noise = 100
                                                                                                                , rng_key = rng_key
                                                                                                                , num_of_measurements = 50
                                                                                                                , pct_training_data=0.50)                                
    hj_peaks_and_intensities = np.vstack([noisy_time_at_peaks, noisy_peaks]).T
    hj_peaks_and_intensities = hj_peaks_and_intensities.astype(float)

    hjout = pd.DataFrame({ "time": hj_peaks_and_intensities[:,0], "peak":hj_peaks_and_intensities[:,1]})
    hjout.to_csv("./fig_predictions_toprow_and_densitiesbottomrow/hj_peaks_and_intensities.csv",index=False)
    
    samples = SEIRH_Forecast(rng_key = rng_key
                             , training_data = training_data
                             , N  = N
                             , ps = ps
                             , total_window_of_observation = total_window_of_observation
                             , peak_time_and_values = True
                             , peak_values_only     = False
                             , peak_times_only      = False
                             , hj_peaks_and_intensities    = hj_peaks_and_intensities)

    peaks = samples["peak_time_and_value"][:,1]
    times = samples["peak_time_and_value"][:,0]

    peaks_and_times = pd.DataFrame({"peaks":peaks,"times":times})
    peaks_and_times.to_csv("./fig_predictions_toprow_and_densitiesbottomrow/hjdata__peaks_and_times.csv",index=False)

    predicted_inc_hosps= np.median(samples["inc_hosps"], 0)
    lower_2p5,lower25, upper75, upper97p5 = np.percentile( samples["inc_hosps"], [2.5, 25, 75, 97.5], 0)

    quantiles = pd.DataFrame({"t":np.arange(0, len(predicted_inc_hosps)), "median":predicted_inc_hosps, "lower2p5":lower_2p5, "lower25": lower25, "upper75":upper75, "upper97p5":upper97p5 })
    quantiles.to_csv("./fig_predictions_toprow_and_densitiesbottomrow/hjdata__quantiles.csv",index=False)
    
    pickle.dump( samples, open("hjdata__samples.pkl", "wb"))
