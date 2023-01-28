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
            with numpyro.handlers.mask(mask=mask):
                sim = numpyro.sample("sim_inc_hosps", dist.NegativeBinomial2(inc_hosps[:T], 10), obs = training_data)
        else:
            sim = numpyro.sample("sim_inc_hosps", dist.NegativeBinomial2(inc_hosps[:T], 10), obs = training_data)
                
    nuts_kernel = NUTS(model)
    mcmc        = MCMC( nuts_kernel , num_warmup=1500, num_samples=2000,progress_bar=True)

    mcmc.run(rng_key, extra_fields=('potential_energy',)
             , training_data = training_data
             , ttl = N
             , ps = ps
             , times = total_window_of_observation
             , mask = mask)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    return samples

def generate_multiple_noisy_measurements_of_peak(time_at_peak,inc_hosps, noisy_hosps, noise, rng_key, num_of_measurements):
    peak_value = inc_hosps[time_at_peak]

    noisy_peaks = np.asarray(dist.NegativeBinomial2( peak_value, noise ).sample(rng_key, sample_shape = (num_of_measurements,)  ))
    
    repeated_noisy_hosps = np.repeat(noisy_hosps[np.newaxis],num_of_measurements,0).T
    repeated_noisy_hosps[time_at_peak] = noisy_peaks

    return repeated_noisy_hosps.T

def generate_multiple_noisy_measurements_of_peak_and_location(time_at_peak,inc_hosps, noisy_hosps, noise, rng_key, num_of_measurements):
    peak_value = inc_hosps[time_at_peak]
    noisy_peaks = np.asarray(dist.NegativeBinomial2( peak_value, noise ).sample(rng_key, sample_shape = (num_of_measurements,)  ))

    #--uniform distribution of time at peak
    noisy_time_at_peaks = time_at_peak + np.random.randint(low = -14, high=14, size = (num_of_measurements,))
    
    
    repeated_noisy_hosps = np.repeat(noisy_hosps[np.newaxis],num_of_measurements,0).T

    for n,(noisy_time, noisy_value) in enumerate(zip(noisy_time_at_peaks, noisy_peaks)):
        repeated_noisy_hosps[noisy_time,n] = noisy_value
    return repeated_noisy_hosps.T, noisy_peaks, noisy_time_at_peaks

def mm2inch(x):
    return x/25.4

def stamp(ax,s):
    ax.text(0.0125,0.975,s=s,fontweight="bold",fontsize=10,ha="left",va="top",transform=ax.transAxes)

if __name__ == "__main__":
    rng_key     = random.PRNGKey(0)    #--seed for random number generation
    N     = 12*10**6                   #--Population of the state of PA
    ps    = 0.12                       #--Percent of the population that is considered to be susceptible
    total_window_of_observation = 210  #--total number of time units that we will observe
    
    
    #--generate simulated data
    inc_hosps, noisy_hosps, time_at_peak = generate_data(rng_key=rng_key, sigma = 1./2, gamma = 1./1,  noise = 100, T = total_window_of_observation)
    training_data__withpeak, truth_data, mask = collect_training_data(inc_hosps,noisy_hosps
                                                            ,time_at_peak
                                                            ,0.20
                                                            ,plus_peak=True)


    #--forecast with peak data included
    samples__wpeak = SEIRH_Forecast(rng_key, training_data__withpeak, mask, N, ps, total_window_of_observation )
    predicted_inc_hosps__wpeak = samples__wpeak["inc_hosps"].mean(0)

    lower_2p5__wpeak,lower25__wpeak, upper75__wpeak, upper97p5__wpeak = np.percentile( samples__wpeak["inc_hosps"], [2.5, 25, 75, 97.5], 0)

    #--
    training_data__wopeak, truth_data, mask = collect_training_data(inc_hosps,noisy_hosps
                                                            ,time_at_peak
                                                            ,0.20
                                                            , plus_peak=False)
    #--forecast with out peak data included
    samples__wopeak = SEIRH_Forecast(rng_key, training_data__wopeak, mask, N, ps, total_window_of_observation )
    
    predicted_inc_hosps__wopeak = samples__wopeak["inc_hosps"].mean(0)
    lower_2p5__w0peak,lower25__w0peak, upper75__w0peak, upper97p5__w0peak = np.percentile( samples__wopeak["inc_hosps"], [2.5, 25, 75, 97.5], 0)


    #--weighted average of forecasts with repeated peak measurements
    num_of_measurements = 100
    noise = 100
    repeated_training_data = generate_multiple_noisy_measurements_of_peak(time_at_peak, inc_hosps, noisy_hosps, noise, rng_key, num_of_measurements = num_of_measurements)

    def process_data_and_make_prediction(d,pct,inc_hosps,time_at_peak):
        training_data__withpeak, truth_data, mask = collect_training_data(inc_hosps
                                                                          ,d
                                                                          ,time_at_peak
                                                                          ,pct
                                                                          ,plus_peak=True)
        samples = SEIRH_Forecast(rng_key, training_data__withpeak, mask, N, ps, total_window_of_observation )
        predicted_inc_hosps__sample = samples["inc_hosps"].mean(0)
        return predicted_inc_hosps__sample
    process = lambda d: process_data_and_make_prediction(d, pct = 0.20, inc_hosps = inc_hosps, time_at_peak = time_at_peak )

    
    rslts = Parallel(n_jobs=20)(delayed(process)(i) for i in repeated_training_data)
    
    store_predictions = np.zeros((num_of_measurements,total_window_of_observation))
    for n,row in enumerate(rslts):
        store_predictions[n,:] = row
    mean_prediction = store_predictions.mean(0)

    lower_2p5__ensemble,lower25__ensemble, upper75__ensemble, upper97p5__ensemble = np.percentile( store_predictions, [2.5, 25, 75, 97.5], 0)

    #--multiple measurrment of noisy peak and noisy value
    repeated_training_data__peakandvalue, noisy_peak_values, noisy_peak_times = generate_multiple_noisy_measurements_of_peak_and_location(time_at_peak,inc_hosps, noisy_hosps, noise, rng_key, num_of_measurements)

    def process_data_and_make_prediction(d,pct,inc_hosps,time_at_peak):
        training_data__withpeak, truth_data, mask = collect_training_data(inc_hosps
                                                                          ,d
                                                                          ,time_at_peak
                                                                          ,pct
                                                                          ,plus_peak=True)
        samples = SEIRH_Forecast(rng_key, training_data__withpeak, mask, N, ps, total_window_of_observation )
        predicted_inc_hosps__sample = samples["inc_hosps"].mean(0)
        return predicted_inc_hosps__sample
    process = lambda d,t: process_data_and_make_prediction(d, pct = 0.20, inc_hosps = inc_hosps, time_at_peak = t )
    
    rslts = Parallel(n_jobs=20)(delayed(process)(d,t) for d,t in zip(repeated_training_data__peakandvalue,noisy_peak_times))
 
    store_predictions_pv = np.zeros((num_of_measurements,total_window_of_observation))
    for n,row in enumerate(rslts):
        store_predictions_pv[n,:] = row
    mean_prediction_peak_and_value = store_predictions_pv.mean(0)

    lower_2p5__ensemble_pv,lower25__ensemble_pv, upper75__ensemble_pv, upper97p5__ensemble_pv = np.percentile( store_predictions_pv, [2.5, 25, 75, 97.5], 0)
    
    
    #-plot
    plt.style.use('science')
    
    fig,axs = plt.subplots(2,2)
    times = np.arange(0,total_window_of_observation)
    
    #--plot truth on all graphs
    for ax in axs.flatten():
        ax.plot(times[1:], inc_hosps, color="black", label = "Truth")

    ax = axs[0,0]
    
    #--plot without peak
    ax.plot(predicted_inc_hosps__wopeak, color= "red", ls='--', label = "Mean prediction")
    ax.scatter(times[:len(training_data__wopeak)], training_data__wopeak, lw=1, color="blue", alpha=1,s=3, label = "Surveillance data (up to day 16) ")
    
    ax.fill_between(times, lower_2p5__w0peak,upper97p5__w0peak,  color= "red" ,ls='--', alpha = 0.25, label = "75 and 95 PI")
    ax.fill_between(times, lower25__w0peak,upper75__w0peak,  color= "red" ,ls='--', alpha = 0.25)

    ax.set_ylim(0,2500)
    ax.set_xlim(0,200)
    ax.set_ylabel("Incident hospitlizations", fontsize=10)

    stamp(ax,"A.")

    ax.legend(fontsize=10,frameon=False)
    
    #--plot with single noisy measurement of intensity
    ax = axs[0,1]

    train_N = len(training_data__withpeak)

    #--plot simulated data    
    ax.scatter(times[:train_N], training_data__withpeak, lw=1, color="blue", alpha=1,s=3, label = "Surveillance data (up to day 16) ")
    
    ax.plot(times, predicted_inc_hosps__wpeak        , color= "purple" ,ls='--')
    ax.fill_between(times, lower_2p5__wpeak,upper97p5__wpeak,  color= "purple" ,ls='--', alpha = 0.25)
    ax.fill_between(times, lower25__wpeak,upper75__wpeak,  color= "purple" ,ls='--', alpha = 0.25)

    ax.set_ylim(0,2500)
    ax.set_xlim(0,200)

    stamp(ax,"B.")

    ax = axs[1,0]
    
    #--plot with multi measurements of peak
    #--plot simulated data       

    ax.scatter(times[:train_N], training_data__withpeak, lw=1, color="blue", alpha=1,s=3)
    for d in repeated_training_data:
        ax.scatter(times[time_at_peak], d[time_at_peak], lw=1, color="blue", alpha=1,s=3)
    
    ax.plot(times, mean_prediction, color= "blue", ls='--')
    ax.fill_between(times, lower_2p5__ensemble,upper97p5__ensemble,  color= "blue" ,ls='--', alpha = 0.25)
    ax.fill_between(times, lower25__ensemble,upper75__ensemble,  color= "blue" ,ls='--', alpha = 0.25)

    ax.set_ylim(0,2500)
    ax.set_xlim(0,200)

    ax.set_ylabel("Incident hospitlizations", fontsize=10)
    ax.set_xlabel("Time (days)", fontsize=10)

    stamp(ax,"C.")
    
    ax = axs[1,1]
    
    #--plot with multi measurements of peak
    #--plot simulated data       

    ax.scatter(times[:train_N], training_data__withpeak, lw=1, color="blue", alpha=1,s=3)
    for d,noisy_peak in zip(repeated_training_data__peakandvalue,noisy_peak_times):
        ax.scatter(times[noisy_peak], d[noisy_peak], lw=1, color="blue", alpha=1,s=3)
    
    ax.plot(times, mean_prediction_peak_and_value, color= "orange", ls='--')
    ax.fill_between(times, lower_2p5__ensemble_pv,upper97p5__ensemble_pv,  color= "orange" ,ls='--', alpha = 0.25)
    ax.fill_between(times, lower25__ensemble_pv,upper75__ensemble_pv,  color= "orange" ,ls='--', alpha = 0.25)

    ax.set_ylim(0,2500)
    ax.set_xlim(0,200)
    ax.set_xlabel("Time (days)", fontsize=10)

    stamp(ax,"D.")
    
    fig.set_tight_layout(True)

    w = mm2inch(183)
    fig.set_size_inches( w, w/1.5 )

    plt.show()
