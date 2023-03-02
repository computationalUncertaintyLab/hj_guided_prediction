#mcandrew

import sys
sys.path.append("../../")

import os
import pickle

import numpy as np
import pandas as pd

from jax import random

from chimeric_forecast.generate_simulated_data import generate_data
from chimeric_forecast.chimeric_forecast__meld7 import chimeric_forecast

from chimeric_forecast.generate_humanjudgment_prediction_data import generate_humanjudgment_prediction_data

import numpyro
import numpyro.distributions as dist

import argparse

if __name__ == "__main__":
    rng_key     = random.PRNGKey(123986)    #--seed for random number generation
    total_window_of_observation = 210
    times = np.arange(1,210+1)

    population = 12*10**6
    
    #--generate several noisy curves and record the peak time and intensity
    sim_data = {"sim":[]
                , "time":[]
                , "inc_hosps":[]
                , "noisy_inc_hosps":[]
                , "peak_time":[]
                , "peak_intensity":[]
                , "i0":[]
                , "e0":[]
                , "ps":[]
                , "sigma":[]
                , "r0":[]
                } 

    #--randomly geenrated from the same prior as the model fit.
    random_i0    = dist.Uniform(1./population, 2./population).sample( random.PRNGKey(123986), (1*10**3,) )
    random_e0    = dist.Uniform(1./population, 2./population).sample( random.PRNGKey(123934), (1*10**3,) )
    
    #random_ps    = dist.Beta( 0.10*5, (1-0.10)*5 ).sample( random.PRNGKey(20200301), (1*10**3,) )
    #random_ph    = dist.Beta(0.025*5, (1-0.025)*5).sample( random.PRNGKey(20230302), (1*10**3,) )

    random_ps = [0.10]*10**3
    random_ph = [0.025]*10**3

    random_sigma = dist.Beta( 1/2., (1-(1./2)) ).sample( random.PRNGKey(20210301), (1*10**3,) )
    random_r0    = dist.Uniform(0.75,4).sample( random.PRNGKey(20220302), (1*10**3,) )

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int) 

    args = parser.parse_args()
    N = args.N

    random_i0    = random_i0[N:N+10]
    random_e0    = random_e0[N:N+10]
    random_ps    = random_ps[N:N+10]
    random_sigma = random_sigma[N:N+10]
    random_r0    = random_r0[N:N+10]
    random_ph    = random_ph[N:N+10]
    
    #--these were not random and instead fixed.
    fixed_gamma = 1.
    fixed_kappa = 1./7
    noise       = 5.

    def simulation(i0,e0,ps,sigma,r0,ph, sim=0):
        qtheta0s = {"sim":[], "q_peak_time_hj":[], "q_peak_intensity_hj":[], "q_peak_time_surv":[],"q_peak_intensity_surv":[]}
        surv_data = generate_data(rng_key = random.PRNGKey(np.random.randint(low=0,high=9999999))
                                  , r0    = r0
                                  , I0    = i0
                                  , E0    = e0
                                  , H0    = 0.
                                  , ps    = ps
                                  , sigma = sigma
                                  , gamma = fixed_gamma
                                  , kappa = fixed_kappa
                                  , ph    = ph
                                  , noise = noise
                                  , continuous = False)
        inc_hosps, noisy_hosps, time_at_peak, peak_value = surv_data.simulate_surveillance_data()

        sim_data["sim"].extend( [sim]*total_window_of_observation  )

        sim_data["i0"].extend( [i0]*total_window_of_observation )
        sim_data["e0"].extend( [e0]*total_window_of_observation )
        sim_data["ps"].extend( [ps]*total_window_of_observation )
        sim_data["sigma"].extend( [sigma]*total_window_of_observation )
        sim_data["r0"].extend( [r0]*total_window_of_observation )
        
        sim_data["time"].extend( times )
        sim_data["inc_hosps"].extend(inc_hosps)
        sim_data["noisy_inc_hosps"].extend(noisy_hosps)
        sim_data["peak_time"].extend([time_at_peak]*total_window_of_observation)
        sim_data["peak_intensity"].extend([peak_value]*total_window_of_observation)

        #--add in HJ predictions
        hj = generate_humanjudgment_prediction_data(true_incident_hospitalizations = inc_hosps
                                                    , time_at_peak = time_at_peak
                                                    , span    = 7.
                                                    , noise   = 25.
                                                    , number_of_humanjudgment_predictions=100
                                                    , rng_key = random.PRNGKey(np.random.randint(low=0,high=9999999)) )
        hj_times, hj_peaks, hj_data = hj.generate_predictions()

        #--train model
        time_before_peak = time_at_peak - 4*7

        try: #May not be able to find valid initial parameters depednign on the data
            forecast = chimeric_forecast(rng_key = random.PRNGKey(np.random.randint(low=0,high=9999999))
                                                      , surveillance_data  = noisy_hosps[:time_before_peak]
                                                      , humanjudgment_data = hj_data
                                                      , past_season_data   = None
                                         )
            forecast.fit_model()

            quantiles_for_incident_hosps = forecast.compute_quantiles()
            quantiles_for_incident_hosps["t"] = np.arange(0,210)
            #--posterior samples of peak
            peak_times_and_intensities = forecast.posterior_samples["peak_time_and_value"]

            #compare posterior samples to truth
            qtheta0_peak_time      = float(np.mean( time_at_peak > peak_times_and_intensities[:,0]))
            qtheta0_peak_intensity = float(np.mean( peak_value > peak_times_and_intensities[:,1]))

            qtheta0s["sim"].append(sim)
            qtheta0s["q_peak_time_hj"].append(qtheta0_peak_time)
            qtheta0s["q_peak_intensity_hj"].append(qtheta0_peak_intensity)

            #--SURVEILLANCE ONLY

            forecast_surv = chimeric_forecast(rng_key = random.PRNGKey(np.random.randint(low=0,high=9999999))
                                                      , surveillance_data  = noisy_hosps[:time_before_peak]
                                                      , humanjudgment_data = None
                                                      , past_season_data   = None
                                         )
            forecast_surv.fit_model()

            quantiles_for_incident_hosps = forecast_surv.compute_quantiles()
            quantiles_for_incident_hosps["t"] = np.arange(0,210)

            #--posterior samples of peak
            peak_times_and_intensities = forecast_surv.posterior_samples["peak_time_and_value"]

            #compare posterior samples to truth
            qtheta0_peak_time      = float(np.mean( time_at_peak > peak_times_and_intensities[:,0]))
            qtheta0_peak_intensity = float(np.mean( peak_value > peak_times_and_intensities[:,1]))

            qtheta0s["q_peak_time_surv"].append(qtheta0_peak_time)
            qtheta0s["q_peak_intensity_surv"].append(qtheta0_peak_intensity)
        except:
            return {"sim":[],"q_peak_time_surv":[],"q_peak_intensity_surv":[],"q_peak_time_hj":[],"q_peak_intensity_hj":[]}

        return qtheta0s

    from joblib import Parallel, delayed
    rslts = Parallel(n_jobs=10)(delayed(simulation)(i0,e0,ps,sigma,r0,ph,sim) for  sim,(i0,e0,ps,sigma,r0,ph) in enumerate(zip(random_i0,random_e0,random_ps,random_sigma,random_r0,random_ph)) )

    d = {"sim":[], "q_peak_time_surv":[], "q_peak_intensity_surv":[],"q_peak_time_hj":[],"q_peak_intensity_hj":[]}
    for rslt in rslts:
        if rslt["sim"] == []:
            continue
        for k,v in rslt.items():
            d[k].append(v[0])
    qtheta0s = pd.DataFrame(d)
   
    qtheta0s.to_csv("./cut_time_series_plushj_quantile_peaks__{:02d}.csv".format(N),index=False)
