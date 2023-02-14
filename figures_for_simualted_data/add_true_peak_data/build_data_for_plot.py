#mcandrew

import sys
sys.path.append("../../")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from jax import random

from chimeric_forecast.chimeric_forecast import chimeric_forecast
from chimeric_forecast.generate_simulated_data import generate_data

if __name__ == "__main__":
    
    times = np.arange(0,210)
    rng_key = random.PRNGKey(20200315)
    peak_sims = {"sim":[], "model":[], "peak_time":[], "peak_intensity":[]}
    
    for sim in range(2000):
        r0 = 1 + np.random.random()*3
        inc_hosps, noisy_hosps, time_at_peak, peak_value = generate_data(rng_key=rng_key,r0=r0).simulate_surveillance_data()

        #--4 weeks before peak
        four_weeks_before = time_at_peak - 4*7

        fig,axs = plt.subplots(2,2)

        #--surveillance only
        model = "surv"
        rng_key  = random.PRNGKey(20151009)
        forecast = chimeric_forecast(rng_key = rng_key
                                     , surveillance_data = noisy_hosps)
        posterior_samples = forecast.fit_model()

        #--quantiles
        quantiles = forecast.compute_quantiles()

        ax = axs[0,0]
        ax.plot(times, inc_hosps, color="black")
        ax.plot(time_at_peak, peak_value, "ro")

        ax.fill_between(times, quantiles["2.500"], quantiles["97.500"],alpha=0.30, color = "red")
        ax.title("surv only")
        
        #--compute percentile for peak time and peak intensity
        peak_samples = posterior_samples["peak_time_and_value"]
        q_time, q_intensity = np.mean(peak_samples < np.array([time_at_peak, peak_value]),0 )

        peak_sims["sim"].append(sim)
        peak_sims["model"].append(model)
        peak_sims["peak_time"].append(q_time)
        peak_sims["peak_intensity"].append(q_intensity)
        #----------------------------------------------------------------------------------------------

        #--model that includes true hj independent
        model = "peak"
        rng_key  = random.PRNGKey(20151009)

        true_peak_time_and_intensity           = np.array([time_at_peak, peak_value]).reshape(1,-1)
        repeated__true_peak_time_and_intensity = np.repeat(true_peak_time_and_intensity,10)
        
        
        forecast = chimeric_forecast(rng_key = rng_key
                                     , surveillance_data = noisy_hosps
                                     , peak_time_and_values_independent = True
                                     , humanjudgment_data = true_peak_time_and_intensity )
        posterior_samples = forecast.fit_model()

        #--quantiles
        quantiles = forecast.compute_quantiles()

        ax = axs[0,1]
        ax.plot(times, inc_hosps, color="black")
        ax.plot(time_at_peak, peak_value, "ro")
        
        ax.fill_between(times, quantiles["2.500"], quantiles["97.500"],alpha=0.30, color = "blue")
        ax.title("time and intesntiy indpt")
        #----------------------------------------------------------------------------------------------

        
        #--model that includes true time
        model = "peak"
        rng_key  = random.PRNGKey(20151009)

        true_peak_time_and_intensity           = np.array([time_at_peak, peak_value]).reshape(1,-1)
        repeated__true_peak_time_and_intensity = np.repeat(true_peak_time_and_intensity,10)
        
        
        forecast = chimeric_forecast(rng_key = rng_key
                                     , surveillance_data = noisy_hosps
                                     , peak_times_only   = True
                                     , humanjudgment_data = time_at_peak )
        posterior_samples = forecast.fit_model()

        #--quantiles
        quantiles = forecast.compute_quantiles()

        ax = axs[1,0]
        ax.plot(times, inc_hosps, color="black")
        ax.plot(time_at_peak, peak_value, "ro")
 
        ax.fill_between(times, quantiles["2.500"], quantiles["97.500"],alpha=0.30, color = "blue")
        ax.title("time only")
        #----------------------------------------------------------------------------------------------
        
        #--compute percentile for peak time and peak intensity
        peak_samples = posterior_samples["peak_time_and_value"]
        q_time, q_intensity = np.mean(peak_samples < np.array([time_at_peak, peak_value]),0 )

        peak_sims["sim"].append(sim)
        peak_sims["model"].append(model)
        peak_sims["peak_time"].append(q_time)
        peak_sims["peak_intensity"].append(q_intensity)



        #--model that includes true time
        model = "peak"
        rng_key  = random.PRNGKey(20151009)

        true_peak_time_and_intensity           = np.array([time_at_peak, peak_value]).reshape(1,-1)
        repeated__true_peak_time_and_intensity = np.repeat(true_peak_time_and_intensity,10)
        
        
        forecast = chimeric_forecast(rng_key = rng_key
                                     , surveillance_data = noisy_hosps
                                     , peak_values_only = True
                                     , humanjudgment_data = peak_value )
        posterior_samples = forecast.fit_model()

        #--quantiles
        quantiles = forecast.compute_quantiles()

        ax = axs[1,1]
        ax.plot(times, inc_hosps, color="black")
        ax.plot(time_at_peak, peak_value, "ro")
 
        ax.fill_between(times, quantiles["2.500"], quantiles["97.500"],alpha=0.30, color = "blue")
        ax.title("intensity only")
        #----------------------------------------------------------------------------------------------
        
        #--compute percentile for peak time and peak intensity
        peak_samples = posterior_samples["peak_time_and_value"]
        q_time, q_intensity = np.mean(peak_samples < np.array([time_at_peak, peak_value]),0 )

        peak_sims["sim"].append(sim)
        peak_sims["model"].append(model)
        peak_sims["peak_time"].append(q_time)
        peak_sims["peak_intensity"].append(q_intensity)

        plt.show()
        break
    


        
    
