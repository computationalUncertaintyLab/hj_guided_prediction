#mcandrew

import sys
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots

from jax import random

from chimeric_forecast.generate_simulated_data import generate_data
from chimeric_forecast.chimeric_forecast import chimeric_forecast
   
if __name__ == "__main__":
    rng_keys     = [random.PRNGKey(20201017), random.PRNGKey(19870420)]    #--seed for random number generation
    total_window_of_observation = 210
    times = np.arange(1,210+1)

    if os.path.isdir("./for_crowdsourcing/"):
        pass
    else:
        os.makedirs("./for_crowdsourcing/")

    #--randomly geenrate R0 to be plus minus 0.5 of 1.75
    #--This will be the same two curves for all treatements
    sim_r0 = 1.75 + (np.random.random(size=(2,)) - 0.5)
        
    for week in [6,5,4,3,2,1,0,-1,-2]:
        for noise in [0.25, 1.0,2.5, 5., 10.]:
            for model_included in [0,1]:
                for season in [0,1]:

                    if os.path.isdir("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/".format(week, model_included, noise)):
                        pass
                    else:
                        os.makedirs("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/".format(week, model_included, noise))

                    #--generate simulated surveillance data
                    surv_data = generate_data(rng_key = rng_keys[season], r0=sim_r0[season], noise = noise)
                    inc_hosps, noisy_hosps, time_at_peak, peak_value = surv_data.simulate_surveillance_data()
                
                    #--store these dataframes
                    pd.DataFrame({"r0":[sim_r0]}).to_csv("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_r0__{:d}__{:d}__{:.2f}_{:02d}.csv".format(week, model_included, noise, week, model_included, noise,season))
                    
                    pd.DataFrame({"training_data":noisy_hosps}).to_csv("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_training_data__{:d}__{:d}__{:.2f}_{:02d}.csv".format(week, model_included, noise, week, model_included, noise,season))
                    pd.DataFrame({"truth_data":inc_hosps}).to_csv("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_truth_data__{:d}__{:d}__{:.2f}_{:02d}.csv".format(week, model_included, noise, week, model_included, noise, season))
                    pd.DataFrame({"time_at_peak":[time_at_peak]}).to_csv("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_time_at_peak__{:d}__{:d}__{:.2f}_{:02d}.csv".format(week, model_included, noise, week, model_included, noise,season))




                    
