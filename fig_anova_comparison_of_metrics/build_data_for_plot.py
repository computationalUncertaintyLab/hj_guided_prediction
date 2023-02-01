#mcandrew

import sys
sys.path.append("../chimeric_forecast")

import numpy as np
import pandas as pd
import pickle

from chimeric_forecast import chimeric_forecast 
from generate_simulated_data import generate_data
from generate_humanjudgment_prediction_data import generate_humanjudgment_prediction_data 

if __name__ == "__main__":

    WEEK = 5 #--how many weeks before the peak to cut the data 
    
    #--set randomization key
    from jax import random
    rng_key = random.PRNGKey(0)

    #--generate simulated surveillance data
    surv_data = generate_data(rng_key = rng_key)
    inc_hosps, noisy_hosps, time_at_peak, peak_value = surv_data.simulate_surveillance_data()

    #--generate 100 simulated human judgment predictins that we will then bootstrap 50 at a time without replacement
    hj_data  = generate_humanjudgment_prediction_data(true_incident_hospitalizations = inc_hosps
                                                      , time_at_peak = time_at_peak
                                                      , number_of_humanjudgment_predictions = 100
                                                      , rng_key  = rng_key ) 
    noisy_time_at_peaks, noisy_peak_values, noisy_human_predictions = hj_data.generate_predictions()

    #--save human judgement data for records
    hj_preds = pd.DataFrame(noisy_human_predictions)
    hj_preds.columns = ['noisy_time_at_peak', 'noisy_peak_values']

    hj_preds.to_csv("./hj_preds.csv", index=False)

    #--bootstrap 30**2 times
    np.random.shuffle(data)
    

    

    
    #--save truth data for records
    truthdata = pd.DataFrame({"times": np.arange(0,surv_data.total_window_of_observation), "hosps":inc_hosps, "noisy_hosps":noisy_hosps})
    truthdata["time_at_peak"] = time_at_peak
    truthdata["peakvalue"]    = inc_hosps[time_at_peak]

    truthdata.to_csv("./truth.csv",index=False)

    #--cut surveillance data to a specific week before the peak
    cut_inc_hosps, cut_noisy_hosps, cut_point = surv_data.cut_data_based_on_week(WEEK)

    #--save noisy truth data for records
    noisy_truthdata = pd.DataFrame({"times": np.arange(0,cut_point), "hosps":cut_inc_hosps, "noisy_hosps":cut_noisy_hosps})
    noisy_truthdata["time_at_peak"] = time_at_peak
    noisy_truthdata["peakvalue"]    = inc_hosps[time_at_peak]

    noisy_truthdata.to_csv("./noisy_truth.csv",index=False)

    #--three models to run: just prior, surv data only, chimeric
    filenames = ["prior", "survdata", "surv_plus_hj"]
    options   = [(None, None, None), (cut_noisy_hosps, None, None), (cut_noisy_hosps, noisy_human_predictions, True)]
    
    for f,o in zip(filenames,options) :
        d, h, o = o #--these are the options that control what information gets to the model
        
        #--fit model that uses only prior information
        forecast = chimeric_forecast(rng_key = rng_key, surveillance_data = d, humanjudgment_data = h, peak_time_and_values = o )
        forecast.fit_model()

        #--record samples of the peak times and intensities to use for bottom row of plot (plot.py)
        samples = forecast.posterior_samples
        peaks   = samples["peak_time_and_value"][:,1]
        times   = samples["peak_time_and_value"][:,0]

        peaks_and_times = pd.DataFrame({"peaks":peaks,"times":times})
        peaks_and_times.to_csv("./{:s}__peaks_and_times.csv".format(f),index=False)

        #--record quantiles from forecast to use in top row of plot (plot.py)
        quantiles_for_incident_hosps = forecast.compute_quantiles()
        quantiles_for_incident_hosps["t"] = np.arange(0,surv_data.total_window_of_observation)

        quantiles_for_incident_hosps.to_csv("./{:s}__quantiles.csv".format(f), index=False)

        pickle.dump( forecast, open("{:s}__forecast.pkl".format(f), "wb"))
