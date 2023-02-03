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

    WEEK = 5 # how many weeks before the peak to cut the data 
    
    #--set randomization key
    from jax import random
    rng_key = random.PRNGKey(0)

    #--generate simulated surveillance data
    surv_data = generate_data(rng_key = rng_key)
    inc_hosps, noisy_hosps, time_at_peak, peak_value = surv_data.simulate_surveillance_data()

    #--generate two past seasons
    past0_surv_data = generate_data(rng_key = random.PRNGKey(20120905), r0 = 1.75 + (np.random.random()-0.5) )
    past0_inc_hosps, past0_noisy_hosps, past0_time_at_peak, past0_peak_value = past0_surv_data.simulate_surveillance_data()
    
    past1_surv_data = generate_data(rng_key = random.PRNGKey(20160507), r0 = 1.75 + (np.random.random()-0.5) )
    past1_inc_hosps, past1_noisy_hosps, past1_time_at_peak, past1_peak_value = past1_surv_data.simulate_surveillance_data()

    #--format past true peak data
    s0 = pd.DataFrame({"true_time_at_peak": [past0_time_at_peak], "true_peak_value": [past0_peak_value]})
    s1 = pd.DataFrame({"true_time_at_peak": [past1_time_at_peak], "true_peak_value": [past1_peak_value]})

    past_season_peak_data = pd.concat([s0,s1])
    past_season_peak_data.to_csv("./past_season_peak_data.csv",index=False)

    #--generate simulated human judgment predictins
    hj_data  = generate_humanjudgment_prediction_data(true_incident_hospitalizations = inc_hosps, time_at_peak = time_at_peak, rng_key  = rng_key ) 
    noisy_time_at_peaks, noisy_peak_values, noisy_human_predictions = hj_data.generate_predictions()

    #--save human judgement data for records
    hj_preds = pd.DataFrame(noisy_human_predictions)
    hj_preds.columns = ['noisy_time_at_peak', 'noisy_peak_values']

    hj_preds.to_csv("./hj_preds.csv", index=False)
    
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
    filenames = ["prior", "survdata", "surv_plus_hj","surv_plus_past_season"]
    options   = [(None, None, None), (cut_noisy_hosps, None, None),(cut_noisy_hosps, noisy_human_predictions, True), (cut_noisy_hosps, past_season_peak_data.asnumpy(), True)]
    
    for f,o  in zip(filenames,options) :
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
