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

    #--set randomization key
    from jax import random
    rng_key = random.PRNGKey(0)

    #--collect all forecatsing data in one file
    all_quantile_data = pd.DataFrame()
    all_peak_data     = pd.DataFrame()
    
    #--factors
    for week in [6,5,4,3,2,1,0,-1,-2]:
        for noise in [0.25, 1.0, 10.]:
            for model_included in [0,1]: #--we assume that times are more similar and intesite (ie smaller variance)

                #--generate simulated surveillance data
                if model_included:
                    model_inc=0.75
                else:
                    model_inc=1.
                
                surv_data = generate_data(rng_key = rng_key, r0=1.75, noise = noise)
                inc_hosps, noisy_hosps, time_at_peak, peak_value = surv_data.simulate_surveillance_data()

                #--save truth data for records
                truthdata = pd.DataFrame({"times": np.arange(0,surv_data.total_window_of_observation), "hosps":inc_hosps, "noisy_hosps":noisy_hosps})
                truthdata["time_at_peak"] = time_at_peak
                truthdata["peakvalue"]    = inc_hosps[time_at_peak]

                truthdata.to_csv("./data/truth__{:d}__{:d}__{:.2f}.csv".format(week, model_included, noise),index=False)
                
                #--cut surveillance data to a specific week before the peak
                cut_inc_hosps, cut_noisy_hosps, cut_point = surv_data.cut_data_based_on_week(week)

                #--save noisy truth data for records
                noisy_truthdata = pd.DataFrame({"times": np.arange(0,cut_point), "hosps":cut_inc_hosps, "noisy_hosps":cut_noisy_hosps})
                noisy_truthdata["time_at_peak"] = time_at_peak
                noisy_truthdata["peakvalue"]    = inc_hosps[time_at_peak]

                noisy_truthdata.to_csv("./noisy_truth.csv",index=False)
                
                #--store these dataframes    
                pd.DataFrame({"training_data":cut_noisy_hosps}).to_csv("./data/training_data__{:d}__{:d}__{:.2f}.csv".format(week, model_included, noise, week, model_included, noise))
                pd.DataFrame({"truth_data":cut_inc_hosps}).to_csv(     "./data/truth_data__{:d}__{:d}__{:.2f}.csv".format(week, model_included, noise, week, model_included, noise))
                pd.DataFrame({"time_at_peak":[time_at_peak]}).to_csv(  "./data/time_at_peak__{:d}__{:d}__{:.2f}.csv".format(week, model_included, noise, week, model_included, noise))


                hj_data  = generate_humanjudgment_prediction_data(true_incident_hospitalizations = inc_hosps
                                                      , time_at_peak = time_at_peak
                                                      , number_of_humanjudgment_predictions = 100
                                                      , noise = 5.95*model_inc #--we assume that times are more similar and intesite (ie smaller variance)
                                                      , rng_key  = rng_key ) 
                noisy_time_at_peaks, noisy_peak_values, noisy_human_predictions = hj_data.generate_predictions()

                #--save human judgement data for records
                hj_preds = pd.DataFrame(noisy_human_predictions)
                hj_preds.columns = ['noisy_time_at_peak', 'noisy_peak_values']

                hj_preds.to_csv("./data/hj_preds__{:d}__{:d}__{:.2f}.csv".format(week, model_included, noise), index=False)
                
                #--Three models to run: just prior, surv data only, and surve plus hj
                filenames = ["prior", "survdata", "survdata_plus_hj"]
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
                    peaks_and_times.to_csv("./plotdata/{:s}__peaks_and_times__{:d}__{:d}__{:.2f}.csv".format(f, week, model_included, noise),index=False)

                    #--record quantiles from forecast to use in top row of plot (plot.py)
                    quantiles_for_incident_hosps = forecast.compute_quantiles()
                    quantiles_for_incident_hosps["t"] = np.arange(0,surv_data.total_window_of_observation)

                    quantiles_for_incident_hosps.to_csv("./plotdata/{:s}__quantiles__{:d}__{:d}__{:.2f}.csv".format(f, week, model_included, noise), index=False)

                    pickle.dump( forecast, open("./plotdata/{:s}__forecast__{:d}__{:d}__{:.2f}.pkl".format(f, week, model_included, noise), "wb"))

                    #--add to large dfs
                    quantiles_for_incident_hosps["week"]           = week
                    quantiles_for_incident_hosps["noise"]          = noise
                    quantiles_for_incident_hosps["model_included"] = model_included
                    quantiles_for_incident_hosps["model"]          = f
                    
                    all_quantile_data = pd.concat([all_quantile_data, quantiles_for_incident_hosps])

                    peaks_and_times["week"]           = week
                    peaks_and_times["noise"]          = noise
                    peaks_and_times["model_included"] = model_included
                    peaks_and_times["model"]          = f
 
                    all_peak_data = pd.concat([all_peak_data, peaks_and_times])
    all_quantile_data.to_csv("./plotdata/all_quantile_data.csv",index=False)
    all_peak_data.to_csv("./plotdata/all_peak_data.csv",index=False)
