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
    rng_key = random.PRNGKey(20180611)

    #--collect all forecatsing data in one file
    all_quantile_data = pd.DataFrame()
    all_peak_data     = pd.DataFrame()
    all_pit_data      = pd.DataFrame()

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
    
    #--factors
    for week in [4,3,2,1,-2]:
        for noise in [2.50, 5.0, 10.]:
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
                filenames = ["prior", "survdata","surv_data_plus_past", "survdata_plus_hj"]
                options   = [(None, None, None), (cut_noisy_hosps, None, None),  (cut_noisy_hosps, past_season_peak_data.to_numpy(), True), (cut_noisy_hosps, noisy_human_predictions, True)]
    
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

                    #--compute PITS from X weeks before to 4 weeks after

                    #--compute the proportion of times the sample posterior is smaller than the truth
                    nsamples = samples["inc_hosps"].shape[0]
                    long_samples          = samples["inc_hosps"].T.flatten()
                    repeated_times        = np.arange(0,210).repeat(nsamples) 

                    long_samples = pd.DataFrame({"sample":long_samples,"times":repeated_times})

                    #--merge in truth
                    data_ready_for_pit = long_samples.merge(truthdata, on = ["times"])

                    def PIT(x):
                        return pd.Series({"PIT": np.mean(x["sample"]<x["hosps"]) })
                    PITS_for_incident_hosps = data_ready_for_pit.groupby(["times"]).apply(PIT)
                    
                    PITS_for_incident_hosps.to_csv("./plotdata/{:s}__PITS__{:d}__{:d}__{:.2f}.csv".format(f, week, model_included, noise), index=False)

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


                    PITS_for_incident_hosps["week"]           = week
                    PITS_for_incident_hosps["noise"]          = noise
                    PITS_for_incident_hosps["model_included"] = model_included
                    PITS_for_incident_hosps["model"]          = f
                    
                    all_pit_data = pd.concat([all_pit_data, PITS_for_incident_hosps])
                    
    all_quantile_data.to_csv("./plotdata/all_quantile_data.csv",index=False)
    all_peak_data.to_csv("./plotdata/all_peak_data.csv",index=False)
    all_pit_data.to_csv("./plotdata/all_pit_data.csv",index=False)
