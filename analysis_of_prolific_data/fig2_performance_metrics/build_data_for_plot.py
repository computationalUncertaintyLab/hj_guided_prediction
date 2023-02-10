#mcandrew

import sys
sys.path.append("../../chimeric_forecast")

import numpy as np
import pandas as pd
import pickle

from chimeric_forecast__weighted_ll import chimeric_forecast 
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
    all_cover_data      = pd.DataFrame()

    #--factors
    for WEEK in [4,3,2,1,-2]:
        for NOISE in [2.50, 5.0, 10.]:
            for MODEL in [0,1]: #--we assume that times are more similar and intesite (ie smaller variance)

                inc_hosps     = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/truth_data__{:d}__{:d}__{:.2f}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE))
                noisy_hosps   = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/training_data__{:d}__{:d}__{:.2f}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE))
                time_at_peak  = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/time_at_peak__{:d}__{:d}__{:.2f}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE))
                peak_value    = float(inc_hosps.truth_data[ time_at_peak.time_at_peak ] )

                hjdata = pd.read_csv("../../data_from_prolific/prolific_data.csv")
                                
                subset_hj_data = hjdata.loc[ (hjdata.week==WEEK) & (hjdata.noise==NOISE) & (hjdata.model_included==MODEL) ]
                print(WEEK)
                print(subset_hj_data.shape)
                
                noisy_time_at_peaks     = subset_hj_data.loc[:, ["peak_time"] ]
                noisy_peak_values       = subset_hj_data.loc[:, ["peak_intensity"] ] 
                noisy_human_predictions = subset_hj_data.loc[:, ["peak_time", "peak_intensity"] ]

                past0_inc_hosps    = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_truth_data__{:d}__{:d}__{:.2f}_{:02d}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE   ,0))
                past0_noisy_hosps  = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_training_data__{:d}__{:d}__{:.2f}_{:02d}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE,0))
                past0_time_at_peak = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_time_at_peak__{:d}__{:d}__{:.2f}_{:02d}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE ,0))
                past0_peak_value   = float(past0_inc_hosps.truth_data[ past0_time_at_peak.time_at_peak ] )

                past1_inc_hosps     = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_truth_data__{:d}__{:d}__{:.2f}_{:02d}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE   ,1))
                past1_noisy_hosps   = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_training_data__{:d}__{:d}__{:.2f}_{:02d}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE,1))
                past1_time_at_peak  = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_time_at_peak__{:d}__{:d}__{:.2f}_{:02d}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE ,1))
                past1_peak_value    = float(past1_inc_hosps.truth_data[ past1_time_at_peak.time_at_peak ] )

                s0 = pd.DataFrame({"true_time_at_peak": [float(past0_time_at_peak.time_at_peak)], "true_peak_value": [past0_peak_value]})
                s1 = pd.DataFrame({"true_time_at_peak": [float(past1_time_at_peak.time_at_peak)], "true_peak_value": [past1_peak_value]})

                past_season_peak_data = pd.concat([s0,s1])
                past_season_peak_data.to_csv("./past_season_peak_data__{:d}_{:d}_{:.2f}.csv".format(WEEK,MODEL,NOISE),index=False)
                
                #--Three models to run: just prior, surv data only, and surve plus hj
                filenames = ["prior", "survdata","surv_data_plus_past", "survdata_plus_hj"]
                options   = [(None, None, None)
                             , (noisy_hosps.training_data.to_numpy(), None, None)
                             , (noisy_hosps.training_data.to_numpy(), past_season_peak_data.to_numpy(), True)
                             , (noisy_hosps.training_data.to_numpy(), noisy_human_predictions.to_numpy(), True)]
    
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
                    peaks_and_times.to_csv("./plotdata/{:s}__peaks_and_times__{:d}__{:d}__{:.2f}.csv".format(f, WEEK, MODEL, NOISE),index=False)

                    #--record quantiles from forecast to use in top row of plot (plot.py)
                    quantiles_for_incident_hosps = forecast.compute_quantiles()
                    quantiles_for_incident_hosps["t"] = np.arange(0,210)

                    quantiles_for_incident_hosps.to_csv("./plotdata/{:s}__quantiles__{:d}__{:d}__{:.2f}.csv".format(f, WEEK, MODEL, NOISE), index=False)
                    pickle.dump( forecast, open("./plotdata/{:s}__forecast__{:d}__{:d}__{:.2f}.pkl".format(f, WEEK, MODEL, NOISE), "wb"))

                    #--compute PITS from X weeks before to 4 weeks after

                    
                    #--compute the proportion of times the sample posterior is smaller than the truth
                    nsamples = samples["inc_hosps"].shape[0]
                    long_samples          = samples["inc_hosps"].T.flatten()
                    repeated_times        = np.arange(0,210).repeat(nsamples) 

                    long_samples = pd.DataFrame({"sample":long_samples,"times":repeated_times})

                    #--merge in truth
                    inc_hosps.columns = ["times","hosps"]
                    data_ready_for_pit = long_samples.merge(inc_hosps, on = ["times"])

                    def PIT(x):
                        return pd.Series({"PIT": np.mean(x["sample"]<x["hosps"]) })
                    PITS_for_incident_hosps = data_ready_for_pit.groupby(["times"]).apply(PIT).reset_index()

                    def cover(x):
                        ps = np.arange(0,100+5,5)
                        pcts = np.percentile(x["sample"], ps)

                        true = x.hosps.max()
                        
                        index = 0

                        percentile = []
                        covered = []
                        for i in range(10):
                            lower = pcts[index]
                            upper = pcts[-(index+1)]

                            cov = 1 if lower<true and upper > true else 0
                            
                            covered.append( cov )
                            percentile.append( ps[-(index+1)] - ps[index] )
                            
                            index+=1
                        return pd.DataFrame({"percentile":percentile, "covered":covered})
                    covers = data_ready_for_pit.groupby(["times"]).apply(cover).reset_index()
                    
                    PITS_for_incident_hosps.to_csv("./plotdata/{:s}__PITS__{:d}__{:d}__{:.2f}.csv".format(f, WEEK, MODEL, NOISE), index=False)
                    covers.to_csv("./plotdata/{:s}__COVER__{:d}__{:d}__{:.2f}.csv".format(f, WEEK, MODEL, NOISE), index=False)

                    #--add to large dfs
                    quantiles_for_incident_hosps["week"]           = WEEK
                    quantiles_for_incident_hosps["noise"]          = NOISE
                    quantiles_for_incident_hosps["model_included"] = MODEL
                    quantiles_for_incident_hosps["model"]          = f
                    
                    all_quantile_data = pd.concat([all_quantile_data, quantiles_for_incident_hosps])

                    peaks_and_times["week"]           = WEEK
                    peaks_and_times["noise"]          = NOISE
                    peaks_and_times["model_included"] = MODEL
                    peaks_and_times["model"]          = f
 
                    all_peak_data = pd.concat([all_peak_data, peaks_and_times])


                    PITS_for_incident_hosps["week"]           = WEEK
                    PITS_for_incident_hosps["noise"]          = NOISE
                    PITS_for_incident_hosps["model_included"] = MODEL
                    PITS_for_incident_hosps["model"]          = f
                    
                    all_pit_data = pd.concat([all_pit_data, PITS_for_incident_hosps])

                    covers["week"]           = WEEK
                    covers["noise"]          = NOISE
                    covers["model_included"] = MODEL
                    covers["model"]          = f
                    
                    all_cover_data = pd.concat([all_cover_data, covers])
                    
    all_quantile_data.to_csv("./plotdata/all_quantile_data.csv",index=False)
    all_peak_data.to_csv("./plotdata/all_peak_data.csv",index=False)
    all_pit_data.to_csv("./plotdata/all_pit_data.csv",index=False)
    all_cover_data.to_csv("./plotdata/all_cover_data.csv",index=False)
