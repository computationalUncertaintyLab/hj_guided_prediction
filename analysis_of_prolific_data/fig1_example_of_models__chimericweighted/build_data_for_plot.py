#mcandrew

import sys
sys.path.append("../../chimeric_forecast")

import numpy as np
import pandas as pd
import pickle

from chimeric_forecast__meld7 import chimeric_forecast 
from generate_simulated_data import generate_data
from generate_humanjudgment_prediction_data import generate_humanjudgment_prediction_data 

from jax import random

if __name__ == "__main__":

    rng_key = random.PRNGKey(2020)

    for WEEK in [2]:#[4,3,2,1,-2]:
        for NOISE in [2.5]: # [10.,5.,2.5]:
            for MODEL in [0]:#[0,1]:

                #--set randomization key
                from jax import random

                #--generate simulated surveillance data
                inc_hosps     = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/truth_data__{:d}__{:d}__{:.2f}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE))
                inc_hosps     = inc_hosps.truth_data.to_numpy() 
                
                noisy_hosps   = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/training_data__{:d}__{:d}__{:.2f}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE))
                time_at_peak  = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/time_at_peak__{:d}__{:d}__{:.2f}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE))
                peak_value    = float(inc_hosps[ time_at_peak.time_at_peak ] )

                #--generate two past seasons
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

                #--generate simulated human judgment predictins
                hjdata = pd.read_csv("../../data_from_prolific/prolific_data.csv")

                #--FOR EXAMPLE ONLY CONSIDER ONE TYPE OF FORECAST
                subset_hj_data = hjdata.loc[ (hjdata.week==WEEK) & (hjdata.noise==NOISE) & (hjdata.model_included==MODEL) ]
                print(WEEK)
                print(subset_hj_data.shape)
                
                noisy_time_at_peaks     = subset_hj_data.loc[:, ["peak_time"] ].astype(float).to_numpy()
                noisy_peak_values       = subset_hj_data.loc[:, ["peak_intensity"] ].astype(float).to_numpy() 
                noisy_human_predictions = subset_hj_data.loc[:, ["peak_time", "peak_intensity"] ].astype(float).to_numpy()

                #--remove nonsensical answers from the crowd
                noisy_human_predictions = noisy_human_predictions[ (noisy_human_predictions[:,0]<=210) & (noisy_human_predictions[:,0]>=0) & (noisy_human_predictions[:,1]<=2500) & (noisy_human_predictions[:,1]>=0)  ]

                #--save the humans from themselves
                peak_10,peak_90 = np.percentile( noisy_human_predictions[:,1], [10,90] )
                time_10,time_90 = np.percentile( noisy_human_predictions[:,0], [10,90] )

                noisy_human_predictions = noisy_human_predictions[ (time_10 < noisy_human_predictions[:,0]) & (noisy_human_predictions[:,0] < time_90) & (peak_10 < noisy_human_predictions[:,1]) & (noisy_human_predictions[:,1] < peak_90)]
                
                #--three models to run: just prior, surv data only, chimeric
                filenames = ["prior", "survdata", "surv_plus_hj","surv_plus_past_season"]
                options   = [(None, None, None)
                             ,(noisy_hosps.training_data.to_numpy(), None, None)
                             ,(noisy_hosps.training_data.to_numpy(), noisy_human_predictions, past_season_peak_data.to_numpy())
                             ,(noisy_hosps.training_data.to_numpy(), None, past_season_peak_data.to_numpy())]

                PITS = pd.DataFrame({"model":[],"week":[],"noise":[],"model_included":[], "PIT":[]})
                for f,o  in zip(filenames,options) :
                    d, h, p = o #--these are the options that control what information gets to the model

                    #--fit model that uses only prior information
                    print(p)
                    forecast = chimeric_forecast(rng_key = rng_key, surveillance_data = d, humanjudgment_data = h, past_season_data = p )
                    forecast.fit_model()

                    #--record samples of the peak times and intensities to use for bottom row of plot (plot.py)
                    samples = forecast.posterior_samples
                    peaks   = samples["peak_time_and_value"][:,1]
                    times   = samples["peak_time_and_value"][:,0]

                    peaks_and_times = pd.DataFrame({"peaks":peaks,"times":times})
                    peaks_and_times.to_csv("./{:s}__peaks_and_times__{:d}_{:d}_{:.2f}.csv".format(f,WEEK,MODEL,NOISE),index=False)

                    # PIT = np.mean(samples["inc_hosps"][:,86] < peak_value)
                    # PIT = pd.DataFrame({"model":[model],"week":[week],"noise":[noise],"model_included":[model_included], "PIT":[PIT]})
                    # PITS = pd.concat([PITS,PIT])  
                    
                    #--record quantiles from forecast to use in top row of plot (plot.py)
                    quantiles_for_incident_hosps = forecast.compute_quantiles()
                    quantiles_for_incident_hosps["t"] = np.arange(0,210)

                    quantiles_for_incident_hosps.to_csv("./{:s}__quantiles__{:d}_{:d}_{:.2f}.csv".format(f,WEEK,MODEL,NOISE), index=False)

                    pickle.dump( forecast, open("{:s}__forecast__{:d}_{:d}_{:.2f}.pkl".format(f,WEEK,MODEL,NOISE), "wb"))
                PITS.to_csv()
