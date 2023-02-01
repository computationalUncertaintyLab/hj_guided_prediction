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
   
def mm2inch(x):
    return x/25.4

if __name__ == "__main__":
    rng_key     = random.PRNGKey(0)    #--seed for random number generation
    total_window_of_observation = 210

    if os.path.isdir("./for_crowdsourcing/"):
        pass
    else:
        os.makedirs("./for_crowdsourcing/")

    for week in [6,5,4,3,2,1,0,-1,-2]:
        for model_included in [0,1]:
            if os.path.isdir("./for_crowdsourcing/data_collection__{:d}__{:d}/".format(week, model_included)):
                pass
            else:
                os.makedirs("./for_crowdsourcing/data_collection__{:d}__{:d}/".format(week, model_included))

            #--generate simulated surveillance data
            surv_data = generate_data(rng_key = rng_key)
            inc_hosps, noisy_hosps, time_at_peak, peak_value = surv_data.simulate_surveillance_data()
            cut_inc_hosps, cut_noisy_hosps, cut_point = surv_data.cut_data_based_on_week(week)
 
            #--store these dataframes    
            pd.DataFrame({"training_data":cut_noisy_hosps}).to_csv("./for_crowdsourcing/data_collection__{:d}__{:d}/training_data__{:d}__{:d}.csv".format(week, model_included, week, model_included))
            pd.DataFrame({"truth_data":cut_inc_hosps}).to_csv("./for_crowdsourcing/data_collection__{:d}__{:d}/truth_data__{:d}__{:d}.csv".format(week, model_included, week, model_included))
            pd.DataFrame({"time_at_peak":[time_at_peak]}).to_csv("./for_crowdsourcing/data_collection__{:d}__{:d}/time_at_peak__{:d}__{:d}.csv".format(week, model_included, week, model_included))

            break
        break
            
            #-plot
            plt.style.use(['science','grid'])

            fig,ax = plt.subplots()
            times = np.arange(1,total_window_of_observation+1)

            #--plot without peak
            ax.scatter(times[:cut_point], cut_noisy_hosps, lw=1, color="blue", alpha=1,s=3, label = "Surveillance data")

            if model_included:
                #--compute model trained on surveillance data
                forecast = chimeric_forecast(rng_key = rng_key, surveillance_data = noisy_hosps, humanjudgment_data = None, peak_time_and_values=False )
                forecast.fit_model()

                #--compute quantiles for incident hospitalizations
                quantiles_for_incident_hosps = forecast.compute_quantiles()
                
                #--save object for our records
                pickle.dump(forecast, open("./for_crowdsourcing/data_collection__{:d}__{:d}/forecast__{:d}__{:d}.pkl".format(week, model_included, week, model_included) ,"wb") )
                quantiles_for_incident_hosps.to_csv("./for_crowdsourcing/data_collection__{:d}__{:d}/quantiles__{:d}__{:d}.csv".format(week, model_included, week, model_included))

                #--plot the median, 50 PI and the 95PI
                ax.plot(times, quantiles.loc[:,"0.500"], color= "red", ls='--', label = "Median prediction")
                ax.fill_between(times,quantiles.loc[:,"0.025"] , quantiles.loc[:,"0.975"] , color= "red"  ,ls='--', alpha = 0.25, label = "50 and 95 PI")
                ax.fill_between(times,quantiles.loc[:,"0.250"] , quantiles.loc[:,"0.750"] ,  color= "red" ,ls='--', alpha = 0.25)

            #--set ylim, xlim, and ticks
            ax.set_ylim(0,2500)
            ax.set_xlim(1,210)
            ax.set_xticks([1,25,50,75,100,125,150,175,200,210])

            #--label axes
            ax.set_ylabel("Incident hospitalizations", fontsize=10)
            ax.set_xlabel("Time (days)", fontsize=10)

            #--add legend for user
            ax.legend(fontsize=10,frameon=False)
            fig.set_tight_layout(True)

            #--size figure
            w = mm2inch(183)
            fig.set_size_inches( w, w/1.5 )

            #--save figure
            plt.savefig("./for_crowdsourcing/data_collection__{:d}__{:d}/plot__{:d}__{:d}.pdf".format(week, model_included, week, model_included))
            plt.close()
