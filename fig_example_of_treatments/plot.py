#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed

import scienceplots

import os
import pickle

def mm2inch(x):
    return x/25.4

if __name__ == "__main__":
    N     = 12*10**6                   #--Population of the state of PA
    ps    = 0.12                       #--Percent of the population that is considered to be susceptible
    total_window_of_observation = 210  #--total number of time units that we will observe

    #-plot
    plt.style.use(['science','grid'])

    noise_level = 10
    week = 1
    model_included = 1
    
    
    fig,axs = plt.subplots(2,3)
    times = np.arange(1,total_window_of_observation+1)

    for n, noise_level in enumerate([10, 1, 0.25]):
        ax = axs[0,n]
        
        #--plot without peak
        training_data = pd.read_csv("../for_testing/data_collection__{:d}__{:.2f}__{:d}/training_data__{:d}__{:.2f}__{:d}.csv".format(week, noise_level, model_included, week, noise_level, model_included) )
        ax.scatter(times[:len(training_data)], training_data, lw=1, color="blue", alpha=1,s=3, label = "Surveillance data")

        quantiles = pd.read_csv("../for_testing/data_collection__{:d}__{:.2f}__{:d}/quantiles__{:d}__{:.2f}__{:d}.csv".format(week, noise_level, model_included, week, noise_level, model_included))

        predicted_inc_hosps = quantiles["mean"]
        lower_2p5           = quantiles["lower_2p5"]
        lower25             = quantiles["lower25__w0peak"]
        upper75             = quantiles["upper75__w0peak"]
        upper97p5           = quantiles["upper97p5__w0peak"]

        
        ax.plot(times, predicted_inc_hosps, color= "red", ls='--', label = "Median prediction")
        ax.fill_between(times, lower_2p5,upper97p5,  color= "red" ,ls='--', alpha = 0.25, label = "75 and 95 PI")
        ax.fill_between(times, lower25  ,upper75,  color= "red" ,ls='--', alpha = 0.25)


        ax.set_ylim(0,2500)
        ax.set_xlim(1,210)
        ax.set_xticks([1,50,100,150,210])

        ax.set_ylabel("Incident hospitalizations", fontsize=10)
        ax.set_xlabel("Time (days)", fontsize=10)

        if n==0:
            ax.text(0.99,0.99,"1 wk before peak\nLow noise", fontsize=10, ha="right",va="top",transform=ax.transAxes)
        if n==1:
            ax.text(0.99,0.99,"1 wk before peak\nMedium noise", fontsize=10, ha="right",va="top",transform=ax.transAxes)
        if n==2:
            ax.text(0.99,0.99,"1 wk before peak\nHigh noise", fontsize=10, ha="right",va="top",transform=ax.transAxes)

    week = -2
    for n, noise_level in enumerate([10, 1, 0.25]):
        ax = axs[1,n]
        
        #--plot without peak
        training_data = pd.read_csv("../for_testing/data_collection__{:d}__{:.2f}__{:d}/training_data__{:d}__{:.2f}__{:d}.csv".format(week, noise_level, model_included, week, noise_level, model_included) )
        ax.scatter(times[:len(training_data)], training_data, lw=1, color="blue", alpha=1,s=3, label = "Surveillance data")

        quantiles = pd.read_csv("../for_testing/data_collection__{:d}__{:.2f}__{:d}/quantiles__{:d}__{:.2f}__{:d}.csv".format(week, noise_level, model_included, week, noise_level, model_included))

        predicted_inc_hosps = quantiles["mean"]
        lower_2p5           = quantiles["lower_2p5"]
        lower25             = quantiles["lower25__w0peak"]
        upper75             = quantiles["upper75__w0peak"]
        upper97p5           = quantiles["upper97p5__w0peak"]

        
        ax.plot(times, predicted_inc_hosps, color= "red", ls='--', label = "Median prediction")
        ax.fill_between(times, lower_2p5,upper97p5,  color= "red" ,ls='--', alpha = 0.25, label = "75 and 95 PI")
        ax.fill_between(times, lower25  ,upper75,  color= "red" ,ls='--', alpha = 0.25)

        ax.set_ylim(0,2500)
        ax.set_xlim(1,210)
        ax.set_xticks([1,50,100,150,210])

        ax.set_ylabel("Incident hospitalizations", fontsize=10)
        ax.set_xlabel("Time (days)", fontsize=10)

        if n==0:
            ax.text(0.99,0.99,"2 wks after peak\nLow noise", fontsize=10, ha="right",va="top",transform=ax.transAxes)
        if n==1:
            ax.text(0.99,0.99,"2 wks after peak\nMedium noise", fontsize=10, ha="right",va="top",transform=ax.transAxes)
        if n==2:
            ax.text(0.99,0.99,"2 wks after peak\nHigh noise", fontsize=10, ha="right",va="top",transform=ax.transAxes)


    
    fig.set_tight_layout(True)

    w = mm2inch(183)
    fig.set_size_inches( w, w/1.5 )

    plt.savefig("example_of_treatments.pdf")
    plt.close()

