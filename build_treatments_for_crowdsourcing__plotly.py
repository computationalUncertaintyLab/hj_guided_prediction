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

#import plotly.plotly as py

import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go

if __name__ == "__main__":
    rng_key     = random.PRNGKey(0)    #--seed for random number generation
    total_window_of_observation = 210
    times = np.arange(1,210+1)

    # username='your_username'
    # api_key='your_api_key'
    # chart_studio.tools.set_credentials_file(username='mcandrew'
    #                                         ,api_key='FxbopNZZbWhkAlViZd73')
    
    
    if os.path.isdir("./for_crowdsourcing/"):
        pass
    else:
        os.makedirs("./for_crowdsourcing/")

    for week in [6,5,4,3,2,1,0,-1,-2]:
        for noise in [0.25, 1.0, 2.5, 5., 10.]:
            for model_included in [0,1]:

                if os.path.isdir("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/".format(week, model_included, noise)):
                    pass
                else:
                    os.makedirs("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/".format(week, model_included, noise))

                #--generate current season simulated surveillance data
                surv_data = generate_data(rng_key = rng_key, r0=1.75, noise = noise)
                inc_hosps, noisy_hosps, time_at_peak, peak_value = surv_data.simulate_surveillance_data()
                cut_inc_hosps, cut_noisy_hosps, cut_point = surv_data.cut_data_based_on_week(week)

                print(week)
                print(time_at_peak)
                print(cut_point)
                
                #--store these dataframes    
                pd.DataFrame({"training_data":cut_noisy_hosps}).to_csv("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/training_data__{:d}__{:d}__{:.2f}.csv".format(week, model_included, noise, week, model_included, noise))
                pd.DataFrame({"truth_data":cut_inc_hosps}).to_csv("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/truth_data__{:d}__{:d}__{:.2f}.csv".format(week, model_included, noise, week, model_included, noise))
                pd.DataFrame({"time_at_peak":[time_at_peak]}).to_csv("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/time_at_peak__{:d}__{:d}__{:.2f}.csv".format(week, model_included, noise, week, model_included, noise))

                if model_included:
                    #--fit model
                    forecast = chimeric_forecast(rng_key = rng_key, surveillance_data = cut_noisy_hosps, humanjudgment_data = None, peak_time_and_values=False)
                    forecast.fit_model()

                    #--compute quantiles for incident hospitalizations
                    quantiles_for_incident_hosps = forecast.compute_quantiles()


                    lower25 = go.Scatter(  x=times
                                         , y=quantiles_for_incident_hosps["25.000"]
                                         , fill=None
                                         , mode       ='lines'
                                         , line_color ='rgba(255,0,0,0.075)'
                                         , hovertext  = '50 Prediction interval'
                                         , opacity    = 0.001
                                         , showlegend = False
                               )
                    upper75 = go.Scatter(x=times
                                         , y=quantiles_for_incident_hosps["75.000"]
                                         , fill       = 'tonexty' # fill area between trace0 and trace1
                                         , mode       = 'lines'
                                         , line_color = 'rgba(255,0,0,0.075)'
                                         , hovertext  = '50 Prediction interval'
                                         , opacity    = 0.01
                                         , fillcolor='rgba(255,0,0,0.075)'
                                         , showlegend = True
                                         , name       = "50th prediction interval")

                    lower2p5 = go.Scatter(x=times
                                          , y=quantiles_for_incident_hosps["2.500"]
                                          , fill=None
                                          , mode='lines'
                                          , line_color='rgba(255,0,0,0.075)'
                                          , opacity=0.001
                                          , hovertext = "95 Prediction interval", name="95PI"
                                          , showlegend = False
                               )
                    upper97p5 = go.Scatter(x=times
                                           , y=quantiles_for_incident_hosps["97.500"]
                                           , fill='tonexty' # fill area between trace0 and trace1
                                           , mode='lines'
                                           , line_color='rgba(255,0,0,0.075)'
                                           , opacity=0.01
                                           , fillcolor='rgba(255,0,0,0.075)'
                                           , hovertext = "95th prediction interval"
                                           , name="95th prediction interval"
                                           , showlegend=True)

                    median   = go.Scatter(x=times
                                          , y=quantiles_for_incident_hosps["50.000"]
                                          , mode='lines'
                                          , line_color='grey'
                                          , line = dict(dash="solid")
                                          , opacity=0.50
                                          , hovertext = "Median"
                                          , name="Median")

                noisy_hosps_scatter = go.Scatter(x = times[:cut_point]
                                                 ,y = cut_noisy_hosps
                                                 ,name = 'Surveillance data'
                                                 ,marker=dict(color='blue')
                                                 ,showlegend=True
                                                 ,mode = "markers"
                                                 ,hovertext = ["Day {:d}, {:d} Hosps".format(t,y) for (t,y) in zip(times[:cut_point], cut_noisy_hosps)]
                                                 )


                #--add in simulated data from two past seasons to guide the crowdsourcer

                #--generate current season simulated surveillance data
                #--We will only consider the past season data geenrated from the model_included = 0 treatment so that we can compare model included vs not

                model_included__no = 0
                past_season_0 = pd.read_csv("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_training_data__{:d}__{:d}__{:.2f}_{:02d}.csv".format(week, model_included__no, noise, week, model_included__no, noise, 0) )
                past_season_1 = pd.read_csv("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/past_season_training_data__{:d}__{:d}__{:.2f}_{:02d}.csv".format(week, model_included__no, noise, week, model_included__no, noise, 1) )

                print(past_season_0)
                
                noisy_hosps_scatter__season0 = go.Scatter(x = times
                                                 ,y = past_season_0.training_data
                                                 ,mode='lines'
                                                 ,line_color='grey'
                                                 ,name = 'Data from two seasons ago'
                                                 ,showlegend=True
                                                 ,hovertext = ["Day {:d}, {:d} Hosps".format(t,y) for (t,y) in zip(times[:cut_point], cut_noisy_hosps)]
                                                 )
                noisy_hosps_scatter__season1 = go.Scatter(x = times
                                                 ,y = past_season_1.training_data
                                                 ,mode='lines'
                                                 ,line_color='grey'
                                                 ,name = 'Data from last season'
                                                 ,showlegend=True
                                                 ,hovertext = ["Day {:d}, {:d} Hosps".format(t,y) for (t,y) in zip(times[:cut_point], cut_noisy_hosps)]
                                                 )

                
                layout =  dict(
                    xaxis = dict(title = 'Time (Days)'),
                    yaxis = dict(title = 'Incident hospitalizations'),
                    margin = dict(
                        l=70,
                        r=10,
                        b=50,
                        t=10
                    )
                )
                if model_included:
                    data = [lower25, upper75, lower2p5, upper97p5, median, noisy_hosps_scatter, noisy_hosps_scatter__season0, noisy_hosps_scatter__season1 ]
                else:
                    data = [noisy_hosps_scatter, noisy_hosps_scatter__season0, noisy_hosps_scatter__season1]
                fig =  go.Figure(data = data, layout=layout)


                fig.update_xaxes(ticks="inside", range = [0,210]
                                 , tickvals = [ 7*x for x in np.arange(0,30+2,2)] + [210]
                                 , minor_ticks="inside"
                                 , showline=True
                                 , linecolor="black"
                                 ,showspikes = True
                                 ,spikemode  = 'across'
                                 ,spikesnap = 'cursor'
                                 ,spikedash = 'solid'
                                 ,spikecolor="black"
                                 ,spikethickness=1
                                 ,showgrid=True)
                fig.update_yaxes(ticks="inside"
                                 , minor_ticks="inside"
                                 , range = [-5,2500]
                                 , showline=True
                                 , linecolor="black"
                                 ,showspikes = True
                                 ,spikemode  = 'across'
                                 ,spikesnap = 'cursor'
                                 ,spikedash = 'solid'
                                 ,spikecolor="black"
                                 ,spikethickness=1
                                 ,showgrid=True)

                fig.update_layout(spikedistance=1000, hoverdistance=100)
                fig.update_layout(template='simple_white')

                fig.write_html("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/plot__{:d}__{:d}__{:.2f}.html".format(week, model_included, noise, week, model_included, noise))
                fig.write_image("./for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/plot__{:d}__{:d}__{:.2f}.png".format(week, model_included, noise, week, model_included, noise))

                #--write out url
                # url = py.plot(fig, filename = 'plot__{:d}__{:d}__{:.2f}'.format(week, model_included, noise), auto_open=False)

                # fout = open("./url__{:d}__{:d}__{:.2f}.png".format(week, model_included, noise),"w")
                # fout.write(url)
                # fout.close()
