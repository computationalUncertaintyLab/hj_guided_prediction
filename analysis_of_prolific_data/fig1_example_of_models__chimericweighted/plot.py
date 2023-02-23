#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

def mm2inch(x):
    return x/25.4

def stamp(ax,s):
    ax.text(0.0125,0.975,s=s,fontweight="bold",fontsize=10,ha="left",va="top",transform=ax.transAxes)

if __name__ == "__main__":
    
    for WEEK in [2]:#[4,3,2,1,-2]:
        for NOISE in [2.5]: # [10.,5.,2.5]:
            for MODEL in [0]:#[0,1]:

                #-plot
                plt.style.use('science')
                
                fig,axs = plt.subplots(2,3)

                inc_hosps     = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/truth_data__{:d}__{:d}__{:.2f}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE))
                inc_hosps     = inc_hosps.truth_data.to_numpy() 

                surv = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/training_data__{:d}__{:d}__{:.2f}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE))
                surv = surv.training_data.to_numpy()

                recorded_time = len(surv)

                times      = np.arange(0,recorded_time)
                total_time = np.arange(0,210)

                time_at_peak  = pd.read_csv("../../for_crowdsourcing/data_collection__{:d}__{:d}__{:.2f}/time_at_peak__{:d}__{:d}__{:.2f}.csv".format(WEEK,MODEL,NOISE,WEEK,MODEL,NOISE))
                time_at_peak  = int(time_at_peak.time_at_peak)

                peak_value    = float(inc_hosps[ time_at_peak] )

                #--prior
                prior       = pd.read_csv("./prior__quantiles__{:d}_{:d}_{:.2f}.csv".format(WEEK,MODEL,NOISE))
                surv        = pd.read_csv("./survdata__quantiles__{:d}_{:d}_{:.2f}.csv".format(WEEK,MODEL,NOISE))
                hj          = pd.read_csv("./surv_plus_hj__quantiles__{:d}_{:d}_{:.2f}.csv".format(WEEK,MODEL,NOISE))

                past_season = pd.read_csv("./surv_plus_past_season__quantiles__{:d}_{:d}_{:.2f}.csv".format(WEEK,MODEL,NOISE))
                    
                colors = ["blue", "orange", "purple"]
                stamps = ["A.","B.","C."]
                for n,quants in enumerate([surv, past_season, hj ]):
                    median_prediction = quants["50.000"]
                    lower2p5          = quants["2.500"]
                    lower10           = quants["10.000"]
                    lower25           = quants["25.000"]
                    
                    lower37           = quants["37.500"] 
                    upper62           = quants["62.500"] 
                    upper75           = quants["75.000"] 
                    upper90           = quants["90.000"]
                    upper97p5         = quants["97.500"]

                    #--setup axis
                    ax = axs[0,n]

                    #--plot true incident hospitalizations
                    ax.plot(total_time[:recorded_time], inc_hosps[:recorded_time], color="black", ls="--")
                    ax.plot(total_time[recorded_time:], inc_hosps[recorded_time:], color="black", ls = "-", label = "Truth")
                    ax.axvline(recorded_time, color = "black", ls = "--")

                    if n==1:
                        eps=1.5
                        ax.text(recorded_time+eps, 2000*0.96, "Time under\nsurveillance", va="top",fontsize=10)

                    #--plot the median, 50PI, and 95PI
                    ax.plot(total_time, median_prediction           , color = colors[n], lw=1)
                    ax.fill_between(total_time, lower25, upper75 , color = colors[n], lw=0 ,alpha=0.25, label="50 and 80 PI")
                    ax.fill_between(total_time, lower37 , upper62   , color = colors[n], lw=0 ,alpha=0.25)
                    ax.fill_between(total_time, lower10 , upper90   , color = colors[n], lw=0 ,alpha=0.25)

                    #--set xlim and xticks in intervals of weeks
                    ax.set_xlim(0,210)
                    ax.set_xticks([7*x for x in np.arange(0,30,6)] + [210])

                    #--set ylim and yticks in intervals of weeks
                    ax.set_ylim(0,4000)
                    #ax.set_yticks([250,500,800])

                    if n>=1:
                        ax.set_yticklabels([])

                    #--add in stamp
                    stamp(ax,stamps[n])

                    #--setup x and y labels
                    ax.set_xlabel("Time (days)", fontsize=10) 
                    if n==0:
                        ax.set_ylabel("Incident hospitalizations", fontsize=10)
                        ax.legend(fontsize=10,frameon=False, handletextpad= 0.0)

                #--peaks
                prior          = pd.read_csv("./prior__peaks_and_times__{:d}_{:d}_{:.2f}.csv".format(WEEK,MODEL,NOISE))
                surv           = pd.read_csv("./survdata__peaks_and_times__{:d}_{:d}_{:.2f}.csv".format(WEEK,MODEL,NOISE))
                hj             = pd.read_csv("./surv_plus_hj__peaks_and_times__{:d}_{:d}_{:.2f}.csv".format(WEEK,MODEL,NOISE))
                surv_plus_past = pd.read_csv("./surv_plus_past_season__peaks_and_times__{:d}_{:d}_{:.2f}.csv".format(WEEK,MODEL,NOISE))

                past_season_data_peaks  = pd.read_csv("./past_season_peak_data__{:d}_{:d}_{:.2f}.csv".format(WEEK,MODEL,NOISE))


                #--generate simulated human judgment predictins
                hjdata = pd.read_csv("../../data_from_prolific/prolific_data.csv")

                #--FOR EXAMPLE ONLY CONSIDER ONE TYPE OF FORECAST
                subset_hj_data = hjdata.loc[ (hjdata.week==WEEK) & (hjdata.noise==NOISE) & (hjdata.model_included==MODEL) ]

                noisy_time_at_peaks     = subset_hj_data.loc[:, ["peak_time"] ]
                noisy_peak_values       = subset_hj_data.loc[:, ["peak_intensity"] ] 
                noisy_human_predictions = subset_hj_data.loc[:, ["peak_time", "peak_intensity"] ].astype(float)

                hjdata__peaks_and_times = noisy_human_predictions.to_numpy()

                stamps = ["E.","F.","G."]
                for n,peak_data in enumerate([surv, surv_plus_past, hj ]):
                    ax = axs[1,n]
                    sns.kdeplot( x="times", y="peaks", data = peak_data,ax=ax, fill=True, clip = (0,np.inf), color = colors[n] )

                    ax.axvline( time_at_peak, color="black" )
                    ax.axhline( peak_value, color ="black" )

                    if n==1:
                        ax.scatter( past_season_data_peaks.true_time_at_peak, past_season_data_peaks.true_peak_value, color="black", s=2, marker="s", label="Past two\nseasons"  )
                        ax.legend(frameon=False, handletextpad=-0.65, loc="upper right")

                    if n==2:
                        ax.scatter( hjdata__peaks_and_times[:,0], hjdata__peaks_and_times[:,1], color="black", s=2, marker="s", label="Human\njudgment"  )
                        ax.legend(frameon=False, handletextpad=-0.65)

                    #--set the xlim and ylim of the plot
                    ax.set_xlim(0,215)

                    if n<2:
                        ax.set_ylim(200,2000)
                    else:
                        ax.set_ylim(200,2000)

                    #--set the xticks and xlabel
                    ax.set_xticks([7*x for x in np.arange(0,30,6)] + [210])
                    ax.set_xlabel("Day of\npeak hospitalizations", fontsize = 10)

                    #--set the ylabel
                    ax.set_yticks(np.arange(500,2000+500,500))
                    if n>0:
                        ax.set_yticklabels([])
                    if n==0:
                        ax.set_ylabel("Peak hospitalizations", fontsize = 10)
                    else:
                        ax.set_ylabel("", fontsize = 10)

                    #--add stamp
                    stamp(ax,stamps[n])


                w = mm2inch(183)
                fig.set_size_inches(w,w/1.5)

                fig.set_tight_layout(True)

                plt.savefig("fig_crowd_of_three_levels_of_data__{:d}_{:d}_{:.2f}.pdf".format(WEEK, MODEL, NOISE))
                plt.close()
