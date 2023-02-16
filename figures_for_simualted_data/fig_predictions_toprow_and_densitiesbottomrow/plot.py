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

    #-plot
    plt.style.use('science')
    
    fig,axs = plt.subplots(2,4)
    
    truth = pd.read_csv("./truth.csv")
    inc_hosps = truth.hosps

    surv = pd.read_csv("./noisy_truth.csv")

    recorded_time = len(surv)

    times      = np.arange(0,recorded_time)
    total_time = np.arange(0,210)

    time_at_peak = truth.time_at_peak.max()
    peak_value   = truth.peakvalue.max()
    
    #--prior
    prior       = pd.read_csv("./prior__quantiles.csv")
    surv        = pd.read_csv("./survdata__quantiles.csv")
    hj          = pd.read_csv("./surv_plus_hj__quantiles.csv")
    past_season = pd.read_csv("./surv_plus_past_season__quantiles.csv")

    colors = ["red", "blue", "orange", "purple"]
    stamps = ["A.","B.","C.","D."]
    for n,quants in enumerate([ prior, surv, past_season, hj ]):
        median_prediction = quants["50.000"]
        lower2p5          = quants["2.500"]
        lower25           = quants["25.000"]
        upper75           = quants["75.000"]
        upper97p5         = quants["97.500"]

        #--setup axis
        ax = axs[0,n]

        #--plot true incident hospitalizations
        ax.plot(total_time[:recorded_time], inc_hosps[:recorded_time], color="black", ls="--")
        ax.plot(total_time[recorded_time:], inc_hosps[recorded_time:], color="black", ls = "-", label = "Truth")
        ax.axvline(recorded_time, color = "black", ls = "--")

        if n==1:
            eps=1.5
            ax.text(recorded_time+eps, 800*0.96, "Time under\nsurveillance", va="top",fontsize=10)
        
        #--plot the median, 50PI, and 95PI
        ax.plot(total_time, median_prediction           , color = colors[n], lw=1)
        ax.fill_between(total_time, lower2p5, upper97p5 , color = colors[n], lw=0 ,alpha=0.25, label="50\nand\n95 PI")
        ax.fill_between(total_time, lower25 , upper75   , color = colors[n], lw=0 ,alpha=0.25)

        #--set xlim and xticks in intervals of weeks
        ax.set_xlim(0,210)
        ax.set_xticks([7*x for x in np.arange(0,30,6)] + [210])

        #--set ylim and yticks in intervals of weeks
        ax.set_ylim(0,2000)
        ax.set_yticks([250,500,800])
 
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
    prior          = pd.read_csv("./prior__peaks_and_times.csv")
    surv           = pd.read_csv("./survdata__peaks_and_times.csv")
    hj             = pd.read_csv("./surv_plus_hj__peaks_and_times.csv")
    surv_plus_past = pd.read_csv("./surv_plus_past_season__peaks_and_times.csv")
    

    past_season_data_peaks        = pd.read_csv("./past_season_peak_data.csv")
    hjdata__peaks_and_times = pd.read_csv("./hj_preds.csv")
    
    stamps = ["E.","F.","G.","H."]
    for n,peak_data in enumerate([ prior, surv, surv_plus_past, hj ]):
        ax = axs[1,n]
        sns.kdeplot( x="times", y="peaks", data = peak_data,ax=ax, fill=True, clip = (0,np.inf), color = colors[n] )

        ax.axvline( time_at_peak, color="black" )
        ax.axhline( peak_value, color ="black" )

        if n==2:
            print(past_season_data_peaks)
            ax.scatter( past_season_data_peaks.true_time_at_peak, past_season_data_peaks.true_peak_value, color="black", s=2, marker="s", label="Past two\nseasons"  )
            ax.legend(frameon=False, handletextpad=-0.65, loc="upper right")


        if n==3:
            ax.scatter( hjdata__peaks_and_times.noisy_time_at_peak, hjdata__peaks_and_times.noisy_peak_values, color="black", s=2, marker="s", label="Human\njudgment"  )
            ax.legend(frameon=False, handletextpad=-0.65)

        #--set the xlim and ylim of the plot
        ax.set_xlim(0,215)

        if n<2:
            ax.set_ylim(200,1100)
        else:
            ax.set_ylim(200,1100)

        #--set the xticks and xlabel
        ax.set_xticks([7*x for x in np.arange(0,30,6)] + [210])
        ax.set_xlabel("Day of\npeak hospitalizations", fontsize = 10)

        #--set the ylabel 
        if n>0:
            ax.set_yticklabels([])
        if n==0:
            ax.set_ylabel("Peak hospitalizations", fontsize = 10)

            ax.set_yticks([300,600,900,1100])
            ax.set_yticklabels([300,600,900,1100])
        else:
            ax.set_ylabel("", fontsize = 10)

        #--add stamp
        stamp(ax,stamps[n])
            

    w = mm2inch(183)
    fig.set_size_inches(w,w/1.5)

    fig.set_tight_layout(True)
    
    plt.savefig("fig_simulation_of_three_levels_of_data.pdf")
    plt.close()
