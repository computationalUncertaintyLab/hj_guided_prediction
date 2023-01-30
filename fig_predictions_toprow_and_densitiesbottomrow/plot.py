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
    
    fig,axs = plt.subplots(2,3)
    times = np.arange(0,210)

    truth = pd.read_csv("./truth.csv")
    inc_hosps = truth.hosps

    time_at_peak = truth.time_at_peak.max()
    peak_value   = truth.peakvalue.max()
    
    #--prior
    prior  = pd.read_csv("./prior__only__quantiles.csv")
    surv   = pd.read_csv("./survdata__quantiles.csv")
    hj     = pd.read_csv("./hjdata__quantiles.csv")

    colors = ["red", "blue", "purple"]
    stamps = ["A.","B.","C."]
    for n,quants in enumerate([ prior, surv, hj ]): 
        median_prediction = quants["median"]
        lower2p5          = quants["lower2p5"]
        lower25           = quants["lower25"]
        upper75           = quants["upper75"]
        upper97p5         = quants["upper97p5"]
        
        ax = axs[0,n]

        ax.plot(times[:-1], inc_hosps, color="black", label = "Truth")

        ax.plot(median_prediction, color = colors[n], lw=1)
        ax.fill_between(times, lower2p5, upper97p5, color = colors[n], lw=0,alpha=0.25, label="75 and 95 PI")
        ax.fill_between(times, lower25, upper75, color = colors[n], lw=0,alpha=0.25)
        
       
        ax.set_xlim(0,210)

        ax.set_xticks([7*x for x in np.arange(0,30,6)] + [210])

        if n==1:
            ax.set_yticklabels([])

        if n<2:
            ax.set_ylim(0,2500)
        else:
            ax.set_ylim(0,700)
            ax.set_yticks([250,500,700])
            
        stamp(ax,stamps[n])

        ax.set_xlabel("Time (days)", fontsize=10)
        
        if n==0:
            ax.set_ylabel("Incident hospitalizations", fontsize=10)
            ax.legend(fontsize=10,frameon=False, handletextpad= 0.0)


    #--peaks
    prior  = pd.read_csv("./prior__only__peaks_and_times.csv")
    surv   = pd.read_csv("./survdata__peaks_and_times.csv")
    hj     = pd.read_csv("./hjdata__peaks_and_times.csv")

    hjdata__peaks_and_times = pd.read_csv("./hj_peaks_and_intensities.csv")
    
    stamps = ["D.","E.","F."]
    for n,peak_data in enumerate([ prior, surv, hj ]):
        ax = axs[1,n]
        sns.kdeplot( x="times", y="peaks", data = peak_data,ax=ax, fill=True, clip = (0,np.inf), color = colors[n] )

        ax.axvline( time_at_peak, color="black" )
        ax.axhline( peak_value, color ="black" )

        if n==2:
            ax.scatter( hjdata__peaks_and_times.time, hjdata__peaks_and_times.peak, color="black", s=2, marker="s", label="Human\njudgment"  )
            ax.legend(frameon=False, handletextpad=0.0)
        ax.set_xlim(0,210)

        if n<2:
            ax.set_ylim(0,2500)
        else:
            ax.set_ylim(400,700)

        ax.set_xticks([7*x for x in np.arange(0,30,6)] + [210])
        
        ax.set_xlabel("Day of peak hospitalizations", fontsize = 10)


        if n==1:
            ax.set_yticklabels([])
        
        if n==0 or n==2:
            ax.set_ylabel("Peak hospitalizations", fontsize = 10)
        else:
            ax.set_ylabel("", fontsize = 10)

        stamp(ax,stamps[n])
            

    w = mm2inch(183)
    fig.set_size_inches(w,w/1.5)

    fig.set_tight_layout(True)
    
    plt.savefig("fig_simulation_of_three_levels_of_data.pdf")
    plt.close()
