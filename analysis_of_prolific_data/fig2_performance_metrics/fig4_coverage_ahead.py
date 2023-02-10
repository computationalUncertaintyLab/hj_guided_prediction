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

    PEAK = 86
    coverage = pd.read_csv("./plotdata/all_cover_data.csv")

    coverage["time_of_forecast"] = PEAK - 7*coverage["week"]
    coverage["time_at_forecast"] = coverage.times - coverage.time_of_forecast

    coverage["p"] = coverage["percentile"]/100.

    #--remove prior model
    coverage = coverage.loc[coverage.model!="prior"]
    
    coverage["model"] = coverage["model"].replace("prior","No data")
    coverage["model"] = coverage["model"].replace("survdata","Surveillance data")
    coverage["model"] = coverage["model"].replace("survdata_plus_hj","Surveillance plus HJ")
    coverage["model"] = coverage["model"].replace("surv_data_plus_past","Surv. plus peaks")
    
    plt.style.use(['science','grid'])

    fig,axs = plt.subplots(2,2)
    axs = axs.flatten()

    letters = ["A.","B.","C.","D."]
    for n,(week,ax) in enumerate(zip([4,3,2,1],axs)):
        
        subset = coverage.loc[ (coverage.week.isin([4,3,2,1])) & (coverage.time_at_forecast>=week*7) & (coverage.time_at_forecast<=(week+1)*7) ]
        
        sns.lineplot(x="p",y="covered",hue="model",data=subset,ax=ax, palette= ["blue","orange","purple"])
        ax.plot([0,1],[0,1],color="black")

        if week!=4:
            ax.get_legend().remove()
        else:
            ax.legend(frameon=False, fontsize=10, labelspacing=0.025, handletextpad=0.025, loc="center", bbox_to_anchor=(0.35,0.85))

        ax.set_xticks([0,0.25,0.50,0.75,1.0])
        ax.set_yticks([0,0.25,0.50,0.75,1.0])

        ax.set_ylim(0,1.1)

        stamp(ax,letters[n])

        ax.text(0.99,0.99, "{:d} weeks ahead".format(week) if week >1 else "{:d} week ahead".format(week), va="top",ha="right", fontsize=10,transform=ax.transAxes )
        
        ax.set_xlabel("Prediction interval percentile", fontsize=10)
        ax.set_ylabel("Empirical coverage", fontsize=10)
            

    fig.set_tight_layout(True)
        
    w = mm2inch(183)

    fig.set_size_inches(w, w/1.5)

    plt.savefig("./coverage__acoss_all_forecast_time_points.pdf")
    plt.close()


