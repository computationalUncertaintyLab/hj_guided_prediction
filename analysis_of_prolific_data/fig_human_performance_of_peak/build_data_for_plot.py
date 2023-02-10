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

    hj = pd.read_csv("../../data_from_prolific/prolific_data.csv")
    hj = hj[["week","noise","model_included","peak_time","peak_intensity"]]

    #--remove erroneous predictions
    hj = hj.loc[(hj.peak_time<=210) & (hj.peak_time>=0) & (hj.peak_intensity>=0) & (hj.peak_intensity<=2500)]

    PEAK = 86
    PEAK_VALUE = 649.15

    hj["diff_time"] = hj.peak_time - PEAK
    hj["diff_intensity"] = hj.peak_intensity - PEAK_VALUE

    plt.style.use(['science','grid'])

    fig,axs = plt.subplots(2,3)

    ax = axs[0,0]
    sns.boxplot(x="week",y="diff_time", data = hj, ax=ax, order=[4,3,2,1,-2],color="thistle")
    ax.set_ylabel("Human prediction\nminus peak time",fontsize=10)
    ax.set_xlabel("")

    stamp(ax,"A.")
    
    ax = axs[0,1]
    sns.boxplot(x="model_included",y="diff_time", data = hj,ax=ax,color="thistle")
    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.set_xticklabels(["No model\nguidance", "Model\nguidance"], fontsize=10)

    stamp(ax,"B.")
    
    ax = axs[0,2]
    sns.boxplot(x="noise",y="diff_time", data = hj,ax=ax,color="thistle")
    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.set_xticklabels(["High\nnoise", "Medium", "Low"], fontsize=10)

    stamp(ax,"C.")

    ax = axs[1,0]
    sns.boxplot(x="week",y="diff_intensity", data = hj, ax=ax, order=[4,3,2,1,-2],color="thistle")
    ax.set_ylabel("Human prediction\nminus peak intensity",fontsize=10)

    ax.set_xlabel("Weeks before underlying peak", fontsize=10)

    stamp(ax,"D.")
    
    ax = axs[1,1]
    sns.boxplot(x="model_included",y="diff_intensity", data = hj,ax=ax,color="thistle")
    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.set_xticklabels(["No model\nguidance", "Model\nguidance"], fontsize=10)

    stamp(ax,"E.")


    ax = axs[1,2]
    sns.boxplot(x="noise",y="diff_intensity", data = hj,ax=ax,color="thistle")
    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.set_xticklabels(["High\nnoise", "Medium", "Low"], fontsize=10)

    stamp(ax,"F.")

    fig.set_tight_layout(True)
    w = mm2inch(183)

    fig.set_size_inches(w, w/1.5)

    plt.savefig("./human_performance.pdf")
    plt.close()
