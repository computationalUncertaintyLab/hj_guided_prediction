#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots

from glob import glob

def mm2inch(x):
    return x/25.4

def stamp(ax,s):
    ax.text(0.0125,0.975,s=s,fontweight="bold",fontsize=10,ha="left",va="top",transform=ax.transAxes)

if __name__ == "__main__":

    # cut_data = pd.read_csv("cut_time_series_quantile_peaks.csv")
    # cut_data["model"] = "Surveillance only"

    D = pd.DataFrame()
    for fil in glob("./cut_time_series_plushj_quantile_peaks__*.csv"):
        d = pd.read_csv(fil)
        D = pd.concat([D, d])

        
    d = D
    d["sim"] = np.arange(d.shape[0])
    
    d_intensity = d.melt(id_vars=["sim"], value_vars = ["q_peak_intensity_surv","q_peak_intensity_hj"])
    d_intensity = d_intensity.rename(columns = {"value":"q_peak_intensity"})
    d_intensity["model"] = [ "Chimeric" if "hj" in x else "Surveillance" for x in d_intensity.variable ]
    
    d_times     = d.melt(id_vars=["sim"], value_vars = ["q_peak_time_surv","q_peak_time_hj"])
    d_times     = d_times.rename(columns = {"value":"q_peak_time"})
    d_times["model"] = [ "Chimeric" if "hj" in x else "Surveillance" for x in d_times.variable ]

    d = d_times.merge(d_intensity, on = ["sim","model"])
    
    plt.style.use('science')
    
    fig,axs = plt.subplots(1,2)

    def cdf(x,var):
        values = x[var]
        values,N = sorted(values), len(values)
        return pd.DataFrame({"x":values,"px":np.arange(0.,N)/N})
    
    ax = axs[0]
    cdfs = d.groupby(["model"], group_keys=True).apply( lambda x: cdf(x,"q_peak_time")).reset_index()

    colors = ["blue", "orange", "purple","red"]
    for n,(model,subset) in enumerate(cdfs.groupby(["model"])):
        ax.step(subset.x.values,subset.px.values, alpha=0.80, color = colors[n], label = "{:s}".format(model))
    ax.plot([0,1],[0,1], color="black", lw=1)

    ax.set_xlabel("Theoretical Quantile",fontsize=10)
    ax.set_ylabel("Empiricial quantile" ,fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    ax.text(0.95,0.05,"Peak time",fontsize=10,ha="right",va="bottom",transform=ax.transAxes)
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    ax.set_xticks(np.arange(0,1+0.25,0.25))
    ax.set_yticks(np.arange(0,1+0.25,0.25))
 
    stamp(ax,"A.")

    ax = axs[1]
    cdfs = d.groupby(["model"], group_keys=True).apply(lambda x: cdf(x,"q_peak_intensity")).reset_index()

    for n,(model,subset) in enumerate(cdfs.groupby(["model"])):
        ax.step(subset.x.values,subset.px.values, alpha=0.80, color = colors[n], label = "{:s}".format(model))
    ax.plot([0,1],[0,1], color="black", lw=1)
    ax.legend()
    
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("Theoretical Quantile",fontsize=10)
    ax.set_ylabel("Empiricial quantile" ,fontsize=10)
 
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    ax.text(0.95,0.05,"Peak intensity",fontsize=10,ha="right",va="bottom",transform=ax.transAxes)

    ax.set_xticks(np.arange(0,1+0.25,0.25))
    ax.set_yticks(np.arange(0,1+0.25,0.25))
    
    stamp(ax,"B.")

    w = mm2inch(183)
    fig.set_size_inches(w,w/1.5)

    fig.set_tight_layout(True)
    
    plt.savefig("fig_model_specification.pdf")
    plt.close()
