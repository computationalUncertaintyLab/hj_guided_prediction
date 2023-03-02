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

    d = pd.read_csv("./merged_data.csv")
   
    plt.style.use('science')
    
    fig,axs = plt.subplots(3,4)

    def cdf(x,var):
        values = x[var]
        values,N = sorted(values), len(values)
        return pd.DataFrame({"x":values,"px":np.arange(0.,N)/N})

    #--times
    cdfs = d.groupby(["model","noise","time"], group_keys=True).apply( lambda x: cdf(x,"q_peak_intensity")).reset_index()

    colors = ["blue", "orange", "purple","red"]
    for row,noise in enumerate(cdfs.noise.unique()):
        for col, time in enumerate(cdfs.time.unique()):

            ax = axs[row,col]
            subset = cdfs.loc[(cdfs.noise==noise) & (cdfs.time==time),:]
            
            for n,(model,sub) in enumerate(subset.groupby(["model"])):
                ax.step(sub.x.values,sub.px.values, alpha=0.80, color = colors[n], label = "{:s}".format(model))
            ax.plot([0,1],[0,1], color="black", lw=1)

            ax.set_aspect('equal', adjustable='box')

            ax.set_xlim(0,1)
            ax.set_ylim(0,1)

            ax.set_xticks(np.arange(0,1+0.25,0.25))
            ax.set_yticks(np.arange(0,1+0.25,0.25))
            
            if col>0:
                ax.set_yticklabels([])
                ax.set_ylabel("",fontsize=10) 
            else:
                ax.set_yticklabels(np.arange(0,1+0.25,0.25), fontsize=8)
                ax.set_ylabel("Empiricial quantile" ,fontsize=10)              
                
            if row < 2:
                ax.set_xticklabels([])
                ax.set_xlabel("",fontsize=10)
            else:
                ax.set_xticklabels(np.arange(0,1+0.25,0.25), fontsize=8)
                ax.set_xlabel("Theoretical Quantile",fontsize=10)
            
            ax.text(x=0.05,y=0.95,s="Noise= {:.2f}\nWkuntil peak={:d}".format(noise,time)
                    ,ha="left",va="top"
                    ,transform=ax.transAxes,fontsize=10)

            if row==2 and col==3:
                ax.legend(frameon=False,fontsize=10, handletextpad=0.01, )
            ax.set_aspect('equal', adjustable='box')

    w = mm2inch(183)
    fig.set_size_inches(w,w/1.5)
    
    plt.savefig("fig_calibration_intensities.pdf")
    plt.close()
