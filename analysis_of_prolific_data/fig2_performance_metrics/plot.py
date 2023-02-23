#mcandrew

import sys
sys.path.append("../../chimeric_forecast")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

import scienceplots
from chimeric_forecast__meld7 import chimeric_forecast

def mm2inch(x):
    return x/25.4

def stamp(ax,s):
    ax.text(0.0125,0.975,s=s,fontweight="bold",fontsize=10,ha="left",va="top",transform=ax.transAxes)

from glob import glob
if __name__ == "__main__":

    all_quantile_data = pd.DataFrame()
    for fil in glob("../fig1_example_of_models__chimericweighted/*quantiles*.csv"):
        d = pd.read_csv(fil)

        model = fil.split("/")[-1].split("__")[0]
        d["model"] = model
        
        week,model,noise = fil.split("/")[-1].split("__")[-1].split("_")
        noise = float(noise.replace(".csv",""))
        d["week"] = int(week)
        d["model_included"] = model
        d["noise"] = noise

        all_quantile_data = pd.concat([all_quantile_data, d])

    all_peak_data = pd.DataFrame()
    for fil in glob("../fig1_example_of_models__chimericweighted/*peaks*.csv"):
        d = pd.read_csv(fil)

        model = fil.split("/")[-1].split("__")[0]
        d["model"] = model
        
        week,model,noise = fil.split("/")[-1].split("__")[-1].split("_")
        noise = float(noise.replace(".csv",""))
        d["week"] = int(week)
        d["model_included"] = model
        d["noise"] = noise

        all_peak_data = pd.concat([all_peak_data, d])

    all_PITS = pd.DataFrame()
    peak_value = 649.1538305605717
    for fil in glob("../fig1_example_of_models__chimericweighted/*forecast*pkl"):
        d = {"week":[],"model_included":[],"model":[],"noise":[],"PIT":[]}

        print(fil)
        forecast = pickle.load(open(fil,"rb"))

        samples = forecast.posterior_samples
        PIT = np.mean(samples["inc_hosps"][:,86] < peak_value)
        d["PIT"] = [float(PIT)]
        
        model = fil.split("/")[-1].split("__")[0]
        d["model"] = [model]
        
        week,model,noise = fil.split("/")[-1].split("__")[-1].split("_")
        noise = float(noise.replace(".pkl",""))
        d["week"] = [int(week)]
        d["model_included"] = [model]
        d["noise"] = [noise]

        d = pd.DataFrame(d)
        
        all_PITS = pd.concat([all_PITS, d])
        
    true_data = pd.read_csv("../../for_crowdsourcing/data_collection__0__0__10.00/truth_data__0__0__10.00.csv")
    true_peak = int(pd.read_csv("../../for_crowdsourcing/data_collection__0__0__10.00/time_at_peak__0__0__10.00.csv").time_at_peak)
    true_peak_value = true_data.truth_data.values[true_peak]

    all_peak_data.loc[all_peak_data.model != "prior"]

    
    #--peak MAE, peak PIT timing
    #--peak MAE, peak PIT intensiyt

    all_peak_data["true_peak_time"]  = true_peak
    all_peak_data["true_peak_value"] = true_peak_value

    all_peak_data["diff_time"]  = all_peak_data["times"] - all_peak_data["true_peak_time"]
    all_peak_data["diff_value"] = all_peak_data["peaks"] - all_peak_data["true_peak_value"]

    all_peak_data["abs_diff_time"]  = abs(all_peak_data["times"] - all_peak_data["true_peak_time"])
    all_peak_data["abs_diff_value"] = abs(all_peak_data["peaks"] - all_peak_data["true_peak_value"])

    #--remove prior
    all_peak_data = all_peak_data.loc[all_peak_data.model!="prior"]
    
    all_peak_data["model"] = all_peak_data["model"].replace("prior","No data")
    all_peak_data["model"] = all_peak_data["model"].replace("survdata","Surveillance data")
    all_peak_data["model"] = all_peak_data["model"].replace("surv_plus_hj","Surveillance plus HJ")
    all_peak_data["model"] = all_peak_data["model"].replace("surv_plus_past_season","Surv. plus peaks")


    all_PITS["model"] = all_PITS["model"].replace("prior","No data")
    all_PITS["model"] = all_PITS["model"].replace("survdata","Surveillance data")
    all_PITS["model"] = all_PITS["model"].replace("surv_plus_hj","Surveillance plus HJ")
    all_PITS["model"] = all_PITS["model"].replace("surv_plus_past_season","Surv. plus peaks")
    
    all_quantile_data["model"] = all_quantile_data["model"].replace("prior","No data")
    all_quantile_data["model"] = all_quantile_data["model"].replace("survdata","Surveillance data")
    all_quantile_data["model"] = all_quantile_data["model"].replace("surv_plus_hj","Surveillance plus HJ")
    all_quantile_data["model"] = all_quantile_data["model"].replace("surv_plus_past_season","Surv. plus peaks")




    flierprops = dict(marker='o', markerfacecolor='black', markersize=1,  markeredgecolor='black')
    plt.style.use(['science','grid'])    
    fig,axs = plt.subplots(2,3)

    ax = axs[0,0]
    g = sns.boxplot(hue="model", y="diff_time", x="week"
                    , hue_order=["Surveillance data","Surv. plus peaks","Surveillance plus HJ"]
                    #, showfliers = False
                    , fliersize = 0
                    , data = all_peak_data
                    , ax=ax
                    , palette = ['blue','orange','purple']
                    , order=[4,3,2,1,-2])
    ax.set_ylabel("Posterior samples\nminus true peak day", fontsize=10)
    ax.set_xlabel("Weeks before underlying peak", fontsize=10)

    leg = ax.legend(frameon=False, loc = "upper right", labelspacing = 0.025, handletextpad=0.05, fontsize=10, bbox_to_anchor=(1.0,1.035))
    for lh in leg.legendHandles: 
        lh.set_alpha(0.7)
    
    stamp(ax,"A.")

    for patch in g.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))

    ax.set_ylim(-55,100)
        
    ax = axs[0,1]
    g = sns.boxplot(hue="model", y="diff_time", x="model_included"
                    , hue_order=["Surveillance data","Surv. plus peaks","Surveillance plus HJ"]
                    , showfliers = False
                    , fliersize=0
                    , data = all_peak_data
                    #, line_kws = dict(color="black")
                    , ax=ax
                    , palette = ['blue','orange','purple'])
    ax.set_ylabel("", fontsize=10)

    ax.get_legend().remove()
     
    ax.set_xlabel("", fontsize=10)
    ax.set_xticklabels(["No model\nguidance", "Model\nguidance"], fontsize=10)

    # for patch in g.patches:
    #     r, g, b, a = patch.get_facecolor()
    #     patch.set_facecolor((r, g, b, .7))
    
    ax.set_ylim(-55,100)
    ax.set_yticklabels([])
    stamp(ax,"B.")
    
    ax = axs[0,2]
    g = sns.boxplot(hue="model", y="diff_time", x="noise"
                    , hue_order=["Surveillance data","Surv. plus peaks","Surveillance plus HJ"]
                    #, showfliers = False
                    , fliersize=0
                    , data = all_peak_data, ax=ax, palette = ['blue','orange','purple'])
    ax.set_ylabel("", fontsize=10)

    ax.get_legend().remove()
     
    ax.set_xlabel("", fontsize=10)
    ax.set_xticklabels(["High noise", "Medium", "Low"], fontsize=10)

    for patch in g.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))
    
    ax.set_ylim(-55,100)
    ax.set_yticklabels([])
   
    
    stamp(ax,"C.")

    #--intensity
    ax = axs[1,0]
    g = sns.pointplot(hue="model", y="PIT", x="week"
                      , data = all_PITS
                      , hue_order=["Surveillance data","Surv. plus peaks","Surveillance plus HJ"], dodge=0.50
                      ,ax=ax, palette = ['blue','orange','purple'], order=[4,3,2,1,-2], join=False, capsize=0.05)

    sns.stripplot(hue="model", y="PIT", x="week"
                  , data = all_PITS
                  , size = 0.25
                  , jitter=True
                  , alpha=0.25
                  , marker="D"
                  , hue_order=["Surveillance data","Surv. plus peaks","Surveillance plus HJ"], dodge=True
                  , ax=ax, palette = ['blue','orange','purple'], order=[4,3,2,1,-2])
        
    for points in g.collections:
        size = points.get_sizes().item()
        points.set_sizes( [12.5 for x in all_peak_data.week.unique()])

    ax.set_ylabel("Probability integral transform\nfor peak intensity", fontsize=10)
    ax.set_xlabel("Weeks before underlying peak", fontsize=10)

    ax.set_ylim(0.,1)
    ax.set_yticks([0,0.25,0.50,0.75,1.0])

    ax.axhline(0.50, color="black",lw=1)
    
    ax.get_legend().remove()
    stamp(ax,"D.")

    ax = axs[1,1]
    g = sns.pointplot(hue="model", y="PIT", x="model_included", data = all_PITS
                      , hue_order=["Surveillance data","Surv. plus peaks","Surveillance plus HJ"]
                      ,ax=ax, palette = ['blue','orange','purple'], join=False, capsize=0.05,dodge= 0.50 )

    sns.stripplot(hue="model", y="PIT", x="model_included"
                  , data = all_PITS
                  , size = 0.25
                  , jitter=True
                  , alpha=0.25
                  , marker="D"
                  , hue_order=["Surveillance data","Surv. plus peaks","Surveillance plus HJ"], dodge=True
                  , ax=ax, palette = ['blue','orange','purple'])
    
    ax.set_ylabel("", fontsize=10)

    for points in g.collections:
        size = points.get_sizes().item()
        points.set_sizes( [12.5 for x in all_peak_data.week.unique()])
 

    ax.get_legend().remove()
     
    ax.set_xlabel("", fontsize=10)
    ax.set_xticklabels(["No model\nguidance", "Model\nguidance"], fontsize=10)

    ax.set_ylim(0.,1)
    ax.set_yticks([0,0.25,0.50,0.75,1.0])
    ax.set_yticklabels([])

    ax.axhline(0.50, color="black",lw=1)
    
    stamp(ax,"E.")
    
    ax = axs[1,2]
    g = sns.pointplot(hue="model", y="PIT", x="noise", data = all_PITS
                      , hue_order=["Surveillance data","Surv. plus peaks","Surveillance plus HJ"], join=False
                      ,ax=ax, palette = ['blue','orange','purple'], capsize=0.05,dodge=0.50 )

    sns.stripplot(hue="model", y="PIT", x="noise"
                  , data = all_PITS
                  , size = 0.25
                  , jitter=True
                  , alpha=0.25
                  , marker="D"
                  , hue_order=["Surveillance data","Surv. plus peaks","Surveillance plus HJ"], dodge=True
                  , ax=ax, palette = ['blue','orange','purple'])
 
    
    for points in g.collections:
        size = points.get_sizes().item()
        points.set_sizes( [12.5 for x in all_peak_data.week.unique()])

    ax.set_ylabel("", fontsize=10)

    ax.set_xlabel("", fontsize=10)
    ax.set_xticklabels(["High noise", "Medium", "Low"], fontsize=10)

    ax.get_legend().remove()

    ax.axhline(0.50, color="black",lw=1)

    ax.set_ylim(0.,1)
    ax.set_yticks([0,0.25,0.50,0.75,1.0])
    ax.set_yticklabels([])
    stamp(ax,"F.")
    
    fig.set_tight_layout(True)
    w = mm2inch(183)

    fig.set_size_inches(w, w/1.5)

    plt.savefig("./anova_on_peak_timing_and_intensity.pdf")
    plt.close()

    
    #--setup two anovas
    import statsmodels.formula.api as smf
    
    mod = smf.ols(formula='abs_diff_time ~ C(model) + C(week) + C(model_included) + C(noise)-1', data=all_peak_data)
    res = mod.fit()
    print(res.summary())


    mod = smf.ols(formula='abs_diff_value ~ C(model) + C(week) + C(model_included) + C(noise)-1', data=all_peak_data)
    res = mod.fit()
    print(res.summary())
