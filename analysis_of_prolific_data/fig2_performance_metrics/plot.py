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

    all_quantile_data = pd.read_csv("./plotdata/all_quantile_data.csv")
    all_peak_data     = pd.read_csv("./plotdata/all_peak_data.csv")
   
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
    all_peak_data["model"] = all_peak_data["model"].replace("survdata_plus_hj","Surveillance plus HJ")
    all_peak_data["model"] = all_peak_data["model"].replace("surv_data_plus_past","Surv. plus peaks")

    plt.style.use(['science','grid'])
    
    fig,axs = plt.subplots(2,3)

    ax = axs[0,0]
    g = sns.boxplot(hue="model", y="diff_time", x="week"          , fliersize=0, data = all_peak_data, ax=ax, palette = ['blue','orange','purple'], order=[4,3,2,1,-2])
    ax.set_ylabel("Post. samples\nminus true peak day", fontsize=10)
    ax.set_xlabel("Weeks before underlying peak", fontsize=10)

    leg = ax.legend(frameon=False, loc = "upper right", labelspacing = 0.025, handletextpad=0.05, fontsize=10, bbox_to_anchor=(1.0,1.035))
    for lh in leg.legendHandles: 
        lh.set_alpha(0.7)
    
    stamp(ax,"A.")

    for patch in g.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))

    ax.set_ylim(-55,85)
        
    ax = axs[0,1]
    g = sns.boxplot(hue="model", y="diff_time", x="model_included", fliersize=0, data = all_peak_data, ax=ax, palette = ['blue','orange','purple'])
    ax.set_ylabel("", fontsize=10)

    ax.get_legend().remove()
     
    ax.set_xlabel("", fontsize=10)
    ax.set_xticklabels(["No model\nguidance", "Model\nguidance"], fontsize=10)

    for patch in g.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))
    
    ax.set_ylim(-55,85)
    ax.set_yticklabels([])
    stamp(ax,"B.")
    
    ax = axs[0,2]
    g = sns.boxplot(hue="model", y="diff_time", x="noise"         , fliersize=0, data = all_peak_data, ax=ax, palette = ['blue','orange','purple'])

    ax.set_ylabel("", fontsizegg=10)

    ax.set_xlabel("", fontsize=10)
    ax.set_xticklabels(["High noise", "Medium", "Low"], fontsize=10)

    ax.get_legend().remove()


    for patch in g.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))

    ax.set_ylim(-55,85)
    ax.set_yticklabels([])
    stamp(ax,"C.")

    #--intensity
    ax = axs[1,0]
    g = sns.pointplot(hue="model", y="diff_value", x="week", data = all_peak_data, ax=ax, join=True, palette = ['blue','orange','purple'], capsize=0.05, order=[4,3,2,1,-2],dodge=True)
    for points in g.collections:
        size = points.get_sizes().item()
        points.set_sizes( [7.5 for x in all_peak_data.week.unique()])

    ax.set_ylabel("Post. samples\nminus true peak intensity", fontsize=10)
    ax.set_xlabel("Weeks before underlying peak", fontsize=10)

    ax.set_ylim(-500,1000)
    ax.get_legend().remove()
    stamp(ax,"D.")

    ax = axs[1,1]
    g = sns.pointplot(hue="model", y="diff_value", x="model_included", data = all_peak_data, ax=ax, palette = ['blue','orange','purple'], capsize=0.05,dodge=True)
    ax.set_ylabel("", fontsize=10)

    for points in g.collections:
        size = points.get_sizes().item()
        points.set_sizes( [7.5 for x in all_peak_data.week.unique()])
 

    ax.get_legend().remove()
     
    ax.set_xlabel("", fontsize=10)
    ax.set_xticklabels(["No model\nguidance", "Model\nguidance"], fontsize=10)

    ax.set_ylim(-500,1000)
    ax.set_yticklabels([])
    
    stamp(ax,"E.")
    
    ax = axs[1,2]
    g = sns.pointplot(hue="model", y="diff_value", x="noise", data = all_peak_data, ax=ax, palette = ['blue','orange','purple'], capsize=0.05,dodge=True)
    
    for points in g.collections:
        size = points.get_sizes().item()
        points.set_sizes( [7.5 for x in all_peak_data.week.unique()])

    ax.set_ylabel("", fontsize=10)

    ax.set_xlabel("", fontsize=10)
    ax.set_xticklabels(["High noise", "Medium", "Low"], fontsize=10)

    ax.get_legend().remove()


    ax.set_ylim(-500,1000)
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
