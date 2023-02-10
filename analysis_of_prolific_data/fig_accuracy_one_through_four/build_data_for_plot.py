#mcandrew

import sys
sys.path.append("../../chimeric_forecast/")

from chimeric_forecast__weighted_ll import chimeric_forecast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
import pickle


def crps(samples, truth):
    from scipy.interpolate import PchipInterpolator as pchip
    from scipy.interpolate import interp1d
    from scipy.integrate import quad

    #--sort data
    L = len(samples)
    samples, probs = sorted([float(x) for x in samples]), np.arange(0.,L)/L

    minx,maxx = min(samples), max(samples) 
    
    #--build linear interpolation for cdf
    cdf = interp1d(samples,probs)

    def CDF(x,minx,maxx,cdf):
        if x<minx:
            return 0
        if x>maxx:
            return 1.
        else:
            return cdf(x)
    
    indicator = lambda x: 1 if x>truth else 0 

    g = lambda x: (CDF(x,minx,maxx,cdf) - indicator(x))**2

    crps, error = quad(g,0,2500)
    return crps

if __name__ == "__main__":

    #--extarct true values over time (same for all treatments)
    truth = pd.read_csv("../../for_crowdsourcing/data_collection__1__0__5.00/truth_data__1__0__5.00.csv")
    true_inc_hosps = truth.truth_data.values

    PEAK_TIME = 86
    
    #--from pickles of posterior samples to absolute error and WIS
    performance_data = pd.DataFrame()

    for f in glob("../fig2_performance_metrics/plotdata/*.pkl"):
        forecast = pickle.load(open(f,"rb"))

        #--extract params from slug
        forecast_model, _, week,model,noise = f.split("/")[-1].split("__")

        noise       = float(noise.replace(".pkl",""))
        week, model = int(week), int(model)

        #--compute medians over time
        medians_over_time = forecast.compute_quantiles()["50.000"]

        #--absolute error
        aes = abs(medians_over_time - true_inc_hosps)

        #--collect the one day through 28 day errors
        forecasts = medians_over_time[ (PEAK_TIME - 7*week)+1 : (PEAK_TIME - 7*week)+28+1 ]
        
        forecast_errors = aes[ (PEAK_TIME - 7*week)+1 : (PEAK_TIME - 7*week)+28+1 ]

        #--CRPS per time point
        horizons = np.arange( (PEAK_TIME - 7*week)+1, (PEAK_TIME - 7*week)+28+1)
        scores = []
        for horizon in horizons:
            samples_at_times = forecast.posterior_samples["inc_hosps"][:,horizon]
            score = crps( samples_at_times, true_inc_hosps[horizon])
            scores.append(score)
            
        forecast_performance = pd.DataFrame({"model":forecast_model
                                             ,"week":week
                                             ,"noise":noise
                                             ,"model_guidance":model
                                             ,"medians":forecasts
                                             ,"ae":forecast_errors
                                             ,"scores":scores
                                             ,"horizon":np.arange(1,28+1)})
        performance_data = pd.concat([performance_data,forecast_performance])
        performance_data.to_csv("./performance_data.csv",index=False)
