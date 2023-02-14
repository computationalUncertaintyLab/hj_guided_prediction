#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots

if __name__ == "__main__":
        #--import data from the prolific survey and the information each particpant inputted
    hj = pd.read_csv("../../data_from_prolific/prolific_data.csv")
    hj = hj[["week","noise","model_included","peak_time","peak_intensity"]]

    #--remove erroneous predictions
    hj = hj.loc[(hj.peak_time<=210) & (hj.peak_time>=0) & (hj.peak_intensity>=0) & (hj.peak_intensity<=2500)]
    
    #--real peak and peak intensity
    PEAK = 86
    PEAK_VALUE = 649.15
    
    #--create columns that show the difference between the precidtced peak/peak intensity and the real
    hj["diff_time"] = hj.peak_time - PEAK
    hj["diff_intensity"] = hj.peak_intensity - PEAK_VALUE

    