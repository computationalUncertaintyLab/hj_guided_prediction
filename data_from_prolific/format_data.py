#mcandrew

import sys
import numpy as np
import pandas as pd

import json

if __name__ == "__main__":

    d = pd.read_csv("./Chimeric_forecasting_February 5, 2023_13.03.csv")
    first_row, second_row  = d.iloc[0], d.iloc[1]

    d = pd.read_csv("./Chimeric_forecasting_February 5, 2023_13.03.csv", skiprows=2)
    d.columns = first_row.index# [  json.loads(x)["ImportId"] for x in second_row.values]

    #--dictionary that maps questions to the treatments
    from_question_2_treatments = {  "Q1": [-2, 10. ,1]
                                   ,"Q2": [-2, 5.  ,1]
                                   ,"Q3": [-2, 2.5 ,1]
                                   ,"Q4": [-2, 10. ,0]
                                   ,"Q5": [-2, 5.  ,0]
                                   ,"Q6": [-2, 2.5 ,0]

                                   ,"Q7": [4, 10. ,1]
                                   ,"Q8": [4, 5.  ,1]
                                   ,"Q9": [4, 2.5 ,1]
                                   ,"Q10": [4, 10. ,0]
                                   ,"Q11": [4, 5.  ,0]
                                   ,"Q12": [4, 2.5 ,0]
                                    
                                   ,"Q13": [3, 10. ,1]
                                   ,"Q14": [3, 5.  ,1]
                                   ,"Q15": [3, 2.5 ,1]
                                   ,"Q16": [3, 10. ,0]
                                   ,"Q17": [3, 5.  ,0]
                                   ,"Q18": [3, 2.5 ,0]
                                    
                                   ,"Q19": [2, 10. ,1]
                                   ,"Q20": [2, 5.  ,1]
                                   ,"Q21": [2, 2.5 ,1]
                                   ,"Q22": [2, 10. ,0]
                                   ,"Q23": [2, 5.  ,0]
                                   ,"Q24": [2, 2.5 ,0]
                                    
                                   ,"Q25": [1, 10. ,1]
                                   ,"Q26": [1, 5.  ,1]
                                   ,"Q27": [1, 2.5 ,1]
                                   ,"Q28": [1, 10. ,0]
                                   ,"Q29": [1, 5.  ,0]
                                   ,"Q30": [1, 2.5 ,0]
                                  }
    formatted_data = {"PID":[], "startdate":[], 'enddate':[], 'duration':[]
                      , "week":[], "noise":[], "model_included":[]
                      , "peak_time":[], "peak_intensity":[]
                      , "factors__current_season":[], "factors__past_season":[], "factors__past_season2":[],"factors__model":[]
                      ,"survey_summary_from_participant":[]}
    for index, row in d.iterrows():

        if 'No, I do not consent to participate in this study' == row.Q1:
            continue
        
        PID = row.Q2
        formatted_data["PID"].append(PID)

        startdate = row.StartDate
        formatted_data["startdate"].append(startdate)
        
        enddate = row.EndDate
        formatted_data["enddate"].append(enddate)

        duration = row["Duration (in seconds)"]
        formatted_data["duration"].append(duration)

        #--find question that was answered
        for num in np.arange(1,30+1):
            q = "Q{:d}".format(num)
            peak_time      = q+"_1"
            peak_intensity = q+"_2"
            
            if np.isnan(row[peak_intensity])==False: #--if there exists an answer
                week, noise, model = from_question_2_treatments[q]
                peak_time      = row[peak_time]
                peak_intensity = row[peak_intensity]
                break
        formatted_data["week"].append(week)
        formatted_data["noise"].append(noise)
        formatted_data["model_included"].append(model)

        formatted_data["peak_time"].append(peak_time)
        formatted_data["peak_intensity"].append(peak_intensity)

        #--parse factors
        try:
            considerations = row["Considerations"].split(",")
            cs = [0,0,0,0,0]
            for consideration in considerations:
                if "current season" in consideration:
                    cs[0]=1
                elif "previous season" in consideration:
                    cs[1]=1
                elif "two seasons" in consideration:
                    cs[2]=1
                elif "model":
                    cs[3]=1
                else: #--must be other
                    cs[4]=1
            formatted_data["factors__current_season"].append(cs[0])
            formatted_data["factors__past_season"].append(cs[1])
            formatted_data["factors__past_season2"].append(cs[2])
            formatted_data["factors__model"].append(cs[3])
        except:
            formatted_data["factors__current_season"].append(0)
            formatted_data["factors__past_season"].append(0)
            formatted_data["factors__past_season2"].append(0)
            formatted_data["factors__model"].append(0)
 
        formatted_data["survey_summary_from_participant"].append(row.Summary)
    formatted_data = pd.DataFrame(formatted_data)

    #--remove prolific id and replace with random integer
    formatted_data["PID"] = np.random.choice(2*len(formatted_data),size=len(formatted_data), replace=False)

    formatted_data.to_csv("./prolific_data.csv", index=False)
