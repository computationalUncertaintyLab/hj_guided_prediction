#mcandrew

class generate_humanjudgment_prediction_data(object):
    def __init__(self
                 ,true_incident_hospitalizations      = None #--The true number of incident hosps.
                 ,time_at_peak                        = None #--The true day that incident hosps peak.
                 ,span                                = 28   #--Number of days before and after that humans predict the time at peak.
                 ,noise                               = 5.95 #--The "noise" around the peak. This is the variance of the NB2.
                 ,number_of_humanjudgment_predictions = 50   #--Number of human judgment predictions of the peak time and intensity.
                 ,rng_key                             = None):

        self.true_incident_hospitalizations = true_incident_hospitalizations
        self.time_at_peak = time_at_peak
        self.span         = span
        self.noise        = noise
        self.number_of_humanjudgment_predictions = number_of_humanjudgment_predictions
        self.rng_key      = rng_key
        
    def generate_predictions(self):
        import numpy as np
        
        import numpyro
        import numpyro.distributions as dist
        
        #--We assume predictions from humans is Uniformly distributed around the time at peak
        noisy_time_at_peaks = self.time_at_peak + np.random.randint(low = -1*self.span, high=self.span, size = (self.number_of_humanjudgment_predictions,))

        #--compute the peak intensity
        peak_value        = self.true_incident_hospitalizations[self.time_at_peak]
        noisy_peak_values = np.asarray(dist.NegativeBinomial2( peak_value, self.noise ).sample(self.rng_key, sample_shape = (self.number_of_humanjudgment_predictions,)  ))

        #--Combine these two vectos into two columns
        noisy_predictions = np.stack([noisy_time_at_peaks,noisy_peak_values]).T
        
        #--attached to object and return
        self.noisy_time_at_peaks = noisy_time_at_peaks
        self.noisy_peak_values   = noisy_peak_values
       
        return noisy_time_at_peaks, noisy_peak_values, noisy_predictions

if __name__ == "__main__":
    pass
    


