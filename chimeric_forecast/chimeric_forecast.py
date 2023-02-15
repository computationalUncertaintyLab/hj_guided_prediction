#mcandrew
class chimeric_forecast(object):
    def __init__(self
                 ,rng_key                       = None
                 , surveillance_data           = None
                 , population                  = 12*10**6
                 , total_window_of_observation = 210
                 , peak_time_and_values        = None
                 , peak_times_only             = None
                 , peak_values_only            = None
                 , humanjudgment_data          = None
                 , gamma_preset                = 1./1
                 , kappa_preset                = 1/.7
                 , sigma_preset                = None
                 , phi_preset                  = None
                 , ps_preset                   = None
                 , E0_preset                   = None
                 , I0_preset                   = None
                 , surveillance_concentration  = None
                 ):
        self.surveillance_data           = surveillance_data
        self.population                  = population
        self.total_window_of_observation = total_window_of_observation
        self.peak_time_and_values        = peak_time_and_values
        self.peak_times_only             = peak_times_only
        self.peak_values_only            = peak_values_only
        self.humanjudgment_data          = humanjudgment_data
        self.gamma_preset                = gamma_preset
        self.kappa_preset                = kappa_preset
        self.sigma_preset                = sigma_preset
        self.phi_preset                  = phi_preset                  
        self.ps_preset                   = ps_preset                   
        self.E0_preset                   = E0_preset
        self.I0_preset                   = I0_preset
        self.surveillance_concentration  = surveillance_concentration
        
        self.rng_key                     = rng_key

    def model_specificiation(self
                             , gamma_preset = 1./1
                             , kappa_preset = 1./7
                             ):

        import jax
        from jax import random
        import jax.numpy as jnp

        import numpyro
        import numpyro.distributions as dist

        #--the number of days from 0 to the total_window_of_observation
        times = jnp.arange(0,self.total_window_of_observation)

        #--priors 
        r0     = numpyro.sample("R0" , dist.Uniform(0.75,4))             #--prior for R0

        if self.sigma_preset is None:
            sigma  = numpyro.sample("sigma", dist.Beta( 1/2., (1-(1./2)) ) ) #--prior for sigma, the latent period
        else:
            sigma = self.sigma_preset

        if self.phi_preset is None:
            phi    = numpyro.sample("phi", dist.Beta(0.025*5, (1-0.025)*5))  #--prior for phi, the percent hospitalized during the infectious period
        else:
            phi = self.phi_preset
            
        gamma = gamma_preset                                             #--gamma, the infectious period, is preset by the user
        kappa = kappa_preset                                             #--kappa, the duration of time in the hospitalization period before movng to R, is preset by the user

        if self.ps_preset is None:
            ps = numpyro.sample("ps", dist.Beta( 0.10*5, (1-0.10)*5 ) )      #--The percent of individuals who are susceptible
        else:
            ps = self.ps_preset

        if self.E0_preset is None:
            E0 = numpyro.sample( "E0", dist.Uniform(1./self.population, 20./self.population) )       #--The proportion of exposed individuals at time 0
        else:
            E0 = self.E0_preset

        if self.I0_preset is None:
            I0 = numpyro.sample( "I0", dist.Uniform(1./self.population, 20./self.population) )       #--The proportion of infectious individuals at time 0
        else:
            I0 = self.I0_preset

        if self.surveillance_concentration is None:
            surveillance_concentration = numpyro.sample("surveillance_concentration" , dist.Uniform(0.01,10))
        else:
            surveillance_concentration = self.surveillance_concentration
            
        S0 = ps*1. - I0                                                  #--The proportion of susceptible individuals at time 0
        H0 = 0                                                           #--The proportion of hospitalized individuals at time 0
        R0 = 1. - S0 - I0 - E0                                           #--The proportion of removed/recovered individuals at time 0

        beta = r0*gamma*(1./ps)                                          #--The effectice contact rate

        def evolve(carry,array, params):       #--Determinsitic system used to evolve states according to SEIR dynamics
            s,e,i,h,r,c = carry                #--States are SEIHR and then a sixth state C which records incidetn hospitalizations
            sigma,beta,gamma,kappa = params    #--Parameters needed are sigma, beta, gamma and kappa

            s2e = s*beta*i
            e2i = sigma*e
            i2h = gamma*i*phi
            i2r = gamma*i*(1-phi)
            h2r = kappa*h

            ns = s-s2e
            ne = e+s2e - e2i
            ni = i+e2i - (i2h+i2r)
            nh = h+i2h - h2r
            nr = r+i2r + h2r

            nc = i2h

            states = jnp.vstack( (ns,ne,ni,nh,nr, nc) )
            return states, states

        #--Final states over the total_window_of_observation
        final, states = jax.lax.scan( lambda x,y: evolve(x,y, (sigma, beta, gamma,kappa) ), jnp.vstack( (S0,E0,I0,H0,R0, H0) ), times)   

        #--Record these states for posthoc evaluation if needed
        states = numpyro.deterministic("states",states)

        #--Extract the proportion of incident hospitalizations
        inc_hosp_proportion = states[:,-1]

        #--The proportion of incident hospitalizations cannot be smaller than machine epsilon or greater than 1
        inc_hosp_proportion = jnp.clip(inc_hosp_proportion,a_min=0.,a_max= ps)

        #--Compute the number of incident hospitalizations as the proportion times the total number of individuals
        inc_hosps = (inc_hosp_proportion*self.population).reshape(-1,)

        #--Record the number of incident hospitalizations
        inc_hosps = numpyro.deterministic("inc_hosps",inc_hosps)

        #--add machine epsilon
        inc_hosps = inc_hosps + jnp.finfo(float).eps
        
        #--Extract from the above vector the peak time and the peak value
        peak_time, peak_value = jnp.argmax(inc_hosps), jnp.max(inc_hosps)

        #--Record the joint vector of peak time and intensity
        joint_peak_data = numpyro.deterministic("peak_time_and_value", jnp.array([peak_time,peak_value]))
        
        #--surveillance data
        if self.surveillance_data is None:
            pass
        else:
            T    = len(self.surveillance_data)    #--T is the number of time units of surveillance data that we have. T < total_window_of_observation
            mask = ~jnp.isnan(inc_hosps)          #--This is used to exclude missing surveillance data fro mthe log likelihood

            #--Compute the log likelihood. The 1./3 is preset
            with numpyro.handlers.mask(mask=mask[:T]):
                ll_surveillance = numpyro.sample("ll_surveillance", dist.NegativeBinomial2(inc_hosps[:T], surveillance_concentration), obs = self.surveillance_data)

        #--human judgment predictions of peak time and intensity
        if self.humanjudgment_data is None:
            pass
        else:
            #--If the user has collected both peak times and peak intensities
            if self.peak_time_and_values:
                d = self.humanjudgment_data.shape[1] #--Number of dimensions or the number of aspects of the trajectory of inc hosps (2)
                N = self.humanjudgment_data.shape[0] #--Number of predictions collected from the crowd of humans

                theta = numpyro.sample("theta", dist.HalfCauchy((100**2)*jnp.ones(d)))   #--Vector of variances for each of the d variables

                concentration = jnp.ones(1)                                       #--A uniform distribution over correlation matrices
                corr_mat = numpyro.sample("corr_mat", dist.LKJ(d, concentration))
                std = jnp.sqrt(theta)                                             #--Standard deviation that governs the joint density for peak time and intensity

                #--Compute the covariance matrix
                #--We can also use a faster formula `cov_mat = jnp.outer(theta, theta) * corr_mat`
                cov_mat = jnp.matmul(jnp.matmul(jnp.diag(std), corr_mat), jnp.diag(std))

                #--Vector that includes the peak time and peak intensity from our SEIR dynamics 
                mu = jnp.array([peak_time, peak_value]).reshape(2,)

                #--Loglikelihood
                with numpyro.plate("observations", N):
                    obs = numpyro.sample("obs", dist.MultivariateStudentT(loc=mu, scale_tril=cov_mat, df=1), obs= self.humanjudgment_data)

            #--If the user has collected only both peak times and not collected peak intensities
            elif self.peak_times_only:
                s = numpyro.sample("s", dist.HalfCauchy(1*jnp.ones(1,))) #--standard deviation
                ll_peak_times = numpyro.sample("ll_peak_times", dist.Normal( peak_time, s )        , obs = self.humanjudgment_data.reshape(-1,) )
            #--If the user has collected only both peak intensity and not collected peak times
            elif self.peak_values_only:
                ll_peak_values = numpyro.sample("ll_peak", dist.NegativeBinomial2( peak_value, 0.3 ), obs = self.humanjudgment_data.reshape(-1,) )

    def fit_model(self, print_summary=True):
        from numpyro.infer import NUTS, MCMC
        #from numpyro.infer import init_to_feasible
        import numpy as np
        
        #--Sample with NUTS
        nuts_kernel = NUTS(self.model_specificiation)#, init_strategy=init_to_feasible())
        mcmc        = MCMC( nuts_kernel , num_warmup=4000, num_samples=2000,progress_bar=True)
        mcmc.run(self.rng_key, extra_fields=('potential_energy',))

        #--print summary is optional
        if print_summary:
            mcmc.print_summary()
            
        #--Collect samples from NUTS
        samples = mcmc.get_samples()

        #--check for any trajetories that are nan and remove them
        non_nan_trajectories = np.where( np.isnan(samples["inc_hosps"]).sum(1)==0)[0]

        if len(non_nan_trajectories)==0:
            return samples
    
        stripped_samples = {}
        for k,v in samples.items():
            stripped_samples[k] = v[non_nan_trajectories]

        #--attached and return completed posterior samples
        self.posterior_samples = stripped_samples
        return stripped_samples

    def compute_quantiles(self,qs = [2.5, 25.0, 50., 75., 97.5],attribute = "inc_hosps"):
        import numpy as np
        import pandas as pd
        quants = np.percentile( self.posterior_samples["{:s}".format(attribute)], qs, 0).T

        quants_df = {}
        for n,q in enumerate(qs):
            quants_df["{:4.3f}".format(q)] = quants[:,n]
        quants_df = pd.DataFrame(quants_df)
            
        #--attach and return computed quantiles
        self.quants    = quants
        self.quants_df = quants_df
        return quants_df

if __name__ == "__main__":
    #--set randomization key
    from jax import random
    rng_key = random.PRNGKey(0)

    #--include other objects in this folder
    from generate_simulated_data import generate_data
    from generate_humanjudgment_prediction_data import generate_humanjudgment_prediction_data

    #--generate simulated surveillance data
    surv_data = generate_data(rng_key = rng_key)
    inc_hosps, noisy_hosps, time_at_peak, peak_value = surv_data.simulate_surveillance_data()

    #--generate simulated human judgment predictins
    hj_data  = generate_humanjudgment_prediction_data(true_incident_hospitalizations = inc_hosps, time_at_peak = time_at_peak, rng_key  = rng_key ) 
    noisy_time_at_peaks, noisy_peak_values, noisy_human_predictions = hj_data.generate_predictions()

    #--fit model
    forecast = chimeric_forecast(rng_key = rng_key, surveillance_data = noisy_hosps, humanjudgment_data = noisy_human_predictions, peak_time_and_values=True )
    forecast.fit_model()

    #--compute quantiles for incident hospitalizations
    quantiles_for_incident_hosps = forecast.compute_quantiles()
