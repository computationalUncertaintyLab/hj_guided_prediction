#mcandrew
class chimeric_forecast(object):
    def __init__(self
                 ,rng_key                       = None
                 , surveillance_data           = None
                 , population                  = 12*10**6
                 , total_window_of_observation = 210
                 , peak_time_and_values        = None
                 , peak_time_and_values_independent = None
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
        self.peak_time_and_values_independent = peak_time_and_values_independent
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


    def svi_on_params(self
                      , gamma_preset = 1./1
                      , kappa_preset = 1./7):
        
        from jax import random
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.distributions import constraints
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        def guide():
            r0_0 = numpyro.param("r0_0", 1.1   , constraint=constraints.interval(0.75,2.))
            r0_1 = numpyro.param("r0_1", 3.   , constraint=constraints.interval(2.,4.))
            
            r0 = numpyro.sample( "r0", dist.Uniform(r0_0, r0_1) )

            sigma_0 = numpyro.param("sigma_0", 10*(1./10), constraint=constraints.positive)
            sigma_1 = numpyro.param("sigma_1", 10*(1 - (1./10)), constraint=constraints.positive)
            sigma = numpyro.sample("sigma", dist.Beta(sigma_0, sigma_1))
            
            ps_0 = numpyro.param("ps_0", 10*(0.1), constraint=constraints.positive)
            ps_1 = numpyro.param("ps_1", 10*(1 - 0.1), constraint=constraints.positive)
            ps = numpyro.sample("ps", dist.Beta(ps_0, ps_1))

            phi_0 = numpyro.param("phi_0", 10*0.10, constraint=constraints.positive)
            phi_1 = numpyro.param("phi_1", 10*0.90, constraint=constraints.positive)
            phi = numpyro.sample("phi", dist.Beta(phi_0, phi_1))

            E0_0 = numpyro.param("E0_0", 1./self.population, constraint=constraints.positive)
            E0_1 = numpyro.param("E0_1", 5./self.population, constraint=constraints.positive)
            E0   = numpyro.sample("E0", dist.Uniform(E0_0,E0_1))

            I0_0 = numpyro.param("I0_0", 1./self.population, constraint=constraints.positive)
            I0_1 = numpyro.param("I0_1", 5./self.population, constraint=constraints.positive)
            I0   = numpyro.sample("I0", dist.Uniform(I0_0,I0_1))

            s_p  =  numpyro.param("s_0" , 1, constraint=constraints.positive)
            s    = numpyro.sample("s", dist.HalfCauchy(s_p))

            s2_p =  numpyro.param("s2_0", 1, constraint=constraints.positive)
            s2   = numpyro.sample("s2", dist.HalfCauchy(s2_p))
        
        optimizer = numpyro.optim.ClippedAdam(step_size=0.0005)
        
        svi = SVI(lambda : self.model_specificiation(prior=True), guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(random.PRNGKey(2020), 5000)
        
        self.svi = svi_result
        self.prior_params = svi_result.params
        
        return svi_result
        
    def model_specificiation(self
                             , gamma_preset = 1./1
                             , kappa_preset = 1./7
                             , prior = None):

        import jax
        from jax import random
        import jax.numpy as jnp

        import numpyro
        import numpyro.distributions as dist

        #--the number of days from 0 to the total_window_of_observation
        times = jnp.arange(0,self.total_window_of_observation)

        if self.humanjudgment_data is not None:
            if prior == False:
                r0_0    = self.prior_params["r0_0"]
                r0_1    = self.prior_params["r0_1"]

                sigma_0    = self.prior_params["sigma_0"]
                sigma_1    = self.prior_params["sigma_1"]

                ps_0    = self.prior_params["ps_0"]
                ps_1    = self.prior_params["ps_1"]

                phi_0    = self.prior_params["phi_0"]
                phi_1    = self.prior_params["phi_1"]

                E0_0    = self.prior_params["E0_0"]
                E0_1    = self.prior_params["E0_1"]

                I0_0    = self.prior_params["I0_0"]
                I0_1    = self.prior_params["I0_1"]

            else:
                r0_0, r0_1    = 0.75, 4.0
                
                sigma_0    = 0.5
                sigma_1    = 0.5

                ps_0    = 0.5
                ps_1    = 0.5

                phi_0    = 0.5
                phi_1    = 0.5

                E0_0    = 1./self.population
                E0_1    = 20./self.population

                I0_0    = 1./self.population
                I0_1    = 20./self.population
        else:
            r0_0, r0_1    = 0.75, 4.0
                
            sigma_0    = 0.5
            sigma_1    = 0.5

            ps_0    = 0.5
            ps_1    = 0.5

            phi_0    = 0.5
            phi_1    = 0.5

            E0_0    = 1./self.population
            E0_1    = 20./self.population

            I0_0    = 1./self.population
            I0_1    = 20./self.population
                
        r0     = numpyro.sample("r0", dist.Uniform(r0_0,r0_1))
        
        sigma  = numpyro.sample("sigma", dist.Beta(sigma_0, sigma_1)) #--prior for sigma, the latent period
        phi    = numpyro.sample("phi"  , dist.Beta(phi_0, phi_1))  #--prior for phi, the percent hospitalized during the infectious period

        gamma = gamma_preset                                             #--gamma, the infectious period, is preset by the user
        kappa = kappa_preset                                             #--kappa, the duration of time in the hospitalization period before movng to R, is preset by the user

        ps    = numpyro.sample("ps", dist.Beta(ps_0, ps_1))  #--prior for ps, the percent hospitalized during the infectious period

        E0 = numpyro.sample( "E0", dist.Uniform(E0_0, E0_1 ) )       #--The proportion of exposed individuals at time 0
        I0 = numpyro.sample( "I0", dist.Uniform(I0_0, I0_1) )       #--The proportion of exposed individuals at time 0

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

        if prior==True:
            if self.humanjudgment_data is not None:
                s  = numpyro.sample("s" , dist.HalfCauchy(1.))
                s2 = numpyro.sample("s2", dist.HalfCauchy(1.))
                ll_peak_values = numpyro.sample("ll_peak"      , dist.Normal(peak_value,s) , obs = self.humanjudgment_data[:,1] )
                ll_peak_times = numpyro.sample("ll_peak_times" , dist.Normal(peak_time,s2) , obs = self.humanjudgment_data[:,0] )

        else:
            #--surveillance data
            if self.surveillance_data is None:
                pass
            else:
                T    = len(self.surveillance_data)    #--T is the number of time units of surveillance data that we have. T < total_window_of_observation
                mask = ~jnp.isnan(inc_hosps)          #--This is used to exclude missing surveillance data fro mthe log likelihood

                #--Compute the log likelihood. The 1./3 is preset
                with numpyro.handlers.mask(mask=mask[:T]):
                    ll_surveillance = numpyro.sample("ll_surveillance", dist.Poisson(inc_hosps[:T]), obs = self.surveillance_data)

    def fit_model(self, print_summary=True):
        from numpyro.infer import NUTS, MCMC, init_to_value, init_to_median, init_to_uniform
        import numpy as np

        if self.humanjudgment_data is not None:
            self.svi_on_params()
        
        #--Sample with NUTS
        nuts_kernel = NUTS(lambda : self.model_specificiation(prior=False), dense_mass=True, max_tree_depth=10)
        mcmc        = MCMC( nuts_kernel , num_warmup=4000, num_samples=4000,progress_bar=True)
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
        quants = np.percentile( self.posterior_samples, qs, 0).T

        quants_df = {}
        for n,q in enumerate(qs):
            quants_df["{:4.3f}".format(q)] = quants[:,n]
        quants_df = pd.DataFrame(quants_df)
            
        #--attach and return computed quantiles
        self.quants    = quants
        self.quants_df = quants_df
        return quants_df
       
    def compute_quantiles_from_sample(self,qs = [2.5, 25.0, 50., 75., 97.5],attribute = "inc_hosps"):
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
    import matplotlib.pyplot as plt
    import numpy as np
    
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
    hj_data  = generate_humanjudgment_prediction_data(true_incident_hospitalizations = inc_hosps
                                                      , noise = 20.
                                                      , number_of_humanjudgment_predictions=4000
                                                      , time_at_peak = time_at_peak
                                                      , rng_key  = rng_key
                                                      , bias_time = 10 ) 
    noisy_time_at_peaks, noisy_peak_values, noisy_human_predictions = hj_data.generate_predictions()

    #--fit model
    cut = time_at_peak - 7*4
    forecast = chimeric_forecast(rng_key = rng_key, surveillance_data = noisy_hosps[:cut]
                                 , humanjudgment_data = noisy_human_predictions
                                 , peak_time_and_values=True
                                 , surveillance_concentration = 1. )
    forecast.fit_model()
    
    
    forecast2 = chimeric_forecast(rng_key = rng_key, surveillance_data = noisy_hosps[:cut]
                                  , humanjudgment_data = None
                                  , peak_time_and_values=True
                                  , surveillance_concentration = 30. )
    forecast2.fit_model()

    #--compute quantiles for incident hospitalizations
    quantiles_for_incident_hosps1 = forecast.compute_quantiles_from_sample()

    #--compute quantiles for incident hospitalizations
    quantiles_for_incident_hosps2 = forecast2.compute_quantiles_from_sample()

    fig,axs = plt.subplots(1,2)

    ax = axs[0]
    
    times = np.arange(0,210)
    ax.plot(times, inc_hosps, color = "black")
    ax.scatter(times[:cut],noisy_hosps[:cut], color="black",s=5)

    ax.scatter( noisy_human_predictions[:,0], noisy_human_predictions[:,1], color = "red", marker = "s", s=5, alpha=0.80 )
    
    ax.fill_between(times, quantiles_for_incident_hosps1["2.500"], quantiles_for_incident_hosps1["97.500"],alpha=0.15, color="blue")
    ax.fill_between(times, quantiles_for_incident_hosps1["25.000"], quantiles_for_incident_hosps1["75.000"],alpha=0.15, color="blue")
    ax.plot(quantiles_for_incident_hosps1["50.000"],color="blue")
    
    ax = axs[1]

    times = np.arange(0,210)
    ax.plot(times, inc_hosps, color = "black")
    ax.scatter(times[:cut],noisy_hosps[:cut], color="black",s=5)

    ax.fill_between(times, quantiles_for_incident_hosps2["2.500"], quantiles_for_incident_hosps2["75.000"],alpha=0.15, color="red")
    ax.fill_between(times, quantiles_for_incident_hosps2["2.500"], quantiles_for_incident_hosps2["97.500"],alpha=0.15, color="red")
    ax.plot(quantiles_for_incident_hosps2["50.000"],color="red")

    plt.show()
