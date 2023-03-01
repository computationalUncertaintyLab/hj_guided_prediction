#mcandrew
class chimeric_forecast(object):
    def __init__(self
                 ,rng_key                       = None
                 , surveillance_data           = None
                 , population                  = 12*10**6
                 , total_window_of_observation = 210
                 , humanjudgment_data          = None
                 , past_season_data            = None
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
        self.humanjudgment_data          = humanjudgment_data
        self.past_season_data            = past_season_data
        self.gamma_preset                = gamma_preset
        self.kappa_preset                = kappa_preset
        self.sigma_preset                = sigma_preset
        self.phi_preset                  = phi_preset                  
        self.ps_preset                   = ps_preset                   
        self.E0_preset                   = E0_preset
        self.I0_preset                   = I0_preset
        self.surveillance_concentration  = surveillance_concentration

        self.rng_key                     = rng_key

    def shoot_trajectories(self,dynamic_params, init_states, times):
        import jax
        import jax.numpy as jnp
        def evolve(carry,array, params):
            #--Determinsitic system used to evolve states according to SEIR dynamics
            s,e,i,h,r,c = carry                #--States are SEIHR and then a sixth state C which records incidetn hospitalizations
            sigma,beta,gamma,kappa,phi = params    #--Parameters needed are sigma, beta, gamma and kappa

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
        final, states = jax.lax.scan( lambda x,y: evolve(x,y, dynamic_params  ), jnp.vstack(init_states), times)   
        return states

    def find_initial_params(self,maxiter=10**2, gamma_preset = 1., kappa_preset = 1./7):
        import jax.numpy as jnp
        import numpy as np

        hj_values = self.humanjudgment_data[:,1]
        hj_times  = self.humanjudgment_data[:,0]

        value_min,value_max = jnp.min(hj_values), jnp.max(hj_values)
        time_min,time_max   = jnp.min(hj_times), jnp.max(hj_times)

        times = jnp.arange(0,self.total_window_of_observation)
        
        iter = 0
        while iter<maxiter:
            iter+=1
            
            sigma = np.random.beta(0.5,0.5)
            r0    = np.random.uniform(0.75,4)
            gamma = gamma_preset
            kappa = kappa_preset

            phi = np.random.beta(0.5,0.5)

            ps    = np.random.beta(0.5,0.5)

            E0   = np.random.uniform(2,9)
            E0   = E0/self.population
            
            I0   = np.random.uniform(2,9)
            I0   = I0/self.population

            S0 = ps*1. - I0                                                  #--The proportion of susceptible individuals at time 0
            H0 = 0                                                           #--The proportion of hospitalized individuals at time 0
            R0 = 1. - S0 - I0 - E0                                           #--The proportion of removed/recovered individuals at time 0

            beta = r0*gamma*(1./ps)                                          #--The effectice contact rate
            
            dynamic_params = (sigma, beta, gamma, kappa, phi)
            init   = (S0,E0,I0,H0,R0, H0)
            states = self.shoot_trajectories(dynamic_params,init,times)

            #--Extract the proportion of incident hospitalizations
            inc_hosp_proportion = states[:,-1]

            #--The proportion of incident hospitalizations cannot be smaller than machine epsilon or greater than 1
            inc_hosp_proportion = jnp.clip(inc_hosp_proportion,a_min=jnp.finfo(float).eps,a_max= ps)

            #--Compute the number of incident hospitalizations as the proportion times the total number of individuals
            inc_hosps = (inc_hosp_proportion*self.population).reshape(-1,)

            #--add machine epsilon
            inc_hosps = inc_hosps + jnp.finfo(float).eps

            peak_time, peak_value = jnp.argmax(inc_hosps), jnp.max(inc_hosps)

            if (value_min < peak_value <value_max) and (time_min < peak_time <time_max):
                break

        inits = {"r0":r0,"sigma":sigma,"gamma":gamma,"kappa":kappa,"phi":phi,"ps":ps,"E0":E0*self.population,"I0":I0*self.population}
        print(iter)
        print("find params")
        print(inits)
        
        self.inits = inits
        return inits
        
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

        r0_0, r0_1 = numpyro.param("fitted_r0_0",1.), numpyro.param("fitted_r0_1",1.)
        r0     = numpyro.sample("r0", dist.TruncatedNormal(r0_0,r0_1, low=0.75,high=4.0))
        
        sigma_0, sigma_1 = numpyro.param("fitted_sigma_0",1*0.25), numpyro.param("fitted_sigma_1",1*(1-0.25))
        sigma  = numpyro.sample("sigma", dist.Beta(sigma_0, sigma_1) ) #--prior for sigma, the latent period

        phi_0, phi_1 = numpyro.param("fitted_phi_0",1*0.25), numpyro.param("fitted_phi_1",1*(1-0.25))
        phi    = numpyro.sample("phi"  , dist.Beta(phi_0,phi_1))  #--prior for phi, the percent hospitalized during the infectious period

        gamma = gamma_preset                                             #--gamma, the infectious period, is preset by the user
        kappa = kappa_preset                                             #--kappa, the duration of time in the hospitalization period before movng to R, is preset by the user

        ps_0, ps_1 = numpyro.param("fitted_ps_0",1*0.25), numpyro.param("fitted_ps_1",1*(1-0.25))
        ps    = numpyro.sample("ps", dist.Beta(ps_0,ps_1) )  #--prior for ps, the percent hospitalized during the infectious period
       
        E0_0, E0_1 = numpyro.param("fitted_E0_0",2.5), numpyro.param("fitted_E0_1",1.)
        E0   = numpyro.sample("E0", dist.TruncatedNormal(E0_0,E0_1, low=1,high=10))
        E0   = E0/self.population

        I0_0, I0_1 = numpyro.param("fitted_I0_0",2.5), numpyro.param("fitted_I0_1",1.)
        I0   = numpyro.sample("I0",dist.TruncatedNormal(I0_0,I0_1, low=1,high=10))
        I0   = I0/self.population

        S0 = ps*1. - I0                                                  #--The proportion of susceptible individuals at time 0
        H0 = 0                                                           #--The proportion of hospitalized individuals at time 0
        R0 = 1. - S0 - I0 - E0                                           #--The proportion of removed/recovered individuals at time 0

        beta = r0*gamma*(1./ps)                                          #--The effectice contact rate

        dynamic_params = (sigma, beta, gamma, kappa, phi)
        init = (S0,E0,I0,H0,R0, H0)
        states = self.shoot_trajectories(dynamic_params,init,times)

        
        #--Record these states for posthoc evaluation if needed
        states = numpyro.deterministic("states",states)

        #--Extract the proportion of incident hospitalizations
        inc_hosp_proportion = states[:,-1]

        #--The proportion of incident hospitalizations cannot be smaller than machine epsilon or greater than 1
        inc_hosp_proportion = jnp.clip(inc_hosp_proportion,a_min=jnp.finfo(float).eps,a_max= ps)

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
                hj_values = self.humanjudgment_data[:,1]
                hj_times  = self.humanjudgment_data[:,0]

                min_value, max_value = jnp.min(hj_values), jnp.max(hj_values)
                min_time, max_time = jnp.min(hj_times), jnp.max(hj_times)
                
                if len(hj_times)>10:
                    prob = numpyro.sample("bound", self.lilypad_dist(preds=self.humanjudgment_data), obs = joint_peak_data )

            if self.past_season_data is not None:
                numpyro.sample("past_season_peak", self.lilypad_dist(preds=self.past_season_data[:,1]), obs = joint_peak_data[1] )
                numpyro.sample("past_season_time", self.lilypad_dist(preds=self.past_season_data[:,0]), obs = joint_peak_data[0] )
        else:
            #--surveillance data
            if self.surveillance_data is None:
                pass
            else:
                T    = len(self.surveillance_data)        #--T is the number of time units of surveillance data that we have. T < total_window_of_observation
                mask = ~jnp.isnan(self.surveillance_data) #--This is used to exclude missing surveillance data fro mthe log likelihood

                surv_conc = numpyro.sample("surv_conc", dist.Exponential(1.))
                with numpyro.handlers.mask(mask=mask[:T]):
                    numpyro.sample("ll_surveillance", dist.NegativeBinomial2(inc_hosps[:T], surv_conc ), obs = self.surveillance_data)

    def fit_priors(self):
        def fit_beta(data):
            import numpy as np
            try:
                import scipy.stats
                a,b,_,_ = scipy.stats.beta.fit(data,floc=0,fscale=1)
                return (a,b)
            except: #--if the MLE solver the use MOM
                m = np.mean(data)
                v = np.var(data)

                if v < m*(1-m):
                
                    h = ( ((m*(1-m))/v)-1. )
                    a = m*h
                    b = (1-m)*h
                else:
                    a,b = 0.5,0.5
                return (a,b)
        def fit_normal(data):
            import scipy.stats
            a,b = scipy.stats.norm.fit(data)
            return (a,b)

        self.fitted_r0_0, self.fitted_r0_1 = fit_normal(self.prior_samples["r0"])
        self.fitted_E0_0, self.fitted_E0_1 = fit_normal(self.prior_samples["E0"])
        self.fitted_I0_0, self.fitted_I0_1 = fit_normal(self.prior_samples["I0"])

        self.fitted_sigma_0, self.fitted_sigma_1 = fit_beta(self.prior_samples["sigma"])
        self.fitted_phi_0  , self.fitted_phi_1 = fit_beta(self.prior_samples["phi"])
        self.fitted_ps_0   , self.fitted_ps_1 = fit_beta(self.prior_samples["ps"])

    def mix_with_vague(self):
        def fit_beta(data):
            import numpy as np
            try:
                import scipy.stats
                a,b,_,_ = scipy.stats.beta.fit(data,floc=0,fscale=1)
                return (a,b)
            except: #--if the MLE solver the use MOM
                m = np.mean(data)
                v = np.var(data)

                if v < m*(1-m):
                
                    h = ( ((m*(1-m))/v)-1. )
                    a = m*h
                    b = (1-m)*h
                else:
                    a,b = 0.5,0.5
                return (a,b)
            
        def fit_normal(data):
            import scipy.stats
            a,b = scipy.stats.norm.fit(data)
            return (a,b)

        
        self.prior_r0_0, self.prior_r0_1 = 0. , 10.
        self.prior_E0_0, self.prior_E0_1 = 1, 2.
        self.prior_I0_0, self.prior_I0_1 = 1, 2.
        
        self.prior_sigma_0, self.prior_sigma_1 = 1*(0.5), 1*(0.5)
        self.prior_phi_0  , self.prior_phi_1   = 1*(0.5), 1*(0.5)
        self.prior_ps_0   , self.prior_ps_1    = 1*(0.5), 1*(0.5)

        def mix_beta(a,b,c,d):
            import numpy as np
            a,b = fit_beta(0.5*np.random.beta(a,b,1000) + 0.5*np.random.beta(c,d,1000))
            return a,b

        def mix_normal(a,b,c,d):
            import numpy as np
            a,b = fit_normal(0.5*np.random.normal(a,b,1000) + 0.5*np.random.normal(c,d,1000))
            return a,b
 
        self.fitted_r0_0, self.fitted_r0_1 = mix_normal(self.fitted_r0_0, self.fitted_r0_1,self.prior_r0_0,self.prior_r0_1)
        self.fitted_E0_0, self.fitted_E0_1 = mix_normal(self.fitted_E0_0, self.fitted_E0_1,self.prior_E0_0,self.prior_E0_1)
        self.fitted_I0_0, self.fitted_I0_1 = mix_normal(self.fitted_I0_0, self.fitted_I0_1,self.prior_I0_0,self.prior_I0_1)
       
        self.fitted_sigma_0, self.fitted_sigma_1 = mix_beta(self.fitted_sigma_0, self.fitted_sigma_1,self.prior_sigma_0,self.prior_sigma_1)
        self.fitted_phi_0, self.fitted_phi_1     = mix_beta(self.fitted_phi_0, self.fitted_phi_1,self.prior_phi_0,self.prior_phi_1)
        self.fitted_ps_0, self.fitted_ps_1       = mix_beta(self.fitted_ps_0, self.fitted_ps_1,self.prior_ps_0,self.prior_ps_1)

        
    #--custom distribution to define crowd region
    import numpyro
    class lilypad_dist(numpyro.distributions.Distribution):
        def __init__(self,preds,method="silverman", *args, **kwargs):
            import jax.numpy as jnp
            super().__init__(*args, **kwargs)

            self.preds  = preds
            self.method = method
            
        def sample(self, key, sample_shape=()):
            raise NotImplementedError

        def log_prob(self,values):
            import jax.numpy as jnp
            import jax
            
            kde = jax.scipy.stats.gaussian_kde(self.preds.T, bw_method= self.method )
            return kde.logpdf(values.T).reshape(-1,)
       
    def fit_model(self, prior=None, print_summary=True):
        import numpyro
        from numpyro.infer import NUTS, MCMC, init_to_value, init_to_median, init_to_uniform, init_to_feasible
        import numpy as np

        if self.humanjudgment_data is not None:
            self.find_initial_params()
            
            #--fit HJ data for prior specification
            nuts_kernel = NUTS(self.model_specificiation, dense_mass=True, max_tree_depth=10, init_strategy = init_to_value(values=self.inits))

            mcmc        = MCMC( nuts_kernel , num_warmup=4000, num_samples=4000,progress_bar=True)
            mcmc.run(self.rng_key, extra_fields=('potential_energy',), prior=True)

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
            self.prior_samples = stripped_samples

            #--approximate and mix priors
            self.fit_priors()
            #self.mix_with_vague()

            #--update model to train on surveillance data
            updated_model = numpyro.handlers.substitute(self.model_specificiation
                                                        , {"fitted_r0_0"    :self.fitted_r0_0   , "fitted_r0_1": self.fitted_r0_1
                                                           ,"fitted_sigma_0":self.fitted_sigma_0, "fitted_sigma_1":self.fitted_sigma_1
                                                           ,"fitted_phi_0"  :self.fitted_phi_0  , "fitted_phi_1":self.fitted_phi_1
                                                           ,"fitted_ps_0"   :self.fitted_ps_0   , "fitted_ps_1":self.fitted_ps_1
                                                           ,"fitted_E0_0"   :self.fitted_E0_0   , "fitted_E0_1":self.fitted_E0_1
                                                           ,"fitted_I0_0"   :self.fitted_I0_0   , "fitted_I0_1":self.fitted_I0_1})

            nuts_kernel = NUTS(updated_model, dense_mass=True, max_tree_depth=10)
            mcmc        = MCMC( nuts_kernel , num_warmup=4000, num_samples=4000,progress_bar=True)
            mcmc.run(self.rng_key, extra_fields=('potential_energy',), prior=False)

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

        else:
            nuts_kernel = NUTS(self.model_specificiation, dense_mass=True, max_tree_depth=10)#, init_strategy = init_to_value(values=self.inits))
            mcmc        = MCMC( nuts_kernel , num_warmup=4000, num_samples=4000,progress_bar=True)
            mcmc.run(self.rng_key, extra_fields=('potential_energy',), prior=False)

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
            
     
    def compute_quantiles(self,qs = [2.5, 10., 25.0, 37.5, 50., 62.5, 75., 90., 97.5],attribute = "inc_hosps"):
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
    surv_data = generate_data(rng_key = rng_key, noise = 20)    
    inc_hosps, noisy_hosps, time_at_peak, peak_value = surv_data.simulate_surveillance_data()

    
    
    #--generate simulated human judgment predictins
    hj_data  = generate_humanjudgment_prediction_data(true_incident_hospitalizations = inc_hosps
                                                      , noise = 100.
                                                      , number_of_humanjudgment_predictions=100
                                                      , time_at_peak = time_at_peak
                                                      , rng_key  = rng_key
                                                      , bias_time = 0
                                                      , span=14) 
    noisy_time_at_peaks, noisy_peak_values, noisy_human_predictions = hj_data.generate_predictions()


    #--generate twopast seasons at random
    sim_r0 = 1.75 + (np.random.random(size=(2,)) - 0.5)

    rng_keys     = [random.PRNGKey(20201017), random.PRNGKey(19870420)]    #--seed for random number generation
    past_season_data = []
    for season,r0 in enumerate(sim_r0):
        surv_data = generate_data(rng_key = rng_keys[season], r0=sim_r0[season], noise = 10.)
        past_season_inc_hosps, past_season_noisy_hosps, past_season_time_at_peak, past_season_peak_value = surv_data.simulate_surveillance_data()

        noisy_time = np.argmax(past_season_noisy_hosps)
        noisy_peak = np.max(past_season_noisy_hosps)
        past_season_data.append( [noisy_time,noisy_peak] )
    past_season_data = 1.*np.array(past_season_data)

    #--fit model
    cut = time_at_peak - 7*2
    
    forecast = chimeric_forecast(rng_key = rng_key
                                 , surveillance_data = noisy_hosps[:cut]
                                 , past_season_data  = past_season_data 
                                 , humanjudgment_data = noisy_human_predictions)
    forecast.fit_model()

    forecast2 = chimeric_forecast(rng_key = rng_key
                                 , surveillance_data = noisy_hosps[:cut]
                                 , past_season_data  = past_season_data 
                                 , humanjudgment_data = None)
    forecast2.fit_model()

    #--compute quantiles for incident hospitalizations
    quantiles_for_incident_hosps1 = forecast.compute_quantiles()

    #--compute quantiles for incident hospitalizations
    quantiles_for_incident_hosps2 = forecast2.compute_quantiles()

    fig,axs = plt.subplots(1,2)

    ax = axs[0]
    
    times = np.arange(0,210)
    ax.plot(times, inc_hosps, color = "black")
    ax.scatter(times[:cut],noisy_hosps[:cut], color="black",s=5)

    ax.scatter( noisy_human_predictions[:,0], noisy_human_predictions[:,1], color = "red", marker = "s", s=5, alpha=0.80 )
    ax.scatter( past_season_data[:,0], past_season_data[:,1], color = "purple", marker = "o", s=5, alpha=0.80 )
    
    
    ax.fill_between(times, quantiles_for_incident_hosps1["2.500"], quantiles_for_incident_hosps1["97.500"],alpha=0.15, color="blue")
    ax.fill_between(times, quantiles_for_incident_hosps1["25.000"], quantiles_for_incident_hosps1["75.000"],alpha=0.15, color="blue")
    ax.plot(quantiles_for_incident_hosps1["50.000"],color="blue")
    
    ax = axs[1]

    times = np.arange(0,210)
    ax.plot(times, inc_hosps, color = "black")
    ax.scatter(times[:cut],noisy_hosps[:cut], color="black",s=5)

    ax.scatter( past_season_data[:,0], past_season_data[:,1], color = "purple", marker = "o", s=5, alpha=0.80 )
    
    ax.fill_between(times, quantiles_for_incident_hosps2["2.500"], quantiles_for_incident_hosps2["75.000"],alpha=0.15, color="red")
    ax.fill_between(times, quantiles_for_incident_hosps2["2.500"], quantiles_for_incident_hosps2["97.500"],alpha=0.15, color="red")
    ax.plot(quantiles_for_incident_hosps2["50.000"],color="red")

    plt.show()
