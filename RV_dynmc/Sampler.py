import scipy.stats as st

class Sampler:
    '''
    Runs the Bayesian sampling (MCMC, nested, etc)
    '''

    def __init__(self):
        pass

    def log_prior(self,params):
        '''
        calculate log prior probability based on the defined priors and proposed parameters
        '''
        pass

    def log_likelihood(self,params):
        '''
        Calculate log likelihood by running Nbody which gives outputs to each dataset each of which has its own likelihood
        '''
        pass

    def log_posterior(self, params):

        logprior = self.log_prior(params)
        loglike = self.log_like(params)
                
        return logprior + loglike



class prior_none:
    '''
    An improper/unbounded ("flat") prior: logp is 0 everywhere. Has no
    natural initial draw -- rvs() raises, so ParameterRegistry.build_params
    requires an explicit starting value (via its p0 argument) for any
    parameter registered with this prior.
    '''
    def __init__(self,name):
        self.name = name
        
    def logp(self,value):
        return 0

    def rvs(self):
        raise NotImplementedError(
            f"prior_none ('{self.name}') is an improper/unbounded prior with no "
            f"natural initial draw -- supply an explicit starting value for this "
            f"parameter via p0 in ParameterRegistry.build_params()."
        )
   
class prior_gaussian:
    def __init__(self,name,mu,sig):
        self.name = name
        self.mu = mu
        self.sig = sig
        
    def logp(self,value):
        
        return st.norm(loc=self.mu, scale=self.sig).logpdf(value)

    def rvs(self):
        return st.norm(loc=self.mu, scale=self.sig).rvs()
        
class prior_uniform:
    def __init__(self,name,low,scale):
        self.name = name
        self.low = low
        self.scale = scale
        
    def logp(self,value):
        return st.uniform(loc=self.low, scale=self.scale).logpdf(value)

    def rvs(self):
        return st.uniform(loc=self.low, scale=self.scale).rvs()
        
class prior_loguniform:
    def __init__(self,name,low,high):
        self.name = name
        self.low = low
        self.high = high
        
    def logp(self,value):
        return st.loguniform(a=self.low, b=self.high).logpdf(value)

    def rvs(self):
        return st.loguniform(a=self.low, b=self.high).rvs()
    
class prior_skew_gaussian:
    def __init__(self,name,loc,scale,skew):
        self.name = name
        self.a = skew
        self.scale = scale
        self.loc = loc
    
    def logp(self,value):
        return st.skewnorm.logpdf(value,a=self.a,loc=self.loc,scale=self.scale)

    def rvs(self):
        return st.skewnorm.rvs(a=self.a,loc=self.loc,scale=self.scale)