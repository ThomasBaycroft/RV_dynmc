import scipy.stats as st

class Sampler:
    '''
    Runs the Bayesian sampling (MCMC, nested, etc)
    '''

    def __init__(self):
        pass

    def log_prior(self,theta):
        '''
        calculate log prior probability based on the defined priors and proposed parameters
        '''
        pass

    def log_likelihood(self,theta):
        '''
        Calculate log likelihood by running Nbody which gives outputs to each dataset each of which has its own likelihood
        '''
        pass

    def log_posterior(self, theta):

        logprior = self.log_prior(theta)
        loglike = self.log_like(theta)
                
        return logprior + loglike



class prior_none:
    def __init__(self,name):
        self.name = name
        
    def logp(self,value):
        return 0
   
class prior_gaussian:
    def __init__(self,name,mu,sig):
        self.name = name
        self.mu = mu
        self.sig = sig
        
    def logp(self,value):
        
        return st.norm(loc=self.mu, scale=self.sig).logpdf(value)
        
class prior_uniform:
    def __init__(self,name,low,scale):
        self.name = name
        self.low = low
        self.scale = scale
        
    def logp(self,value):
        return st.uniform(loc=self.low, scale=self.scale).logpdf(value)
        
class prior_loguniform:
    def __init__(self,name,low,high):
        self.name = name
        self.low = low
        self.high = high
        
    def logp(self,value):
        return st.loguniform(a=self.low, b=self.high).logpdf(value)
    
class prior_skew_gaussian:
    def __init__(self,name,loc,scale,skew):
        self.name = name
        self.a = skew
        self.scale = scale
        self.loc = loc
    
    def logp(self,value):
        return st.skewnorm.logpdf(value,a=self.a,loc=self.loc,scale=self.scale)