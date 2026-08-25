import warnings
import numpy as np
import scipy.stats as st

class Sampler:
    '''
    Runs the Bayesian sampling (MCMC, nested, etc). This base class is
    algorithm-agnostic: it just exposes log_prior/log_likelihood/
    log_posterior for `params`, suitable for driving by hand or handing to
    an external sampler. Algorithm-specific subclasses (dynesty, emcee,
    ...) will wrap this with the sampler-specific setup/run machinery.
    '''

    def __init__(self, nbody, registry):
        self.nbody = nbody
        self.registry = registry

    def log_prior(self,params):
        '''
        Sum the log-prior density of every free parameter at its current
        value in `params`. self.registry.priors is already ordered to
        match `params`' indices (see ParameterRegistry.set_free), so this
        is a direct zip rather than a lookup per parameter.
        '''
        return float(sum(prior.logp(value) for prior, value in zip(self.registry.priors, params)))

    def log_likelihood(self,params):
        '''
        Run the Nbody integration for `params` (which writes outputs into
        every dataset via obtain_sim_outputs), then sum each dataset's own
        log_likelihood(params).
        '''
        try:
            self.nbody.integrate(params)
        except Exception as e:
            warnings.warn(
                f'Nbody integration failed for this parameter draw ({e!r}); '
                f'treating as zero probability (logL = -inf).'
            )
            return -np.inf

        logL = 0.0
        for data in self.nbody.datas:
            logL += data.log_likelihood(params)
        return logL

    def log_posterior(self, params):
        '''
        log_prior + log_likelihood, skipping the (expensive) Nbody
        integration entirely whenever the prior alone already rules
        `params` out -- e.g. a proposed value outside a uniform prior's
        bounds.
        '''
        logprior = self.log_prior(params)
        if not np.isfinite(logprior):
            return -np.inf

        loglike = self.log_likelihood(params)
        return logprior + loglike

class Dynesty_sampler(Sampler):

    def __init__(self, nbody, registry):
        super().__init__(nbody, registry)

    def prior_transform(self,u):
        return np.array([prior.cdf_inv(value) for prior, value in zip(self.registry.priors, u)])

    def setup_sampler(self,**kwargs):
        from dynesty import NestedSampler

        self.ndim = self.registry.nfree
        self.sampler = NestedSampler(self.log_likelihood,self.prior_transform,self.ndim,**kwargs)

    def run_sampler(self,**kwargs):
        self.sampler.run_nested(**kwargs)
        


class prior_none: ##Needed????
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


#Various priors avaialable, each has:
# logp: calculates log_pdf of the prior at the given value
# cdf_inv: calculates the inverse cumulative distribution function of the prior at the given value
# rvs: samples from the prior with size: size
class prior_gaussian:
    def __init__(self,name,mu,sig):
        self.name = name
        self.mu = mu
        self.sig = sig

        self.distribution = st.norm(loc=self.mu, scale=self.sig)
        
    def logp(self,value):
        return self.distribution.logpdf(value)

    def cdf_inv(self,value):
        return self.distribution.ppf(value)

    def rvs(self,size=1):
        return self.distribution.rvs(size)
        
class prior_uniform:
    def __init__(self,name,low,high):
        self.name = name
        self.low = low
        self.high = high
        self.scale = self.high - self.low

        self.distribution = st.uniform(loc=self.low, scale=self.scale)
        
    def logp(self,value):
        return self.distribution.logpdf(value)

    def cdf_inv(self,value):
        return value*self.scale + self.low

    def rvs(self,size=1):
        return self.distribution.rvs(size)
        
class prior_loguniform:
    def __init__(self,name,low,high):
        self.name = name
        self.low = low
        self.high = high

        self.distribution = st.loguniform(a=self.low, b=self.high)
        
    def logp(self,value):
        return self.distribution.logpdf(value)

    def cdf_inv(self,value):
        return self.distribution.ppf(value)

    def rvs(self,size=1):
        return self.distribution.rvs(size)
    
class prior_skew_gaussian:
    def __init__(self,name,loc,scale,skew):
        self.name = name
        self.a = skew
        self.scale = scale
        self.loc = loc

        self.distribution = st.skewnorm(a=self.a,loc=self.loc,scale=self.scale)
    
    def logp(self,value):
        return self.distribution.logpdf(value)

    def cdf_inv(self,value):
        return self.distribution.ppf(value)

    def rvs(self,size=1):
        return self.distribution.rvs(size)

class prior_beta:
    def __init__(self,name,a,b):
        self.name = name
        self.a = a
        self.b = b

        self.distribution = st.beta(a=self.a, b=self.b)

    def logp(self,value):
        return self.distribution.logpdf(value)
    
    def cdf_inv(self,value):
        return self.distribution.ppf(value)

    def rvs(self,size=1):
        return self.distribution.rvs(size)