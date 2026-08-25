import numpy as np
import pandas as pd
import scipy.stats as st


class Data:
    '''
    Holds data, can read the correct parameters from the simulation
    and calculate the model to compare with the data through a likelihood.
    '''

    def __init__(self, registry):
        self.times = None
        self.model_vals = np.array([])
        self.observed_vals = np.array([])
        self.errors = np.array([])

        self.registry = registry

    def load_data(self,datafile):
        '''
        read in the datafile storing values
        '''
        pass

    def get_times(self):
        '''
        Return the array of times at which the Nbody code needs to produce
        direct output for this dataset (e.g. RV epochs, photometric epochs,
        astrometric epochs). Not used by ETV_Data, which instead works from
        predicted eclipse windows -- see ETV_Data.get_predicted_times().
        '''
        if self.times is None:
            return np.array([])
        return np.atleast_1d(self.times)

    def obtain_sim_outputs(self, outputs):
        '''
        Function run by the nbody code when it reaches the relevant timestep it adds 
        the relevant outputs to this class
        '''
        pass

    def calc_diffs(self):
        '''
        Use the simulation outputs as have been fed in by the nbody to calculate the simulated dataset
        Calculate the differences between simulated and observed data
        '''
        pass

    def log_likelihood(self, params, distribution='Gaussian'):
        '''
        Calculate likelihood based on diffs
        '''
        self.calc_diffs(params)

        res = self.observed_vals - self.model_vals
        if distribution == 'Gaussian':
            logL = np.sum(st.norm(scale=self.errors).logpdf(res))
        else:
            raise NotImplementedError(f'Likelihood using a {distribution} function is not available')
        return logL

    def get_parameter(self, param, params, cache=None):
        '''
        Look up this dataset's value for `param` (e.g. 'jitter', 'vsys', ...)
        from self.registry, given the current flat `params` vector.
        '''
        return self.registry.resolve(self, param, params, cache)

class RV_Data(Data):

    def __init__(self, body, registry, datafile=None,header=0,skiprows=[1],sep='\t',units='kms'):
        super().__init__(registry)
        self.body = body
        self.datatype = 'RV'
        if datafile is not None:
            self.load_data(datafile,header,skiprows,sep,units)
        self.Jitter = 0
        self.vsys = 0


    def load_data(self,datafile,header=0,skiprows=[1],sep='\t',units='kms'):
        data = pd.read_csv(datafile,header=header,skiprows=skiprows,sep=sep,usecols=[0,1,2],names=['times','vrad','svrad'])
        self.times = data.times
        mult = units == 'kms'
        mult *= 999
        mult += 1
        self.vrad = data.vrad * mult
        self.observed_vals = self.vrad.copy()
        self.svrad = data.svrad * mult

        self.model_rvs = np.zeros_like(self.vrad)

    def calc_diffs(self,params):
        cache = {}
        if self.registry.is_registered(self,'jitter'):
            self.Jitter = self.registry.resolve(self, 'jitter', params, cache)
        self.vsys = self.registry.resolve(self, 'vsys', params, cache)

        self.errors = np.hypot(self.svrad,self.Jitter)
        self.model_vals = self.model_rvs + self.vsys

    def obtain_sim_outputs(self,sim,Nbody,index):
        self.model_rvs[index] = Nbody.get_particle_velocity(sim,self.body.id)


class Phot_Data(Data):

    def __init__(self, registry):
        super().__init__(registry)

class ETV_Data(Data):
    '''
    Eclipse-timing data for a pair of bodies. Unlike RV/Phot/astrometry,
    ETV epochs aren't sampled directly -- the Nbody code instead opens a
    window around each approximate observed eclipse time, scans through it
    with a heartbeat to bracket the true contact-point/minimum-separation
    root(s), and refines those roots with safeguarded Newton's method.
    '''

    def __init__(self, body_a, body_b, window_half_width, registry, front_body=None, datafile=None, contacts=False, header=0, skiprows=[1], sep='\t', units='days'):
        super().__init__(registry)
        self.datatype = 'ETV'
        self.body_a = body_a
        self.body_b = body_b
        self.contacts = contacts
        self.epochs = None
        self.window_half_width = window_half_width
        if datafile is not None:
            self.load_data(datafile,header=header,skiprows=skiprows,sep=sep,units=units)
        # which of body_a/body_b is expected to be nearer the observer (smaller z) during the eclipses in this dataset -- e.g. body_b for a primary eclipse of body_a, body_a for the secondary.
        if front_body is not None and front_body not in (body_a, body_b):
            raise ValueError('front_body must be body_a, body_b, or None')
        self.front_body = front_body

        #penalisation in logL of an eclipse not occurring
        self.ne_penalty = -13.4 #-np.inf

        self.Jitter = 0

    def get_predicted_times(self):
        '''
        Approximate eclipse-window centres to search around, taken directly
        from the observed epochs loaded from the datafile.
        '''
        if self.epochs is None:
            return np.array([])
        return np.atleast_1d(self.epochs)

    def load_data(self,datafile,header=0,skiprows=[1],sep='\t',units='days'):
        if self.contacts:
            data = pd.read_csv(datafile,header=header,skiprows=skiprows,sep=sep,usecols=[0,1,2,3,4,5],names=['mid_times','mt_errs','ingress_times','it_errs','egress_times','et_errs'])
            self.ingress_times = data.ingress_times
            self.egress_times = data.egress_times
            self.it_errs = data.it_errs
            self.et_errs = data.et_errs
        else:
            data = pd.read_csv(datafile,header=header,skiprows=skiprows,sep=sep,usecols=[0,1],names=['mid_times','mt_errs'])
        self.mid_times = data.mid_times
        self.mt_errs = data.mt_errs
        self.epochs = self.mid_times.copy()

        self.model_mid_times = np.zeros_like(self.mid_times)
        if self.contacts:
            self.model_ingress_times = np.zeros_like(self.mid_times)
            self.model_egress_times = np.zeros_like(self.mid_times)

            self.observed_vals = np.c_[self.mid_times,self.ingress_times,self.egress_times]
        else:
            self.observed_vals = self.mid_times.copy()

    def obtain_sim_outputs(self, model_times, index):
        '''
        Write the refined root(s) found for one eclipse window into this
        dataset's model arrays at `index` (the epoch's position, matching
        self.epochs / self.model_mid_times / etc. -- see ETVWindow).

        Nbody.refine_events already guarantees at most 1 minimum-separation
        root and at most 2 contact-point roots (raising otherwise, since
        that would mean the window spans more than one eclipse). It does
        not guarantee at least that many: a trial parameter set can simply
        fail to eclipse at all. Those are legitimate outcomes during
        sampling, not bugs, so missing roots are recorded as NaN.
        '''
        contact_times = model_times['contact_times']
        extremum_times = model_times['min_sep_times']

        self.model_mid_times[index] = extremum_times[0] if len(extremum_times) == 1 else np.nan

        if self.contacts:
            self.model_ingress_times[index] = contact_times[0] if len(contact_times) >= 2 else np.nan
            self.model_egress_times[index] = contact_times[1] if len(contact_times) >= 2 else np.nan

    def calc_diffs(self,params):
        cache = {}
        if self.registry.is_registered(self,'jitter'):
            self.Jitter = self.registry.resolve(self, 'jitter', params, cache)

        if self.contacts:
            self.model_vals = np.c_[self.model_mid_times,self.model_ingress_times,self.model_egress_times]
            self.errors = np.hypot(np.c_[self.mt_errs,self.it_errs,self.et_errs],self.Jitter)
        else:
            self.model_vals = self.model_mid_times
            self.errors = np.hypot(self.mt_errs,self.Jitter)

    def log_likelihood(self,params,distribution='Gaussian'):
        '''
        Calculate likelihood based on diffs. Matches the base Data
        signature (params first) so a generic caller -- e.g. Sampler --
        can call data.log_likelihood(params) uniformly across every
        dataset type; unlike the base class this overrides the NaN
        (non-eclipsing trial configuration) handling.
        '''
        self.calc_diffs(params)

        res = self.observed_vals - self.model_vals
        if distribution == 'Gaussian':
            logLs = st.norm(scale=self.errors).logpdf(res)
        else:
            raise NotImplementedError(f'Likelihood using a {distribution} function is not available')
        logLs[np.isnan(logLs)] = self.ne_penalty
        logL = np.sum(logLs)
        return logL

class Gaia_epoch_Data(Data):

    def __init__(self, registry):
        super().__init__(registry)

class Gaia_auxiliary_Data(Data):

    def __init__(self, registry):
        super().__init__(registry)
