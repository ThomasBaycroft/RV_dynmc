import numpy as np


class Data:
    '''
    Holds data, can read the correct parameters from the simulation
    and calculate the model to compare with the data through a likelihood.
    '''

    def __init__(self):
        self.times = None

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

    def likelihood(self):
        '''
        Calculate likelihood based on diffs
        '''
        pass

class RV_Data(Data):

    def __init__(self, body):
        super().__init__()
        self.body = body

class Phot_Data(Data):

    def __init__(self):
        super().__init__()

class ETV_Data(Data):
    '''
    Eclipse-timing data for a pair of bodies. Unlike RV/Phot/astrometry,
    ETV epochs aren't sampled directly -- the Nbody code instead opens a
    window around each approximate observed eclipse time, scans through it
    with a heartbeat to bracket the true contact-point/minimum-separation
    root(s), and refines those roots with safeguarded Newton's method.
    '''

    def __init__(self, body_a, body_b, window_half_width, front_body=None):
        super().__init__()
        self.body_a = body_a
        self.body_b = body_b
        # rough observed mid-eclipse times (e.g. one per cycle in the
        # datafile) used only to centre the search window -- the precise
        # model times come out of the root-finding in Nbody
        self.epochs = None
        # half-width (in the same time units as the simulation) of the
        # window scanned around each entry in self.epochs
        self.window_half_width = window_half_width
        # which of body_a/body_b is expected to be nearer the observer
        # (smaller z) during the eclipses in this dataset -- e.g. body_b for
        # a primary eclipse of body_a, body_a for the secondary. This is
        # NOT interchangeable with swapping body_a and body_b: the
        # sky-projected separation and R_a+R_b are symmetric in body_a/
        # body_b, so swapping them has no effect on which conjunction is
        # found. front_body is what actually distinguishes primary from
        # secondary, via a line-of-sight check during the window scan.
        # Leave as None to accept whichever body is in front (e.g. if a
        # dataset mixes both, or the window is known to bracket only one
        # conjunction anyway).
        if front_body is not None and front_body not in (body_a, body_b):
            raise ValueError('front_body must be body_a, body_b, or None')
        self.front_body = front_body

    def get_predicted_times(self):
        '''
        Approximate eclipse-window centres to search around, taken directly
        from the observed epochs loaded from the datafile.
        '''
        if self.epochs is None:
            return np.array([])
        return np.atleast_1d(self.epochs)

class Gaia_epoch_Data(Data):

    def __init__(self):
        super().__init__()

class Gaia_auxiliary_Data(Data):

    def __init__(self):
            super().__init__()
