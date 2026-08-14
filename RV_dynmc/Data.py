
class Data:
    '''
    Holds data, can read the correct parameters from the simulation
    and calculate the model to compare with the data through a likelihood.
    '''

    def __init__(self):
        pass

    def load_data(self,datafile):
        '''
        read in the datafile storing values
        '''
        pass

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

    def __init__(self):
        super().__init__()

class Gaia_epoch_Data(Data):

    def __init__(self):
        super().__init__()