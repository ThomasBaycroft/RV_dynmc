
class Data:
    '''
    Holds data, can read the correct parameters from the simulation
    and calculate the model to compare with the data through a likelihood.
    '''

    def __init__(self):
        pass

    def load_data(self,datafile):
        pass

    def read_sim_params(self, sim):
        pass

    def calc_diffs(self):
        pass

    def likelihood(self):
        pass

class RV_Data(Data):

    def __init__(self, body):
        super().__init__()
        self.body = body

class Phot_Data(Data):

    def __init__(self):
        super().__init__()

class Gaia_epoch_Data(Data):

    def __init__(self):
        super().__init__()