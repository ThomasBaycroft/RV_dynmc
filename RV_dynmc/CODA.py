from . import Data, RV_Data, Phot_Data, ETV_Data, Gaia_epoch_Data

class CODA:
    '''
    General top-level class of CODA which the user interfaces with mostly.
    '''

    bodies = []
    bodies_ids =[]
    datas = []
    sampler = None
    integrator = None

    def __init__(self):
        pass

    def add_body(self, type, parent_id=None, radius=None):
        '''
        Set the structure of the system by adding bodies. `radius` is
        optional (e.g. not needed for a pure-RV/astrometry body with no
        eclipses) but is required for any body involved in ETV data, since
        eclipse contact-point timing needs R_a + R_b.
        '''
        id = len(self.bodies_ids)
        self.bodies_ids.append(id)
        parent = None
        if parent_id is not None:
            parent = self.bodies[parent_id]
        if type == 'star':
            body = Star(id, parent)
        elif type == 'planet':
            body = Planet(id, parent)
        elif type == 'brown_dwarf':
            body = Brown_Dwarf(id, parent)
        elif type == 'black_hole':
            body = Black_hole(id, parent)
        else:
            raise ValueError('Invalid body type')
        body.radius = radius
        self.bodies.append(body)

    def choose_sampler(self,algorithm, **kwargs):
        '''
        Choose which sampling algorithm to use, add extra arguments to give to the sampler
        '''
        pass

    def choose_integrator(self,integrator, **kwargs):
        '''
        Choose which Nbodyintegrator to use and give any specific arguments required
        '''
        pass

    def add_rv_data(self, datafile, body_id):
        '''
        add a radial velocity datafile to the list of data
        '''
        data = RV_Data(self.bodies[body_id])
        data.load_data(datafile)
        self.datas.append(data)

    def add_etv_data(self, datafile, body_a_id, body_b_id, window_half_width,
                      front_body_id=None):
        '''
        add an eclipse-timing (ETV) datafile for the pair (body_a, body_b)
        to the list of data. `window_half_width` sets how far either side
        of each observed epoch the Nbody code scans to bracket the true
        eclipse time. `front_body_id`, if given, selects which of the two
        bodies is expected to be nearer the observer (smaller z) during
        these eclipses -- e.g. pass the secondary star's id for a primary
        eclipse dataset -- so that only the matching conjunction is found
        even if body_a/body_b also eclipse each other the other way round
        half an orbit later. Leave as None to accept either.
        '''
        front_body = self.bodies[front_body_id] if front_body_id is not None else None
        data = ETV_Data(self.bodies[body_a_id], self.bodies[body_b_id],
                         window_half_width, front_body=front_body)
        data.load_data(datafile)
        self.datas.append(data)

    def setup(self):
        #decide reference time
        #setup theta order
        #print/write a setupfile summary?
        pass






class Celestial_body:
    '''
    A class representing a generic celestial body, organised by id. 
    The child classes are the different types of bodies, by default 
    they are assumed to orbit the centre-of-mass unless a specific 
    parent body is given. This class allows to read the parameters 
    for the specific bodies directly from the simulation.
    '''

    def __init__(self, id):
        self.id = id
        self.radius = None

    def get_parameter(self,sim,parameter):
        pass

class Star(Celestial_body):

    def __init__(self, id, parent=None):
        super().__init__(id)
        self.type = 'star'
        if parent is not None:
            self.parent = parent
        else:
            self.parent = 'COM'
        
class Planet(Celestial_body):

    def __init__(self, id, parent=None):
        super().__init__(id)
        self.type = 'planet'
        if parent is not None:
            self.parent = parent
        else:
            self.parent = 'COM'   

class Brown_Dwarf(Celestial_body):

    def __init__(self, id, parent=None):
        super().__init__(id)
        self.type = 'brown_dwarf'
        if parent is not None:
            self.parent = parent
        else:
            self.parent = 'COM'

class Black_hole(Celestial_body):

    def __init__(self, id, parent=None):
        super().__init__(id)
        self.type = 'black_hole'
        if parent is not None:
            self.parent = parent
        else:
            self.parent = 'COM'