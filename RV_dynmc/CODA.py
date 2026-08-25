from . import Data, RV_Data, Phot_Data, ETV_Data, Gaia_epoch_Data
from .Parameters import ParameterRegistry
from .Nbody import Nbody_rebound
from .Sampler import Sampler

class CODA:
    '''
    General top-level class of CODA which the user interfaces with mostly.
    '''

    def __init__(self):
        self.bodies = []
        self.bodies_ids = []
        self.datas = []
        self.sampler = None
        self.integrator = None
        self.registry = ParameterRegistry()

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

    def set_free(self, owner, param, prior):
        '''
        Register `param` on `owner` (a body or a Data instance) as a free
        parameter with the given prior -- see ParameterRegistry.set_free.
        '''
        return self.registry.set_free(owner, param, prior)

    def set_fixed(self, owner, param, value):
        '''
        Register `param` on `owner` as fixed at a constant value -- see
        ParameterRegistry.set_fixed.
        '''
        self.registry.set_fixed(owner, param, value)

    def set_derived(self, owner, param, depends_on, func):
        '''
        Register `param` on `owner` as computed from other registered
        parameters rather than sampled directly -- see
        ParameterRegistry.set_derived. Typical use: a body's mass computed
        as a mass ratio times another body's mass.
        '''
        self.registry.set_derived(owner, param, depends_on, func)

    def build_params(self, p0=None):
        '''
        Build an initial flat parameter vector from every registered free
        parameter -- see ParameterRegistry.build_params.
        '''
        return self.registry.build_params(p0)

    def choose_integrator(self,integrator, **kwargs):
        '''
        Choose which Nbody backend to use and give any backend-specific
        arguments required. Currently only 'rebound' is supported. Note:
        this `integrator` argument selects the *backend* (rebound vs. a
        future alternative); the specific numerical scheme within rebound
        (e.g. 'ias15', 'whfast') is a separate kwarg forwarded to
        Nbody_rebound -- e.g. choose_integrator('rebound', integrator='whfast').
        '''
        if integrator == 'rebound':
            self.integrator = Nbody_rebound(self.bodies, self.datas, self.registry, **kwargs)
        else:
            raise NotImplementedError(
                f"Nbody backend {integrator!r} is not supported yet -- only "
                f"'rebound' is currently implemented."
            )

    def choose_sampler(self,algorithm='base', **kwargs):
        '''
        Choose which sampling algorithm to use, add extra arguments to give
        to the sampler. 'base' gives a plain Sampler exposing
        log_prior/log_likelihood/log_posterior -- useful for manual
        testing, or for driving an external sampler by hand. Algorithm-
        specific subclasses (dynesty, emcee, ...) will be added later.
        '''
        if self.integrator is None:
            raise RuntimeError('choose_integrator(...) must be called before choose_sampler(...)')
        if algorithm == 'base':
            self.sampler = Sampler(self.integrator, self.registry, **kwargs)
        else:
            raise NotImplementedError(f"Sampler algorithm {algorithm!r} is not implemented yet.")

    def add_rv_data(self, datafile, body_id):
        '''
        add a radial velocity datafile to the list of data
        '''
        data = RV_Data(self.bodies[body_id], self.registry)
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
                         window_half_width, self.registry, front_body=front_body)
        data.load_data(datafile)
        self.datas.append(data)

    def setup(self):
        #decide reference time
        #params order is now handled by self.registry (see set_free/set_fixed/set_derived/build_params)
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

    def get_parameter(self, param, params, registry, cache=None):
        '''
        Look up this body's value for `param` (e.g. 'mass', 'period',
        'eccentricity', 'inclination', 'Omega', 'omega', 'true_longitude',
        'radius') from `registry`, given the current flat `params` vector.
        Thin convenience wrapper around registry.resolve() -- this body
        doesn't need to know or care whether `param` is free, fixed, or
        derived; the registry handles that.
        '''
        return registry.resolve(self, param, params, cache)

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