import numpy as np


class ParameterRegistry:
    '''
    Central registry mapping every (owner, parameter_name) pair used by the
    model -- a Celestial_body's mass/orbital elements, or a Data instance's
    nuisance parameters like Jitter/vsys -- to how its value should be
    obtained:
      - 'free'    : occupies a slot in the flat sampled parameter vector
                    `params`, with an associated prior (see Sampler.py's
                    prior_* classes) giving its log-prior density and an
                    initial-value draw via prior.rvs().
      - 'fixed'   : a constant value, not sampled.
      - 'derived' : computed from other already-registered (owner, param)
                    pairs via an arbitrary function -- e.g. a body's mass
                    computed as (reference body's mass) * (a free mass
                    ratio), rather than sampled directly.

    `owner` can be any hashable object -- typically a Celestial_body or a
    Data instance -- there is nothing body-specific about this class.

    Whatever builds a simulation from `params` (e.g.
    Nbody_rebound.setup_simulation) doesn't need to know or care which mode
    a given (owner, param) is in -- it always just calls resolve(), and the
    registry looks up the mode itself.

    Ordering constraint: a 'derived' parameter's dependencies must already
    be resolvable by the time it is resolved. In practice this means
    depending only on parameters of bodies added earlier (lower `id`) than
    the body being derived -- which is also required for rebound's
    primary-before-secondary hierarchy, so no separate dependency-graph
    solver is needed; resolving bodies in `self.bodies` order is enough.
    '''

    def __init__(self):
        self.specs = {}     # (owner, param) -> spec tuple
        self.priors = []    # prior objects, in params-index order
        self.n_free = 0


    def set_free(self, owner, param, prior):
        '''
        Register (owner, param) as free: it occupies the next slot in the
        flat `params` vector, with `prior` (a prior_* instance from
        Sampler.py) giving its log-prior density and, via prior.rvs(), a
        way to draw an initial value for build_params(). Returns the
        assigned index, mostly useful for debugging/inspection.
        '''
        index = self.n_free
        self.specs[(owner, param)] = ('free', index)
        self.priors.append(prior)
        self.n_free += 1
        return index

    def set_fixed(self, owner, param, value):
        '''
        Register (owner, param) as fixed at a constant value -- not part
        of `params` at all.
        '''
        self.specs[(owner, param)] = ('fixed', value)

    def set_derived(self, owner, param, depends_on, func):
        '''
        Register (owner, param) as computed by func(*values), where
        `values` are the resolved values of each (owner, param) pair
        listed in `depends_on`, in that order. Each dependency should
        belong to a body added earlier (lower id) than `owner` if `owner`
        is a body -- see class docstring.

        Example (mass ratio instead of an absolute mass):
            registry.set_free(star_b, 'mass_ratio', prior_uniform('q_b', 0, 1))
            registry.set_derived(star_b, 'mass',
                                  depends_on=[(star_a, 'mass'), (star_b, 'mass_ratio')],
                                  func=lambda m_a, q: m_a * q)
        '''
        self.specs[(owner, param)] = ('derived', depends_on, func)

    def is_registered(self, owner, param):
        return (owner, param) in self.specs

    def resolve(self, owner, param, params, cache=None):
        '''
        Return the actual value of (owner, param) given the current flat
        `params` vector, resolving 'derived' dependencies recursively.

        `cache`, if given, is a plain dict that should be created fresh
        once per setup_simulation()-style call and shared across every
        resolve() call within it -- this avoids re-resolving a value that
        several derived parameters depend on more than once. Never reuse a
        cache across different `params` vectors, since every resolved
        value depends on `params`.
        '''
        key = (owner, param)
        if cache is not None and key in cache:
            return cache[key]

        if key not in self.specs:
            raise KeyError(
                f'No parameter registered for {param!r} on {owner!r}. '
                f'Use set_free/set_fixed/set_derived to register it first.'
            )

        spec = self.specs[key]
        mode = spec[0]
        if mode == 'fixed':
            value = spec[1]
        elif mode == 'free':
            value = params[spec[1]]
        elif mode == 'derived':
            _, depends_on, func = spec
            args = [self.resolve(dep_owner, dep_param, params, cache)
                    for (dep_owner, dep_param) in depends_on]
            value = func(*args)
        else:
            raise ValueError(f'Unknown parameter mode {mode!r} for {key!r}')

        if cache is not None:
            cache[key] = value
        return value

    def build_params(self, p0=None):
        '''
        Build an initial flat `params` vector: for every free parameter,
        use the value given in `p0` (a dict keyed by (owner, param)) if
        provided, otherwise draw one from its prior via prior.rvs().

        Raises NotImplementedError (propagated from the prior's own
        .rvs()) if a free parameter has no override in `p0` and its prior
        has no natural draw -- e.g. prior_none, an improper/unbounded
        prior -- supply an explicit value for those via p0.
        '''
        p0 = p0 or {}
        params = np.zeros(self.n_free)
        for (owner, param), spec in self.specs.items():
            if spec[0] != 'free':
                continue
            index = spec[1]
            if (owner, param) in p0:
                params[index] = p0[(owner, param)]
            else:
                params[index] = self.priors[index].rvs()
        return params
