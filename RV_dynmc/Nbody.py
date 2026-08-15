import numpy as np

from .Data import ETV_Data

si = {'AU': 149597870700, 'days': 24*3600, 'Msun':1.988409870698051e+30}


class OutputTime:
    '''
    A single unique time at which the Nbody integration needs to produce
    direct output, together with every (dataset, index) pair that has a
    datum at this time -- e.g. two RV instruments with a simultaneous
    point, or an RV and an astrometric point taken together, both end up
    as separate entries on the same OutputTime.

    Built once per integrate() call by collect_output_times(), so that
    outputs() can write results straight to the right dataset/index without
    re-searching every dataset's time array at every timestep.
    '''

    __slots__ = ('t', 'entries')

    def __init__(self, t):
        self.t = t
        self.entries = []  # list of (data, index) pairs

    def add_entry(self, data, index):
        self.entries.append((data, index))

    def __repr__(self):
        return f'OutputTime(t={self.t}, n_entries={len(self.entries)})'


class ETVWindow:
    '''
    A single eclipse-search window: the (t_start, t_end) interval to scan
    for one predicted eclipse epoch of one ETV_Data instance, together with
    `index`, the position of that epoch in the dataset's own arrays
    (data.epochs, data.model_mid_times, etc.) -- mirroring what OutputTime
    does for directly-sampled data, so the refined root(s) can be written
    straight back to the right place with no extra bookkeeping.

    Each window is expected to contain exactly one eclipse: at most two
    contact-point roots (ingress/egress) and at most one minimum-separation
    root. This holds as long as window_half_width is chosen well under half
    the orbital period; Nbody.refine_events raises a clear error if a scan
    ever turns up more than that, since it means a window is spanning more
    than one eclipse.
    '''

    __slots__ = ('t_start', 't_end', 'data', 'index')

    def __init__(self, t_start, t_end, data, index):
        self.t_start = t_start
        self.t_end = t_end
        self.data = data
        self.index = index

    def __repr__(self):
        return f'ETVWindow(t_start={self.t_start}, t_end={self.t_end}, index={self.index})'


class Nbody:
    '''
    Base class for Nbody integration backends. Performs Nbody integration to
    specified times, calculates the relevant observable quantities from this
    to feed back to the data classes.

    Timestep handling follows a single-pass, monotonic-forward design and is
    entirely backend-agnostic, implemented once here:
      - RV/Phot/astrometry data supply exact times they need output at;
        these are grouped by unique time into OutputTime nodes (each
        recording which dataset/index needs writing there), merged into one
        sorted sequence, and the simulation is stepped through them in
        order exactly once, dispatching directly to the right dataset/index
        with no per-timestep search.
      - ETV data instead supply approximate eclipse epochs. Around each one
        an ETVWindow is opened (carrying that epoch's index, same idea as
        OutputTime), scanned to bracket the true root(s) (contact points
        and/or minimum separation), and those brackets are refined with
        safeguarded Newton's method run on a private copy of the
        simulation -- so refinement never disturbs or reorders the main
        forward pass. Each window is expected to contain exactly one
        eclipse; refine_events raises if a scan finds more roots than that
        allows.

    Everything above is generic across integrators. What differs between
    backends (rebound, or any future integrator) is only *how* a simulation
    object is built, stepped forward, copied, and queried for particle
    state/derivatives -- those pieces are left as hooks below for a child
    class such as Nbody_rebound to implement. This class should not be
    instantiated directly.
    '''

    def __init__(self, bodies, datas, **integrator_kwargs):
        self.bodies = bodies
        self.datas = datas
        self.integrator_kwargs = integrator_kwargs
        self.sim = None

    # ------------------------------------------------------------------
    # Backend hooks -- must be implemented by a child class
    # ------------------------------------------------------------------
    def setup_simulation(self, theta):
        '''
        Build and return a fresh simulation object for parameter vector
        `theta`, using self.bodies and self.integrator_kwargs.
        '''
        raise NotImplementedError('setup_simulation must be implemented by an Nbody backend subclass')

    def step_to(self, sim, t):
        '''
        Advance `sim` forward in place to exactly time t.
        '''
        raise NotImplementedError('step_to must be implemented by an Nbody backend subclass')

    def copy_sim(self, sim):
        '''
        Return an independent copy of `sim` that can be integrated forward
        separately without affecting the original.
        '''
        raise NotImplementedError('copy_sim must be implemented by an Nbody backend subclass')

    def get_particles(self, sim):
        '''
        Return the particle/body state collection from `sim`, in whatever
        form the Data subclasses' obtain_sim_outputs() expects.
        '''
        raise NotImplementedError('get_particles must be implemented by an Nbody backend subclass')

    def get_particle_velocity(self, sim, body_id):
        '''
        Return the line-of-sight (radial) velocity of the body with the
        given id at the current state of `sim`, in physical units (i.e.
        already converted out of the simulation's internal unit system).
        Given the sky-plane = x-y, observer at z -> -infinity looking
        towards +z convention, this is the z velocity component
        (positive = receding).
        '''
        raise NotImplementedError('get_particle_velocity must be implemented by an Nbody backend subclass')

    def separation_and_derivative(self, sim, body_a, body_b):
        '''
        Return (sky_sep, d(sky_sep)/dt) between body_a and body_b at the
        current state of `sim`, where sky_sep is the separation projected
        onto the sky plane (x-y; see line_of_sight_sign for the z
        convention). This is what eclipse contact points are defined on.
        '''
        raise NotImplementedError('separation_and_derivative must be implemented by an Nbody backend subclass')

    def separation_derivatives(self, sim, body_a, body_b):
        '''
        Return (d(sky_sep)/dt, d^2(sky_sep)/dt^2) between body_a and
        body_b at the current state of `sim`, for Newton's method on the
        minimum-sky-separation root.
        '''
        raise NotImplementedError('separation_derivatives must be implemented by an Nbody backend subclass')

    def scan_window(self, sim, t_end, data):
        '''
        Advance `sim` forward in place from its current time to t_end,
        monitoring the sky-projected separation between data.body_a and
        data.body_b as finely as the backend allows, and return bracketing
        intervals (sign changes) as {'contact': [(t_lo, t_hi), ...],
        'extremum': [(t_lo, t_hi), ...]} for:
          - contact points: sky_sep - (R_a + R_b) changes sign
          - minimum separation: d(sky_sep)/dt changes sign
        If data.front_body is set, brackets should only be recorded while
        that body is the one nearer the observer (see line_of_sight_sign),
        so that only the matching conjunction (primary or secondary) is
        picked up.
        '''
        raise NotImplementedError('scan_window must be implemented by an Nbody backend subclass')

    def line_of_sight_sign(self, sim, body_a, body_b):
        '''
        Sign convention for which of body_a/body_b is nearer the observer.
        Given the sky-plane = x-y, observer at z -> -infinity looking
        towards +z (so the body with the smaller/more negative z is in
        front): return body_a.z - body_b.z. Negative means body_a is in
        front; positive means body_b is in front.
        '''
        raise NotImplementedError('line_of_sight_sign must be implemented by an Nbody backend subclass')

    # ------------------------------------------------------------------
    # Timestep bookkeeping (backend-agnostic)
    # ------------------------------------------------------------------
    def collect_output_times(self):
        '''
        Build one OutputTime per unique time needed for direct simulation
        output across every non-ETV dataset (RV, photometry, astrometry,
        ...), each carrying every (data, index) pair that has a datum at
        that time. ETV_Data is excluded -- its times are found by
        root-finding, not by direct sampling. Returned sorted by time.
        '''
        nodes = {}
        for data in self.datas:
            if isinstance(data, ETV_Data):
                continue
            for index, t in enumerate(data.get_times()):
                node = nodes.get(t)
                if node is None:
                    node = OutputTime(t)
                    nodes[t] = node
                node.add_entry(data, index)
        return sorted(nodes.values(), key=lambda node: node.t)

    def collect_etv_windows(self):
        '''
        For every ETV_Data instance, build an ETVWindow around each of its
        predicted eclipse epochs, carrying that epoch's index in the
        dataset's arrays. Sorted by t_start so windows can be interleaved
        with the direct-output times in one monotonic pass.
        '''
        windows = []
        for data in self.datas:
            if not isinstance(data, ETV_Data):
                continue
            for index, t_pred in enumerate(data.get_predicted_times()):
                t_start = t_pred - data.window_half_width
                t_end = t_pred + data.window_half_width
                windows.append(ETVWindow(t_start, t_end, data, index))
        windows.sort(key=lambda w: w.t_start)
        return windows

    def build_event_sequence(self):
        '''
        Merge direct-output OutputTimes and ETV windows into one
        time-ordered sequence of events to step through: ('output', node)
        for a direct sample, ('etv_window', window) for an eclipse window.
        '''
        events = [('output', node) for node in self.collect_output_times()]
        events += [('etv_window', window) for window in self.collect_etv_windows()]
        events.sort(key=lambda e: e[1].t if e[0] == 'output' else e[1].t_start)
        return events

    # ------------------------------------------------------------------
    # Main integration pass (backend-agnostic)
    # ------------------------------------------------------------------
    def integrate(self, theta):
        '''
        Perform the integration: step through the merged, time-ordered
        event sequence exactly once, in increasing time order. Direct
        output events dispatch straight to outputs(); ETV window events
        trigger a bracket scan followed by root refinement.
        '''
        sim = self.setup_simulation(theta)
        events = self.build_event_sequence()

        for event in events:
            if event[0] == 'output':
                _, node = event
                self.step_to(sim, node.t)
                self.outputs(sim, node)
            else:
                _, window = event
                # independent copy taken *before* the window is entered,
                # used exclusively for root refinement afterwards so the
                # main pass is never revisited or stepped backward
                snapshot = self.copy_sim(sim)
                self.step_to(sim, window.t_start)
                brackets = self.scan_window(sim, window.t_end, window.data)
                self.refine_events(snapshot, brackets, window)

        return sim

    # ------------------------------------------------------------------
    # Direct output dispatch (RV / Phot / Astrometry) -- backend-agnostic
    # ------------------------------------------------------------------
    def outputs(self, sim, node):
        '''
        Calculate relevant observables from the simulation at this
        OutputTime and hand them to each dataset that has a datum here.
        Since collect_output_times() has already grouped every dataset by
        time, this is a direct dispatch over node.entries -- no per-dataset
        time search needed. Actual observable calculation (projecting to
        RV / sky-plane position / etc.) is left to the Data subclasses.
        '''
        for data, index in node.entries:
            data.obtain_sim_outputs(sim, self, index)

    # ------------------------------------------------------------------
    # ETV: safeguarded Newton root refinement (backend-agnostic)
    # ------------------------------------------------------------------
    def refine_events(self, snapshot, brackets, window):
        '''
        Refine every bracket found during the window scan with safeguarded
        Newton's method, for both contact points and minimum separation,
        and hand the resulting times (and the epoch index they belong to)
        to window.data.

        Each window is expected to contain exactly one eclipse, so at most
        two contact-point roots and at most one minimum-separation root
        should ever come out of it.
        '''
        data = window.data
        contact_times = [
            self.safeguarded_newton(snapshot, t_lo, t_hi, data, kind='contact')
            for (t_lo, t_hi) in brackets['contact']
        ]
        extremum_times = [
            self.safeguarded_newton(snapshot, t_lo, t_hi, data, kind='extremum')
            for (t_lo, t_hi) in brackets['extremum']
        ]

        if len(contact_times) > 2:
            raise ValueError(
                f'Found {len(contact_times)} contact-point crossings in the '
                f'window [{window.t_start}, {window.t_end}] for body pair '
                f'({data.body_a.id}, {data.body_b.id}) -- expected at most 2 '
                f'(ingress/egress). This window is likely spanning more than '
                f'one eclipse; try reducing window_half_width.'
            )
        if len(extremum_times) > 1:
            raise ValueError(
                f'Found {len(extremum_times)} minimum-separation roots in '
                f'the window [{window.t_start}, {window.t_end}] for body '
                f'pair ({data.body_a.id}, {data.body_b.id}) -- expected at '
                f'most 1. This window is likely spanning more than one '
                f'eclipse; try reducing window_half_width.'
            )

        data.obtain_sim_outputs({'contact_times': contact_times,
                                  'min_sep_times': extremum_times}, window.index)

    def safeguarded_newton(self, snapshot, t_lo, t_hi, data, kind='contact',
                            tol=1e-8, max_iter=30):
        '''
        Safeguarded Newton's method: take a Newton step using the
        analytic derivative available from the simulation state (no extra
        integration needed), falling back to bisection whenever the Newton
        step would land outside the current bracket. Runs on a private
        copy of `snapshot`, so the main integration pass is never touched.

        kind='contact'  : root of sep - (R_a + R_b) = 0  (ingress/egress)
        kind='extremum' : root of d(sep)/dt = 0            (minimum separation)
        '''
        sim = self.copy_sim(snapshot)
        body_a, body_b = data.body_a, data.body_b

        def f_and_fprime(t):
            self.step_to(sim, t)
            if kind == 'contact':
                sep, dsep_dt = self.separation_and_derivative(sim, body_a, body_b)
                return sep - (body_a.radius + body_b.radius), dsep_dt
            else:
                dsep_dt, d2sep_dt2 = self.separation_derivatives(sim, body_a, body_b)
                return dsep_dt, d2sep_dt2

        f_lo, _ = f_and_fprime(t_lo)

        t = 0.5 * (t_lo + t_hi)
        for _ in range(max_iter):
            f, fprime = f_and_fprime(t)

            if fprime != 0:
                t_newton = t - f / fprime
            else:
                t_newton = t_lo - 1.0  # force bisection fallback below

            if t_lo < t_newton < t_hi:
                t_next = t_newton
            else:
                t_next = 0.5 * (t_lo + t_hi)

            if f * f_lo < 0:
                t_hi = t
            else:
                t_lo, f_lo = t, f

            if abs(t_next - t) < tol:
                return t_next
            t = t_next

        return t


class Nbody_rebound(Nbody):
    '''
    Nbody backend implemented using rebound. Provides the low-level hooks
    (simulation setup/stepping/copying/state-querying/window-scanning)
    required by the base Nbody class; all timestep bookkeeping and
    root-finding logic is inherited unchanged.
    '''

    def __init__(self, bodies, datas, integrator='ias15', **integrator_kwargs):
        super().__init__(bodies, datas, **integrator_kwargs)
        self.integrator = integrator

    def setup_simulation(self, theta):
        '''
        Build a fresh rebound.Simulation for parameter vector `theta`.
        Converting body orbital elements/masses (from `theta`) into
        sim.add(...) calls is left for the body classes / later work; only
        the integrator setup is handled here.
        '''
        import rebound

        sim = rebound.Simulation()
        sim.units = ('days', 'AU', 'Msun')
        sim.integrator = self.integrator
        ri = getattr(sim, f'ri_{self.integrator}', None)
        for key, value in self.integrator_kwargs.items():
            if ri is not None and hasattr(ri, key):
                setattr(ri, key, value)
            else:
                setattr(sim, key, value)

        # TODO: populate the simulation from theta, e.g.
        # for body in self.bodies:
        #     sim.add(**body.get_parameter(theta, 'rebound_kwargs'))

        self.sim = sim
        return sim

    def step_to(self, sim, t):
        sim.integrate(t, exact_finish_time=1)

    def copy_sim(self, sim):
        return sim.copy()

    def get_particles(self, sim):
        return sim.particles

    def get_particle_velocity(self,sim,id):
        return sim.particles[id].vz * si['AU']/si['days']

    def separation_and_derivative(self, sim, body_a, body_b):
        '''
        Sky-projected separation and its time derivative for a pair of
        bodies. Convention: the sky plane is x-y, and the observer sits at
        z -> -infinity looking towards +z, so only x and y contribute to
        the projected separation (z sets line-of-sight ordering, handled
        separately in line_of_sight_sign).
        '''
        pa, pb = sim.particles[body_a.id], sim.particles[body_b.id]
        dx, dy = pb.x - pa.x, pb.y - pa.y
        dvx, dvy = pb.vx - pa.vx, pb.vy - pa.vy
        sep = np.sqrt(dx**2 + dy**2)
        dsep_dt = (dx * dvx + dy * dvy) / sep
        return sep, dsep_dt

    def separation_derivatives(self, sim, body_a, body_b):
        '''
        d(sky_sep)/dt and d^2(sky_sep)/dt^2, for Newton's method on the
        minimum-sky-separation root g(t) = d(sky_sep)/dt = 0, where g'(t)
        needs the relative in-plane acceleration (already available from
        the simulation, no extra integration required).
        '''
        pa, pb = sim.particles[body_a.id], sim.particles[body_b.id]
        dx, dy = pb.x - pa.x, pb.y - pa.y
        dvx, dvy = pb.vx - pa.vx, pb.vy - pa.vy
        dax, day = pb.ax - pa.ax, pb.ay - pa.ay
        sep = np.sqrt(dx**2 + dy**2)
        dsep_dt = (dx * dvx + dy * dvy) / sep
        d2sep_dt2 = ((dvx**2 + dvy**2) + (dx * dax + dy * day) - dsep_dt**2) / sep
        return dsep_dt, d2sep_dt2

    def line_of_sight_sign(self, sim, body_a, body_b):
        '''
        body_a.z - body_b.z at the current state of `sim`. Negative means
        body_a is nearer the observer (in front); positive means body_b is.
        '''
        pa, pb = sim.particles[body_a.id], sim.particles[body_b.id]
        return pa.z - pb.z

    def front_body(self, sim, body_a, body_b):
        '''
        Return whichever of body_a/body_b is currently nearer the observer.
        '''
        return body_a if self.line_of_sight_sign(sim, body_a, body_b) < 0 else body_b

    def scan_window(self, sim, t_end, data):
        '''
        Integrate sim from its current time up to t_end, using a rebound
        heartbeat to monitor the separation between data.body_a and
        data.body_b at every internal integrator step, and record
        bracketing intervals (sign changes) for:
          - contact points: sep - (R_a + R_b) changes sign
          - minimum separation: d(sep)/dt changes sign
        Returns {'contact': [...], 'extremum': [...]} lists of (t_lo, t_hi)
        brackets. This mutates `sim` (already positioned at the window
        start) forward to t_end as a side effect.

        If data.front_body is set, a bracket is only kept if that body is
        the one nearer the observer at the point the sign change is
        detected -- this is what selects primary vs. secondary eclipses,
        not the order of body_a/body_b.
        '''
        body_a, body_b = data.body_a, data.body_b
        contact_brackets = []
        extremum_brackets = []
        state = {'t': None, 'contact': None, 'deriv': None}

        def heartbeat(sim_ptr):
            s = sim_ptr.contents
            sep, dsep_dt = self.separation_and_derivative(s, body_a, body_b)
            contact_indicator = sep - (body_a.radius + body_b.radius)

            if state['t'] is not None:
                in_front = (data.front_body is None
                            or self.front_body(s, body_a, body_b) is data.front_body)
                if in_front and np.sign(contact_indicator) != np.sign(state['contact']):
                    contact_brackets.append((state['t'], s.t))
                if in_front and np.sign(dsep_dt) != np.sign(state['deriv']):
                    extremum_brackets.append((state['t'], s.t))

            state['t'] = s.t
            state['contact'] = contact_indicator
            state['deriv'] = dsep_dt

        sim.heartbeat = heartbeat
        sim.integrate(t_end, exact_finish_time=1)
        sim.heartbeat = None

        return {'contact': contact_brackets, 'extremum': extremum_brackets}
