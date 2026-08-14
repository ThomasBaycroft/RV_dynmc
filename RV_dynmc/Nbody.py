import numpy as np

from .Data import ETV_Data


class Nbody:
    '''
    Base class for Nbody integration backends. Performs Nbody integration to
    specified times, calculates the relevant observable quantities from this
    to feed back to the data classes.

    Timestep handling follows a single-pass, monotonic-forward design and is
    entirely backend-agnostic, implemented once here:
      - RV/Phot/astrometry data supply exact times they need output at;
        these are merged into one sorted, de-duplicated array and the
        simulation is stepped through them in order exactly once.
      - ETV data instead supply approximate eclipse epochs. Around each one
        a window is opened, scanned to bracket the true root(s) (contact
        points and/or minimum separation), and those brackets are refined
        with safeguarded Newton's method run on a private copy of the
        simulation -- so refinement never disturbs or reorders the main
        forward pass.

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
        Sorted, de-duplicated union of every time that needs direct
        simulation output (RV, photometry, astrometry, ...). ETV_Data is
        excluded -- its times are found by root-finding, not by direct
        sampling.
        '''
        times = [data.get_times() for data in self.datas
                  if not isinstance(data, ETV_Data)]
        times = [t for t in times if len(t) > 0]
        if len(times) == 0:
            return np.array([])
        return np.unique(np.concatenate(times))

    def collect_etv_windows(self):
        '''
        For every ETV_Data instance, build a (t_start, t_end, data) window
        around each of its predicted eclipse epochs, sorted by t_start so
        they can be interleaved with the direct-output times in one
        monotonic pass.
        '''
        windows = []
        for data in self.datas:
            if not isinstance(data, ETV_Data):
                continue
            for t_pred in data.get_predicted_times():
                t_start = t_pred - data.window_half_width
                t_end = t_pred + data.window_half_width
                windows.append((t_start, t_end, data))
        windows.sort(key=lambda w: w[0])
        return windows

    def build_event_sequence(self):
        '''
        Merge direct-output times and ETV windows into one time-ordered
        sequence of events to step through: ('output', t) for a direct
        sample, ('etv_window', t_start, t_end, data) for an eclipse window.
        '''
        events = [('output', t) for t in self.collect_output_times()]
        events += [('etv_window', t_start, t_end, data)
                   for (t_start, t_end, data) in self.collect_etv_windows()]
        events.sort(key=lambda e: e[1])
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
                _, t = event
                self.step_to(sim, t)
                self.outputs(sim, t)
            else:
                _, t_start, t_end, data = event
                # independent copy taken *before* the window is entered,
                # used exclusively for root refinement afterwards so the
                # main pass is never revisited or stepped backward
                snapshot = self.copy_sim(sim)
                self.step_to(sim, t_start)
                brackets = self.scan_window(sim, t_end, data)
                self.refine_events(snapshot, brackets, data)

        return sim

    # ------------------------------------------------------------------
    # Direct output dispatch (RV / Phot / Astrometry) -- backend-agnostic
    # ------------------------------------------------------------------
    def outputs(self, sim, t):
        '''
        Calculate relevant observables from the simulation at time t based
        on each data type and system architecture, and hand them to the
        corresponding data instance. Actual observable calculation
        (projecting to RV / sky-plane position / etc.) is left to the Data
        subclasses -- here we just extract and dispatch the raw state.
        '''
        for data in self.datas:
            if isinstance(data, ETV_Data):
                continue
            if t not in data.get_times():
                continue
            data.obtain_sim_outputs(self.get_particles(sim))

    # ------------------------------------------------------------------
    # ETV: safeguarded Newton root refinement (backend-agnostic)
    # ------------------------------------------------------------------
    def refine_events(self, snapshot, brackets, data):
        '''
        Refine every bracket found during the window scan with safeguarded
        Newton's method, for both contact points and minimum separation,
        and hand the resulting times to `data`.
        '''
        contact_times = [
            self.safeguarded_newton(snapshot, t_lo, t_hi, data, kind='contact')
            for (t_lo, t_hi) in brackets['contact']
        ]
        extremum_times = [
            self.safeguarded_newton(snapshot, t_lo, t_hi, data, kind='extremum')
            for (t_lo, t_hi) in brackets['extremum']
        ]
        data.obtain_sim_outputs({'contact_times': contact_times,
                                  'min_sep_times': extremum_times})

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
