#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 16:40:35 2023

@author: thomasbaycroft
"""

import numpy as np
import rebound
import pandas as pd
import scipy.stats as st
import random
import emcee

class Sampling:
    
    def __init__(self,n_orbits,insts,n_lines):
        
        self.n_orbits = n_orbits
        self.n_insts = len(insts)
        self.n_lines = n_lines
        self.insts = [instrument(inst) for inst in insts]
        self.total_time = 0
        self.numtimes = 0
        self.t0 = 0
        self.priors = []

        
    def add_datafile(self,instrument,times,vrad,svrad,units):
        
        self.total_time += sum(times)
        self.numtimes += len(times)
        
        for inst in self.insts:
            if inst.name == instrument:
                inst.add_data(times,vrad,svrad,units)
                
    def load_default_priors(self):
        
        self.priors.append(prior_none('M0'))
        for i in range(self.n_orbits):
            self.priors.append(prior_none('M'+str(i+1)))
        for i in range(self.n_orbits):
            self.priors.append(prior_none('P'+str(i+1)))
        for i in range(self.n_orbits):
            self.priors.append(prior_none('e'+str(i+1)))
        for i in range(self.n_orbits):
            self.priors.append(prior_uniform('w'+str(i+1),0,2*np.pi))
        for i in range(self.n_orbits):
            self.priors.append(prior_uniform('W'+str(i+1),0,2*np.pi))
        for i in range(self.n_orbits):
            self.priors.append(prior_uniform('f'+str(i+1),0,2*np.pi))
        for i in range(self.n_orbits):
            self.priors.append(prior_gaussian('inc'+str(i+1),np.pi/2,0.01))
            
        for i in range(self.n_insts):
            for j in range(self.n_lines):
                self.priors.append(prior_none('vsys'+str(i)+','+str(j)))
            for j in range(self.n_lines):
                self.priors.append(prior_loguniform('jit'+str(i)+','+str(j),0.01,1000))
                
    def define_prior(self,index,name,dist,a=0,b=1):
        
        if dist=='none':
            self.priors[index] = prior_none(name)
        elif dist in ['uniform','Uniform','U','flat','Flat']:
            self.priors[index] = prior_uniform(name,a,b)
        elif dist in ['loguniform','logUniform','LU','Loguniform','LogUniform']:
            self.priors[index] = prior_loguniform(name,a,b)
        elif dist in ['Gaussian','gaussian','Normal','normal','N']:
            self.priors[index] = prior_gaussian(name,a,b)
        else:
            raise ValueError('Prior distribution type not found, use one of: none, uniform, loguniform, gaussian')
                
    def sim_params_from_theta(self,theta):
        
        model_params = np.array(theta)

        M0 = model_params[0]
        Ms, Ps, es, ws, Ws, fs, incs = [], [], [], [], [], [], []
        for i in range(self.n_orbits):
            Ms.append(model_params[1+i])
            Ps.append(model_params[1+self.n_orbits+i])
            es.append(model_params[1+self.n_orbits*2+i])
            ws.append(model_params[1+self.n_orbits*3+i])
            Ws.append(model_params[1+self.n_orbits*4+i])
            fs.append(model_params[1+self.n_orbits*5+i])
            incs.append(model_params[1+self.n_orbits*6+i])
            
        return M0,Ms,Ps,es,ws,Ws,fs,incs
        
    
    def log_like(self,theta):
        '''
        theta: {orbit parameters}[M0,M1-Mn,P1-Pn,e1-en,w1-wn,W1-Wn,f1-fn,inc1-incn] + {data parameters}[vsys1-k,jit1-k](k the number of lines)(for each instrument)
        times: bjd
        rvs: m/s
        errs: m/s
        masses in solar mass, periods in days, angles in radians, vsys, offsets and jitter in m/s
        '''
        logL = 0
            
        model_params = np.array(theta)

        M0 = model_params[0]
        Ms, Ps, es, ws, Ws, fs, incs = [], [], [], [], [], [], []
        for i in range(self.n_orbits):
            Ms.append(model_params[1+i])
            Ps.append(model_params[1+self.n_orbits+i])
            es.append(model_params[1+self.n_orbits*2+i])
            ws.append(model_params[1+self.n_orbits*3+i])
            Ws.append(model_params[1+self.n_orbits*4+i])
            fs.append(model_params[1+self.n_orbits*5+i])
            incs.append(model_params[1+self.n_orbits*6+i])
            
        num_orb_par = 7*self.n_orbits + 1
        num_dat_par = 2*self.n_lines
            
        physical = True
        for i,e in enumerate(es):
            if e<0 or e>=1:
                physical = False
        for i,M in enumerate([M0]+Ms):
            if M<=0:
                physical = False
        for i,P in enumerate(Ps):
            if P<=0:
                physical = False
        # for i,inc in enumerate(incs):
        #     if abs(inc) > 0.1:
        #         physical = False
                
        if physical:
            for i,inst in enumerate(self.insts):
                dat=inst.datas[0]
                times = dat.times
                # rvs = dat.vrad
                # errs = dat.svrad
                
                model_rvs = self.sim_rvs(M0,Ms,Ps,es,ws,Ws,fs,incs,self.t0,times)
                
                for j,dat in enumerate(inst.datas):
                    # times = dat.times
                    rvs = dat.vrad
                    errs = dat.svrad
                    
                    vsys,jitter = model_params[num_orb_par + num_dat_par*i + j], model_params[num_orb_par + num_dat_par*i + self.n_lines + j]
                    
                    error = errs**2 + jitter**2
        
                    # Compute residuals to model
                    res = (rvs - model_rvs[j] - vsys)

                    logL += np.sum(st.norm(scale=np.sqrt(error)).logpdf(res))
        else:
            logL += -np.inf
    
        return logL
        
    def sim_rvs(self,M0,Ms,Ps,es,ws,Ws,fs,incs,t0,times,body=0):
        
        sim = self.sim_setup(M0,Ms,Ps,es,ws,Ws,fs,incs)
        
        Time = times - t0
        
        RVs = []
        RVs2 = []
        for t in Time:
            sim.integrate(t)                
            vel = -sim.particles[0].vz*(1.496*10**11)/(24*3600)
            RVs.append(vel)
            vel2 = -sim.particles[1].vz*(1.496*10**11)/(24*3600)
            RVs2.append(vel2)
            
        return [RVs,RVs2]
    
    def sim_setup(self,M0,Ms,Ps,es,ws,Ws,fs,incs):
        
        sim = rebound.Simulation()
        sim.units = ('days', 'AU', 'Msun')
        
        sim.add(m=M0)
        for i,m in enumerate(Ms): 
            sim.add(m=m,P=Ps[i],e=es[i],omega=ws[i],Omega=Ws[i],M=fs[i],inc=incs[i])
        
        sim.move_to_com()
        
        return sim

    def log_prior(self,theta):
        
        logP = 0
        for prior,param in zip(self.priors,theta):
            logP += prior.logp(param)
            
        return logP
    
    def log_post(self,theta):
        
        logprior = self.log_prior(theta)
        loglike = self.log_like(theta)
        
        return logprior + loglike
        
        
    def run_emcee(self,chains,steps,x0,prior=False,t0=None,mult=1):
        self.x0 = x0
        
        if t0==None:
            self.t0 = self.total_time/self.numtimes
        else:
            self.t0=t0
        print(self.t0)
            
        if prior:
            self.sampler = emcee.EnsembleSampler(chains, len(x0.T), self.log_post,moves=[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2)])
            print('Sampler set-up, priors included')
        else:
            self.sampler = emcee.EnsembleSampler(chains, len(x0.T), self.log_like,moves=[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2)])
            print('Sampler set-up, no priors included')
            
        try:
            self.sampler.run_mcmc(self.x0, nsteps=steps, progress=True)
        except ValueError:
            print('Initial state has a large condition number, perturbing initalisaition within 1-sigma...')
            self.perturb_x0(times=mult)
            try:
                self.sampler.run_mcmc(self.x0, nsteps=steps, progress=True)
            except ValueError:
                print('Initial state still has a large condition number, will now skip initial state check and run anyway.')
                self.sampler.run_mcmc(self.x0, nsteps=steps, progress=True, skip_initial_state_check=True)
        
        return self.sampler
    
    def perturb_x0(self,times=1):
        
        add = self.x0*0
        for i in range(len(self.x0.T)):
            sigma = np.std(self.x0[:,i])
            add[:,i] = np.random.randn(len(self.x0[:,i]))*sigma
            
        self.x0 += add
        
   
class prior_none:
    def __init__(self,name):
        self.name = name
        
    def logp(self,value):
        return 0
   
class prior_gaussian:
    def __init__(self,name,mu,sig):
        self.name = name
        self.mu = mu
        self.sig = sig
        
    def logp(self,value):
        
        return st.norm(loc=self.mu, scale=self.sig).logpdf(value)
        
class prior_uniform:
    def __init__(self,name,low,scale):
        self.name = name
        self.low = low
        self.scale = scale
        
    def logp(self,value):
        return st.uniform(loc=self.low, scale=self.scale).logpdf(value)
        
class prior_loguniform:
    def __init__(self,name,low,high):
        self.name = name
        self.low = low
        self.high = high
        
    def logp(self,value):
        return st.loguniform(a=self.low, b=self.high).logpdf(value)

class instrument:
    
    def __init__(self,name):
        
        self.name = name
        self.datas = []
        
    def add_data(self,times,vrad,svrad,units):
        
        self.datas.append(data(times,vrad,svrad,units))
        
class data:
    
    def __init__(self,times,vrad,svrad,units):
        self.units = units
        
        if self.units == 'kms':
            self.mult = 1000
        else:
            self.mult = 1
        self.times = np.array(times)
        self.vrad = np.array(vrad)*self.mult
        self.svrad = np.array(svrad)*self.mult
        