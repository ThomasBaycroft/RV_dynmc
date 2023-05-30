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
          
    def define_priors(self):
        
        self.priors.append(prior_none('M0'))
        for i in range(self.n_orbits):
            self.priors.append(prior_none('M'+str(i+1)))
        for i in range(self.n_orbits):
            self.priors.append(prior_none('P'+str(i+1)))
        for i in range(self.n_orbits):
            self.priors.append(prior_none('e'+str(i+1)))
        for i in range(self.n_orbits):
            self.priors.append(prior_none('w'+str(i+1)))
        for i in range(self.n_orbits):
            self.priors.append(prior_none('W'+str(i+1)))
        for i in range(self.n_orbits):
            self.priors.append(prior_none('f'+str(i+1)))
        for i in range(self.n_orbits):
            self.priors.append(prior_Gaussian('inc'+str(i+1),0,0.01))
            
        for i in range(self.n_insts):
            for j in range(self.n_lines):
                self.priors.append(prior_none('vsys'+str(i)+','+str(j)))
            for j in range(self.n_lines):
                self.priors.append(prior_none('jit'+str(i)+','+str(j)))
                
    def log_like(self,theta):
        '''
        theta: {orbit parameters}[M0,M1-Mn,P1-Pn,e1-en,w1-wn,W1-Wn,f1-fn,inc1-incn] + {data parameters}[vsys1-k,jit1-k](k the number of lines)(for each instrument)
        times: bjd
        rvs: m/s
        errs: m/s
        masses in solar mass, periods in days, angles in radians, vsys, offsets and jitter in m/s
        '''
        logL = 0
            
        # Note that you have to separate the vsys and jitter from theta
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
        for i,inc in enumerate(incs):
            if abs(inc) > 0.1:
                physical = False
                
        if physical:
            for i,inst in enumerate(self.insts):
                dat=inst.datas[0]
                times = dat.times
                rvs = dat.vrad
                errs = dat.svrad
                
                model_rvs = self.sim_rvs(M0,Ms,Ps,es,ws,Ws,fs,incs,self.t0,times)
                
                for j,dat in enumerate(inst.datas):
                    # times = dat.times
                    rvs = dat.vrad
                    errs = dat.svrad
                    
                    # model = modelfunc(model_param, times)
                    
                    # Compute increased (squared) error
                    
                    
                    vsys,jitter = model_params[num_orb_par + num_dat_par*i + j], model_params[num_orb_par + num_dat_par*i + self.n_lines + j]
                    
                    
                    error = errs**2 + jitter**2
        
                    # Compute residuals to model
                    res = (rvs - model_rvs[j] - vsys)
                    # print(np.sqrt(sum(res**2)/len(res)))
                    
                    logL += np.sum(st.norm(scale=np.sqrt(error)).logpdf(res))
        else:
            logL += -np.inf
    
        return logL
        
    def sim_rvs(self,M0,Ms,Ps,es,ws,Ws,fs,incs,t0,times,body=0):
        
        sim = rebound.Simulation()
        sim.units = ('days', 'AU', 'Msun')
        
        sim.add(m=M0)
        for i,m in enumerate(Ms): 
            sim.add(m=m,P=Ps[i],e=es[i],omega=ws[i],Omega=Ws[i],f=fs[i],inc=incs[i])
        
        sim.move_to_com()
        
        Time = times - t0
        
        RVs = []
        RVs2 = []
        for t in Time:
            sim.integrate(t)                
            vel = -sim.particles[0].vy*(1.496*10**11)/(24*3600)
            RVs.append(vel)
            vel2 = -sim.particles[1].vy*(1.496*10**11)/(24*3600)
            RVs2.append(vel2)
            
        return [RVs,RVs2]


    def log_prior(self,theta):
        
        logP = 0
        for prior,param in zip(self.priors,theta):
            logP += prior.logp(param)
            
        return logP
    
    def log_post(self,theta):
        
        logprior = self.log_prior(theta)
        loglike = self.log_like(theta)
        
        return logprior + loglike
        
        
    def run_emcee(self,chains,steps,x0,prior=False,t0=None):
        
        if t0==None:
            self.t0 = self.total_time/self.numtimes
        else:
            self.t0=t0
            
        if prior:
            self.define_priors()
            sampler = emcee.EnsembleSampler(chains, len(x0.T), self.log_post)
        else:
            sampler = emcee.EnsembleSampler(chains, len(x0.T), self.log_like)
        sampler.run_mcmc(x0, nsteps=steps, progress=True)
        
        return sampler
        
   
class prior_none:
    def __init__(self,name):
        self.name = name
        
    def logp(self,value):
        return 0
   
class prior_Gaussian:
    def __init__(self,name,mu,sig):
        self.name = name
        self.mu = mu
        self.sig = sig
        
    def logp(self,value):
        
        return st.norm(loc=self.mu, scale=self.sig).logpdf(value)
        
class prior_uniform:
    def __init__(self,name,low,high):
        self.name = name
        self.low = low
        self.high = high
        
    def logp(self,value):
        return st.uniform(a=self.low, b=self.high).logpdf(value)
        
class prior_loguniform:
    def __init__(self,name,low,high):
        self.name = name
        self.low = low
        self.high = high
        
    def logp(self,value):
        st.loguniform(a=self.low, b=self.high).logpdf(value)

class instrument:
    
    def __init__(self,name):
        
        self.name = name
        self.datas = []
        
    def add_data(self,times,vrad,svrad,units):
        
        self.datas.append(data(times,vrad,svrad,units))
        
class data:
    
    def __init__(self,times,vrad,svrad,units):
        
        if units == 'kms':
            mult = 1000
        else:
            mult = 1
        self.times = np.array(times)
        self.vrad = np.array(vrad)*mult
        self.svrad = np.array(svrad)*mult
        