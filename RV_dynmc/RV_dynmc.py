#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:00:45 2023

@author: thomasbaycroft
"""

from . import Sampling
import random
import numpy as np
import matplotlib.pyplot as plt

class RV_dynmc:
    
    def __init__(self,n_orbits,n_lines,insts,chains,steps):
        '''
        Initialise with:
            n_orbits: int; number of orbits being included in the fit
            n_lines: int; number of stars with RV time-series (1 for a single star or SB1)
            insts: list of strings; names of the instruments used for RV data collection
            chains: int; number of MCMC cahins
            steps: number of steps to run MCMC for
        '''
        
        
        self.n_orbits = n_orbits
        self.insts = insts
        self.n_insts = len(insts)
        self.n_lines = n_lines
        self.n_chains = chains
        self.n_steps = steps
        self.ndim = 1 + 7*self.n_orbits + 2*self.n_lines*self.n_insts
        
        self.Sampling = Sampling.Sampling(n_orbits,insts,n_lines)
        
        self.get_column_names()
        
    def get_column_names(self):
        names = []
        
        names.append('M0')
        for i in range(self.n_orbits):
            names.append('M'+str(i+1))
        for i in range(self.n_orbits):
            names.append('P'+str(i+1))
        for i in range(self.n_orbits):
            names.append('e'+str(i+1))  
        for i in range(self.n_orbits):
            names.append('w'+str(i+1))
        for i in range(self.n_orbits):
            names.append('W'+str(i+1))
        for i in range(self.n_orbits):
            names.append('phi'+str(i+1)) 
        for i in range(self.n_orbits):
            names.append('inc'+str(i+1))  
        
        
        for j in range(self.n_insts):
            for i in range(self.n_lines):
                names.append('vsys'+str(i+1)+'_'+self.insts[j])
            for i in range(self.n_lines):    
                names.append('jit'+str(i+1)+'_'+self.insts[j])
                
        self.colnames = names
        
    def define_priors(self,parameters,distributions,par1s,par2s):
        
        self.Sampling.load_default_priors()
        
        for name,dist,a,b in zip(parameters,distributions,par1s,par2s):
            try:
                ind = self.colnames.index(name)
            except ValueError:
                raise ValueError('Parameter '+name+' is not valid, you can see the names of the parameters with RV_dynmc.colnames')
            self.Sampling.define_prior(ind, name, dist,a,b)
        
        
    def add_rv_data(self,insts,datas,units):
        '''
        insts: list of strings, names of the instruments
        datas: list of (dataframes with columns: bjd, vrad1, svrad1, ..., vradn, svradn (for n lines)) (one dataframe for each instrument)
        units: str either 'kms' or 'ms'
        '''
        
        for i,inst in enumerate(insts):
            if self.n_lines==1:
                self.Sampling.add_datafile(inst, datas[i]['bjd'], datas[i]['vrad'], datas[i]['svrad'], units[i])
            else:
                for j in range(self.n_lines):
                    self.Sampling.add_datafile(inst, datas[i]['bjd'], datas[i]['vrad'+str(j+1)], datas[i]['svrad'+str(j+1)], units[i])

    def x0_from_kima(self,posts,binary=True,precessing=False):
        
        x0 = []
        indices = random.sample(list(posts.index),self.n_chains)
        for k,ind in enumerate(indices):
            x0k = []
            if binary:
                x0k.append(float(posts['M_pri'][ind]))
                x0k.append(float(posts['M_sec'][ind]))
                for i in range(self.n_orbits-1):
                    x0k.append(float(posts['M_pl_'+str(i)][ind]))
                if precessing:
                    x0k.append(float(posts['KO_Pano0'][ind]))
                else:
                    x0k.append(float(posts['KO_P0'][ind]))
                for i in range(self.n_orbits-1):
                    x0k.append(float(posts['P'+str(i)][ind]))
                x0k.append(float(posts['KO_ecc0'][ind]))
                for i in range(self.n_orbits-1):
                    x0k.append(float(posts['ecc'+str(i)][ind]))  
                x0k.append(float(posts['KO_w0'][ind]) % (2*np.pi) )
                for i in range(self.n_orbits-1):
                    x0k.append(float(posts['w'+str(i)][ind]) % (2*np.pi))  
                x0k.append(np.random.rand()*2*np.pi)
                for i in range(self.n_orbits-1):
                    x0k.append(np.random.rand()*2*np.pi)
                x0k.append(float(posts['KO_phi0'][ind]) % (2*np.pi))
                for i in range(self.n_orbits-1):
                    x0k.append(float(posts['phi'+str(i)][ind]) % (2*np.pi))  
                x0k.append(np.pi/2+np.random.randn()*0.01)
                for i in range(self.n_orbits-1):
                    x0k.append(np.pi/2+np.random.randn()*0.01)  
            else:
                x0k.append(float(posts['M_pri'][ind])) #may need altering?
                for i in range(self.n_orbits):
                    x0k.append(float(posts['M_pl_'+str(i)][ind]))
                for i in range(self.n_orbits):
                    x0k.append(float(posts['P'+str(i)][ind]))
                for i in range(self.n_orbits):
                    x0k.append(float(posts['ecc'+str(i)][ind]))  
                for i in range(self.n_orbits):
                    x0k.append(float(posts['w'+str(i)][ind]) % (2*np.pi))  
                for i in range(self.n_orbits):
                    x0k.append(np.random.rand()*2*np.pi)
                for i in range(self.n_orbits):
                    x0k.append(float(posts['phi'+str(i)][ind]) % (2*np.pi))  
                for i in range(self.n_orbits):
                    x0k.append(np.pi/2+np.random.randn()*0.01)  
             
            x0k.append(float(posts['vsys'][ind]))
            for i in range(self.n_lines-1):
                x0k.append(float(posts['vsys_sec'][ind]))
            if self.n_insts == 1:
                x0k.append(float(posts['extra_sigma'][ind]))
                for i in range(self.n_lines-1):
                    x0k.append(float(posts['extra_sigma_sec'][ind]))
            else:
                x0k.append(float(posts['jitter1'][ind]))
                for i in range(self.n_lines-1):
                    x0k.append(float(posts['jitter_sec1'][ind]))
            for j in range(1,self.n_insts):
                x0k.append(float(posts['offset'+str(j)][ind])+float(posts['vsys'][ind]))
                for i in range(self.n_lines-1):
                    x0k.append(float(posts['offset_sec'+str(j)][ind])+float(posts['vsys_sec'][ind]))
                x0k.append(float(posts['jitter'+str(j+1)][ind]))
                for i in range(self.n_lines-1):
                    x0k.append(float(posts['jitter_sec'+str(j+1)][ind]))
                    
            x0.append(x0k)
        
        self.x0 = np.array(x0)
            
    
    def x0_from_values(self,mean,sigma):
    
        self.x0 = np.array(mean) + np.random.randn(self.n_chains, self.ndim)*sigma
        
    def run(self,prior=True,mult=1):
        
        self.sam = self.Sampling.run_emcee(self.n_chains, self.n_steps, self.x0,prior=prior,mult=mult)
        
        self.sample_analysis = Samples_analysis(self.colnames, self.sam,self.n_chains,self.Sampling)
        
        return self.sample_analysis

class Samples_analysis:
    
    def __init__(self,names,sampler,n_chains,Sampling):
        self.colnames = names
        self.sampler = sampler
        self.n_chains = n_chains
        self.Sampling = Sampling
        self.samples = self.sampler.get_chain()
        self.chains = list(np.linspace(0,self.n_chains-1,self.n_chains,dtype=int))
        self.discard=0
        self.thin=1
        
    def pop_chains(self,indices):
        for ind in indices:
            self.chains.pop(ind)
            
    def restore_chains(self):
        self.chains = list(np.linspace(0,self.n_chains-1,self.n_chains,dtype=int))
        
        
    def plot_logP(self,evolve=True,discard=None,thin=None):
        
        if discard==None:
            discard = self.discard
        if thin==None:
            thin = self.thin
        
        y = self.sampler.get_log_prob(discard=discard,thin=thin)
        
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        if evolve:
            ax.plot(y[:,self.chains])
        else:
            ax.plot(y.T[self.chains])
        
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel('logP', fontsize=16)
        
    def choose_random(self):
        
        sams = self.sampler.get_chain(discard=self.discard,thin=self.thin)
        
        chain = random.sample(self.chains,1)[0]
        
        sam = random.sample(list(sams[:,chain,:]), 1)[0]
    
        return sam
        
    def random_subsample(self,num):
        
        sams = []
        for i in range(num):
            sams.append(self.choose_random())
            
        return sams
    
    def sims_subsample(self,samples):
        
        sims = []
        for theta in samples:
            M0,Ms,Ps,es,ws,Ws,fs,incs = self.Sampling.sim_params_from_theta(theta)
            sim = self.Sampling.sim_setup(M0, Ms, Ps, es, ws, Ws, fs, incs)
            sims.append(sim)
        
        return sims
    