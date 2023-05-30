#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:00:45 2023

@author: thomasbaycroft
"""

from . import Sampling
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

class RV_dynmc:
    
    def __init__(self,n_orbits,n_lines,insts,chains,steps):
        
        self.n_orbits = n_orbits
        self.insts = insts
        self.n_insts = len(insts)
        self.n_lines = n_lines
        self.chains
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
                names.append('vsys'+str(i)+'_'+self.insts[j])
            for i in range(self.n_lines):    
                names.append('jit'+str(i)+'_'+self.insts[j])
                
        self.colnames = names
        
        
    def add_rv_data(self,insts,datas,units):
        '''
        insts: list of strings, names of the instruments
        datas: list of (dataframes with columns: bjd, vrad1, svrad1, ..., vradn, svradn (for n lines)) (one dataframe for each instrument)
        units: str either 'kms' or 'ms'
        '''
        
        for i,inst in enumerate(insts):
            for j in range(self.n_lines):
                self.Sampling.add_datafile(inst, datas[i]['bjd'], datas[i]['vrad'+str(j+1)], datas[i]['svrad'+str(j+1)], units)

    def x0_from_kima(self,posts,num,binary=True,precessing=False):
        
        x0 = []
        indices = random.sample(list(posts.index),self.chains)
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
                x0k.append(float(posts['KO_w0'][ind]))
                for i in range(self.n_orbits-1):
                    x0k.append(float(posts['w'+str(i)][ind]))  
                x0k.append(np.random.rand()*2*np.pi)
                for i in range(self.n_orbits-1):
                    x0k.append(np.random.rand()*2*np.pi)
                x0k.append(float(posts['KO_phi0'][ind]))
                for i in range(self.n_orbits-1):
                    x0k.append(float(posts['phi'+str(i)][ind]))  
                x0k.append(0)
                for i in range(self.n_orbits-1):
                    x0k.append(0)  
            else:
                x0k.append(float(posts['M_pri'][ind])) #may need altering?
                for i in range(self.n_orbits):
                    x0k.append(float(posts['M_pl_'+str(i)][ind]))
                for i in range(self.n_orbits):
                    x0k.append(float(posts['P'+str(i)][ind]))
                for i in range(self.n_orbits):
                    x0k.append(float(posts['ecc'+str(i)][ind]))  
                for i in range(self.n_orbits):
                    x0k.append(float(posts['w'+str(i)][ind]))  
                for i in range(self.n_orbits):
                    x0k.append(np.random.rand()*2*np.pi)
                for i in range(self.n_orbits):
                    x0k.append(float(posts['phi'+str(i)][ind]))  
                for i in range(self.n_orbits):
                    x0k.append(0)  
             
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
    
        self.x0 = np.array(mean) + np.random.randn(self.chains, self.ndim)*sigma
        
    def run(self,prior=False):
        
        sam = self.Sampling.run_emcee(self.chains, self.steps, self.x0,prior=prior)
        
        self.sample_analysis = Samples_analysis(self.colnames, sam)
        
        return self.sample_analysis

class Samples_analysis:
    
    def __init__(self,names,sampler,n_chains):
        self.colnames = names
        self.sampler = sampler
        self.n_chains = n_chains
        self.samples = self.sampler.get_chains()
        self.chains = list(np.linspace(0,self.n_chains-1,self.n_chains,dtype=int))
        
    def pop_chains(self,indices):
        for ind in indices:
            self.chains.pop(ind)
            
    def restore_chains(self):
        self.chains = list(np.linspace(0,self.n_chains-1,self.n_chains,dtype=int))
        
    def plot_logP(self,evolve=True,burnin=0,thin=1):
        
        y = self.samples.get_log_prob(discard=burnin,thin=thin)
        
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        if evolve:
            ax.plot(y[:,self.chains])
        else:
            ax.plt(y.T[self.chains])
        
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel('logP', fontsize=16)
        
        
        
    
        
        