# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:17:48 2022

@author: till kaeufer

"""


log_dict={'Mstar': 'log', 'Lstar': 'log', 'Teff': 'log', 'fUV': 'log', 'pUV': 'log', 'amin': 'log', 'amax': 'log',
      'apow': 'linear', 'a_settle': 'log', 'Mg0.7Fe0.3SiO3[s]': 'linear', 'amC-Zubko[s]': 'linear', 'fPAH': 'log',
   'PAH_charged': 'linear', 'Mdisk': 'log', 'Rin': 'log', 'Rtaper': 'log', 'Rout': 'log', 'epsilon': 'linear',
   'MCFOST_H0': 'linear', 'MCFOST_BETA': 'linear', 'incl': 'linear'}#,'Dist[pc]':'linear'}

slider_dict={
    'Mstar':{
        'label':r'$log_{10}(M_{star}) [M_{sun}]$',
        'lims':[-0.69, 0.39],
        'x0':0.06,
        'priority':1}
        ,
    
    'Teff':{
        'label':r'$log_{10}(T_{eff})$',
        'lims':[3.5, 4.0], 
        'x0':3.69,
        'priority':1},
    
    'Lstar':{
        'label':r'$log_{10}(L_{star})$',
        'lims':[-1.3, 1.7],
        'x0':0.79,
        'priority':1}, 
    'fUV':{
        'label':r'$log_{10}(f_{UV})$',
        'lims':[-3, -1],
        'x0':-1.57, 
        'priority':1},
    
    'pUV':{
        'label':r'$log_{10}(p_{UV})$',
        'lims':[-0.3, 0.39],
        'x0':-0.02, 
        'priority':1},
    
    'Mdisk':{
        'label':r'$log_{10}(M_{disk})$',
        'lims':[-5, 0],
        'x0':-1.367, 
        'priority':2},
    
    'incl':{
        'label':r'$incl [Deg]$',
        'lims':[0.0, 90.0],
        'x0':20.0,
        'priority':2},
    
    'Rin':{
        'label':r'$log_{10}(R_{in}[AU])$',
        'lims':[-2.00, 2.00], 
        'x0':-1.34,
        'priority':2},
   
     'Rtaper':{
        'label':r'$log_{10}(R_{taper}[AU])$',
        'lims':[0.7, 2.5],
         'x0':1.95, 
        'priority':2},
    
    'Rout':{
        'label':r'$log_{10}(R_{out}[AU])$',
        'lims':[1.3, 3.14],
        'x0':2.556, 
        'priority':2},
    
    'epsilon':{
        'label':r'$\epsilon$',
        'lims':[0, 2.5],
        'x0':1, 
        'priority':2},
    
    'MCFOST_BETA':{
        'label':r'$\beta$',
        'lims':[0.9, 1.4],
        'x0':1.15, 
        'priority':2},
    
    'MCFOST_H0':{
        'label':'$H_0[AU]$',
        'lims':[3, 35],
        'x0':12, 
        'priority':2},    
    
    'a_settle':{
        'label':r'$log_{10}(a_{settle})$',
        'lims':[-5, -1],
        'x0':-3, 
        'priority':3},
    
    'amin':{
        'label':r'$log_{10}(a_{min})$',
        'lims':[-3, -1],
        'x0':-1.5, 
        'priority':3},
    
    
    'amax':{
        'label':r'$log_{10}(a_{max})$',
        'lims':[2.48, 4],
        'x0':3.6, 
        'priority':3},
    
    'apow':{
        'label':r'$a_{pow}$',
        'lims':[3, 5],
        'x0':3.6, 
        'priority':3},
    
    'Mg0.7Fe0.3SiO3[s]':{
        'label':r'Mg0.7Fe0.3SiO3[s]',
        'lims':[0.45, 0.7],
        'x0':0.57, 
        'priority':3},
    
    'amC-Zubko[s]':{
        'label':r'amC-Zubko[s]',
        'lims':[0.05, 0.3],
        'x0':0.18, 
        'priority':3},
    
    'fPAH':{
        'label':r'$log_{10}(f_{PAH})$',
        'lims':[-3.5, 0],
        'x0':-1.5, 
        'priority':3},
    
    'PAH_charged':{
        'label':r'$PAH_{charged}$',
        'lims':[0, 1], 
        'priority':3},
}
   