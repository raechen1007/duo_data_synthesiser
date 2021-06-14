#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:50:36 2019

@author: RaeChen
"""
import pandas as pd
#import analyse
import numpy as np
from numpy.random import gamma

#data = pd.read_csv('Hazy dataset.csv')
#continuous_var = analyse.continuous_var
#hist_infor = analyse.hist_infor
#gamma_paras = analyse.gamma_paras

def categoriseData(data, var, hist_infor):
    
    new_data = data.copy()
    
    for name in var:
        series = data[name]
        replace = np.zeros(series.shape)
        division = hist_infor[(name, 'division')]

        for i in range(series.shape[0]):
            where_i = 0
            
            while where_i < (len(division) - 1):
                
                if series[i] >= division[where_i] and series[i] < division[where_i + 1]: 
                    replace[i] = where_i
                    break
                    
                else:
                    where_i += 1

        
        new_data[name] = replace
        
    return new_data

#cat_data = categoriseData(data, continuous_var, hist_infor)
#cat_data.to_csv('Mod Data_v01.csv', index = False)

'''
################################## Verifying Hypo #############################
'''

def hypoVerfying(data, cat_data, var, gamma_paras):
    
    match = {}
    _tail = {}
    
    for name in var:
        
        ori_series = data[name]
        cat_series = cat_data[name]
        
        division = gamma_paras[(name, 'division')]
        paras = gamma_paras[(name, 'paras')]
        
        restore = np.zeros(ori_series.size)
        
        for i in range(ori_series.size):
            
            value = cat_series[i]
            
            shape, scale = paras[int(value + 1)][0], paras[int(value + 1)][1]
            
            if shape == -998 or scale == -998:
                # -998: there is unique/unique value in the bin
                restore[i] = round((division[int(value)] + division[int(value + 1)])/2)
            
            elif shape == -999 or scale == -999:
                #-999: there is no avaliable value in the bin
                restore[i] = round((division[int(value)] + division[int(value + 1)])/2)
            
            else:
                if scale <= 0:
                    restore[i] = -round(gamma(shape, -scale))
                
                else:
                    restore[i] = round(gamma(shape, scale))
        
        
        score = np.corrcoef(ori_series, restore)[0][1]
        
        _tail[name] = restore
        
        match[name] = score
        
    return match, _tail

#cat_data = pd.read_csv('Mod Data_v01.csv')
#match_board, tail = hypoVerfying(data, cat_data, continuous_var, gamma_paras) 

'''
bins = {'LIMIT_BAL': 8, 'AGE': 8, 'BILL_AMT1': 15, 'BILL_AMT2': 15, 
        'BILL_AMT3': 15, 'BILL_AMT4': 15, 'BILL_AMT5': 15, 'BILL_AMT6': 15, 
        'PAY_AMT1': 20, 'PAY_AMT2': 20, 'PAY_AMT3': 20, 'PAY_AMT4': 20, 
        'PAY_AMT5': 20, 'PAY_AMT6': 20}
{'AGE': 0.9514934229827844,
 'BILL_AMT1': 0.9516196578220582,
 'BILL_AMT2': 0.9724818823163258,
 'BILL_AMT3': 0.8521647267088504, --- less than .9
 'BILL_AMT4': 0.9542780494512151,
 'BILL_AMT5': 0.9479364834898556,
 'BILL_AMT6': 0.9319563414820239,
 'LIMIT_BAL': 0.9524603232177469,
 'PAY_AMT1': 0.8960836346455922, --- less than .9
 'PAY_AMT2': 0.6851664336133343, --- less than .9, and no significant improvement
 'PAY_AMT3': 0.9072000697135898, 
 'PAY_AMT4': 0.9116419238811904,
 'PAY_AMT5': 0.9217777002682023,
 'PAY_AMT6': 0.9262835504536046}

bins = {'LIMIT_BAL': 8, 'AGE': 8, 'BILL_AMT1': 15, 'BILL_AMT2': 15, 
        'BILL_AMT3': 15, 'BILL_AMT4': 15, 'BILL_AMT5': 15, 'BILL_AMT6': 15, 
        'PAY_AMT1': 24, 'PAY_AMT2': 50, 'PAY_AMT3': 20, 'PAY_AMT4': 20, 
        'PAY_AMT5': 20, 'PAY_AMT6': 20}
{'AGE': 0.9520060852799245,
 'BILL_AMT1': 0.9507556261767048,
 'BILL_AMT2': 0.9719793849617762,
 'BILL_AMT3': 0.8505283045716918,
 'BILL_AMT4': 0.9544932816850802,
 'BILL_AMT5': 0.9477155670356278,
 'BILL_AMT6': 0.9312213338146995,
 'LIMIT_BAL': 0.9517828968005523,
 'PAY_AMT1': 0.906908907831017,
 'PAY_AMT2': 0.791403763009514,
 'PAY_AMT3': 0.9087543484742974,
 'PAY_AMT4': 0.9123354100831473,
 'PAY_AMT5': 0.9223116153769352,
 'PAY_AMT6': 0.9281853325616332}

###############################################################################
''' 
def synContinuous(tail, exception = 'LIMIT_BAL'):
    data = pd.read_csv('Hazy dataset.csv')        
    cat_data = pd.read_csv('Mod Data_v01.csv')
    
    cat_data = cat_data.drop(['BILL_AMT1', 'BILL_AMT2', 
                          'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                          'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
                          'PAY_AMT5', 'PAY_AMT6'] , axis=1)
    
    for name in list(tail.keys()):
        diff = round(data[name].mean() - np.mean(tail[name]))
        if not name == exception:
            cat_data[name] = tail[name] + diff
            
    return cat_data

#new_data = synContinuous(tail, exception = None)
#new_data.to_csv('', index = False)

    
    
    
    

        
        
        
    
    
    

        
        
    
    
