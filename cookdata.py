#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:55:05 2019

@author: RaeChen
"""
import numpy as np
import pandas as pd

data = pd.read_csv('Mod Data_v02.csv')

def zeroBasedCategorise(data):
    
    modify = {}
    
    var = list(data)
    refined_data = data.copy()
    
    for name in var[:-1]:
        if not data[name].min() == 0:
            
            diff = data[name].min()
            replace = np.zeros(data[name].size)
            
            for i in range(data[name].size):
                replace[i] = data[name][i] - diff
        
            refined_data[name] = replace
            
            modify[(name, 'ori minimum')] = diff
    
    return refined_data, modify

'''
Create new variable for subdata1 and subdata2
'''

def subData(): 
        
    data['_TAIL'] = list(zip(data['BILL_AMT1'], data['BILL_AMT2'], 
                        data['BILL_AMT3'], data['BILL_AMT4'],
                        data['BILL_AMT5'], data['BILL_AMT6'], 
                        data['PAY_AMT1'], data['PAY_AMT2'], 
                        data['PAY_AMT3'], data['PAY_AMT4'],
                        data['PAY_AMT5'], data['PAY_AMT6'],
                        data['PAY_0'], data['PAY_2'], data['PAY_3'],
                        data['PAY_4'], data['PAY_5'], data['PAY_6'],
                        data['AGE'])) 
    
    new_data = data.drop(['BILL_AMT1', 'BILL_AMT2', 
                         'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                         'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
                         'PAY_AMT5', 'PAY_AMT6', 'PAY_0', 'PAY_2', 'PAY_3', 
                         'PAY_4', 'PAY_5', 'PAY_6', 'AGE'], axis=1)
    return new_data

#data = subData()
#
#data, modify = zeroBasedCategorise(data)
#data.to_csv('Mod Data_v03.csv', index = False)

modify = {('SEX', 'ori minimum'): 1}






    


