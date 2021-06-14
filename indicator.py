#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:12:54 2019

@author: RaeChen
"""
import pandas as pd
import numpy as np
from cookdata import zeroBasedCategorise

data = pd.read_csv('Mod Data_v03.csv')
syn_data = pd.read_csv('syn01.csv')
var = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','default payment next month']


def findUniqueness(data, var):
    sub = zeroBasedCategorise(data[var])[0]
    
    data_arr = np.array(sub).T
    data_arr = data_arr.astype(int)
    
    def unique(data): 
        unique = []
        for i in range(data.shape[0]):
            unique.append(len(list(np.unique(data[i]))))
        return tuple(unique)
    
    def create_array(data, shape): #Written by DS: duncan.g.smith@manchester.ac.uk
           arr = np.zeros(shape)
           for case in data.T:
               arr[tuple(case)] += 1
           return arr
    
    def uniqueness_indicator(data, full_table):
        indicator = []
        for case in data.T:
            if full_table[tuple(case)] == 1.0:
                indicator.append(tuple(case))
        return indicator
    
    shape = unique(data_arr)
    full_table = create_array(data_arr, shape)
    indicator = uniqueness_indicator(data_arr, full_table)
    
    sub = list(sub.itertuples(index = False, name = None))
    unique_inds = []
    for case in indicator:
        unique_inds.append(sub.index(case))
        
    return unique_inds

#unique_inds = findUniqueness(data, var)
#unique_data = data.iloc[unique_inds]
#unique_data.to_csv('uniqueness.csv', index = False)
#
#safe_data = data.drop(unique_inds) 
#safe_data.to_csv('Mod Data_v04.csv', index = False)
     


    
    