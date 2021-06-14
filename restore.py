# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:25:04 2019

@author: yingr
"""

import pandas as pd
import analyse
import numpy as np
from numpy.random import gamma
import cookdata

tail = ['BILL_AMT1', 'BILL_AMT2', 
        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
        'PAY_AMT5', 'PAY_AMT6', 'PAY_0', 'PAY_2', 'PAY_3', 
        'PAY_4', 'PAY_5', 'PAY_6', 'AGE']

exception = 'LIMIT_BAL'
modify = cookdata.modify   

data = pd.read_csv('syn03_raw.csv')

def restoreTail(data, tail):
    for i in range(len(tail)):
        var = []
        for case in data['_TAIL']:
            case = list(iter("".join(c for c in case if c not in "()[] ").split(",")))
            var.append(int(float(case[i])))
        
        data[tail[i]] = var

    data = data.drop(['_TAIL'], axis = 1)

    return data

def restoreException(data, exception):
    series = data[exception]
    
    gamma_paras = analyse.gamma_paras
        
    division = gamma_paras[(exception, 'division')]
    paras = gamma_paras[(exception, 'paras')]
    
    restore = np.zeros(series.size)
        
    for i in range(series.size):
        
        value = series[i]
        
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
                
        
    data[exception] = restore


def restoreNonZeroBase(data, modify):
    for key in list(modify.keys()):
        data[key[0]] += modify[key]

data = restoreTail(data, tail)    
restoreNonZeroBase(data, modify)  
restoreException(data, exception)  
    
var = analyse.var  
data = data[var]

data.to_csv('syn03.csv', index = False)
        
    