# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:57:00 2019

@author: yingr
"""

import soga as sg
import copy
    
data_file = 'Mod Data_v03.csv' 
data = sg.get_data(data_file)
#data = sg.restructure(data, 0)

m = data.shape[1] #number of cases in the dataset
shape = sg.unique(data[:-1]) #shape of full-contingency table in the dataset
full_table = sg.create_array(data[:-1].astype(int), shape) #full_table
size = 100
var = ['SEX','EDUCATION','MARRIAGE',
       'default payment next month', 'LIMIT_BAL']


pc, pm=0.1, 0.01

population = sg.mutate(data, size, .01) #initial population
pop=sg.Population(data, m, shape, full_table, size, population) #class of initial population

fit_var=pop.population_fit_var() #baseline = (0.20765619513652503, 0.10555555555555556)
fit_val=pop.population_fit_val()

err=min(fit_val)
err_var=fit_var[fit_val.index(err)]

err_min_val=[err]
err_min_var=[err_var]

final_bin=[] #bin to store all ever-appeared candidates' information

generation=0

while generation < 10 and err_var[1] > 0.12:
    pop_old=copy.deepcopy(population) #retain parental population
    err_old=err
    
    parents=sg.fixed_parent(population, size, sg.tournament, 2, fit_val)
    population=sg.fixed_offspring(size, parents, pc, pm)
    pop=sg.Population(data, m, shape, full_table, size, population) #offspring population

    fit_var=pop.population_fit_var()    
    fit_val=pop.population_fit_val()
    
    err=min(fit_val) 
     
    generation+=1
    
    err_var=fit_var[fit_val.index(err)]
    
    err_min_val.append(err)
    err_min_var.append(err_var)
    
    final_bin.append(fit_var)
    print("%.7f" % err_var[0], "%.7f" % err_var[1], generation) 

syn = pop.population[fit_val.index(err)]

syn = syn.T

import pandas as pd
syn_data = pd.DataFrame(syn, columns = ['SEX','EDUCATION','MARRIAGE',
            'default payment next month', 'LIMIT_BAL', '_TAIL'])

syn_data.to_csv('syn03_raw.csv', index = False)

#7 generations, 7 mins
