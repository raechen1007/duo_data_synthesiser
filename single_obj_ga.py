# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:40:14 2019

@author: RaeChen
"""

import numpy as np
from numpy.random import choice
from random import sample, random
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

'''
--------------------------------General functions------------------------------
'''
'restructure data to Keys+Target format '
def restructure(data, target_index):
    '''
        restructure data to Keys+Target format 
    '''
    new = data.copy()
    new[target_index], new[-2] = new[-2], new[target_index].copy()
    return new

def unique(data): #get the shape of data
    unique=[]
    for i in range(data.shape[0]):
        unique.append(int(max(data[i])-min(data[i])+1))
    return tuple(unique)
    
def slices(n):
    indices = range(n+1)
    while True:
        a, b = sorted(sample(indices, 2))
        yield slice(a, b)
                
'''
---------------------------Initial Population functions------------------------
'''

def mutate(data, size, pum):
    def mutation(data, pum):
        mutant=data.copy()
        
        if pum <= 1.0 and pum >= 0:
            n,m = data.shape
            for i in range(n - 1):
                for j in range(m):
                    if random() < pum:
                        mutant[i][j] = choice(data[i].astype(int), size = 1,
                                              replace = True)[0]
            return mutant
        else:
            raise ValueError('invalid pum value')
    
    population = []
    for i in range(size):
        population.append(mutation(data, pum))
    return population

'''
------------------------------Selection functions------------------------------
'''

def tournament(fitnesses, size, k):
   selection = []
   for _ in range(size):
       competitors = sample(fitnesses, k)
       fittest = min(competitors)
       index = fitnesses.index(fittest)
       selection.append(index)
   return selection

def np_tournament(fit_var, size, size_comparison, delta):
    n=size #number of candidates
    selection=[]
    while len(selection)<n:
        competitors=sample(range(n), 2) #two candidates are picked randomly from fit_var
        remain=[x for x in range(n) if x not in competitors] #remained candidates
        comparison_set=sample(remain, size_comparison) #determined unber of rival candidates
        
        '''dominated tournament'''
        count_win=[0,0] #count_win= number of winning times
        for candidate in comparison_set:
            if fit_var[competitors[0]][0]<=fit_var[candidate][0] and fit_var[competitors[0]][1]<=fit_var[candidate][1]:
                count_win[0]+=1
            if fit_var[competitors[1]][0]<=fit_var[candidate][0] and fit_var[competitors[1]][1]<=fit_var[candidate][1]:
                count_win[1]+=1
                
        if max(count_win) == size_comparison and count_win[0] != count_win[1]:
            selection.append(competitors[count_win.index(max(count_win))])
            
        else:    
            '''non-dominated tournament'''
            count_niche=[0, 0] #count_niche=number of surronding candidates
            for i in range(n):
                d_0=sqrt((fit_var[competitors[0]][0]-fit_var[i][0])**2+(fit_var[competitors[0]][1]-fit_var[i][1])**2)/sqrt(2)
                d_1=sqrt((fit_var[competitors[1]][0]-fit_var[i][0])**2+(fit_var[competitors[1]][1]-fit_var[i][1])**2)/sqrt(2)
                if d_0<=delta:
                    count_niche[0]+=1
                if d_1<=delta:
                    count_niche[1]+=1
            selection.append(competitors[count_niche.index(min(count_niche))])
        
    return selection
    
'''
-----------------------------Crossover functions-------------------------------
'''

def WCPC(arr1, arr2, pc):
    if not arr1.shape==arr2.shape:
        raise ValueError("Arrays are different shapes")
    else:
        res1, res2 = arr1.copy().T, arr2.copy().T
        m=arr1.shape[1] #m:cases
        for i in range(m):
            if random()<pc:
                res1[i], res2[i]=res2[i], res1[i].copy()
        return res1.T, res2.T
    
'''
--------------------------------Mutation functions-----------------------------
''' 
    
def uniformMutation(arr, pum):
    mutant=arr.copy()
        
    if pum <= 1.0 and pum >= 0:
        n,m = arr.shape
        for i in range(n - 1):
            for j in range(m):
                if random() < pum:
                    mutant[i][j] = choice(arr[i].astype(int), size = 1,
                                          replace = True)[0]
        return mutant
    else:
        raise ValueError('invalid pum value')
        
'''
-------------------------------Fitness-----------------------------------------
'''

def create_array(data, shape): #Written by DS: duncan.g.smith@manchester.ac.uk
       arr = np.zeros(shape)
       for case in data.T:
           arr[tuple(case)] += 1
       return arr
       
def JS_distance(P, Q):#Written by DS: duncan.g.smith@manchester.ac.uk
   mean = 0.5*P + 0.5*Q
   JS_divergence=((- mean * np.where(mean > 0, np.log2(mean), 0)).sum() -
           0.5 * (- P * np.where(P > 0, np.log2(P), 0)).sum() -
           0.5 * (- Q * np.where(Q > 0, np.log2(Q), 0)).sum())
   return JS_divergence ** 0.5

def full_contingency_divergence(m, shape, full_table, syndata):
    P = full_table/m
    Q = create_array(syndata[:-1].astype(int), shape)/m
    delta = JS_distance(P,Q)
    return delta
    
def dcapUnique(ori_data, syn_data, shape, ori_full_table):
    
    syn_full_table=create_array(syn_data, shape)
    '''
        unique keys from original data
    '''
    count=np.unique(ori_data[:-1].T, return_counts=True, axis=0)
    unique_ori_key=[]
    for i in range(len(count[1])):
        if count[1][i]==1:
            unique_ori_key.append(count[0][i])
    
    '''
    unique key that has unique target in the original data, aka reference table     
    '''
    unique_ori_target=[]
    for key in unique_ori_key:
        unique_ori_target.append(np.where(ori_full_table[tuple(key)]==1)[0][0])
    reference_table=np.column_stack((np.array(unique_ori_key), np.array(unique_ori_target)))
    '''
        finding frequencies of key values appeared in reference table and also 
        in synthetic data
    '''
    corr_syn_key=[]
    for key in unique_ori_key:
        corr_syn_key.append(sum(syn_full_table[tuple(key)]))
    '''
        Finding frequencies of key+target values appeared in reference table and 
        also in synthetic data
    '''
    corr_syn_target=[] 
    for value in reference_table:
        corr_syn_target.append(syn_full_table[tuple(value)])
        
    paa=np.array([corr_syn_target[i]/corr_syn_key[i] for i in range(len(corr_syn_key))])
    paa[np.isnan(paa)]=0 #recode nan to 0
    
    dcap=np.mean(paa)
    return dcap
       
def fitness_variable(m, shape, full_table, ori_data, population, size):
    fit_var=[]
    for i in range(size):
        fit_var.append((full_contingency_divergence(m, shape, full_table, 
                                                    population[i]), 
                        dcapUnique(ori_data[:-1].astype(int), 
                                   population[i][:-1].astype(int), 
                                   shape, full_table)))
    return fit_var

def fitness_scalar(fit_var):
    fit_val=[]
    for pair in fit_var:
        fit_val.append(sqrt(pair[0]**2+pair[1]**2)/sqrt(2))
    return fit_val
    
'''
-------------------------------Progress----------------------------------------
'''
        
class Population(object):
    '''
    Population class including the original data, the shape of full contingency table and the table itself, population
    and population fitness values.
    '''
    def __init__(self, data, m, shape, full_table, size, population, fit_var=None, fit_val=None):
        self.data = data
        self.m = m
        self.shape = shape
        self.full_table = full_table
        self.size = size
        self.population = population
    
    def population_fit_var(self):
        self.fit_var = fitness_variable(self.m, self.shape, 
                                        self.full_table, self.data, 
                                        self.population, self.size)
        return self.fit_var
    
    def population_fit_val(self):
        self.fit_val = fitness_scalar(self.fit_var)
        return self.fit_val
        
'''
---------------------------------------Process Fn------------------------------
'''
def get_data(data_file):
    testdata=pd.read_csv(data_file)
    data=np.array(testdata).T
    return data
    
def fixed_parent(population, size, selection_fn, t, fitval):
    selection=selection_fn(fitval, size, t)
    parent=[]
    for i in selection:
        parent.append(population[i].copy())
    return parent

def fixed_offspring(size, parents, pc, pm):
    even=list(range(0, size-1, 2))
    offspring=[]
    for j in even:
        for i in range(2):
            offspring.append(WCPC(parents[j],parents[j+1], pc)[i])
    for k in range(size):
        offspring[k]=uniformMutation(offspring[k],pm)
    return offspring

def process_plot(generation, err_min):
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    axes = plt.gca()
    axes.set_xlim([0,generation])
    plt.plot(err_min)
    plt.title('Minimum Scalar Fitness from each Generation')
    plt.xlabel('Generation')
    plt.ylabel('Minimum Scalar Fitness')

def process3d_plot(the_bin):
    x,y=zip(*the_bin)
    df=pd.DataFrame({'divergence':list(x), 'risk':list(y), 'generation':range(len(x))})
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['divergence'], df['risk'], df['generation'], c='blue', s=5)
    ax.set_xlabel('Divergence')
    ax.set_ylabel('Disclosure Risks')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.view_init(45, 250)
    plt.show()

def save_syndata(file_name, var, fit_val, err, population):    
    ind=fit_val.index(err)
    best=population[ind]
    syndata={}
    for i in range(len(var)):
        syndata.update({var[i]:best[i]})
    sd=pd.DataFrame(data=syndata)
    sd.to_csv(file_name, sep=',')
    
def bi_obj_plot(_bin):
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    axes = plt.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    x,y = zip(*_bin)
    plt.scatter(x,y)
    plt.title('Divergence vs Risks Plot')
    plt.xlabel('Divergence')
    plt.ylabel('Disclosure Risks')

def final_bin_to_csv(final_bin, file_name):
    new=[]
    for ls in final_bin:
        for item in ls:
            new.append(item)
    
    divergence, risk=[], []
    for pair in new:
        divergence.append(pair[0])
        risk.append(pair[1])
        
    import csv
    from itertools import zip_longest
    result=[divergence, risk]
    export_data = zip_longest(*result, fillvalue = '')
    with open(file_name, 'w', encoding="ISO-8859-1", newline='') as myfile:
          wr = csv.writer(myfile)
          wr.writerow(("divergence", "risk"))
          wr.writerows(export_data)
    myfile.close()
