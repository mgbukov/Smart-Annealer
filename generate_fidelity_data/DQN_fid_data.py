import Hamiltonian
import numpy as np
import random
import pickle

import time
import sys
import os
import gc

# make system update output files regularly
sys.stdout.flush()

### define save directory for data
# read in local directory path
str1=os.getcwd()
str2=str1.split('\\')
n=len(str2)
my_dir = str2[n-1]


#################

L=2
delta_time=0.05
max_t_steps=30

# load data
file = '../data/protocols_L-'+str(L)+'_dt-'+str(delta_time).replace('.','p')+'_NT-'+str(max_t_steps)+'.pkl'

with open(file,'rb') as data_file:
	Data=pickle.load(data_file) 
	data_file.close()

print(Data.shape)
Protocols=Data[0]
Fidelities=Data[1]


for h in Protocols:
	


