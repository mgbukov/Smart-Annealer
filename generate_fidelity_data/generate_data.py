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

N_samples=100000

max_t_steps = 30 
delta_time = 0.05

# define model params
L = 2 # system size
if L==1:
	J = 0.0 # required by PBC
else:
	J = 1.0 #/0.809 # zz interaction
hz = 1.0 #0.9045/0.809 #1.0 # hz field
hx_i= -2.0 # initial hx coupling
hx_f= +2.0 # final hx coupling

# define dynamic params of H(t)
b=hx_i
lin_fun = lambda t: b
# define Hamiltonian
H_params = {'J':J,'hz':hz}
H = Hamiltonian.Hamiltonian(L,fun=lin_fun,**H_params)


# calculate initial state
if L==1:
	E_i, psi_i = H.eigh()
else:
	E_i, psi_i = H.eigsh(time=0,k=2,which='BE',maxiter=1E10,return_eigenvectors=True)
	#E_i, psi_i = H.eigsh(time=0,k=1,sigma=-0.1,maxiter=1E10,return_eigenvectors=True)
E_i = E_i[0]
psi_i = psi_i[:,0]
# calculate final state
b = hx_f
if L==1:
	E_f, psi_target = H.eigh()
else:
	E_f, psi_target = H.eigsh(time=0,k=2,which='BE',maxiter=1E10,return_eigenvectors=True)
E_f = E_f[0]
psi_target = psi_target[:,0]


print("number of states is:", H.Ns)
print("initial and final energies are:", E_i, E_f)
print("overlap btw initial and target state is:", abs(psi_i.dot(psi_target)**2) )

 

##### pre-calculate unitaries
state_i=[-4.0]
Hamiltonian.Unitaries(delta_time,L,J,hz,8.0,4.0,-4.0,state_i,save=True,save_str='_bang')

# define time vector
times=delta_time*np.arange(max_t_steps)


Protocols=np.zeros((N_samples,len(times)))
Fidelities=np.zeros((N_samples,))
Fidelities_DQN=np.zeros((N_samples,len(times)))
for i in range(N_samples):
	# calculate random protocol
	protocol = [4.0*random.choice([-1.0, 1.0]) for i in range(len(times))]

	Fidelities[i] = Hamiltonian.MB_observables(psi_i,times,protocol,[8],-4.0,L,J=J,hx_i=hx_i,hx_f=hx_f,hz=hz,bang=True,psi_target=psi_target)
	Protocols[i,:]=protocol
	for j in range(len(times)):
		protocol_new=protocol.copy()
		protocol_new[j]=-protocol_new[j]

		Fidelities_DQN[i,j] = Hamiltonian.MB_observables(psi_i,times,protocol_new,[8],-4.0,L,J=J,hx_i=hx_i,hx_f=hx_f,hz=hz,bang=True,psi_target=psi_target)

	print(i)

# save data
with open(my_dir + '/../data/protocols_L-'+str(L)+'_dt-'+str(delta_time).replace('.','p')+'_NT-'+str(max_t_steps)+'.pkl','wb') as f:
	expm_dict = pickle.dump([Protocols,Fidelities,Fidelities_DQN],f)





