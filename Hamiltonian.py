from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.operators import exp_op
from quspin.tools.measurements import ent_entropy

import numpy as np

import time
import sys
import os
import pickle



def Hamiltonian(L,J,hz,fun=None,fun_args=[]):
	######## define physics
	basis = spin_basis_1d(L=L,kblock=0,pblock=1,pauli=False) #
	

	zz_int =[[J,i,(i+1)%L] for i in range(L)]
	x_field=[[1.0,i] for i in range(L)]
	z_field=[[hz,i] for i in range(L)]

	static = [["zz",zz_int],["z",z_field]]
	dynamic = [["x",x_field,fun,fun_args]]

	kwargs = {'dtype':np.float64,'basis':basis,'check_symm':False,'check_herm':False,'check_pcon':False}
	H = hamiltonian(static,dynamic,**kwargs)

	return H


def Unitaries(delta_time,L,J,hz,action_min,var_max,var_min,state_i,save=False,save_str=''):

	# define Hamiltonian
	b=0.0
	lin_fun = lambda t: b 
	H = Hamiltonian(L,fun=lin_fun,**{'J':J,'hz':hz})

	# number of unitaries
	n = int((var_max - var_min)/action_min)
	
	# preallocate dict
	expm_dict = {}
	for i in range(n+1):
		# define matrix exponential; will be changed every time b is overwritten
		b = state_i[0]+i*action_min
		expm_dict[i] = np.asarray( exp_op(H,a=-1j*delta_time).get_mat().todense() )
		
	# calculate eigenbasis of H_target
	b=+2.0
	_,V_target=H.eigh()

	if save:

		### define save directory for data
		# read in local directory path
		str1=os.getcwd()
		str2=str1.split('\\')
		n=len(str2)
		my_dir = str2[n-1]
		# create directory if non-existant
		save_dir = my_dir+"data/unitaries"
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		save_dir="data/unitaries/"


		# save file
		with open(save_dir + 'unitaries_L-'+str(L)+'_dt-'+str(delta_time).replace('.','p')+'.pkl','wb') as f:
			pickle.dump(expm_dict,f)
		
	else:
		return expm_dict



def MB_observables(psi,times,protocol,pos_actions,var_min,L,J=1.0,hx_i=2.0,hx_f=-2.0,hz=1.0,bang=True,psi_target=None):

	"""
	this function returns instantaneous observalbes during ramp 
	OR 
	when fin_vals=True only the final values at the end of the ramp
	----------------
	observables:
	----------------
	Fidelity
	E: energy above instantaneous GS
	delta_E: energy fluctuations
	Sd: diagonal entropy
	Sent: entanglement entropy
	"""


	if L<2:
		print("function only analyses manybody chains! Exiting...")
		return [],[],[],[],[]
		
	# read in local directory path
	str1=os.getcwd()
	str2=str1.split('\\')
	n=len(str2)
	my_dir = str2[n-1]

	# define Hamiltonian
	b=hx_f  
	lin_fun = lambda t: b
	# define Hamiltonian
	H = Hamiltonian(L,fun=lin_fun,**{'J':J,'hz':hz})


	with open(my_dir + '/data/unitaries/unitaries_L-'+str(L)+'_dt-'+str(times[2]-times[1]).replace('.','p')+'.pkl','rb') as f:
		expm_dict = pickle.load(f)

	
	# preallocate variables
	Fidelity=[]

	
	i=0
	while True:

		# instantaneous fidelity
		Fidelity.append( abs(psi.conj().dot(psi_target))**2 )
		
		if i == len(protocol):
			break
		else:
			# go to next step
			b=protocol[i] # --> induces a change in H
			#psi = exp_H.dot(psi)
			psi = expm_dict[int(np.rint((b - var_min)/min(pos_actions)))].dot(psi)
			i+=1


	return Fidelity[-1]


