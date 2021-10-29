import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spsp		
import scipy.special as sps 		# for an N-choose-K function called 'comb' # 
import scipy.optimize as opt 		# for Hungarian Method to resort cosine similarity matrix to make it diagonally dominant
import time
import os.path 						# to check if a file or directory exists already
import itertools as it

from sklearn.metrics import auc
from sklearn.metrics.pairwise import cosine_similarity



# def set_dir_tree():
# 	# (0). Check what Operating System you are on (either my machine or Cortex cluster) and adjust directory structure accordingly.
# 	if sys.platform == 'darwin' or sys.platform == 'darwin15':
# 		print('on Mac OS - assuming Chris laptop')
# 		#dirPre = '../../'
# 		dirHome = '/Users/chriswarner/Desktop/Grad_School/Berkeley/Work/Fritz_Work/Projects/G_Field_Retinal_Data/home/G_Field_Retinal_Data/'
# 		dirScratch = '/Users/chriswarner/Desktop/Grad_School/Berkeley/Work/Fritz_Work/Projects/G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/'
# 	elif sys.platform == 'linux' or sys.platform == 'linux2':
# 		print('on Linux - assuming NERSC Cluster')
# 		#dirPre = '/clusterfs/cortex/scratch/cwarner/'
# 		dirHome = '/global/homes/w/warner/G_Field_Retinal_Data/'
# 		dirScratch = '/global/cscratch1/sd/warner/G_Field_Retinal_Data/'

# 	return dirHome, dirScratch







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def sig(r):	
	# The standard logistic function with k=1 an f=0. Fritz maintains that we do not need the linear fit logistic function ("sig" above)
	P = 1/(1+np.exp(-r))	
	return P



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def inv_sig(P):
	# inverting the standard logistic function to get rho values (between -inf and +inf) from P values (between 0 and 1)
	r = np.log( P/(1-P) )	
	r[np.isposinf(r)] 	=  20 	#0.9*np.finfo(type(p[0])).max
	r[np.isneginf(r)] 	= -20 	#0.9*np.finfo(type(p[0])).min
	r[np.isnan(r)] 		=  20 	#0.9*np.finfo(type(p[0])).min
	return r 






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def noiseOnProbability(Inpt,sig,verbose=False):	
	# Add some gaussian noise to a Matrix, Vector, Scalar but resample if an entry goes outside the range of probability [0,1]

	M = Inpt
	M = M + np.random.normal(0, sig, size=M.shape)
	#
	ind = np.where( np.bitwise_or(M>1, M<0) )
	while ind[0].any():
		if verbose:
			print(ind[0].size)
		M[ind] = Inpt[ind] + np.random.normal( 0, sig, size=ind[0].size )
		ind = np.where( np.bitwise_or(M>1, M<0) )
	return M	








# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def set2boolVec(S,N):
	# Convert a set (indicating) nonzero entries into a boolean/binary vector.
	# S = a set of non-zero elements in binary vector
	# N = number of elements in or length of binary vector 
	V = np.zeros( (1,N) ).astype(int)
	V[0,list(S)] = 1
	return V[0]







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def synthesize_model(N,M,M_mod, C,K, Cmin,Cmax, bernoulli_Pi,mu_Pi,sig_Pi, mu_Pia,sig_Pia, verbose=False):
#
# Synthesize model (i.e., construct q, ri and ria parameter arrays).

# 	Synthesize in P- or probability-space and then pipe Q, Pi, Pia through the inverse
# 	logistic function (inv_sig) to get q, ri, ria. Why? The user input parameters on
# 	assumed distributions are more intuitive and can be fit to data better. 

# 	- Q (the scalar probability of a 1 in Z-vector) is determined by K & M. Q is used in
# 		construct_YnZ to ensure that Z-vectors have reasonable cardinality, i.e., that
# 		relatively few cell assemblies are active in any one observed spike-word / y-vector.
	
# 	Assume Bernoulli and Gaussian probability distributions for parameters in Pi and Pia.

# 	- Pi (the vector probability of each cell being silent outside any cell assemblies) is 
# 		constructed in TWO STEPS. 
# 			#
# 		1st, a binary Pi vector is constructed by drawing values from a bernoulli 
# 			distribution with the bernoulli_Pi parameter defining #1's or the number
# 		 	of cells that are more likely to be silent outside cell assemblies.
# 			#
# 		2nd, those binary values are Noised by drawing values for delta Pi from a 
# 			Gaussian distribution displaced from 0 or 1 by mu_Pi and with a spread
# 			of sig_Pi

# 	-Pia (the matrix probability of each cell being silent in each cell assembly) is 
# 		constructed in THREE steps.
# 			#
# 		1st, a binary matrix is constructed by drawing values from a bernoulli distribution
# 			with bernoulli_Pia parameter given by C/N (Note the similarity to Q in this). 
# 			The number of 0's in a column of Pia or equivalently the number of active cells 
# 			in a cell assembly can vary and so we limit the bounds by the Cmin & Cmax 
# 			parameters. If a column exceeds these bounds, that column is simply resampled.
# 			#
# 		2nd, those binary values in Pia are Noised exactly the same as Pi.

# 	Finally, after {Q, Pi, Pia} model parameters are synthesized containing real numbers 
# 	between 0 and 1, those values are piped pointwise through the logistic function



	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (1). Probability of a Za element of the Z-vector being active.
	Q=np.array([K/M]) # *np.ones(M) # to make any Z-vector ~K-hot.


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (2). Probability that cell is silent in general (regardless of cell assemblies)
	# 	   Pi =  P(Y_i=0 | Z-vec=0)
	#
	#		(a). Draw cells from a bernoulli distribution whether they will be quiet outside
	#			a cell assembly (Pi~1) or noisy (Pi~0).
	#		(b). For each population seperately, draw Pi values from a gaussian that is displaced
	#			from 0 or 1 by mu_Pi and has a spread of sig_Pi
	if verbose:
		print('Construct Pi vector')

	Pi = np.random.binomial( 1, bernoulli_Pi, (N) ).astype(float)						# (a).
	quiet = (Pi==1)
	Pi[quiet] = Pi[quiet] - np.abs( np.random.normal( mu_Pi, sig_Pi, quiet.sum() ) )
	#noisy = np.bitwise_not(quiet)
	# Pi[noisy] = Pi[noisy] + np.abs( np.random.normal( mu_Pi, sig_Pi, noisy.sum() ) )	# (b).


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (3). Probability that cell is silent in a particular cell assembly. 
	# 	   Bernoulli distribution = Binomial with n (number of draws) = 1.
	# Plus some Gaussian Noise to P_ia to move away from deterministic.
	# Constrain number of cells participating to be within bounds (Cmax & Cmin)
	# to correct for the spread / std on a binomial distribution.

	# print('Construct Pia so that: ')
	# print('			(1). Make sure #cells in each Cell assembly > ',Cmin,' and < ',Cmax)
	

	# Pia1 = np.random.binomial( 1, C/N, (N,M) )
	# CperA = Pia1.sum(axis=0) # how many cells active in each cell assembly
	# n = np.where( np.bitwise_or(CperA<Cmin,CperA>Cmax) )[0]
	# while n.any():
	# 	Pia1[:,n] = (np.random.binomial( 1, C/N, (N,n.size) ) ).astype(float)
	# 	CperA = Pia1.sum(axis=0) # how many cells active in each cell assembly
	# 	n = np.where( np.bitwise_or(CperA<Cmin,CperA>Cmax) )[0]
	# 	print(n.size)
	# 	print(Pia1.T)



	# ALGORITHM TO REDUCE OVERLAP IN CELL ASSEMBLIES.
	#
	# Set an overlap threshold to ensure Cell Assemblies do not share more cells than intended
	if verbose:
		print('Construct Pia so that: ')
		print('			(1). Make sure #cells in each Cell assembly > ',Cmin,' and < ',Cmax)
		print('			(2). Overlap between cell assemblies (shared cells) is small.')
	Pia1 = np.random.binomial( 1, C/N, (N,M) )
	
	


	# (1). Make sure that cell assemblies have the right number of active cells in them (between Cmin and Cmax) 
	
	CperA = Pia1.sum(axis=0) # how many cells active in each cell assembly
	n = np.where( np.bitwise_or(CperA<Cmin,CperA>Cmax) )[0]
	while n.any():
		Pia1[:,n] = (np.random.binomial( 1, C/N, (N,n.size) ) ).astype(float)
		CperA = Pia1.sum(axis=0) # how many cells active in each cell assembly
		n = np.where( np.bitwise_or(CperA<Cmin,CperA>Cmax) )[0]
		if verbose:
			print(n.size)
			print(Pia1.T)
	AperC = Pia1.sum(axis=1) # how many cells assemblies a cell participates in.



	
	# (2). Shift active cells around (within assemblies) to decrease their overlap or dot-product while keeping the number of cells active in an assembly unchanged
	OVL = np.matmul(Pia1.T,Pia1) / np.sqrt(CperA[:,None]*CperA[None,:])
	np.fill_diagonal(OVL,0)
	OVL_totPrev = OVL.sum()
	#
	a=0
	while a < 100000:
		a+=1
		Pia2 = Pia1
		#
		i = np.random.randint(0,M)							# randomly choose cell assembly.
		cellsInCA = np.where(Pia2[:,i])[0] 
		#
		AperC_sorted = np.argsort(AperC) 					# activate cell that does not participate in many assemblies.
		xx=0
		j = AperC_sorted[xx]
		while Pia2[j,i] == 1:								# make sure the cell we are activating is not active already.
			xx+=1
			j = AperC_sorted[xx]




		try:
			k = cellsInCA[np.random.randint(0,cellsInCA.size)]	# randomly choose an active cell to inactivate
		except:
			return None, None, None, None
			print('break because sometimes I get, ValueError: low >= high')
		#
		Pia2[k,i] = 0 										# inactivate an active cell in CA i
		Pia2[j,i] = 1 										# activate an inactive cell in CA i
		#
		AperC = Pia2.sum(axis=1) # how many cells assemblies a cell participates in.
		CperA = Pia2.sum(axis=0) # how many cells active in each cell assembly
		#
		OVL = np.matmul(Pia2.T,Pia2) / np.sqrt(CperA[:,None]*CperA[None,:])
		np.fill_diagonal(OVL,0)
		#
		if OVL_totPrev==0 and OVL.sum()==0:
			break
		#
		if OVL.sum() < OVL_totPrev: # if the overall overlap is decreased, keep the change.
			if verbose:
				print(a,': In cell assembly # ',i,', moving cell ',k,' to ',j,'. Decreased OVL from ',OVL_totPrev,' to ',OVL.sum())
			Pia1 = Pia2
			OVL_totPrev = OVL.sum()
			
			AperC = Pia1.sum(axis=1) # how many cells assemblies a cell participates in.
			if verbose:
				print('Pia sum = ',Pia1.sum())
				print('# Assemblies per cell: ', AperC, 'and # Cells per assembly: ',CperA)	

	Pia = (1 - Pia1 ).astype(float) # P(Y_i=0 | Z_a=1) = C/N

	if verbose:
		print('AperC = ',AperC)
		print(not AperC.all())
	# Check that each cell participates in at least one assembly. Maybe get rid of this. 
	if not AperC.all():
		print('Error: Some cells participate in zero cell assemblies.')
		return None, None, None, None



	# Add some noise / variablility to Pia. Not all binary values. Probabalistic model.
	inCA = (Pia==0)
	notIn = np.bitwise_not(inCA)
	#Pia[notIn] = Pia[notIn] - np.abs( np.random.normal( mu_Pia, sig_Pia, notIn.sum() ) ) # FRITZ SAYS MAYBE NOT NECESSARY.
	Pia[inCA] = Pia[inCA] + np.abs( np.random.normal( mu_Pia, sig_Pia, inCA.sum() ) )



	# Allow dimensionality of model to be different from the dimensionality of the model that creates the data (M_mod <,=,> M).
	Pia_mod = np.ones((N,M_mod))
	if M_mod>=M:
		Pia_mod[:,:M] = Pia
	else:
		Pia_mod = Pia[:,:M_mod]


	# Convert probabilities (P) to non-probabilities (p) by piping values backwards through the logistic function.
	#	P = rc.sig(p) and p = rc.inv_sig(P)
	q 		= inv_sig(Q)
	ri 		= inv_sig(Pi)
	ria 	= inv_sig(Pia)	
	ria_mod = inv_sig(Pia_mod)

	if verbose:
		print('got to end and q = ',q)
	return q, ri, ria, ria_mod






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def generate_data(numSWs, q, ri, ria, Cmin, Cmax, Kmin, Kmax, verbose=False):

# Given a model parameters {q, ri, ria}, this constructs synthetic data (Y-vector,
# 	Z-vector pairs). These can then be used for inference and learning steps.
# 			#
# 		1st, model parameters {q, ri, ria} are converted into probabilities 
# 			{Q, Pi, Pia} by piping them through the logistic function.
# 			#
# 		2nd, a sparse binary Z-vector is constructed by drawing values from a 
# 			bernoulli distribution with the probability of drawing a 1 being Q.
# 			#
# 		3rd, a binary Y-vector is constructed by multiplying the Z-vector by the
# 			Pia matrix and then pointwise multiplying that vector with the Pi vector.
# 			This gives the probability that a cell is silent, considering active
# 			cell assemblies (Pia*Z) and the cells propensity to be silent outside 
# 			any assemblies (Pi). The 0 or 1 value for an element in the Y-vector (Yi)
# 			is drawn from a binomial distribution with the probability of drawing a 1
# 			being { 1 - (Pia*Z)*Pi }. 
# 	
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# (a). Z indicates which Cell Assemblies are active to generate a
# spike word. K allows determines the number allowed active.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# (b). Build up spike word (Y) from cell assembly(/ies)
#	   p(Y_i=0 | Z)

	# Convert model parameters to probabilities.
	Q=sig(q)
	Pi=sig(ri)
	Pia=sig(ria)

	M = ria.shape[1]

	Qa = Q*np.ones(M)

	print('Constructing Data. Z & Y pairs. ' ,numSWs, 'Spikewords!')
	Z_list = []
	Y_list = []

	for i in range(numSWs):

		Z = np.where( np.random.binomial(1,Qa) )[0]
		zsOn = len(Z)
		Py0 = Pi**(1-zsOn/M)*(Pia[:,Z]).prod(axis=1)
		Y = np.where( np.random.binomial( 1,(1-Py0) ) )[0]

		while zsOn<Kmin or zsOn>Kmax: # or Y.size<Cmin: 
		# Note: dont check Y.size>Cmax because multiple cell assemblies can
		#  be active and C defines number cells in a single cell assembly.
			Z = np.where( np.random.binomial(1,Qa) )[0]
			zsOn = len(Z)
			Py0 = Pi**(1-zsOn/M)*(Pia[:,Z]).prod(axis=1)
			Y = np.where( np.random.binomial( 1,(1-Py0) ) )[0]
		
		Y_list.append( set(Y) )
		Z_list.append( set(Z) )

	return Y_list, Z_list	










# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def LMAP(q, ri, ria, yTru, zHyp, zActivationHistory, verbose=False):
	# This computes the log joint probability of y-vec and z-vec given the model (Pia, Pi, q). 
	# It returns a single scalar value. Could return an N-vector too.

	N = ria.shape[0]
	M = ria.shape[1]

	Q   = sig(q)
	Pi  = sig(ri)
	Pia = sig(ria)

	# Make Qa vector from Q and zActivationHistory to encourage all CA's to be equally active.
	# Like an "Egalitarian Homeostasis" (Laurent Perrinet). 
	zRelActInv = ( zActivationHistory/zActivationHistory.mean() )**-1
	Qa = Q*zRelActInv

	Y = set2boolVec(yTru,N) # set of active cells / CAs into boolean vector [0s, & 1s]
	Z = set2boolVec(zHyp,M)

	yTru 	= list(yTru)
	zHyp 	= list(zHyp)


	zsOn	= len(zHyp) 				# Number of active cell assemblies (1's) in z-vector.
	#
	Q_part	= np.log( sps.comb(M,zsOn) ) + zsOn*np.log(Q) + (M-zsOn)*np.log(1-Q) # scalar Q.

	Qa_part = Z*np.log(Qa) + (1-Z)*np.log(1-Qa)
	Pi_part  = (1-Y)*(1-zsOn/M)*np.log(Pi) 
	Pia_part = (1-Y)*np.log(Pia[:,zHyp]).sum(axis=1)
	mix_part = Y*np.log( 1 - Pi**(1-zsOn/M)*(Pia[:,zHyp]).prod(axis=1) )

	#
	pyiEq0_gvnZ = Pi**(1-zsOn/M)*( Pia[:,zHyp] ).prod(axis=1)
	
	#
	pjoint 	= Qa_part.sum() + Pi_part.sum() + Pia_part.sum() + mix_part.sum()
	cond 	= Pi_part.sum() + Pia_part.sum() + mix_part.sum()

	if verbose: # for debugging
		print('Q part  : ',Q_part.shape,' : ',Q_part)
		print('Qa part  : ',Qa_part.shape,' : ',Qa_part)
		print('Pi part : ',Pi_part.shape,' : ',Pi_part)
		print('Pia part: ',Pia_part.shape,' : ',Pia_part)
		print('mix part: ',mix_part.shape,' : ',mix_part)

	return pjoint, cond, Q_part, Pi_part, Pia_part, mix_part, pyiEq0_gvnZ # returning all parts just for debugging.






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def LMAP_qScalar(q, ri, ria, yTru, zHyp, verbose=False):
	# This computes the log joint probability of y-vec and z-vec given the model (Pia, Pi, q). 
	# It returns a single scalar value. Could return an N-vector too.

	N = ria.shape[0]
	M = ria.shape[1]

	Q   = sig(q)
	Pi  = sig(ri)
	Pia = sig(ria)

	yTru 	= list(yTru)
	zHyp 	= list(zHyp)

	zsOn	= len(zHyp) 				# Number of active cell assemblies (1's) in z-vector.
	#
	Q_part	 = np.log( sps.comb(M,zsOn) ) + zsOn*np.log(Q) + (M-zsOn)*np.log(1-Q) # all return an N-vector of p(y_i | z-vec) for all i in N neurons.
	# NOTE: Without the M choose |z| term, inference completely fails (finds all z=0 vectors) for any NoisyInit.
	#       But, With added  M choose |z| term, approx inference sometimes finds that z=0 vector is not highest pjoint, which is mathematically wrong.
	# 		M choose |z| term = " np.log( sps.comb(M,zsOn) ) + "

	Qa_part = 1

	Pi_part  = (1-Y)*(1-zsOn/M)*np.log(Pi) 
	Pia_part = (1-Y)*np.log(Pia[:,zHyp]).sum(axis=1)
	mix_part = Y*np.log( 1 - Pi**(1-zsOn/M)*(Pia[:,zHyp]).prod(axis=1) )

	#
	x = Pi**(1-zsOn/M)*( Pia[:,zHyp] ).prod(axis=1)
	
	#
	pjoint = Q_part + Pi_part.sum() + Pia_part.sum() + mix_part.sum()


	if verbose: # for debugging
		print('q part  : ',Q_part.shape,' : ',Q_part)
		print('Pi part : ',Pi_part.shape,' : ',Pi_part)
		print('Pia part: ',Pia_part.shape,' : ',Pia_part)
		print('mix part: ',mix_part.shape,' : ',mix_part)

	return pjoint, Q_part, Pi_part, Pia_part, mix_part, x # returning all parts just for debugging.




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# def LMAP_1stOrdr_approx(q, ri, ria, yTru, flg_include_Zeq0_infer=False, verbose=False):

# 	N = ria.shape[0]
# 	M = ria.shape[1]

# 	Q   	= sig(q)
# 	Pi  	= sig(ri)
# 	Pia 	= sig(ria)

# 	Y 		= set2boolVec(yTru,N) # convert set of active cells to boolean vector
# 	ysOn	= len(yTru) 				# Number of active cells (1's) in y-vector.

# 	CinZ  	= M*np.log(1-Q)  - ysOn + ( (1-2*Y.astype(int))*np.log(Pi) ).sum() 
# 	Alpha  	= np.log(Q/(1-Q)) - (1/M)*( (1-2*Y.astype(int))*np.log(Pi) ).sum()
# 	Beta 	= ( (1-2*Y.astype(int))[:,None]*np.log(Pia) ).sum(axis=0)
	
# 	# Add the z = {} inferred vector. No cell assemblies active.
# 	#np.concatenate((pJt_z,np.array([Alpha+CinZ])))
# 	if flg_include_Zeq0_infer:
# 		Beta = np.concatenate((Beta,np.array([0])))

# 	pJt_z = Alpha + Beta + CinZ

# 	return pJt_z, Alpha, Beta, CinZ

# def LMAP_2ndOrdr_approx(q, ri, ria, yind, verbose=False):
# 	# This is a second order (pairwise) approximation to the full log-joint (computed in LMAP). This is not working
# 	#
# 	#
# 	# log p(y-vec,z-vec) ~ Cinz + sum_{a=1}^M {z_a * (Alpha + Beta_a)} + sum_{a=1}^M sum_{a'=1}^M {z_a * z_a' * Gamma_{a,a'} }
# 	#	where:
# 	#	Cinz is
# 	#	Alpha	is
# 	#	Beta 	is
# 	#	Gamma 	is

# 	N = ria.shape[0]
# 	M = ria.shape[1]

# 	Q   = sig(q)
# 	Pi  = sig(ri)
# 	Pia = sig(ria)

# 	y 		= set2boolVec(yind,N) # convert set of active cells to boolean vector
# 	ysOn	= len(y)				# Number of active cells (1's) in y-vector.

# 	PiPiaTerm = np.log(Pia) - (1/M)*np.log(Pi)[:,None]

# 	CinZ  	= M*np.log(1-Q) + ( (1-4*y)*np.log(Pi) ).sum() - ysOn - (9/2)*( np.log(Pi)**2 ).sum()

# 	Alpha 	= np.log(Q/(1-Q)) 

# 	Beta  	= ( (1-4*y)[:,None] * PiPiaTerm ).sum(axis=0) - 9*( (y * np.log(Pi))[:,None] * PiPiaTerm ).sum(axis=0)

# 	Gamma	= -(9/2)*np.matmul(y[None,:]*PiPiaTerm.T,y[:,None]*PiPiaTerm)
# 	np.fill_diagonal(Gamma,0)

	
# 	pJt_z = Alpha + (Beta[:,None] + Beta[None,:]) + Gamma + CinZ

# 	# print('debugging 2nd order approx thing')
# 	# print('y = ',y)
# 	# print('y*log(Pi) = ',(y * np.log(Pi))[:,None])
# 	# print('Gamma = ',Gamma.min(),'  ',Gamma.max())
# 	# print('PiPiaTerm = ',PiPiaTerm.min(),'  ',PiPiaTerm.max())


# 	return pJt_z, Alpha, Beta, Gamma, CinZ	






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# def geom_series_log(x,ordr=4):
# 	# Geometric Series expansion of log(1-x)

# 	gx = 0
# 	for n in range(1,ordr+1):
# 		gx += x**n / n 
# 	return gx



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def MAP_inferZ_Comb(yTru, zTru, q, ri, ria, CAs, pJoint_1H, pCond_1H, z0_ind, zActivationHistory, verbose=False):


	# Overall Goal: Compute joint probability of combinations of Cell Assemblies with high 
	# individual pjoints.


	# parameters of the inference algorithm that user can tune. Maybe should depend on q.
	numCAbelowZ0 = 9
	maxNumCAcombos = 10 # THESE VALS WERE 5 & 6

	N = ria.shape[0]
	M = ria.shape[1]	

	Q 	= sig(q)
	Pi 	= sig(ri)
	Pia = sig(ria)

	# A list of sets of which cells are members of all CAs. Useful for diagnosis later.
	if verbose:
		print('cellsInCA not valid. No longer doing this.')
		#cellsInCA = [set(np.where( (1-Pia[:,i])>0.5 )[0]) for i in range(M)]

	# Grab best 1-Hot z-vector and its pjoint value. Can also be z=0 vector.
	z_Best = CAs[0]
	pj_Best = pJoint_1H[0]
	cond_Best = pCond_1H[0]

	# Compute number of combinations of CAs to look at. Uses input parameters.
	numCAcombos = z0_ind + numCAbelowZ0  
	numCAcombos = np.min([ numCAcombos, maxNumCAcombos ])


	CAs_top = list( CAs[:numCAcombos] ) # top N Cell Assemblies (consider combinations of them).
	PJs_top = list( pJoint_1H[:numCAcombos] ) # single CA (1-hot z-vec) pjoint values. Top N.
	#
	# get rid of the z=0 vector so we dont consider it in the combinations.
	try:
		CAs_top.pop(z0_ind)
		PJs_top.pop(z0_ind)
	except:
		xyz=1
	#	print('z=0 vector not in there.')

	if verbose:
		print('  ')
		print(' --- --- --- --- --- --- --- --- ---  --- --- --- --- --- --- --- --- --- ')
		print('Checking combinations of Cell Assemblies with high individual joint probability.')
		#
		print('Number of pjoint values better than z=0 vector::  ',z0_ind) # number of positive pjoint values from 1st order approximation.
		print('Number of Cell Assembly combos to check = ',len(CAs_top))	
		#
		print('***Tru Values: z.',zTru,' --> y.',yTru)
		#
		print(' --- --- --- --- --- --- --- --- ---  --- --- --- --- --- --- --- --- --- ')
		if z0_ind>0:
			print('Best single: pj.',pj_Best,' &  z.',z_Best,' --> y.',cellsInCA[z_Best])
		else:
			print('Best single: pj.',pj_Best,' &  z=0')
		#


		# print(' --- --- --- --- --- --- --- --- ---  --- --- --- --- --- --- --- --- --- ')
		# print(' Top ',numCAcombos,' single CAs.  1-Hot z vectors')
		# print('***Tru Values: z.',zTru,' --> y.',yTru)
		# for j in range( len(CAs_top) ):
		# 	print( j, 'pj:',PJs_top[j],' & z.',CAs_top[j],' --> y.',cellsInCA[CAs_top[j]])



	t0=time.time()
	for j in range(len(CAs_top)):
		SSS = list( it.combinations(CAs_top,j+1) )
		#
		for k in range(len(SSS)):
			z_Comb = set(SSS[k])
			pj_Comb,cond_Comb,_,_,_,_,_ = LMAP(q, ri, ria, yTru, z_Comb, zActivationHistory, verbose=False)
			#
			if pj_Comb >= pj_Best:
				pj_Best = pj_Comb
				cond_Best = cond_Comb
				z_Best = z_Comb
				if verbose:
					print(' --- --- --- --- --- --- --- --- ---  --- --- --- --- --- --- --- --- --- ')
					print('pj.',pj_Comb,' z.',z_Comb,' --> y.',[cellsInCA[z] for z in z_Comb])

	t1=time.time()
	if verbose:
		print('Time: ',t1-t0,' for all CA combos of size ',1,' to ',len(CAs_top))
	#


	#
	if z_Best==M:
		z_Best = set() 
		pyiEq1_gvnZ = 1 - Pi
	else:
		if type(z_Best) != set:
			z_Best = set( [ z_Best ])
		#
		# Here instead of yHyp computed by thresholding p(yi=1|z) at 0.5, pass out the whole pyiEq1_gvnZ.
		# 		yHyp = set( np.where( (1-Pia[ :, list(z_Best) ].prod(axis=1) * Pi)>0.5)[0] )
		zsOn = len(z_Best)
		pyiEq0_gvnZ = Pi**(1-zsOn/M)*(Pia[:,list(z_Best)]).prod(axis=1)
		pyiEq1_gvnZ = 1 - pyiEq0_gvnZ

	if verbose:
		print(' --- --- --- --- --- --- --- --- ---  --- --- --- --- --- --- --- --- --- ')
		print('z_Best = ',z_Best, 'and pj_Best = ',pj_Best,' in MAP_inferZ_1sComb ')
		print('yTru = ', len(yTru), yTru)

	
	return z_Best, pyiEq1_gvnZ, np.float64(pj_Best), np.float64(cond_Best)





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def inferZ_andComb(qp, rip, riap, yTru, zTru, pjt_tol, flg_include_Zeq0_infer, yLo, yHi, zActivationHistory, verbose=False):

	N = riap.shape[0]
	M = riap.shape[1]

	Piap = sig(riap)
	Pip = sig(rip)

	# If |y| too small, force inference to find the z=0 vector.
	if len(yTru) <= yLo:
		zHyp = set()
		pyiEq1_gvnZ = 1 - Pip
		pjt,cond,_,_,_,_,_ = LMAP(qp, rip, riap, yTru, zHyp, zActivationHistory, verbose=False)
		#
		return zHyp, pyiEq1_gvnZ, pjt, cond

	


	# (2). Preallocate memory and compute Full LMAP for the z=0 vector if flg_include_Zeq0_inference = True
	if flg_include_Zeq0_infer:
		pJoint_1H = np.zeros(M+1)
		pCond_1H = np.zeros(M+1)
		zHyp = set()					# Add the z=0 vector to the end too as a possibility.
		pjt,cond,_,_,_,_,_ = LMAP(qp, rip, riap, yTru, zHyp, zActivationHistory, verbose=False)
		#
		pJoint_1H[M] 	= pjt
		pCond_1H[M] 	= cond
	else:
		pJoint_1H = np.zeros(M)
		pCond_1H = np.zeros(M)


	# (3). Compute full LMAP for each 1-hot z-vector - to use it in the MAP_inferZ_Comb function below.
	zHyp = set()
	for a in range(M):
		zHyp.add(a)	
		pjt,cond,_,_,_,_,_ = LMAP(qp, rip, riap, yTru, zHyp, zActivationHistory, verbose=False)
		#
		pJoint_1H[a] 	= pjt
		pCond_1H[a] 	= cond
		#print('z = ',zHyp,' :: pjt = ',pjt)
		zHyp.remove(a)
	pJoint_1H = pJoint_1H.round(decimals=pjt_tol)
	pj_1H_inds = np.argsort(pJoint_1H)[::-1]
	z0_ind = np.where( pj_1H_inds==M )[0][0]




	# (6). Try z-vectors which are combinations of 1-hot z-vectors with the highest LMAP values and compute the Full LMAP of these combined z-vectors
	numPosPjts = z0_ind # Number of cell assemblies that are individually more likely to be active than the z=0 vector.
	zHyp, pyiEq1_gvnZ, pj_zHyp, cond_zHyp = MAP_inferZ_Comb(yTru, zTru, qp, rip, riap, pj_1H_inds, pJoint_1H[pj_1H_inds], pCond_1H[pj_1H_inds], numPosPjts, zActivationHistory, verbose=verbose)



	return zHyp, pyiEq1_gvnZ, pj_zHyp, cond_zHyp











# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def compute_infZstats_single(z, zHyp, M_GT, M_Mod, verbose=False): # y, yHyp, N,

	#if len(zHyp)>0:
	zUnion = list( np.union1d(list(zHyp),list(z)).astype(int) ) 
	zIntersect = list( np.intersect1d(list(zHyp),list(z)).astype(int) ) 
	zDiff = list( np.setdiff1d( zUnion, zIntersect ).astype(int) )
	#
	zSubed = list( np.setdiff1d( zUnion, list(zHyp) ).astype(int) ) 
	zAdded = list( np.setdiff1d( zUnion, list(z)).astype(int) )

	Kinf		= len(zHyp)
	KinfDiff	= len(zHyp) - len(z)
	#
	zCapture 	= len(zIntersect)
	zMissed 	= len(z) - zCapture
	zExtra 		= len(zHyp) - zCapture

	if (len(zAdded)>0 and len(zSubed)==0): 			# if CA is just added, not mixed up. -> (M,j)
		zSubed = [M_GT]
	#
	if (len(zAdded)==0 and len(zSubed)>0):			# if CA is just missed, not mixed up. -> (i,M)
		zAdded = [M_Mod]

	if verbose:
		print(' --- z ---')
		print('zHyp = ', zHyp, '   &  zTrue = ', z)
		#print('zUnion = ', zUnion,'    &  zDiff = ', zDiff)
		print('zAdded = ', zAdded,'  &  zSubed = ', zSubed,' &  zIntersect = ', zIntersect)

	return zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, Kinf, KinfDiff






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 				
def compute_infZstats_allSamples(Z_GT, Z_Mod, M_GT, M_Mod, verbose): # ind_matchMod2GT, ind_matchGT2Mod,# Y_train, Y_inferred, N, 

	num_samps = len(Z_GT)

	print('len Z_GT = ',num_samps)
	print('len Z_Mod = ',len(Z_Mod))

	# Compute statistics on inferred Cell / Cell Assembly activity comparing them to true y and z.  
	# NOTE: Will have to adapt this when true z-vector is not known.
	Kinf_EM 	= np.zeros(num_samps) 							# number of active cell assemblies inferred |z|.
	KinfDiff_EM = np.zeros(num_samps)							# difference between true |z| and inferred |z|.
	#
	zCapture_EM = np.zeros(num_samps) 							# number of active CAs correctly active inferred z.
	zMissed_EM 	= np.zeros(num_samps) 							# number of active CAs incorrectly inactive in inferred z. 
	zExtra_EM 	= np.zeros(num_samps) 							# number of inactive CAs incorrectly active in inferred z.
	#
	inferCA_Confusion_EM 	= np.zeros( (M_Mod+1,M_GT+1) ) 		# confusion matrix for Cell Assemblies - when one is active and another inferred.
	zInferSampled_EM 		= np.zeros( M_Mod+1 ) 					# count / histogram of number of times each CA is inferred (after permute translation).
	zInferSampledRaw_EM 	= np.zeros( M_Mod+1 ) 					# count / histogram of number of times each CA is inferred (raw means without permute translation).
	zInferSampledT_EM 		= np.zeros( M_GT+1 )					# count / histogram of number of times each CA is active in ground truth.
	#

	for sample in range(num_samps):

		zTru = Z_GT[sample]
		zHyp = Z_Mod[sample]

		zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, Kinf, KinfDiff \
		= compute_infZstats_single( zTru, zHyp, M_GT, M_Mod, verbose=verbose )
		#
		Kinf_EM[sample] 	= Kinf
		KinfDiff_EM[sample]	= KinfDiff
		zCapture_EM[sample] = zCapture
		zMissed_EM[sample] 	= zMissed 
		zExtra_EM[sample] 	= zExtra
		#
		# Compute confusion matrix between inferred z's (zHyp) and true z's (z). (see below for details ...)
		inferCA_Confusion_EM[ np.ix_(zAdded,zSubed) ]+=1		# add to off-diagonal (i,j) for mix-ups
		inferCA_Confusion_EM[( zIntersect,zIntersect )]+=1 		# add to diagonal (i,i) for correct inference
		zInferSampled_EM[list(zHyp)]+=1 						# keeping track of how many times each cell assembly was active in an inference sample.
		zInferSampledT_EM[list(zTru)]+=1 
		#
	
	return Kinf_EM, KinfDiff_EM, zCapture_EM, zMissed_EM, zExtra_EM, inferCA_Confusion_EM, zInferSampled_EM, zInferSampledT_EM






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def mapTo_matchInds(Z, ind_match, verbose=False):
	#
	# Maybe dont even use it :/

		#z = list(Z)
		#print(Z)
		if len(Z)>0:
			Z_match = set([ind_match.index(z) for z in Z])
		else:
			Z_match = set()	

		if verbose:
			# print('N=',N,' M=',M,' M_mod=',M_mod)
			# print('# unique begining of ind_match ', len( np.unique(ind_match[:M_mod]) ) )
			print(Z, ' --> ', Z_match)

		return Z_match






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# def compute_Frequentist_Sample_Model(Z_train, Y_train, verbose=False):





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def compute_Confusion_Stats(Confusion,flgAddDrop):
	# From a confusion matrix, count up all correct (diagonal entries), mixed up (off diagonals),
	# Dropped (bottom row), Added (last column) and total (all off diagonals entries except for 
	# last row & column.)

	if flgAddDrop:
		Right = Confusion[:-1,:-1].diagonal().sum()
		Dropped = Confusion[-1].sum()
		Added = Confusion[:,-1].sum()
		MixedUp = Confusion.sum() - (Right + Added + Dropped)
		#
		return np.array( [Right, MixedUp, Dropped, Added] ).astype(int)

	else:
		Right = Confusion.diagonal().sum()
		MixedUp = Confusion.sum() - Right
		#
		return np.array( [Right, MixedUp] ).astype(int)

		TH = 0.5
		np.max(CA_ovl-np.eye(M))
		np.sum(CA_ovl-np.eye(M))/((M*(M-1)))
		(np.where((CA_ovl-np.eye(M))>TH)[0]).size/2
	


	





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def init_parameters(q, ri, ria, param_init, sig_init, C_noise):

	N = ria.shape[0]
	M = ria.shape[1]

	if(param_init=='True'): # {Initialization #1}. Start model at the correct answer. Confirm it does not wander away.
		riap 	= ria 
		rip 	= ri 
		qp 		= q 
		param_init_param = ''
	# --------------------------------------------
	elif(param_init=='NoisyTrue'): # {Initialization #2}. Start model at the correct answer + some noise. Confirm it converges to correct solution.
		Q   = sig(q)
		Pi  = sig(ri)
		Pia = sig(ria)
		#
		riap 	= inv_sig( noiseOnProbability(Pia,sig_init[2]) )
		rip 	= inv_sig( noiseOnProbability(Pi,sig_init[1]) )
		qp 		= inv_sig( noiseOnProbability(Q,sig_init[0]) )
		#
		sig_init_str = str(np.round(sig_init,2))
		param_init_param = str( '_sl' + sig_init_str.replace('.','pt') )

	# --------------------------------------------
	elif(param_init=='RandomUniform'): # {Initialization #3}. Initialize all Random Uniform.
		riap 	= inv_sig( np.random.uniform(low=0, high=1, size=(N,M)) )
		rip 	= inv_sig( np.random.uniform(low=0, high=1, size=(N)) )
		qp 		= inv_sig( np.random.uniform(low=0, high=1, size=q.shape) )
		param_init_param = ''
	# --------------------------------------------
	elif(param_init=='DiagonalPia' or param_init=='DiagonalPia_ratePiFixed'): # {Initialization #3}. Initialize Pia to noisy diagonal

		success = False
		while not success:
			riap 	= inv_sig( noiseOnProbability(C_noise[2]*(1-np.eye(N,M)), sig_init[2]) )	
			rip 	= inv_sig( noiseOnProbability(C_noise[1]*np.ones_like(ri),sig_init[1]) )	
			qp 		= inv_sig( noiseOnProbability(C_noise[0]*np.ones_like(q), sig_init[0]) )	
			#
			sig_init_str = str(np.round(sig_init,2))
			C_noise_str = str(np.round(C_noise,2))
			param_init_param = str( C_noise_str.replace('.','pt') + '_sl' + sig_init_str.replace('.','pt') )

			print('any nans? ',np.any(np.isnan(riap)) )
			print('max riap ',riap.max() )
			print('min riap ',riap.min() )

			if not ( np.any(np.isnan(riap)) or np.any(np.isnan(rip)) or np.isnan(qp) ):
				print('SUCCCESS')
				success=True
			else:
				print('Error, there is a NAN in riap. Success = ', success ,'REDO.')
				# plt.imshow(1-sig(riap))
				# plt.colorbar()
				# plt.show()


	# --------------------------------------------
	elif(param_init=='NoisyConst'): # {Initialization #4}. Initialize all P's to 0.5 with some gaussian variability for symmetry breaking (that is r's to 0)
		riap 	= inv_sig( noiseOnProbability(C_noise[2]*np.ones_like(ria),sig_init[2]) )	
		rip 	= inv_sig( noiseOnProbability(C_noise[1]*np.ones_like(ri),sig_init[1]) )	
		qp 		= inv_sig( noiseOnProbability(C_noise[0]*np.ones_like(q),sig_init[0]) )	
		#
		sig_init_str = str(np.round(sig_init,2))
		C_noise_str = str(np.round(C_noise,2))
		param_init_param = str( C_noise_str.replace('.','pt') + '_sl' + sig_init_str.replace('.','pt') )
	# --------------------------------------------
	else:
		print('duh?, What you talkin bout Willis?')	

	return qp, rip, riap, param_init_param






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 											
def learn_model_params(qp, rip, riap, q, ri, ria, yTru, zHyp, zActivationHistory, learning_rate, sample, samps2snapshot, lRateScale, verbose=False):	

	N = riap.shape[0]
	M = riap.shape[1]

	# Convert learned model parameters to probabilities.
	Qp 	 = sig(qp)
	Pip  = sig(rip)
	Piap = sig(riap)

	# Convert true model parameters to probabilities.
	Q 	= sig(q)
	Pi 	= sig(ri)
	Pia = sig(ria)

	yTru = list(yTru)
	zHyp = list(zHyp)

	Y = set2boolVec(yTru,N) # set of active cells / CAs into boolean vector [0s, & 1s]
	Z = set2boolVec(zHyp,M)


	zRelActInv = ( zActivationHistory/zActivationHistory.mean() )**-1
	Qap = Qp*zRelActInv

	# Compute derivatives of LMAP w.r.t model parameters {ri, ria, q} : 
	#
	# (0). Compute common quantities for simplicity below...
	zsOn = Z.sum()/M 					# Scalar: Percentage of cell assemblies that are active in z-vector.
	C 	 = Pip**(1-zsOn)*(Piap[:,zHyp]).prod(axis=1) # Conditional probability y is off. p(yi=0|z)

	Crat = C/(1-C)
	Crat[Crat>1e6]=1e6
	Crat[np.isnan(Crat)]=1e6

	#
	# (1). { d(log.p(y,z)) / d(q) }.  Scalar value.
	# dq = (1-Qp)*zsOn - (1-zsOn)*Qp ( old way without the Qa-vector )
	dq = (1-Qp)*( Z - (1-Z)*( Qap/(1-Qap) ) ).sum()

	#
	# (2). { d(log.p(y,z)) / d(ri) }.  N-vector
	dri = (1-zsOn)*(1-Pip)*( (1-Y) - Y*Crat )

	#
	# (3). { d(log.p(y,z)) / d(ria) }. NxM-matrix
	dria = Z[None,:]*(1-Piap)*( (1-Y) - Y*Crat )[:,None]


	# collect stats of derivatives w.r.t. each parameter for plotting and debugging
	q_deriv		= dq
	ri_deriv	= [ np.sign(dri.mean())*np.abs(dri).mean(), 	np.abs(dri).std(), 	np.abs(dri).max(),
						dri.mean(), 	dri.std(),		dri.max(), 		dri.min()	]
	ria_deriv	= [ np.sign(dria.mean())*np.abs(dria).mean(), 	np.abs(dria).std(), np.abs(dria).max(),
						dria.mean(), 	dria.std(),		dria.max(), 		dria.min()	]


	# Apply Gradients.
	riap 	= riap 	+ lRateScale[0] * learning_rate * dria
	rip 	= rip 	+ lRateScale[1] * learning_rate * dri
	qp 		= qp 	+ lRateScale[2] * learning_rate * dq

	# Compute LMAP - p(y-vec,z-vec)			
	pjoint,_,_,_,_,_,_ = LMAP(qp, rip, riap, yTru, zHyp, zActivationHistory)

	# Convert learned model parameters to probabilities.
	Qp 	 = sig(qp)
	Pip  = 1-sig(rip)
	Piap = 1-sig(riap)

	#
	Q 	= sig(q)
	Pi 	= 1-sig(ri)
	Pia = 1-sig(ria)

	Pi_SE =  [ (Pi-Pip).mean(), (Pi-Pip).std(), (Pi-Pip).max(), (Pi-Pip).min(), Pi.size ] 	
	Pi_AE =  [ np.abs(Pi-Pip).mean(), np.abs(Pi-Pip).std(), np.abs(Pi-Pip).max(), Pi.size  ] 
	#
	Q_SE =  Q-Qp # abs err contains no new information since there is just one number.


	if sample in samps2snapshot:
		print(sample)
		ria_snapshot = riap
		ri_snapshot  = rip
		q_snapshot 	 = qp
	else:
		ria_snapshot = np.array([False])
		ri_snapshot  = np.array([False])
		q_snapshot   = np.array([False])


	# Print out errors (in y, Pia, Pi and q) for debugging and diagnostics
	# Display iteration and error.
	#
	if verbose:
		print('- - - - Parameter Derivatives in Learning at sample# ',sample,' - - - -')
		print('dq: ',dq )
		print('dri  {max, mean, meanAbs, min}: ',dri.max(), dri.mean(), np.abs(dri).mean(),  dri.min() )
		print('dria {max, mean, meanAbs,min}: ',dria.max(), dria.mean(), np.abs(dria).mean(), dria.min() )
		print('+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +')
		print('')


	# Introduce stopping / convergence criteria: (a). derivs below some threshold OR (b). max number of iterations.
	# print('Sample #',sample,' :: Derivatives of parameters in learning to determine convergence.')
	# print('dq   :: ', dq)
	# print('dri  :: ', dri)
	# print('dria :: ', dria)



	return qp, rip, riap, pjoint, Q_SE, Pi_SE, Pi_AE, q_deriv, ri_deriv, ria_deriv, ria_snapshot, ri_snapshot, q_snapshot




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def compute_errs_Pia_snapshots( ria_snapshots, ria_GT, ind, cosSim, lenDif ):


	num_snaps = ria_snapshots.shape[0]

	# Compute Absolute Value and Signed Errors of Pia using translations.
	Pia_AE 		= np.zeros( (num_snaps, 4) )	# mean, std, max, numElements
	PiaOn_SE 	= np.zeros( (num_snaps, 5) )	# mean, std, max, min, numElements
	PiaOn_AE 	= np.zeros( (num_snaps, 4) )	
	PiaOff_SE 	= np.zeros( (num_snaps, 5) )	
	PiaOff_AE	= np.zeros( (num_snaps, 4) )	
	num_simSig_snaps = np.zeros( num_snaps )  # number of dot products between Pia and Piap are significant (>0.1)


	simil_signif = np.where(cosSim[0]>0.1)[0]

	#
	if len(simil_signif)==0:
		return Pia_AE, PiaOn_SE, PiaOn_AE, PiaOff_SE, PiaOff_AE, num_dpSig_snaps 


	for i in range( num_snaps ):
		#print(i) 
		Piap_Mod = sig(ria_snapshots[i])
		Pia_GT = sig(ria_GT)

		M_GT = Pia_GT.shape[1]
		M_Mod = Piap_Mod.shape[1]

		print(M_GT, M_Mod)

		x = (1 - Pia_GT)[:,ind[1] ] 		# GT model.
		x = x[ :, simil_signif] 			# index into GT model where there is significant overlap of CAs with learned model

		y = (1 - Piap_Mod)[:,ind[0]]		# Learned model at snapshot
		y = y[ :, simil_signif] 			# index into learned model where there is significant overlap of CAs with GT model


		#
		Pia_AE[i,:] = np.array([ np.abs(x-y).mean(), np.abs(x-y).std(), np.abs(x-y).max(), x.size ]) 
		#
		# Compute Error on Signed Values (SE) in all Pia elements of Cell that are members of CA.
		maskMemb 	= x > 1e-8
		PiaM 		= x[maskMemb]
		PiapM 		= y[maskMemb]
		PiaOn_SE[i,:] 	= [ (PiaM-PiapM).mean(), (PiaM-PiapM).std(), (PiaM-PiapM).max(), (PiaM-PiapM).min(), maskMemb.sum() ] 
		PiaOn_AE[i,:] 	= [ np.abs(PiaM-PiapM).mean(), np.abs(PiaM-PiapM).std(), np.abs(PiaM-PiapM).max(), maskMemb.sum() ] 
		#
		# Compute Error on Signed Values (SE) inall Pia elements of Cell that are not members of CAs
		notMemb 	= (1-maskMemb).astype(bool)
		PiaN 		= x[notMemb]
		PiapN 		= y[notMemb]
		PiaOff_SE[i,:] 	= [ (PiaN-PiapN).mean(), (PiaN-PiapN).std(), (PiaN-PiapN).max(), (PiaN-PiapN).min(), notMemb.sum() ] 
		PiaOff_AE[i,:] 	= [ np.abs(PiaN-PiapN).mean(), np.abs(PiaN-PiapN).std(), np.abs(PiaN-PiapN).max(), notMemb.sum() ] 
		#
		num_simSig_snaps[i] = len(simil_signif)

	print(num_simSig_snaps)	

			
	return Pia_AE, PiaOn_SE, PiaOn_AE, PiaOff_SE, PiaOff_AE, num_simSig_snaps 	






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def run_EM_algorithm( qp, rip, riap, q, ri, ria, q_init, ri_init, ria_init, Zs_train, Zs_test, SWs_train, SWs_test, xVal_snapshot, xVal_batchSize, samps2snapshot_4Pia, \
					pjt_tol, learning_rate, lRateScale, flg_include_Zeq0_infer, yLo, yHi, flg_EgalitarianPrior, flg_sample_longSWs_1st, verbose_EM ):



	print('Learning PG Model on real retina with EM algorithm: ')
	print('Note: In run_EM_algorithm. Not using num_SWs right now in real Data. Using len(SWs_train) and len(SWs_test). Do I use it in Synth data?')
	print('Note: In run_EM_algorithm. Not using Zs_train or Zs_test within for real data. Do I use it in Synth data?')


	num_EM_samps = len(SWs_train) # Use each training data point once. IN construct_crossvalidation, they were randomly chosen without replacement.
	M = riap.shape[1]

	# Preallocate memory to hold Learning and Inference results.
	Z_inferred_train	= list() # which cell assemblies get inferred and when.
	pyiEq1_gvnZ_train	= list() # which cells get inferred and when. (conditional prob)
	pj_zHyp_train  		= np.zeros(num_EM_samps)
	cond_zHyp_train 	= np.zeros(num_EM_samps)
	#
	Z_inferred_test		= list() # which CAs get inferred and when.
	pyiEq1_gvnZ_test	= list() # which cells get inferred and when. (conditional prob)
	pj_zHyp_test  		= np.zeros(num_EM_samps)
	cond_zHyp_test 		= np.zeros(num_EM_samps)
	#
	ria_snapshots		= list()
	ri_snapshots 		= list()
	q_snapshots			= list()
	#
	# pjt_mnDs_train 	= list()
	# pjt_mnDs_test		= list()

	pj_zTru_Batch		= list()
	pj_zTrain_Batch		= list()
	pj_zTest_Batch		= list()
	#
	cond_zTru_Batch		= list()
	cond_zTrain_Batch	= list()
	cond_zTest_Batch	= list()
	#
	Q_SE 				= np.zeros(  num_EM_samps    ) 
	Pi_SE 				= np.zeros( (num_EM_samps,5) ) # 4 allows room for mean, std, max, min, numElements. 
	Pi_AE 				= np.zeros( (num_EM_samps,4) ) # 3 allows room for mean, std, max, numElements.
	#
	q_deriv				= np.zeros(  num_EM_samps    ) 
	ri_deriv 			= np.zeros( (num_EM_samps,7) ) # 7 allows room for mean, std, max, etc of derivatives.
	ria_deriv 			= np.zeros( (num_EM_samps,7) )
	#
	zActivationHistory 	= np.ones( M ) # keep a running tally of activations of each CA to weight scalar Q value
											# and make a Q vector where Qa is Q weighted by individual activation and
											# the mean activation. (set to 1's to avoid inf's and nan's since a ratio
											# is being computed to form Q_a or Q-vector.)

	
	# RUN FULL EXPECTATION MAXIMIZATION ALGORITHM
	success = False
	while not success:
		success = True

		# NOTE: no longer randomly sampling train data vector because we are sampling each data point from it 
		# 		randomly without replacement. And the order has already been randomized in construct_crossValidation_SWs.
		#		Maybe for test data, we can randomly sample multiple data points and take mean pjoint to smooth out the 
		# 		curve for cross validation purposes.
		try:
			tstUB = np.random.choice( np.arange(len(SWs_test)), size=len(SWs_train), replace=False ) 
		except:
			tstUB = np.random.choice( np.arange(len(SWs_test)), size=len(SWs_train), replace=True ) # a bandaid for now if there are more training samples than test samples
		
		# if flg_sample_longSWs_1st != 'Dont': 
		# 	trnUB = np.random.choice(np.arange(len(SWs_train)), size=len(SWs_train), replace=False) 


		for sample in range(num_EM_samps):

			# (A). INFERENCE STEP (INFER Z VECTOR GIVEN FIXED MODEL AND OBSERVED Y VECTOR)
			if verbose_EM:
				print('')
				print('~ ~ ~ ~ ><((((> ~ ~ ~ ~ ><((((> ~ ~ ~ ~ ><((((> ~ ~ ~ ~ ><((((> ~ ~ ~ ~')
				print('EM Iteration  #',sample)


			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
			#
			# (1). Run Inference on sampled Spike Word
			#
			yTru_test  = list(SWs_test[ tstUB[sample] ])		# actual cells involved in sample from test set.
			yTru_train = list(SWs_train[sample]) 				# actual cells involved in sample from training set.
			# yTru_trainUB = list(SWs_train[trnUB[sample]])
			#
			try:
				zTru_test = list(Zs_test[ tstUB[sample] ]) 	
				zTru_train = list(Zs_train[sample]) 		
				# zTru_trainUB = list(Zs_train[trnUB[sample]])
			except:
				zTru_test = list() 				# This should happen with real data when we dont know zTrue	
				zTru_train = list() 						
				# zTru_trainUB = list()
			#	
			# # 
			#
			# Compute joint and conditional probabilities using Test set data.
			zHyp_test, pyiEq1_gvnZ, pj_zHyp_test1, cond_zHyp_test1 = inferZ_andComb( qp, rip, riap, yTru_test, zTru_test, \
				pjt_tol, flg_include_Zeq0_infer, yLo, yHi, zActivationHistory, verbose=verbose_EM) 
			#										
			pj_zHyp_test[sample] 	= pj_zHyp_test1 
			cond_zHyp_test[sample] 	= cond_zHyp_test1
			Z_inferred_test.append(zHyp_test)
			pyiEq1_gvnZ_test.append(pyiEq1_gvnZ) 	# HAVE TO MAKE THESE LIST OF LISTS.
			#
			# #
			#
			# Compute joint and conditional probabilities of observed Y using inferred Z and learned model. (biased sampling)
			zHyp_train, pyiEq1_gvnZ, pj_zHyp_train1, cond_zHyp_train1 = inferZ_andComb( qp, rip, riap, yTru_train, zTru_train, \
				pjt_tol, flg_include_Zeq0_infer, yLo, yHi, zActivationHistory, verbose=verbose_EM ) 
			#
			pj_zHyp_train[sample] 	= pj_zHyp_train1
			cond_zHyp_train[sample]	= cond_zHyp_train1
			Z_inferred_train.append(zHyp_train)
			pyiEq1_gvnZ_train.append(pyiEq1_gvnZ)
			#
			# #
			#
			# # Compute joint and conditional probabilities of observed Y using true Z and true model. (unbiased sampling)
			# _,_, pj_zHyp_trainUB1, cond_zHyp_trainUB1 = inferZ_andComb(qp, rip, riap, yTru_trainUB, zTru_trainUB, \
			# 	pjt_tol, flg_include_Zeq0_infer, yLo, yHi, zActivationHistory, verbose=verbose_EM)
			# #
			# pj_zHyp_trainUB[sample] 	= pj_zHyp_trainUB1
			# cond_zHyp_trainUB[sample] 	= cond_zHyp_trainUB1
			
			#
			# # Compute joint and conditional probabilities of observed Y using true Z and true model.
			# try:
			# 	pj_zTru1, cond_zTru1, _,_,_,_,_ = LMAP(q, ri, ria, yTru_train, zTru_train, zActivationHistory)
			# 	pj_zTru[sample] 				= pj_zTru
			# 	cond_zTru[sample] 				= cond_zTru
			# except:
			# 	z=1 # do nothing.



			################################################################

			## BATCH COMPUTE Joints and Conditionals with Fixed Model 
			## and fixed zActivationHistory at set time intervals.

			if np.mod(sample,xVal_snapshot)==0:

				# THESE ARE UNBIASED BECAUSE OF THIS RANDOM CHOICE SAMPLING RIGHT HERE..
				ind_test 	= np.random.choice(np.arange(len(SWs_test)), size=xVal_batchSize, replace=False) 
				ind_train 	= np.random.choice(np.arange(len(SWs_train)), size=xVal_batchSize, replace=False) 
				#
				yTru_testB 	= [ SWs_test[i] for i in ind_test ]
				yTru_trainB = [ SWs_train[i] for i in ind_train ]
				#
				try:
					zTru_testB 	= [ Zs_test[i] for i in ind_test ]
					zTru_trainB = [ Zs_train[i] for i in ind_test ]
				except:
					zTru_testB 	= [ list() for i in ind_test ]
					zTru_trainB = [ list() for i in ind_train ]




				# preallocate memory for joint and conditional values as we loop over samples in batch.	
				pj_zHyp_testB 		= np.zeros(xVal_batchSize)
				cond_zHyp_testB 	= np.zeros(xVal_batchSize)
				pj_zHyp_trainB 		= np.zeros(xVal_batchSize)
				cond_zHyp_trainB 	= np.zeros(xVal_batchSize)
				pj_zTruB 			= np.zeros(xVal_batchSize)
				cond_zTruB			= np.zeros(xVal_batchSize)


				for bs in range(xVal_batchSize):

					# # # # # # # # # # # # # # # # # # # # # # # #
					#
					# (1). For test data which is unbiased already with learned model parameters
					_,_, pj_zHyp_test1, cond_zHyp_test1 = inferZ_andComb( qp, rip, riap, yTru_testB[bs], zTru_testB[bs], \
							pjt_tol, flg_include_Zeq0_infer, yLo, yHi, zActivationHistory, verbose=verbose_EM)							
					#
					pj_zHyp_testB[bs] = pj_zHyp_test1
					cond_zHyp_testB[bs] = cond_zHyp_test1

				
					# # # # # # # # # # # # # # # # # # # # # # # #
					#
					# (2). For unbiased sampling from train data with learned model parameters
					_,_, pj_zHyp_train1, cond_zHyp_train1 = inferZ_andComb( qp, rip, riap, yTru_trainB[bs], zTru_trainB[bs], \
							pjt_tol, flg_include_Zeq0_infer, yLo, yHi, zActivationHistory, verbose=verbose_EM)							
					#
					pj_zHyp_trainB[bs] = pj_zHyp_train1
					cond_zHyp_trainB[bs] = cond_zHyp_train1


					# # # # # # # # # # # # # # # # # # # # # # # #
					#
					# (3). For unbiased sampling from train data with ground truth model parameters.
					_,_, pj_zTru1, cond_zTru1 = inferZ_andComb( q, ri, ria, yTru_trainB[bs], zTru_trainB[bs], \
							pjt_tol, flg_include_Zeq0_infer, yLo, yHi, zActivationHistory, verbose=verbose_EM)							
					#
					pj_zTruB[bs] = pj_zTru1
					cond_zTruB[bs] 	= cond_zTru1


				# Take mean and STD across samples in batch.
				pj_zTru_Batch.append([ pj_zTruB.mean(), pj_zTruB.std() ])
				pj_zTrain_Batch.append([ pj_zHyp_trainB.mean(), pj_zHyp_trainB.std() ])
				pj_zTest_Batch.append([ pj_zHyp_testB.mean(), pj_zHyp_testB.std() ])
				#
				cond_zTru_Batch.append([ cond_zTruB.mean(), cond_zTruB.std() ])
				cond_zTrain_Batch.append([ cond_zHyp_trainB.mean(), cond_zHyp_trainB.std() ])
				cond_zTest_Batch.append([ cond_zHyp_testB.mean(), cond_zHyp_testB.std() ])
		


			################################################################


			
			#
			# Update zActivation history vector to adjust Qa vector.
			if flg_EgalitarianPrior:
				zActivationHistory[list(zHyp_train)]+=1


			#
			# (B). LEARNING STEP ON TRAINING DATA. Update model parameter using inferred Z's and observed Y vector.
			qp, rip, riap, pjoint1, Q_SE1, Pi_SE1, Pi_AE1, q_deriv1, ri_deriv1, ria_deriv1, ria_snapshot, \
			ri_snapshot, q_snapshot = learn_model_params( qp, rip, riap, q, ri, ria, yTru_train, zHyp_train, \
				zActivationHistory, learning_rate, sample, samps2snapshot_4Pia, lRateScale, verbose=verbose_EM ) # PiaOn_SE1, PiaOn_AE1, PiaOff_SE1, PiaOff_AE1,
				

			# Collect up changes in parameters and learning statistics for each sample. For plots later.
			Q_SE[sample] 		= Q_SE1
			Pi_SE[sample,:] 	= Pi_SE1
			Pi_AE[sample,:] 	= Pi_AE1
			#
			if ria_snapshot.any():
				ria_snapshots.append(ria_snapshot)
				ri_snapshots.append(ri_snapshot)
				q_snapshots.append(q_snapshot)
				#
				# iterNum = len(q_snapshots)-1
				# pjt_mnDs_train.append(pjoint_train[iterNum-1:iterNum].mean())
				# pjt_mnDs_test.append(pjoint_test[iterNum-1:iterNum].mean())
			#
			q_deriv[sample] 	 	= q_deriv1
			ri_deriv[sample,:] 		= ri_deriv1
			ria_deriv[sample,:] 	= ria_deriv1
			#

			if ( np.any(np.isnan(qp)) or np.any(np.isnan(rip)) or np.any(np.isnan(riap)) ):
				print('Inside run_EM_algorithm, found a nan in parameters. Starting over. Ugh.')
				qp = q_init
				rip = ri_init
				riap = ria_init
				print('Any nan in q_init?', np.any(np.isnan(q_init)))
				print('Any nan in ri_init?', np.any(np.isnan(ri_init)))
				print('Any nan in ria_init?', np.any(np.isnan(ria_init)))
				success=False
				break


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# Convert lists to sets because np.savez and np.load effectively do this automatically.	
	ria_snapshots 		= np.asarray(ria_snapshots) # convert list of 2D-arrays into a 3D-array
	ri_snapshots 		= np.asarray(ri_snapshots)
	q_snapshots 		= np.asarray(q_snapshots)
	Z_inferred_train 	= np.asarray(Z_inferred_train) # convert list of sets to an array of sets
	Z_inferred_test 	= np.asarray(Z_inferred_test) # convert list of sets to an array of sets



	return qp, rip, riap, Z_inferred_train, Z_inferred_test, pyiEq1_gvnZ_train, pyiEq1_gvnZ_test, \
		ria_snapshots, ri_snapshots, q_snapshots, q_deriv, ri_deriv, ria_deriv, Q_SE, Pi_SE, Pi_AE, \
		pj_zHyp_train, pj_zHyp_test, pj_zTru_Batch, pj_zTrain_Batch, pj_zTest_Batch, \
		cond_zHyp_train, cond_zHyp_test, cond_zTru_Batch, cond_zTrain_Batch, cond_zTest_Batch, zActivationHistory
		
		# maybe pass out "zActivationHistory" also...











# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def compute_conditional_scalar(z, y, ria, ri):


	N = ria.shape[0]
	M = ria.shape[1]

	Pi = sig(ri)
	Pia = sig(ria)

	zsOn = len(z)
	pyiEq0_gvnZ = Pi**(1-zsOn/M)*(Pia[:,list(z)]).prod(axis=1)
	pyiEq1_gvnZ = 1 - pyiEq0_gvnZ

	Yon_bool = np.zeros(N).astype(bool)
	Yon_bool[list(y)] = True
	Yoff_bool = (1-Yon_bool).astype(bool)

	cond = np.log(pyiEq1_gvnZ[Yon_bool]).sum() + np.log(pyiEq0_gvnZ[Yoff_bool]).sum()

	return cond






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def compute_dataGen_Histograms( Y_list, Z_list, M, N ):
	#
	# have to do this try except statement because snapshot 0 makes a problem. Small bandaid.
	try:
		yDtype = type(Y_list[0])
	except:
		yDtype = type(Y_list)
	#
	if yDtype is set: 
		print('Y contains sets. Must be observed y-vec, not inferred p(yi=1|z)')
		Ydata = 'obs'
	elif yDtype is np.ndarray: 
		print('Y contains ndarrays. Must be vector of probabilities - p(yi=1|z)')
		Ydata = 'inf'
	else:
		print('The type of Y does not make sense.')
	#
	numSWs = len(Y_list)
	#
	Ycell_hist  = np.zeros(N)				# how often each cell is active (single cell stat)
	Zassem_hist = np.zeros(M+1).astype(int) # how often each assembly is active (single CA stat)
	#
	Ylen_hist	= np.zeros(numSWs)			   # how many cells are active in each spike-word
	Zlen_hist	= np.zeros(numSWs).astype(int) # how many cell assemblies are active at one time
	#
	CA_coactivity = np.zeros( (M,M) ).astype(int) # matrix of pairwise coactivity of cell assemblies
	Cell_coactivity = np.zeros( (N,N) )			  # matrix of pairwise coactivity of cells
	#
	for i in range(numSWs):
		#
		# # # # # # # # # # # # # #
		# For Cells Y
		if Ydata == 'obs':
			try:
				ysOn = list(Y_list[i]) # for Y_inferred, it is possible that |y|=0
				Ycell_hist[ysOn] +=1
			except:
				print('no cells active.')
				# ysOn=list() 
				# Ycell_hist[N]	 +=1 
				#	
			Ylen_hist[i] = len(ysOn)
			Cell_coactivity[np.ix_(ysOn,ysOn)] 	+=1
		elif Ydata == 'inf':
			Ylen_hist[i] 	= Y_list[i].sum()
			Cell_coactivity += np.outer(Y_list[i],Y_list[i])
			Ycell_hist 		+= Y_list[i]

		else:
			print('Never gets here.')
		#
		# # # # # # # # # # # # # #
		# For Cell Assemblies Z
		try:
			zsOn = list(Z_list[i]) 
			Zassem_hist[zsOn] +=1	
		except:
			zsOn=list() # This except statement is to deal with real data when we dont know Ztrue and we construct one just to generate this plot.
			Zassem_hist[M]	  +=1
		#
		Zlen_hist[i] = len(zsOn)
		CA_coactivity[np.ix_(zsOn,zsOn)] +=1
		

	# set diagonal of CA_coactivity to zero (because it swamps the other information).
	CA_coactivity[( range(M),range(M) )] = 0
	Cell_coactivity[( range(N),range(N) )] = 0	

	nY = Ylen_hist
	nZ = Zlen_hist

	return Ycell_hist, Zassem_hist, nY, nZ, CA_coactivity, Cell_coactivity
	


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def compute_CA_Overlap(ria):

	# # MAYBE USE MAT MUL HERE? LIKE:
	# xx = np.matmul((1-Pia).T,(1-Piap))
	# x = (1-Pia).sum(axis=0)
	# y = (1-Piap).sum(axis=0)

	# plt.imshow(xx / np.sqrt(x[:,None]*y[None,:]) )
	# plt.colorbar()
	# plt.title('Permutation of learned Pia matches GT Pia better?')
	# plt.show()

	M = ria.shape[1]
	Pia = sig(ria) 
	CA_ovl = np.zeros( (M,M) )
	for i in range(M):
		for j in range(i,M):
			CA_ovl[i,j] = np.dot((1-Pia[:,i]),(1-Pia[:,j]))/(np.linalg.norm((1-Pia[:,i]))*np.linalg.norm((1-Pia[:,j])))
			CA_ovl[j,i] = CA_ovl[i,j]

	return CA_ovl	






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def translate_CAs_LrnAndTru(A, B, verbose=True):
	#





	M_a = A.shape[1] 	# number of columns / CAs in matrix A
	M_b = B.shape[1] 	# number of columns / CAs in matrix B
	N = A.shape[0]		# number of rows / cells in matrix A&B




	# Preallocate memory ...
	translate = -1*np.ones( np.max([M_a,M_b]) ).astype(int)  	# for each CA in matrix A, the corresponding best fit CA in matrix B
	translate2 = -1*np.ones( np.max([M_a,M_b]) ).astype(int) 	# for each CA in matrix A, the corresponding 2nd best fit CA in matrix B
	dot_prod = np.empty( np.max([M_a,M_b]) ) 					# for each CA in matrix A, overlap of best fit CA in matrix B
	dot_prod2 = np.empty( np.max([M_a,M_b]) )					# for each CA in matrix A, overlap of 2nd best fit CA in matrix B


	# # Compute matrix of Dot products between all columns in matrix A & all columns in 
	# # matrix B normalized by their average length (Cosine similarity of CA pairs).
	# xx = np.matmul(A.T,B) 	# np.dot ??
	# x = A.sum(axis=0) 		# np.sqrt( (xx**2).sum(axis=0) ) ??
	# y = B.sum(axis=0)
	# Perm = xx / np.sqrt(x[:,None]*y[None,:]) 

	Perm = cosSim(A.T,B.T)






	# If there are more columns in matrix B, then we can find a unique CA in B for each CA in A.
	# That is, dont repeat CAs before we use every single one in A. 
	if M_a > M_b:
		unique=False 
	else:
		unique=True	

	print('unique = ',unique,'. In A,', M_a,' and in B, ',M_b)	


	########################################################################################
	#
	# Sorting the Perm (Cosine Similarity Matrix) in order to make it diagonally dominant.
	# 	1st, sort columns by their largest overlap between two columns in the Pia matrix pair --> {index of translate}
	# 	2nd, for each column, find the entry that has the largest overlap in that column 	  --> {value of translate}
	#
	rowSort = np.argsort( Perm.max(axis=1) )[::-1]  # order columns of Perm by the size of their brightest element. Map strongest matching learned/model Cell Assembly first. /Perm.mean(axis=1)
	#

	print(rowSort, rowSort.shape)

	for i in rowSort:
		colSort = np.argsort( Perm[i] )[::-1]		# Choose the GT CA that best matches this Learned/model CA.
		if verbose and False:
			print('--------------------------------------------------------')
			print('i = ',i)
			print('rowSort = ',rowSort)
			print('')
			print('colsort = ', colSort)
			print('')
			print('translate = ',translate)
			print('')
		j = 0									#
		while colSort[j] in translate and unique: 	# this j and while loop to deal with duplicates in translate vector due to ambiguities in Perm matrix.
			j+=1									# increment j if the best match in colSort is already in translate vector.
			if verbose:
				print(' j = ',j,' : colSort = ',colSort[j])
		translate[i] = colSort[j]					# index indicates GT CA and value indicates corresponding learned/model CA.
		dot_prod[i] = Perm[ i,colSort[j] ]
		#	
		# this if statement doesnt really make sense :?
		if j>0:
			translate2[i] = colSort[0]
			dot_prod2[i] = Perm[ i,colSort[0] ]
		else:
			translate2[i] = colSort[j+1]	# index indicates GT CA and value indicates corresponding learned/model CA.
			dot_prod2[i] = Perm[ i,colSort[j+1] ] 

		
		if verbose and False:
			print('j = ',j, ' and best CA = ',translate, ' and 2nd best = ',translate2)
			print('dotProds = best CA = ',dot_prod.round(2), ' and 2nd best = ',dot_prod2.round(2))
			#
			#  Good diagnostic plot that shows 2 Pia matrices and 2 slices of CA's that were matches.
			f,ax = plt.subplots(2,2)
			ax[0][0].imshow(A)
			ax[0][0].set_title('A - from')
			#
			ax[0][1].imshow(B)
			ax[0][1].set_title('B - to')
			#
			ax[1][0].plot(A[:,i])
			ax[1][0].set_title( str('CA#'+str(i)+' in B') )
			#
			ax[1][1].plot(B[:,translate[i]])
			ax[1][1].set_title( str('CA#'+str(translate[i])+' in B') )
			#
			plt.show()




	print('dp shape', dot_prod.shape)



	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# In the 'translate' vector, we want to make sure that there are no repeat CA's in matrix B
	# that are fit to multiple CA's in matrix A until every CA in matrix A is represented. That
	# is, I  want match each CA in A to a unique CA in B. After I have M_a unique CAs in matrix B,
	# (M_a < M_b), then I can and have to use


	#if all the ground truth cell assemblies are not represented first...
	#
	trans_preShuff = translate
	ind = list(np.argsort(dot_prod)[::-1])
	trans_shuff = list(translate[ind])
	trans2_shuff = list(translate2[ind])
	dotprod_shuff = dot_prod[ind]
	dotprod2_shuff = dot_prod2[ind]
	ind_shuff = list(ind)
	CA_in = set()
	CA_out = set(np.arange(M_b))


	if verbose:
		print('#######')

		print('trans_preShuff = ',trans_preShuff.shape)
		print(' ')
		print('ind = ', len(ind))
		print(' ')
		print('trans_shuff = ',len(trans_shuff))
		print(' ')
		print('trans2_shuff = ',len(trans2_shuff))
		print(' ')
		print('dotprod_shuff = ',dotprod_shuff.round(2).shape)
		print(' ')
		print('dotprod2_shuff = ',dotprod2_shuff.round(2).shape)
		print(' ')
		print('ind_shuff = ', len(ind_shuff))

		print('Ma = ',M_a,'  Mb = ',M_b)

	
	#
	i=0
	cntr = 0
	drop_warn = ''
	while len(CA_in) < M_b: #np.min([M_b,M_a]):

		if verbose:
			print('----------------------------------')
			print('trans_shuff = ', trans_shuff)
			print('')
			print('CA_in = ',CA_in)
			print('')
			print('CA_out = ',CA_out)
			print('')


		if cntr > 1000:
			print('I cant sort this shit. Too many iterations.')
			break

		if trans_shuff[i] not in CA_in:
			CA_in.add( trans_shuff[i] )
			CA_out.remove( trans_shuff[i] )
			if False and verbose:
				print(trans_shuff[i], trans_shuff, CA_in)
			i+=1


		else: # if CA i has already been included, go to next CA (j=i, then j+=1)
			if verbose:
				print('Arrg: ',trans_shuff[i], trans_shuff, CA_in)
			j=i
			while trans_shuff[j] not in CA_out:
				j+=1
				print( j)
				if j == M_a:
					drop_warn = str('CA drop :'+str(CA_out))
					if verbose:
						print(drop_warn,' j = ',j)
					CA_in.add( 'one missing' )
					cntr += 1
					j-=1
					break
			x = trans_shuff[i:j] + trans_shuff[j+1:]
			y = trans_shuff[j]
			trans_shuff[i] = y
			trans_shuff[i+1:] = x
			#
			x = trans2_shuff[i:j] + trans2_shuff[j+1:]
			y = trans2_shuff[j]
			trans2_shuff[i] = y
			trans2_shuff[i+1:] = x
			#
			x = dotprod_shuff[i:j] + dotprod_shuff[j+1:]
			y = dotprod_shuff[j]
			dotprod_shuff[i] = y

			# print('x = ',x.shape)
			# print('i = ',i)
			# print('j = ', j)
			# print('dotprod_shuff[i+1:] = ', dotprod_shuff[i+1:].shape)

			dotprod_shuff[i+1:] = x
			#
			x = dotprod2_shuff[i:j] + dotprod2_shuff[j+1:]
			y = dotprod2_shuff[j]
			dotprod2_shuff[i] = y
			dotprod2_shuff[i+1:] = x
			#
			x = ind_shuff[i:j] + ind_shuff[j+1:]
			y = ind_shuff[j]
			ind_shuff[i] = y
			ind_shuff[i+1:] = x






	dot_prod = np.array(dotprod_shuff)
	dot_prod2 = np.array(dotprod2_shuff)
	translate = np.array(trans_shuff)
	translate2 = np.array(trans2_shuff)
	ind = np.array(ind_shuff)


	if verbose:
		print(' in translate A to B')
		print('translate B after shuffle: ',translate)
		print('translate B before shuffle: ',trans_preShuff)
		print('indx into A before shuffle: ',ind)


	if False:
		f,ax = plt.subplots(2,3)
		ax[0][0].imshow( Perm )
		ax[0][0].set_title('orig perm')
		ax[0][1].imshow( Perm[:,translate])
		ax[0][1].set_title('correct permuted')
		ax[0][2].imshow( Perm[translate])
		ax[0][2].set_title('wrong permuted')
		#
		ax[1][2].imshow( A )
		ax[1][2].set_title('A')
		#
		ax[1][0].imshow( B )
		ax[1][0].set_title('B')
		plt.suptitle('Permutation of learned Pia matches GT Pia better?')
		plt.show()	

		
	return np.vstack([translate,translate2]), np.vstack([dot_prod,dot_prod2]), trans_preShuff, ind, Perm, drop_warn	







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def matchModels_AtoB_cosSim(A, B, noSelf=False, verbose=False):
	#
	# THOROUGH EXPLANATION BELOW:
	#
	# Goal: Mapping columns in A to columns in matrix B. I.e., to permute the columns 
	# ---- 	of matrix A so that they align with the columns in matrix B with the highest
	# 		cosine similarity { cosSim = (A.B)/(||A||*||B||) }.
	#
	# Inputs: 	Matrix A and Matrix B are both Pia matrices where columns - dimension 1 or M -
	# ------  	represent cell assemblies (CAs). The number of rows - dimension 0 or N, representing 
	# 			cells, is the same for both matrices. The number of columns can be equal but 
	# 			dont have to be. One model overcomplete or undercomplete relative to the other 
	#			model or ground truth.
	# 
	# Direction of Mapping and Number of CAs:	(Mapping from A to B.)
	# --------------------------------------
	#
	#			(1). A=Model and B=GT. 
	# 				 -----------------
	#				Interpetation: Find the best CA in the learned model to match each GT CA.
	# 				
	# 				(a). Complete (M_a=M_b), 
	# 					 ------------------
	# 					 Each model CA is mapped to one unique GT CA. <-- Bijective mapping.
	#	
	#				(b). Undercomplete (M_a<M_b),
	# 					 -----------------------
	# 					 First, each GT CA is mapped to by one unique model CA - which
	# 					 is the best match for it and does not match any other model 
	# 					 CAs better.  <-- Injective mapping for first M_a.
	#					 #
	#					 Second, the remaining GT CAs map to any model CAs that best match
	# 					 them, even if that model CA matches another GT CA better and has 
	# 					 already been used. Including these matches beyond the first M_a, 
	# 					 every A maps to a B, but some A's must map to multiple B's.
	# 					 #
	# 					 The mappings are sorted into a list so that the first M_a matches form
	# 					 an Injective mapping and the ones that reuse model CAs are further down.
	#
	# 				(c). Overcomplete (M_a>M_b),
	# 					 ----------------------
	#					 First, Each GT CA is mapped to by one unique model CA - which
	# 					 is the best match for it and does not match any other model 
	# 					 CAs better. <-- Injective mapping for first M_b.
	#					 #
	#					 Second, the remaining model CAs map to any GT CAs that best match
	# 					 them, even if that GT CA matches another model CA better and has 
	# 					 already been used. Including these matches beyond the first M_b, 
	# 					 every B maps to an A, but some B's are be mapped to by multiple A's. 
	#
	#
	#					 
	#
	# 			(2). A=GT and B=Model.
	# 				 -----------------
	# 				Interpretation: Find the best GT CA to match each CA in the learned model.
	#
	# 				(a). If model is complete (M_a=M_b),
	# 					 ------------------------------
	# 					 Each GT CA is mapped to one and only one Model CA. <-- Bijective Mapping.
	#
	#
	#				(b). Undercomplete (M_a>M_b),
	# 					 -----------------------
	# 					 First, each model CA is mapped to by one unique model CA - which is the best match for 
	# 					 it and does not match any other GT CAs better.  <-- Injective mapping for first M_b.
	#					 #
	#					 Second, the remaining GT CAs map to any model CAs that best match them, even 
	# 					 if that model CA matches another GT CA better and has already been used. 
	# 					 Including these matches beyond the first M_b, 
	# 					 every A maps to a B, but some A's must map to multiple B's.
	#
	# 				(c). Overcomplete (M_a<M_b),
	# 					 ----------------------
	#					 First, Each GT CA maps one unique model CA - which is the best match for it and does 
	# 					 not match any other model CAs better. <-- Injective mapping for first M_a.
	#					 #
	#					 Second, the remaining model CAs are mapped to by any GT CAs that best match them, even
	# 					 if that GT CA matches another model CA better and has already been used. 
	# 					 Including these matches beyond the first M_a, 
	# 					 every A maps to an B, but some B's are be mapped to by multiple A's. 
	#
	#
	# 
	# Outputs: Subject to change.
	# -------
	#
	# 			np.vstack([ind_A,ind_A2])
	# 			np.vstack([dot_prod,dot_prod2]), 
	# 			trans_preShuff, 
	#			ind, 
	# 			Perm, 
	# 			drop_warn	






	M_a = A.shape[1] 	# number of columns / CAs in matrix A 
	M_b = B.shape[1] 	# number of columns / CAs in matrix B  
	M_sml = np.min([M_a,M_b])
	N = A.shape[0]		# number of rows / cells in matrix A&B

	if verbose:
		print('Goal: find best column in A that best matches a given column in B')
		print('M_a = ',M_a,' and M_b = ',M_b)
		print('')
	

	# Preallocate memory ...
	#
	# for each CA in matrix B, cosine similarity of best fit CA in matrix A
	cos_sim = np.empty( M_b ) 	
	len_dif = np.empty( M_b ) 
	len_sum = np.empty( M_b ) 			
	# for each CA in matrix B, cosine similarity of 2nd best fit CA in matrix A
	cos_sim2 = np.empty( M_b )
	len_dif2 = np.empty( M_b )
	#
	# index in B matrix. Sorted in descending order by cos_sim. 
	ind_B 	= -1*np.ones( M_b ).astype(int) 
	# for each CA in matrix B, the corresponding best fit CA in matrix A
	ind_A 	= -1*np.ones( M_b ).astype(int)  	
	# for each CA in matrix B, the corresponding 2nd best fit CA in matrix A
	ind_A2 	= -1*np.ones( M_b ).astype(int) 	
	 	
	


	# Compute Cost function - Cosine Similarity between columns in matrix A and columns in matrix B.
	cosSimMat = cosine_similarity(B.T,A.T) # shape = M_b x M_a


	# # # MAYBE LATER: Compute Cost function - Could use symmetrized KL-divergence (averaged across all cells) between each CA pair.
	# import scipy.stats as st 
	# #rc.compute_sym_KL_Pias(B,A)
	# #
	# KLbernMat = np.zeros( (M_b,M_a) )
	# #
	# for ia in range(M_a):
	# 	for ib in range(M_b):
	# 		#for j in range(N):
	# 		xxx = [ st.entropy( [A[j,ia],1-A[j,ia]], [B[j,ib],1-B[j,ib]] ) for j in range(N) ] # A,B matrix is NxM
	# 		KLbernMat[ib,ia] = np.array(xxx).mean()
	
	# plt.imshow(KLbernMat)
	# plt.colorbar()
	# plt.show()
	# #yyy


	# Reorder cosine similarity matrix using the Hungarian Method. 
	# NOTE: I dont think this will work for a non square cost matrix.
	HungRowCol 	= opt.linear_sum_assignment(1-cosSimMat)
	#
	srtHungCS 	= np.argsort( cosSimMat[ np.ix_( HungRowCol[0], HungRowCol[1] ) ].diagonal() )[::-1]
	HungRowColSrt = list() # resort matrix by size of Cosine Similarity - so diagonal is descending order.
	HungRowColSrt.append( HungRowCol[0][srtHungCS] )
	HungRowColSrt.append( HungRowCol[1][srtHungCS] )
	#
	cos_simHM 	= cosSimMat[ np.ix_( HungRowColSrt[0], HungRowColSrt[1] ) ].diagonal()


	if verbose:
		print('Hungarian Method:')

		print(len(HungRowCol),len(HungRowCol[0]), len(HungRowCol[1]) )
		print('Row',HungRowColSrt[0])
		print('Col',HungRowColSrt[1])

		print( 'CS Hung ',cosSimMat[ np.ix_( HungRowColSrt[0], HungRowColSrt[1] ) ].diagonal() )
	
		f,ax = plt.subplots(1,2)
		ax[0].imshow(cosSimMat,vmin=0,vmax=1)
		ax[0].set_title('Before HM resort')
		#
		ax[1].imshow(cosSimMat[ np.ix_( HungRowColSrt[0], HungRowColSrt[1] ) ],vmin=0,vmax=1)
		ax[1].set_title('After HM resort')
		plt.show()




	# Doing this in for loops because there are so many ways it can fuck up dimensions and 
	# error depending on M_a, M_b and N with these kinds of calls
	# 	summ = (B[None,:] + A.T[:,None])
	#	diff = (B[None,:] - A.T[:,None])
	#	lenDifMat = 1 - ( np.linalg.norm(diff,axis=2) / np.linalg.norm(summ,axis=2) ).T
	lenDifMat = np.zeros([M_b,M_a])
	for b in range(M_b):
		for a in range(M_a):
			summ = np.linalg.norm( B[:,b] + A[:,a] )
			diff = np.linalg.norm( B[:,b] - A[:,a] )
			lenDifMat[b][a] =  1 - diff/summ
	# if M_b > M_a:
	# 	summ = (B.T[:,None] + A[None,:])
	# 	diff = (B.T[:,None] - A[None,:])
	# 	lenDifMat = 1 - ( np.linalg.norm(diff,axis=2) / np.linalg.norm(summ,axis=2) ) # |A-B|/|A+B| = 1 if orthogonal, = 0 if identical
	# elif M_b < M_a:
	# 	summ = (B[None,:] + A.T[:,None])
	# 	diff = (B[None,:] - A.T[:,None])
	# 	lenDifMat = 1 - ( np.linalg.norm(diff,axis=2) / np.linalg.norm(summ,axis=2) ).T # |A-B|/|A+B| = 1 if orthogonal, = 0 if identical
	# else: # M_a==M_b and M_a > N:
	# 	summ = (B.T[:,:,None] + A[None,:,:])
	# 	diff = (B.T[:,:,None] - A[None,:,:])
	# 	lenDifMat = 1 - ( np.linalg.norm(diff,axis=1) / np.linalg.norm(summ,axis=1) ).T # |A-B|/|A+B| = 1 if orthogonal, = 0 if identical
	# #elif M_a==M_b and M_a < N:



	cosSimNM = np.diag( cosSimMat)
	lenDifNM = np.diag( lenDifMat)



	# if noSelf:
	# 	np.fill_diagonal(cosSimMat,0)
	# 	np.fill_diagonal(lenDifMat,0)
		


	if verbose:
		print(B.shape)
		print(A.shape)


	if verbose:		
		print(summ.shape)
		print(diff.shape)
		print(lenDifMat.shape)
		print(cosSimMat.shape)

	# Difference in length of vectors (CAs) becomes interesting when their angle between them is very small.

	#lenDifMat = (B_norm[:,None] - A_norm.T[None,:])/N # When positive, B values larger

	LenAngMat = cosSimMat

	# find best column in A that best matches a given column in B


	########################################################################################
	#
	# Sorting the cosSimMat (Cosine Similarity Matrix) in order to make it diagonally dominant.
	# 	1st, sort rows by their max --> {index of ind_A}
	# 	2nd, for each row, find the location of the max entry --> {value of ind_A}
	# 	CONSTRAINT: Must use unique columns for
	#



	# (1). Sort the cosSim matrix rows (CAs in B) by the highest cosSim,
	rowSort = np.argsort( LenAngMat.max(axis=1) )[::-1]  # could also normalize by the mean... /cosSim.mean(axis=1)

	#
	if verbose and False:
		# Plot cosSim and results of rowSort on cosSim
		print('rowSort of cosSim',rowSort, rowSort.shape)
		print('cosSim shape ', LenAngMat.shape)
		f,ax = plt.subplots(1,2)
		ax[0].imshow(LenAngMat, vmin=0, vmax=1)
		ax[0].set_xlabel('A matrix')
		ax[0].set_ylabel('B matrix')
		ax[0].set_title('Cosine Similarity')
		ax[0].set_yticks(np.arange(M_b))
		ax[0].set_yticklabels(np.arange(M_b),fontsize=9)
		#
		ax[1].imshow(LenAngMat[rowSort], vmin=0, vmax=1)
		ax[1].set_xlabel('A matrix')
		ax[1].set_ylabel('B matrix')
		ax[1].set_title('sorting rows by largest cosSim')
		ax[1].set_yticks(np.arange(M_b))
		ax[1].set_yticklabels(rowSort,fontsize=9)
		plt.show()


	#print('rowSort size = ',len(rowSort))

	for cntr,b in enumerate(rowSort):

		#
		# can only enforce a unique mapping, ie. that mapBtoA list has no repeat elements
		# until the minimum of M_a,M_b is reached.
		unique_map = (ind_A>-1).sum()<M_sml 
		


		#
		# For row b, find the column location of the largest value in it.
		colSort = np.argsort( LenAngMat[b] )[::-1]		# cosSim.shape = M_b x M_a
		a=0									#
		while colSort[a] in ind_A and unique_map: 	# this a and while loop to deal with duplicates in ind_A list due to ambiguities in cosSimMat
			a+=1										# increment a if the best match in colSort has already been used in the ind_A list.
			if verbose:
				print(' a (CA in A) = ',a,' : colSort = ',colSort[a])
		#
		ind_B[cntr] = b
		ind_A[cntr] = colSort[a]					# index indicates GT CA and value indicates corresponding learned/model CA.
		cos_sim[cntr] = cosSimMat[ ind_B[cntr], ind_A[cntr] ]
		len_dif[cntr] = lenDifMat[ ind_B[cntr], ind_A[cntr] ]
		
		#
		#print(M_b, M_a)
		if a<(M_sml-1):
			ind_A2[cntr] = colSort[a+1]	# index indicates GT CA and value indicates corresponding learned/model CA.
			cos_sim2[cntr] = cosSimMat[ ind_B[cntr], colSort[a+1] ] 
			len_dif2[cntr] = lenDifMat[ ind_B[cntr], colSort[a+1] ]	
		else:
			ind_A2[b] = colSort[a]
			cos_sim2[cntr] = cosSimMat[ ind_B[cntr], colSort[a] ] 
			len_dif2[cntr] = lenDifMat[ ind_B[cntr], colSort[a] ]	




		if verbose:
			print('--------------------------------------------------------')
			print('b (CA in B) = ',ind_B[cntr],' maps to ')
			print('a (CA in A) = ',ind_A[cntr])
			print('cos sim = ',cos_sim[cntr])
			print('cos sim = ',len_dif[cntr])
			print('num mapped = ', (ind_A>-1).sum() )
			print('')

		
		if verbose and False:
			#
			# Plot both A and B matrices as well as slices through them of CAs thought to match up.
			# If the plotted slices are not very similar something is not working. If the numbers
			# reported above the matrices dont match the column with the plot in middle also,
			# something is wrong.
			#
			print('a (CA in A) = ',a, ' and best CA = ',ind_A, ' and 2nd best = ',ind_A2)
			print('')
			print('cosine sims = best CA = ',cos_sim.round(2), ' and 2nd best = ',cos_sim2.round(2))
			print('')
			print('cos_sim shape', cos_sim.shape)
			#
			#  Good diagnostic plot that shows 2 Pia matrices and 2 slices of CA's that were matches.
			f,ax = plt.subplots(1,3)
			ax[0].imshow(A)
			ax[0].set_title( str('CA#'+ str( ind_A[b] ) +' in A') )
			#
			ax[2].imshow(B)
			ax[2].set_title( str('CA#'+ str(b) +' in B') )
			#
			ax[1].plot(A[:,ind_A[b]], label='CA in A')
			ax[1].plot(B[:,b], label='CA in B')
			ax[1].set_title('Goal: find column in A that best matches a given column in B')
			ax[1].legend()
			plt.show()

	








	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# In the 'ind_A' vector, we want to make sure that there are no repeat CA's in matrix B
	# that are fit to multiple CA's in matrix A until every CA in matrix A is represented. That
	# is, I  want match each CA in A to a unique CA in B. After I have M_a unique CAs in matrix B,
	# (M_a < M_b), then I can and have to use




	if verbose:
		print('Compare Hungarian Method to our Heuristic method of resorting:')
		print('HMrow',HungRowColSrt[0])
		print('Our B', ind_B)
		print('')
		print('HMcol',HungRowColSrt[1])
		print('Our A', ind_A)
		print('')
		print( 'HM CosSim: ', cos_simHM.round(3) )
		print( 'Our CosSim: ', cos_sim.round(3) )



	
	if len( np.unique(ind_A[:M_sml]) )!=M_sml:
		print('First M_a mappings in ind_A are not unique. And they should be.')
		print('Num unique = ',len( np.unique(ind_A[:M_sml]) ))
		print('Num M_sml = ',M_sml)
		return
	
	return 	list(np.vstack([ind_B, ind_A, ind_A2])), \
			np.vstack([cos_sim, cos_sim2]), cosSimNM, \
			np.vstack([len_dif, len_dif2]), lenDifNM, \
			cosSimMat, lenDifMat, HungRowColSrt, cos_simHM 	

						#, len_difNM,










	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #







	# #if all the ground truth cell assemblies are not represented first...
	# #
	# ind_A_preShuff = ind_A
	# indSrtCosSim = list(np.argsort(cos_sim)[::-1]) 
	# ind_A_shuff = list(ind_A[ind])
	# ind_A2_shuff = list(ind_A2[ind])
	# cos_sim_shuff = cos_sim[ind]
	# cos_sim2_shuff = cos_sim2[ind]
	# indSrtCosSim_shuff = list(indSrtCosSim)
	# CA_in = set()
	# CA_out = set(np.arange(M_b))


	# if verbose:
	# 	print('#######')

	# 	print('trans_preShuff = ',ind_A_preShuff.shape)
	# 	print(' ')
	# 	print('indSrtCosSim = ', len(indSrtCosSim))
	# 	print(' ')
	# 	print('trans_shuff = ',len(ind_A_shuff))
	# 	print(' ')
	# 	print('trans2_shuff = ',len(ind_A2_shuff))
	# 	print(' ')
	# 	print('cos_sim_shuff = ',cos_sim_shuff.round(2).shape)
	# 	print(' ')
	# 	print('cos_sim2_shuff = ',cos_sim2_shuff.round(2).shape)
	# 	print(' ')
	# 	print('ind_shuff = ', len(indSrtCosSim_shuff))

	# 	print('Ma = ',M_a,'  Mb = ',M_b)



	# xxx	

	
	# #
	# i=0
	# cntr = 0
	# drop_warn = ''
	# while len(CA_in) < M_b: #np.min([M_b,M_a]):

	# 	if verbose:
	# 		print('----------------------------------')
	# 		print('trans_shuff = ', ind_A_shuff)
	# 		print('')
	# 		print('CA_in = ',CA_in)
	# 		print('')
	# 		print('CA_out = ',CA_out)
	# 		print('')


	# 	if cntr > 1000:
	# 		print('I cant sort this shit. Too many iterations.')
	# 		break

	# 	if ind_A_shuff[i] not in CA_in:
	# 		CA_in.add( ind_A_shuff[i] )
	# 		CA_out.remove( ind_A_shuff[i] )
	# 		if False and verbose:
	# 			print(tind_A_shuff[i], ind_A_shuff, CA_in)
	# 		i+=1


	# 	else: # if CA i has already been included, go to next CA (j=i, then j+=1)
	# 		if verbose:
	# 			print('Arrg: ',ind_A_shuff[i], ind_A_shuff, CA_in)
	# 		j=i
	# 		while ind_A_shuff[j] not in CA_out:
	# 			j+=1
	# 			print( j)
	# 			if j == M_a:
	# 				drop_warn = str('CA drop :'+str(CA_out))
	# 				if verbose:
	# 					print(drop_warn,' j = ',j)
	# 				CA_in.add( 'one missing' )
	# 				cntr += 1
	# 				j-=1
	# 				break
	# 		x = ind_A_shuff[i:j] + ind_A_shuff[j+1:]
	# 		y = ind_A_shuff[j]
	# 		ind_A_shuff[i] = y
	# 		ind_A_shuff[i+1:] = x
	# 		#
	# 		x = ind_A2_shuff[i:j] + ind_A2_shuff[j+1:]
	# 		y = ind_A2_shuff[j]
	# 		ind_A2_shuff[i] = y
	# 		ind_A2_shuff[i+1:] = x
	# 		#
	# 		x = cos_sim_shuff[i:j] + cos_sim_shuff[j+1:]
	# 		y = cos_sim_shuff[j]
	# 		dotprod_shuff[i] = y

	# 		# print('x = ',x.shape)
	# 		# print('i = ',i)
	# 		# print('j = ', j)
	# 		# print('dotprod_shuff[i+1:] = ', dotprod_shuff[i+1:].shape)

	# 		cos_sim_shuff[i+1:] = x
	# 		#
	# 		x = cos_sim2_shuff[i:j] + cos_sim2_shuff[j+1:]
	# 		y = cos_sim2_shuff[j]
	# 		cos_sim2_shuff[i] = y
	# 		cos_sim2_shuff[i+1:] = x
	# 		#
	# 		x = ind_shuff[i:j] + ind_shuff[j+1:]
	# 		y = ind_shuff[j]
	# 		ind_shuff[i] = y
	# 		ind_shuff[i+1:] = x






	# cos_sim = np.array(cos_sim_shuff)
	# cos_sim2 = np.array(cos_sim2_shuff)
	# ind_A = np.array(ind_A_shuff)
	# ind_A2 = np.array(ind_A2_shuff)
	# ind = np.array(ind_shuff)


	# if verbose:
	# 	print(' in translate A to B')
	# 	print('ind_A B after shuffle: ',ind_A)
	# 	print('ind_A B before shuffle: ',ind_A_preShuff)
	# 	print('indx into A before shuffle: ',ind)


	# if False:
	# 	f,ax = plt.subplots(2,3)
	# 	ax[0][0].imshow( cosSimMat )
	# 	ax[0][0].set_title('orig perm')
	# 	ax[0][1].imshow( cosSimMat[:,ind_A])
	# 	ax[0][1].set_title('correct permuted')
	# 	ax[0][2].imshow( cosSimMat[ind_A])
	# 	ax[0][2].set_title('wrong permuted')
	# 	#
	# 	ax[1][2].imshow( A )
	# 	ax[1][2].set_title('A')
	# 	#
	# 	ax[1][0].imshow( B )
	# 	ax[1][0].set_title('B')
	# 	plt.suptitle('Permutation of learned Pia matches GT Pia better?')
	# 	plt.show()	

		
	# return np.vstack([ind_A,ind_A2]), np.vstack([cos_sim,cos_sim2]), ind_A_preShuff, ind, cosSimMat, drop_warn	





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def extract_spikeWords(spiketrains, SW_bins, fname_SWs):
	# extract spike words from real retinal data	

	print('Spike Word Extraction from real retina data: ')
	print('Trying to open File ', fname_SWs)
		
	if os.path.isfile ( fname_SWs ): # 
		print('Already exists. Not remaking it. Loading it.')
		data = np.load( fname_SWs )
		SWs = data['SWs'] 
		SWtimes = data['SWtimes']
	else:
		print('Does not exist. Have to make it.')
		SWs = list()
		SWtimes = list()
		#
		numTrials = spiketrains.shape[1]
		numCells  = spiketrains.shape[0]

		for T in range(numTrials): # Loop thru trials and find spike words within each one

			SpkTimes = np.unique( np.concatenate(spiketrains[:,T]) ).astype(int)
			print('Trial #',T,' has ',SpkTimes.size,' possible spike words.')
			SWs.append( [] )
			SWtimes.append( [] )
			#
			for st in SpkTimes:
				SWtimes[T].append( st )
				sw = set()
				for c in range(numCells):
					if np.any( np.abs(spiketrains[c][T] - st) <= SW_bins):
						sw.add(c)
				SWs[T].append( sw )	

		np.savez( fname_SWs, SWs=SWs, SWtimes=SWtimes, SW_bins=SW_bins )

	return SWs, SWtimes	







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def construct_crossValidation_SWs(all_SWs, yMin, pct_xVal_train):

	lx = [ len(x) for x in all_SWs ]
	indSWs_gt_yMin = np.where(np.array(lx)>=yMin)[0]
	# xxx = np.where(np.array(lx)>=yMin)
	# print('number of things > yMinSW: ',len(indSWs_gt_yMin) )
	# print('xxx= ',xxx)
	ind = np.random.choice(len(indSWs_gt_yMin), size=len(indSWs_gt_yMin), replace=False)
	split = np.round(pct_xVal_train*len(indSWs_gt_yMin)).astype(int)
	indSWs_gt_yMin_train = indSWs_gt_yMin[ind[:split]].astype(int)
	indSWs_gt_yMin_test = indSWs_gt_yMin[ind[split:]].astype(int)	

	return indSWs_gt_yMin_train, indSWs_gt_yMin_test





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# def compute_PSTHy( SWs, SWtimes, N, numTrials, minTms, maxTms ):

# 	# binnedHists = np.zeros( (numTrials,maxTms) )		
# 	# for t in range(numTrials):
# 	# 	ind = np.where( np.bitwise_and(np.array(SWtimes[t])>minTms, np.array(SWtimes[t])<maxTms  ) )[0]
# 	# 	x = [ SWtimes[t][i] for i in ind]
# 	# 	y = [len(SWs[t][i]) for i in ind]
# 	# 	binnedHists[t][x] = y

# 	# return binnedHists

# 	numTms = maxTms - minTms
# 	PSTH_Y_observed_allSWs = np.zeros( (N, numTrials, numTms) )
# 	#
# 	for tr in range(numTrials):
# 		print('Computing PSTH of observed Ys on trial #',tr)
# 		t1 = time.time()
# 		#
# 		sTs = np.array(SWtimes[tr])
# 		sTs = sTs[np.bitwise_and(sTs>minTms,sTs<maxTms)] # Last trial goes on up to 25sec. Get rid of that.
# 		#
# 		for i,yt in enumerate(sTs):
# 			y = SWs[tr][i]
# 			#
# 			PSTH_Y_observed_allSWs[ list(y), tr, yt ] = 1
# 		#	
# 		print('Time it took: ', time.time()-t1)

# 	return PSTH_Y_observed_allSWs





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def compute_raster_list( Xtimes, X, pX, numX, minTms, maxTms ): #SWs, numTrials, numX,
	#
	# Turn a list of lists (X) 

	# Nested Lists to form a matrix or array.
	# (Cell or CA) x (Trials).
	# Each entry contains a list of times when X was active in that trial.
	#


	numTrials 	= len(X)
	#print(numTrials)

	numTms = maxTms - minTms
	#pjt_tol = 10


	# Set up 1st two dimensions of the raster 3D sparse tensor
	raster_allSWs = list()
	raster_PofYs = list()
	#
	for tr in range(numTrials):				# Dim 1: numTrials
		raster_allSWs.append(list())
		raster_PofYs.append(list())
		#
		for nx in range(numX):     			# Dim 2: numCells or numCAs
			raster_allSWs[tr].append(list())
			raster_PofYs[tr].append(list())


	# Fill each element in 2D matrix with times that X is active in Trial
	print('Making spike rasters for ',numTrials,' trials.')
	t1 = time.time()
	for tr in range(numTrials):				# Dim 1: numTrials
		#
		for nx in range(numX):     			# Dim 2: numCells or numCAs
			#
			sTs = np.array(Xtimes[tr])
			sTs = sTs[np.bitwise_and(sTs>minTms,sTs<maxTms)] # Last trial goes on up to 25sec. Get rid of that.
			#
			for i,yt in enumerate(sTs):
				if nx in X[tr][i]:						# Cell or CA ID.
					raster_allSWs[tr][nx].append(yt)
					raster_PofYs[tr][nx].append( pX[tr][i] )
		#	
	print('Time it took: ', time.time()-t1)

	return raster_allSWs, raster_PofYs	





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def compute_raster( Xtimes, X, numX, minTms, maxTms ): #SWs, numTrials, numX,
	#
	# Turn a list of lists (X) 

	# Nested Lists to form a matrix or array.
	# (Cell or CA) x (Trials).
	# Each entry contains a list of times when X was active in that trial.
	#


	numTrials 	= len(X)
	print(numTrials)

	numTms = maxTms - minTms
	#pjt_tol = 10

	# Set up list of list of lists as a sparse 3D matrix.
	raster_allSWs = np.zeros([numX,numTrials,numTms])

	#
	for tr in range(numTrials):
		print('trial #',tr)
		t1 = time.time()
		#
		sTs = np.array(Xtimes[tr])
		sTs = sTs[np.bitwise_and(sTs>minTms,sTs<maxTms)] # Last trial goes on up to 25sec. Get rid of that.
		#
		for i,yt in enumerate(sTs):
			x 	= X[tr][i]
			raster_allSWs[ list(x), tr, yt ] = 1
		#	
		print('Time it took: ', time.time()-t1)

	return raster_allSWs	




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def inferZ_allSWs( SWs, SWtimes, CAs, numTrials, minTms, maxTms, qp, rip, riap, flg_include_Zeq0_infer, yLo, yHi, verbose=False ):

	numTms = maxTms - minTms
	pjt_tol = 10
	Z_inferred_allSWs = list()
	pyiEq1_gvnZ_allSWs = list()
	pj_inferred_allSWs = list()
	cond_inferred_allSWs = list()

	M = riap.shape[1]
	zActivationHistory = np.ones(M) # once model is learned dont want to use this (just scalar Q now.)

	#
	for tr in range(numTrials):
		
		t1 = time.time()
		#
		sTs = np.array(SWtimes[tr])
		sTs = sTs[np.bitwise_and(sTs>minTms,sTs<maxTms)] # Last trial goes on up to 25sec. Get rid of that.
		#
		print('Inferring Zs for ',len(sTs),' spikewords on trial #',tr)
		#
		Z_inferred_allSWs.append( list() )
		pyiEq1_gvnZ_allSWs.append( list() )
		pj_inferred_allSWs.append( list() )
		cond_inferred_allSWs.append( list() )
		#print(sTs)
		#
		for i,yt in enumerate(sTs):
			if np.mod(i,1000)==0:
				print(i,'/',len(sTs))
			#
			y = SWs[tr][i]
			try:
				z = CAs[tr][i] 	# for synthData
			except:
				z = set() 		# for realData

			zHyp, pyiEq1_gvnZ, pj_zHyp, cond_zHyp  = inferZ_andComb(qp, rip, riap, y, z, pjt_tol, \
				flg_include_Zeq0_infer, yLo, yHi, zActivationHistory, verbose=verbose)
			#
			Z_inferred_allSWs[tr].append( zHyp )
			pyiEq1_gvnZ_allSWs[tr].append( pyiEq1_gvnZ )
			pj_inferred_allSWs[tr].append( pj_zHyp )
			cond_inferred_allSWs[tr].append( cond_zHyp )
		#	
		print('Time it took: ', time.time()-t1)


		# print( type(Z_inferred_allSWs), len(Z_inferred_allSWs) )
		# print( type(Z_inferred_allSWs[tr-st]), len(Z_inferred_allSWs[tr-st]), len(SWs[tr]) )
		# print( type(Z_inferred_allSWs[tr-st][0]) )
		# print( Z_inferred_allSWs[tr-st] )
		# print( SWs[tr] )

	return Z_inferred_allSWs, pyiEq1_gvnZ_allSWs, pj_inferred_allSWs, cond_inferred_allSWs


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def find_good_ones(Pia, TH, minMemb, maxMemb, ZorY):

	#print("WTF do I mean by good ones?")

	if ZorY=='z':
		axs=0
	elif ZorY=='y':
		axs=1
	else:
		print('I dont know what you mean by ',ZorY)
		good_ones = np.array([])
		return good_ones

	if len(Pia.shape)>1:
		good_ones = np.where( np.bitwise_and( (Pia>TH).sum(axis=axs)>=minMemb, (Pia>TH).sum(axis=axs)<=maxMemb ) )[0]
	else:
		if np.bitwise_and( (Pia>TH).sum()>=minMemb, (Pia>TH).sum()<=maxMemb ):
			good_ones = np.where( Pia>TH )[0]
		else:
			good_ones = np.array([])


	return good_ones	






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def compute_pairwise_model_CA_match(data):


	if 'ria' in data[0].keys():
		GT_there = True 		# if synthData
	else:
		GT_there = False		# if realData


	#
	matchMod1toMod2_cosSim	= np.zeros( (len(data), len(data)) )
	matchMod1toMod2_csHM 	= np.zeros( (len(data), len(data)) )
	matchMod1toMod2_lenDif	= np.zeros( (len(data), len(data)) )
	matchMod2toMod1_cosSim	= np.zeros( (len(data), len(data)) )
	matchMod2toMod1_csHM 	= np.zeros( (len(data), len(data)) )
	matchMod2toMod1_lenDif	= np.zeros( (len(data), len(data)) )
	#
	matchMod1toGT_cosSim	= np.zeros( (len(data), 1) ) 
	matchMod1toGT_lenDif	= np.zeros( (len(data), 1) )
	matchMod1toGT_csHM 		= np.zeros( (len(data), 1) )
	matchGTtoMod1_cosSim	= np.zeros( (len(data), 1) )
	matchGTtoMod1_lenDif 	= np.zeros( (len(data), 1) )
	matchGTtoMod1_csHM 		= np.zeros( (len(data), 1) )
	#
	matchMod1toMod2_csNM	= np.zeros( (len(data), len(data)) )
	matchMod1toMod2_ldNM	= np.zeros( (len(data), len(data)) )
	matchMod2toMod1_csNM	= np.zeros( (len(data), len(data)) )
	matchMod2toMod1_ldNM	= np.zeros( (len(data), len(data)) )
	#
	matchMod1toGT_csNM	= np.zeros( (len(data), 1) ) 
	matchMod1toGT_ldNM	= np.zeros( (len(data), 1) )
	matchGTtoMod1_csNM	= np.zeros( (len(data), 1) )
	matchGTtoMod1_ldNM 	= np.zeros( (len(data), 1) )
	#	
	# matchAvgModtoGT_csNM	= np.zeros( (len(data), len(data)) )
	# matchAvgModtoGT_ldNM	= np.zeros( (len(data), len(data)) )
	# matchGTtoAvgMod_csNM	= np.zeros( (len(data), len(data)) )
	# matchGTtoAvgMod_ldNM	= np.zeros( (len(data), len(data)) )
	#
	# matchAvgModtoGT_cosSim	= np.zeros( (len(data), len(data)) )
	# matchAvgModtoGT_lenDif	= np.zeros( (len(data), len(data)) )
	# matchGTtoAvgMod_cosSim	= np.zeros( (len(data), len(data)) )
	# matchGTtoAvgMod_lenDif	= np.zeros( (len(data), len(data)) )
	#



	for i in range(len(data)):
		#
		Pia1 = sig(data[i]['riap'])
		#
		if GT_there:
			Pia = sig(data[i]['ria'])
		#
		for j in range(len(data)):
			#
			Pia2 = sig(data[j]['riap'])	

			print(i,j)


			# print('Pia1: ',Pia1.shape)
			# print('Pia2: ',Pia2.shape)
			# print('Pia: ',Pia.shape)
			# #
			# print('Pia1=Pia2: ',np.all(Pia1.shape==Pia2.shape))
			# print('Pia1=Pia: ',np.all(Pia1.shape==Pia.shape))
			# print('Pia2=Pia: ',np.all(Pia2.shape==Pia.shape))




			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
			#
			# (1a). Find Model1 CAs that best match each Model2 CA.
			#
			ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = matchModels_AtoB_cosSim(A=1-Pia1, B=1-Pia2)
			matchMod1toMod2_cosSim[i][j] 	= cosSim[0].mean()
			matchMod1toMod2_csHM[i][j] 		= cos_simHM.mean()
			matchMod1toMod2_lenDif[i][j] 	= lenDif[0].mean()
			matchMod1toMod2_csNM[i][j]		= csNM.mean()
			matchMod1toMod2_ldNM[i][j]		= ldNM.mean()
			#
			# (1b). Find Model2 CAs that best match each Model1 CA.
			ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = matchModels_AtoB_cosSim(A=1-Pia2, B=1-Pia1)
			matchMod2toMod1_cosSim[i][j] 	= cosSim[0].mean()
			matchMod2toMod1_csHM[i][j] 		= cos_simHM.mean()
			matchMod2toMod1_lenDif[i][j] 	= lenDif[0].mean()
			matchMod2toMod1_csNM[i][j]		= csNM.mean()
			matchMod2toMod1_ldNM[i][j]		= ldNM.mean()


			
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
			# #
			# # (2a). Find match between Ground Truth and Average of 2 models, Pia1 and Pia2.
			# #
			# if GT_there:
			# 	ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = matchModels_AtoB_cosSim(A=1-Pia1, B=1-Pia2)
			# 	PiaAvg = ( Pia1[:,ind[1]] + Pia2[:,ind[0]] )/2
			# 	#
			# 	ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = matchModels_AtoB_cosSim(A=1-PiaAvg, B=1-Pia)
			# 	matchAvgModtoGT_cosSim[i][j] 	= cosSim[0].mean()
			# 	matchAvgModtoGT_lenDif[i][j] 	= lenDif[0].mean()
			# 	matchAvgModtoGT_csNM[i][j]		= csNM.mean()
			# 	matchAvgModtoGT_ldNM[i][j]		= ldNM.mean()
			# 	#
			# 	# (2b). Find Model2 CAs that best match each Model1 CA.
			# 	ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = matchModels_AtoB_cosSim(A=1-Pia, B=1-PiaAvg)
			# 	matchGTtoAvgMod_cosSim[i][j] 	= cosSim[0].mean()
			# 	matchGTtoAvgMod_lenDif[i][j] 	= lenDif[0].mean()
			# 	matchGTtoAvgMod_csNM[i][j]		= csNM.mean()
			# 	matchGTtoAvgMod_ldNM[i][j]		= ldNM.mean()



		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		#
		# (3a). Find Model1 CAs that best match each GT CA.
		if GT_there:
			ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = matchModels_AtoB_cosSim(A=1-Pia1, B=1-Pia)
			matchMod1toGT_cosSim[i]	= cosSim[0].mean()
			matchMod1toGT_csHM[i]	= cos_simHM.mean()
			matchMod1toGT_lenDif[i]	= lenDif[0].mean()
			matchMod1toGT_csNM[i]	= csNM.mean()
			matchMod1toGT_ldNM[i]	= ldNM.mean()

			#
			# (3b). Find GT CAs that best match each Model1 CA.
			ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = matchModels_AtoB_cosSim(A=1-Pia, B=1-Pia1)
			matchGTtoMod1_cosSim[i]	= cosSim[0].mean()
			matchGTtoMod1_csHM[i]	= cos_simHM.mean()
			matchGTtoMod1_lenDif[i]	= lenDif[0].mean()
			matchGTtoMod1_csNM[i]	= csNM.mean()
			matchGTtoMod1_ldNM[i]	= ldNM.mean()



	return 	matchMod1toMod2_cosSim, matchMod1toMod2_lenDif, matchMod1toMod2_csHM, \
			matchMod2toMod1_cosSim, matchMod2toMod1_lenDif, matchMod2toMod1_csHM, \
			matchMod1toGT_cosSim, 	matchMod1toGT_lenDif,	matchMod1toGT_csHM, \
			matchGTtoMod1_cosSim,	matchGTtoMod1_lenDif, 	matchGTtoMod1_csHM, \
			matchMod1toMod2_csNM, 	matchMod1toMod2_ldNM, 	matchMod2toMod1_csNM, 	matchMod2toMod1_ldNM, \
			matchMod1toGT_csNM, 	matchMod1toGT_ldNM, 	matchGTtoMod1_csNM, 	matchGTtoMod1_ldNM








# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def pyi_gvnZ_performance( pyiEq1_gvnZ, yObs, THs ):

	#
	N = len( pyiEq1_gvnZ )
	ysOn = len(yObs)
	yNot = set(range(N)).difference(yObs) # all cells that are inactive in spike word.
	pyiEq0_gvnZ = 1 - pyiEq1_gvnZ
	#
	
	# Compute mean, std and median of p(yi=0|z) wherever yi=0 and p(yi=1|z) wherever yi=1.
	pyiEq0_gvnZ_stats = [ np.mean(pyiEq0_gvnZ[list(yNot)]), np.std(pyiEq0_gvnZ[list(yNot)]), np.median(pyiEq0_gvnZ[list(yNot)]) ]
	pyiEq1_gvnZ_stats = [ np.mean(pyiEq1_gvnZ[list(yObs)]), np.std(pyiEq1_gvnZ[list(yObs)]), np.median(pyiEq1_gvnZ[list(yObs)]) ]


	# Compute number of cells on (yi=1) or off (yi=0) where p(yi=[1|0] | z) 
	# exceeds a threshold (TH) for a vector of thresholds between 0 and 1.
	TP = np.array([ np.sum(pyiEq1_gvnZ[list(yObs)]>TH) for TH in THs ]) # True Positives.
	FP = np.array([ np.sum(pyiEq1_gvnZ[list(yNot)]>TH) for TH in THs ]) # False Positives.
	#
	TN = np.array([ np.sum(pyiEq0_gvnZ[list(yNot)]>TH) for TH in THs ]) # True Negatives.
	FN = np.array([ np.sum(pyiEq0_gvnZ[list(yObs)]>TH) for TH in THs ]) # False Negatives.
	#
	# area under ROC curve for [y=1 inferred, y=0 inferred, y=1 w/ zTrain, y=0 w/ zTrain]
	AUC_yEq1 = auc( FP/(N-ysOn), TP/ysOn 	)
	AUC_yEq0 = auc( FN/ysOn, 	TN/(N-ysOn) )	

	return list(pyiEq0_gvnZ_stats + pyiEq1_gvnZ_stats), [AUC_yEq0, AUC_yEq1], np.vstack([TP, FP, TN, FN])
		   


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def pyi_gvnZ_performance_allSamps( pyi_gnvZ, yObs, ROC_THs ):

	pyi_gvnZ_stats_all 	= list()
	pyi_gvnZ_auc_all 	= list() 
	pyi_gvnZ_ROC_all 	= list()
	#
	
	for samp in range(len(yObs)):
		pyi_gvnZ_stats, pyi_gvnZ_auc, pyi_gvnZ_ROC = pyi_gvnZ_performance( pyi_gnvZ[samp], yObs[samp], ROC_THs )

		pyi_gvnZ_stats_all.append( pyi_gvnZ_stats )
		pyi_gvnZ_auc_all.append( pyi_gvnZ_auc )
		pyi_gvnZ_ROC_all.append( pyi_gvnZ_ROC ) 

	return pyi_gvnZ_stats_all, pyi_gvnZ_auc_all, pyi_gvnZ_ROC_all	






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def sampleSWs_by_length( Y_in, flg_sample_longSWs_first='Dont' ):

	nY = np.array( [len(y) for y in Y_in] ) # Get spike word lengths.
	
	# probabilistic sorting of spike words with probability proportional to length |y|.
	if flg_sample_longSWs_first == 'Prob': 
		nY_srt = np.random.choice(len(nY), len(nY), p=nY/nY.sum(),replace=False) 
		#
	# hard or deterministic (sorta) sorting by spike word length in descending order.	
	elif flg_sample_longSWs_first == 'Hard': 
		nY_srt = list(np.argsort(nY)[::-1]) 
		#
	# no sorting by spike word length. Take them as they were randomly drawn.	
	elif flg_sample_longSWs_first == 'Dont':
		return np.arange(len(nY))
		#
	else:
		print('I do not understand how to sort training data. Not doing anything. Possible choices are {Dont, Prob, Hard}')
		return np.arange(len(nY))
	#
	return nY_srt







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def compute_QQ_diff_metric(nYr_Mov, nYr_Wnz, nYs, Ycell_histR_Mov, Ycell_histR_Wnz, Ycell_histS, \
							Cell_coactivityR_Mov, Cell_coactivityR_Wnz, Cell_coactivityS):


	samps 		= nYs.size
	nSamp_Mov 	= nYr_Mov.size
	nSamp_Wnz 	= nYr_Wnz.size
	N 			= Ycell_histS.size

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# # (1). |y| - Spike Word Length. Cardinality of y-vector.
	nyMax = np.max( [nYr_Mov.max(), nYr_Wnz.max(), nYs.max()] )
	#	
	m1 = np.histogram( nYr_Mov, bins=np.arange(nyMax) )
	n1 = np.histogram( nYr_Wnz, bins=np.arange(nyMax) )
	s1 = np.histogram( nYs, 	bins=np.arange(nyMax) )
	#
	mx1 = np.cumsum(m1[0]/m1[0].sum()) # cumulative sums of normalized distributions.
	nx1 = np.cumsum(n1[0]/n1[0].sum())
	sx1 = np.cumsum(s1[0]/s1[0].sum())
	#
	QQ_yc_sm = np.abs(mx1-sx1).mean()
	QQ_yc_sn = np.abs(nx1-sx1).mean()
	QQ_yc_nm = np.abs(nx1-mx1).mean()
	#
	# m1[0][m1[0]==0]=1 # turn bins with 0 values into 1's. Admittedly a hack. But a small change with many samples.
	# n1[0][n1[0]==0]=1 # otherwise KL-div can be infinite.
	# #s1[0][s1[0]==0]=1
	# #
	# KL_yc_sm[Kind,Cind,a,b,c,d] = st.entropy( s1[0]/s1[0].sum(), m1[0]/m1[0].sum() ) # Movie and middle parameter of bracket
	# KL_yc_sn[Kind,Cind,a,b,c,d] = st.entropy( s1[0]/s1[0].sum(), n1[0]/n1[0].sum() ) # Noise and middle parameter of bracket

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# # (2). single cell y activity frequency - distribution
	indSortYR_Mov 	= np.argsort(Ycell_histR_Mov)[::-1]
	indSortYR_Wnz 	= np.argsort(Ycell_histR_Wnz)[::-1]
	indSortYS 		= np.argsort(Ycell_histS)[::-1]


	if( indSortYS.size == indSortYR_Mov.size+1 and indSortYS.size == indSortYR_Wnz.size+1 ):
		print('inds: ',indSortYR_Mov.shape, indSortYR_Wnz.shape, indSortYS.shape)
		indSortYS = indSortYS[:-1]


	

	#
	m2 = [ Ycell_histR_Mov[indSortYR_Mov] ]
	n2 = [ Ycell_histR_Wnz[indSortYR_Wnz] ]
	s2 = [ Ycell_histS[indSortYS] ]
	#
	mx2 = np.cumsum(m2[0]/m2[0].sum()) # cumulative sums of normalized distributions.
	nx2 = np.cumsum(n2[0]/n2[0].sum())
	sx2 = np.cumsum(s2[0]/s2[0].sum())
	#
	QQ_y1_sm = np.abs(mx2-sx2).mean()
	QQ_y1_sn = np.abs(nx2-sx2).mean()
	QQ_y1_nm = np.abs(nx2-mx2).mean()
	#
	# m2[0][m2[0]==0]=1 # turn bins with 0 values into 1's. Admittedly a hack. But a small change with many samples.
	# n2[0][n2[0]==0]=1 # otherwise KL-div can be infinite.
	# s2[0][s2[0]==0]=1
	y1Min = np.min( [m2[0].min(), n2[0].min(), s2[0].min()] )
	#
	# KL_y1_sm[Kind,Cind,a,b,c,d] = st.entropy( s2[0]/s2[0].sum(), m2[0]/m2[0].sum() ) # Movie and middle parameter of bracket
	# KL_y1_sn[Kind,Cind,a,b,c,d] = st.entropy( s2[0]/s2[0].sum(), n2[0]/n2[0].sum() ) # Noise and middle parameter of bracket


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# # (3). pairwise cell coactivity - weight distribution
	coMax = np.max( [Cell_coactivityR_Mov.max()/nSamp_Mov, Cell_coactivityR_Wnz.max()/nSamp_Wnz, Cell_coactivityS.max()/samps] )
	#
	m3 = np.histogram( Cell_coactivityR_Mov.flatten()/nSamp_Mov, bins=np.linspace(0,coMax,30) )
	n3 = np.histogram( Cell_coactivityR_Wnz.flatten()/nSamp_Wnz, bins=np.linspace(0,coMax,30) )
	s3 = np.histogram( Cell_coactivityS.flatten()/samps, 		 bins=np.linspace(0,coMax,30) )
	#
	mx3 = np.cumsum(m3[0]/m3[0].sum()) # cumulative sums of normalized distributions.
	nx3 = np.cumsum(n3[0]/n3[0].sum())
	sx3 = np.cumsum(s3[0]/s3[0].sum())
	#
	QQ_y2_sm = np.abs(mx3-sx3).mean()
	QQ_y2_sn = np.abs(nx3-sx3).mean()
	QQ_y2_nm = np.abs(nx3-mx3).mean()
	# m3[0][m3[0]==0]=1 # turn bins with 0 values into 1's. Admittedly a hack. But a small change with many samples.
	# n3[0][n3[0]==0]=1 # otherwise KL-div can be infinite.
	# s3[0][s3[0]==0]=1 
	#
	# KL_y2_sm[Kind,Cind,a,b,c,d] = st.entropy( s3[0]/s3[0].sum(), m3[0]/m3[0].sum() ) # Movie and middle parameter of bracket
	# KL_y2_sn[Kind,Cind,a,b,c,d] = st.entropy( s3[0]/s3[0].sum(), n3[0]/n3[0].sum() ) # Noise and middle parameter of bracket

	# if verbose:
	# 	print('	For |y|, mov, wnz:', KL_yc_sm[Kind,Cind,a,b,c,d].round(3), KL_yc_sn[Kind,Cind,a,b,c,d].round(3))
	# 	print('	For single y activity, mov, wnz:', KL_y1_sm[Kind,Cind,a,b,c,d], KL_y1_sn[Kind,Cind,a,b,c,d].round(3))
	# 	print('	For pairwise coactivity, mov, wnz:', KL_y2_sm[Kind,Cind,a,b,c,d].round(3), KL_y2_sn[Kind,Cind,a,b,c,d].round(3))

	return QQ_yc_sm, QQ_yc_sn, QQ_yc_nm, QQ_y1_sm, QQ_y1_sn, QQ_y1_nm, QQ_y2_sm, QQ_y2_sn, QQ_y2_nm, \
			mx1, nx1, sx1, mx2, nx2, sx2, mx3, nx3, sx3, m1, n1, s1, m2, n2, s2, m3, n3, s3





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def Zset_modulo_Ms( Z,M_sml ):

	# Z is a set of training, or test, or inferred Cell Assembly data.

    xx = list(Z) 
    Zmodulo 	= set([ xx[ind] for ind in np.where(xx<M_sml)[0] ])
    Zoutside 	= set( set.difference( set(Z), Zmodulo ) )

    return Zmodulo, Zoutside





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # OLD FUNCTIONS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #