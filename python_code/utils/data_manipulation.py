import sys
import numpy as np
#import pandas as pd
import pickle
import scipy.sparse as spsp
import time
import os.path 						# to check if a file or directory exists already
import platform


def set_dir_tree():

	nd = platform.node()
	print(nd)

	# (0). Check what Operating System you are on (either my machine or Cortex cluster) and adjust directory structure accordingly.
	if 'cori' in nd or 'edison' in nd:
		print('on NERSC Cluster')
		dirHome = '/global/homes/w/warner/Projects/G_Field_Retinal_Data/'
		dirScratch = '/global/cscratch1/sd/warner/Projects/G_Field_Retinal_Data/'

	elif '.brc' in nd or 'cortex' in nd:
		print('on Cortex Cluster')
		dirHome = '/global/home/users/cwarner/Projects/G_Field_Retinal_Data/'
		dirScratch = '/clusterfs/cortex/scratch/cwarner/Projects/G_Field_Retinal_Data/'

	else:
		print('assuming on Chris laptop')
		dirHome = '/Users/chriswarner/Desktop/Grad_School/Berkeley/Work/Fritz_Work/Projects/G_Field_Retinal_Data/home/G_Field_Retinal_Data/'
		dirScratch = '/Users/chriswarner/Desktop/Grad_School/Berkeley/Work/Fritz_Work/Projects/G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/'
	

	return dirHome, dirScratch




def find_M_N(searchDir, specifier1, specifier2, z0_tag):

	# Find the number of cells and cell assemblies from file names and directory structure 
	# given a couple specifiers, like cell type, yMin and other parameters on real data.
	# Find directory (model_str) with unknown N and M that matches cell_type and yMin

	subDirs = os.listdir( searchDir ) 
	model_str = [s for s in subDirs if specifier1 in s and specifier2 in s ]
	#
	if len(model_str) != 1:
		print('Inside find_M_N, I am expecting one match. I have ',len(model_str))
		print(model_str)
		#
	model_str = str(model_str[0]) 	
	a = model_str.find('_N')
	b = model_str.find('_M')	
	c = model_str.find(z0_tag)
	#
	N = int(model_str[a+2:b])
	M = int(model_str[b+2:c])

	return N, M, model_str



def find_numSWs(searchDir, specifier1, specifier2, stim):

	#
	# #
	# Find npz file (model_str) inside model_dir with unknown numSWs, numTrain, numTest but matching stim, msBins, EMsamps and rand.
	filesInDir = os.listdir( searchDir ) 
	model_str = [s for s in filesInDir if specifier1 in s and specifier2 in s ]
	#
	if len(model_str) != 1:
		print('I am expecting one match. I have ',len(model_str))
		print(model_str)
		#
	model_str = str(model_str[0])
	#
	a = model_str.find( str(stim+'_') )
	b = model_str.find('SWs_')
	#
	numSWs 		 = int(model_str[a+1+len(stim):b]) 

	return numSWs, model_str






def load_similarity_mat( flg_sparse, fname ):
	# (1). This function will compute or load similarity matrix.

	if os.path.isfile(fname): # if npz file exists, just load it in.
		print('Loading ',fname)
		t0 = time.time()
		if flg_sparse:  # Using sparse Spike Word Similarity Matrix
			Msim = spsp.load_npz(fname)

		else: # Using dense Spike Word Similarity Matrix
			data = np.load(fname)
			Msim = data['arr_0']
			del data

		t1 = time.time()
		print('Time = ',t1-t0)	

	else:
		print(fname, ' doesnt exist. Try constructing it.')	
		Msim = None
	
	return Msim	


def construct_AND_similarity_mat( numSW, unique_SW_bool, fname ):
	# (1). Actually make a dense Similarity matrix with a bitwise AND operation in
	#  a for loop and convert it to a sparse matrix and save one or both.

	if os.path.isfile(fname): # if npz file exists, just load it in.
		print('You have already constructed that dense similarity matrix. Use "load_similarity_mat" to load it in.')
		Msim = None

	else:
		print( 'Computing Spike Word Overlaps (AND) (this can take a long time)' )
		Msim = np.zeros( (numSW,numSW), dtype=np.int8 )
		for i in range( numSW ):
			print( i, ' / ', numSW )
			t0 = time.time()
			Msim[i] = np.bitwise_and(unique_SW_bool.T[i],unique_SW_bool.T).sum(axis=1).astype(np.uint8)
			t1 = time.time()
			print('Time = ',t1-t0)
		np.savez( fname, Msim )

	return Msim


def Jaccard_sim( MsimAND, lenSW, fname ):
	# compute Jacard Similarity from bitwise AND similarity matrix and 
	#				vector of number of active cells in each spikeword

	if os.path.isfile(fname):
			print( 'That sparse matrix already exists. ', fname, ' Loading it in.' )
			t0 = time.time()
			simTH = spsp.load_npz(fname)
			t1 = time.time()
			print('Time = ',t1-t0)
	else:
		print('Compute Jaccard similarity')
		numSW = lenSW.size
		t0 = time.time()
		simTH = [] # make an empty list
		for i in range(numSW):
			print(i)
			simTH.append( spsp.csr_matrix( (MsimAND.getrow(i)/lenSW[i]).astype(np.float16) ) ) # (just lenSW[i]? might be quicker) put a sparse 1D matrix in each entry of the list
		print('Time = ',time.time()-t0)
		#
		print('Converting list of sparse vectors into a sparse csr matrix (vstack).')	
		t0 = time.time()
		simTH = spsp.vstack(simTH) # stack all 1D sparse csr matrixes into a single 2D one.
		print('Time = ',time.time()-t0)
		#
		print('Size of Sparse Matrix in MB')
		print( (simTH.data.nbytes + simTH.indptr.nbytes + simTH.indices.nbytes)/10**6, 'MB' )
		#
		print('Save sparse matrix.', fname)
		t0 = time.time()
		spsp.save_npz( fname, simTH )
		t1 = time.time()
		print('Time = ',t1-t0)

	return simTH	



def Kulczynski_sim( MsimAND, lenSW, fname ):
	# Kulczynski Similarity is the symmetric version of Jaccard similarity

	if os.path.isfile(fname):
			print( 'That sparse matrix already exists. ', fname, ' Loading it in.' )
			t0 = time.time()
			simTH = spsp.load_npz(fname)
			t1 = time.time()
			print('Time = ',t1-t0)
	else:
		print('Compute Jaccard similarity')
		numSW = lenSW.size
		t0 = time.time()
		simTH = [] # make an empty list
		for i in range(numSW):
			print(i)
			simTH.append( spsp.csr_matrix( (0.5*MsimAND.getrow(i)*(1/lenSW + 1/lenSW[i])).astype(np.float16) ) ) # put a sparse 1D matrix in each entry of the list
		print('Time = ',time.time()-t0)
		#
		print('Converting list of sparse vectors into a sparse csr matrix (vstack).')	
		t0 = time.time()
		simTH = spsp.vstack(simTH) # stack all 1D sparse csr matrixes into a single 2D one.
		print('Time = ',time.time()-t0)
		#
		print('Size of Sparse Matrix in MB')
		print( (simTH.data.nbytes + simTH.indptr.nbytes + simTH.indices.nbytes)/10**6, 'MB' )
		#
		print('Save sparse matrix.', fname)
		t0 = time.time()
		spsp.save_npz( fname, simTH )
		print('Time = ',time.time()-t0)

	return simTH	




def threshold_similarity_mat(Msim,TH,numSW,fname):
	# (1). Create a new csr similarity matrix from Msim from only the values > TH.	
	#
	if os.path.isfile(fname):
		print( 'That sparse matrix already exists. ', fname, ' Loading it in.' )
		t0 = time.time()
		simTH = spsp.load_npz(fname)
		t1 = time.time()
		print('Time = ',t1-t0)

	else:
		print('Time to Threshold matrix and construct new one row by row.')
		t0 = time.time()
		simTH = [] 					# 3 Key Steps: (1). make an empty list
		for i in range(numSW):
			#print(i)
			x = Msim.getrow(i)
			R,C,_ = spsp.find(x>TH)
			D = x.data[np.where(x.data>TH)]
			y = spsp.csr_matrix( (D, (R, C)), shape=(1, numSW) )
			simTH.append(y) 	   # (2). put a sparse 1D matrix in each entry of the list
		simTH = spsp.vstack(simTH) # (3). stack all 1D sparse csr matrixes into a single 2D one.
		print('Time = ',time.time()-t0)
		#
		sparsity = simTH.nnz / numSW**2
		print('Sparsity of thresholded spike word similarity at = ', sparsity)
		#
		print('Size of Sparse Matrix in MB')
		print( (simTH.data.nbytes + simTH.indptr.nbytes + simTH.indices.nbytes)/10**6, 'MB' )
		#
		print('Save sparse matrix.', fname)
		t0 = time.time()
		spsp.save_npz( fname, simTH, sparsity )
		t1 = time.time()
		print('Time = ',t1-t0)

	return simTH







def convert_dense_to_sparse(Md,fname):
	# Convert a dense similarity matrix with many zeros to a sparse matrix. Then save it.

	if os.path.isfile(fname):
		print( 'That sparse matrix already exists. ', fname, ' Loading it in.' )
		t0 = time.time()
		Ms = load_similarity_mat( 1, fname )
		t1 = time.time()
		print('Time = ',t1-t0)

	else:
		print('Convert dense to sparse matrix.')
		t0 = time.time()
		Ms = spsp.csr_matrix(Md)
		t1 = time.time()
		print('Time = ',t1-t0)
		#
		print('Size of Sparse Matrix in MB')
		print( (Ms.data.nbytes + Ms.indptr.nbytes + Ms.indices.nbytes)/10**6 ) 
		#
		print('Save sparse matrix in ', fname)
		t0 = time.time()
		spsp.save_npz( fname, Ms )
		t1 = time.time()
		print('Time = ',t1-t0)	

	return Ms




def cuthill_mckee(A,sym=None):
	## Compute the Cuthill-Mckee reordering of A matrix (sparse. 
	# Makes sparse matrix block diagonal, which is essentially a clustering.

	print('Check type of A.  If it is not sparse, make it sparse.')
	A = spsp.csr_matrix(A)

	if sym is None: #not given in input whether matrix is symmetric. Must check it.
		t=time.time()
		print('Compute transpose')
		if np.any(A != A.transpose()):
			sym = False
		else:
			sym = True
		print(time.time()-t,' Seconds')    
		print( 'sym = ', sym )

	t=time.time()
	print('Compute Cuthill McKee Reordering')
	perm = spsp.csgraph.reverse_cuthill_mckee(A, symmetric_mode=sym)
	print(time.time()-t,' Seconds') 
	return perm	


