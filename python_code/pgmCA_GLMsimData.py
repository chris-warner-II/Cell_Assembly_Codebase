import argparse
import numpy as np
import scipy as sp
from scipy import io as io
import matplotlib.pyplot as plt
import time 						# to time operations for code analysis
import os.path 						# to check if a file or directory exists already
import sys

import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.plot_functions as pf
import utils.retina_computation as rc



def pgmCA_realData(args): # y=DEFAULTS["y"], ):  #

	print('Running EM algorithm to learn PGM parameters on real data.')


	print('NOTE: flg_include_Zeq0_infer defaults to true. Will never be false.')


	# Extract variables from args input into function from command line
	argsRec = args

	print(args)
	globals().update(vars(args))

	# Turn comma delimited strings input into lists.
	lRateScale 	= [float(item) for item in args.lRateScale.split(',')]

	sig_init = np.array([ sigQ_init, sigPi_init, sigPia_init ])

	# Name of data files saved from my preprocessing in MATLAB of raw datafiles from Greg Field's Lab.
	



	GField_spikeData_File = str('GLM_sim_'+whichPop+'_'+whichCells+'_'+whichGLM+'_spikeTrains_CellXTrial.mat')
	GField_STRFdata_File  = str('allCells_STRF_fits_329cells.mat')


	## (1). Set up directories and create dirs if necessary.
	dirHome, dirScratch = dm.set_dir_tree()
	#
	SW_extracted_Dir = str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')
	if not os.path.exists(SW_extracted_Dir):
		os.makedirs(SW_extracted_Dir)
	#
	EM_learning_Dir = str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Models_learned_EM/')
	if not os.path.exists(EM_learning_Dir):
		os.makedirs(EM_learning_Dir)
	#	
	EM_figs_dir = str( dirScratch + 'figs/PGM_analysis/EM_Algorithm/Greg_retinal_data/')
	if not os.path.exists(EM_figs_dir):
		os.makedirs(EM_figs_dir)


		#		
	colorsForPlot = ['blue','green','red','cyan','magenta','yellow','black']
	#
	if flg_include_Zeq0_infer:
		z0_tag='_zeq0'
	else:
		z0_tag='_zneq0'


	print( str(dirScratch + 'data/matlab_data/' + GField_spikeData_File) )

	## (2). Load in ndarray of spiketimes for 2 types of stimulus. They are of size: (#cells x #trials x #spikes)
	spikes = io.loadmat( str(dirScratch + 'data/matlab_data/' + GField_spikeData_File) ) # forPython.mat') )
	
	print(spikes.keys())



	# SW_bin = 2
	# pct_xVal_train = 0.5
	# train_2nd_model = True
	#maxSamps = 500
	


# 	if whichPop_MATfile == 'independent':

# 		print( whichPop_MATfile )

# 	elif whichPop_MATfile == 'coupldNindep':

# 		if whichCells=='OffBT_OffBS':
# 			cellSubTypes = ['offBriskTransient','offBriskSustained']
# 		elif whichCells=='OffBT':
# 			cellSubTypes = ['offBriskTransient']
# 		else:
# 			print( str( 'duh, Do not understand whichCells = '+str(whichCells) ) )


# 	else:
# 		print( ( str( 'duh, Do not understand whichPop_MATfile = '+str(whichPop_MATfile) ) ) )	



# if whichPop == 'subpop':

# elif whichPop == 'fullpop':
# 	whichGLM = 'ind' # only option is independent GLM for full popluation
# 	#
# 	if whichCells == 'offBT':

# 	elif whichCells == 'offBT_offBS':

# 	elif whichCells == 'offBT_onBT':

# 	else:
# 		print( ( str( 'Do not understand whichCells = '+str(whichCells) ) ) )	

# else:
# 	print( ( str( 'Do not understand whichPop = '+str(whichPop) ) ) )	










	triggers = spikes['trigs']
	cellMatlabIDs = spikes['matlab_inds']
	spikesIn = spikes['spikes']


	allCellTypes = spikes['allCellTypes']
	cellTypeIDs = spikes['cellTypeIDs']
	#
	# print( spikesIn.shape )
	# print( spikesIn[0][0].shape )
	# print( spikesIn[0][0] )
	# print( type(spikesIn) )
	# print(allCellTypes)
	# print(cellTypeIDs)
	#
	del spikes
	#
	N 			= spikesIn.shape[0] # number of cells in data.
	numTrials 	= spikesIn.shape[1] # number of trials in data.
	# 
	for T in range(numTrials):
		for c in range(N):
			spikesIn[c][T] = spikesIn[c][T][0] # getting rid of another dimension or some ndarray nested in an ndarray.
			







	## (4). Clean up cell type and cell id data passed in from MATLAB and grab cells from certain subtypes to look for Cell Assemblies in.
	#
	#
	# (a). Covert allCellTypes ndarray of ndarrays weird format passed from matlab to a list.
	cellTypesList = [ allCellTypes[i][0][0] for i in range( np.size(allCellTypes) ) ]
	print('cellTypesList',cellTypesList)
	#
	# (b). Grab entries in cellTypesList that are in particular cellSubTypes we want to compare.
	indx_cellSubTypes = [ allCellTypes[i][0][0] in cellSubTypes for i in range( np.size(allCellTypes) ) ]
	print('indx_cellSubTypes',indx_cellSubTypes)
	#
	# (c). Extract number of cells of each cell Type.
	num_cells_SubType = [ np.sum(cellTypeIDs[i]>0)  for i in range( np.size(allCellTypes) ) ]
	print('num_cells_SubType',num_cells_SubType)
	#
	# (d). Index into cellTypeIDs to grab cell IDs that belong to the cellSubTypes we want to compare.
	cellSubTypeIDs = cellTypeIDs[indx_cellSubTypes].flatten()
	cellSubTypeIDs = cellSubTypeIDs[ cellSubTypeIDs>0 ]
	print( 'cellSubTypeIDs',cellSubTypeIDs )
	#

	#print('cellSubTypeIDs', cellSubTypeIDs)
	try:
		print('cellMatlabIDs',cellMatlabIDs)
		print('cellIDs', cellSubTypeIDs[cellMatlabIDs-1] )
	except:
		print('***cellMatlabIDs',cellMatlabIDs)
		print('***cellIDs', cellSubTypeIDs )


	# (e). Index into spikesIn to look for spike words in spike trains from cells that belong to the cellSubTypes we want to compare.
	spikesInCellSubTypes = spikesIn 
	print(spikesInCellSubTypes.shape)

	N = spikesInCellSubTypes.shape[0] # number of cells that belong to the cellSubTypes we want to compare in data.










	## (3). Load in STRF parameters (Gaussian fits and temporal profiles. Compute distance matrix between pairwise RFs)
	if False:
		cellRFs = io.loadmat( str(dirScratch + 'data/matlab_data/' + GField_STRFdata_File) )
		STRFtime = cellRFs['STRF_TimeParams']
		STRFgauss = cellRFs['STRF_GaussParams']  # [Amplitude, x_mean, x_width, y_mean, y_width, orientation_angle]
		del cellRFs
		#
		# Compute pairwise distance between Receptive Fields in microns on retina
		pix2ret = 15*4 # microns. (15 monitor pixels per movie pixel that we have & 4 microns on retina per monitor pixel). Info from G. Field.
		distRFs = pix2ret*sp.spatial.distance_matrix( np.array([STRFgauss[:,1], STRFgauss[:,3]]).T, np.array([STRFgauss[:,1], STRFgauss[:,3]]).T )
		#
	





	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# Extract spikewords from the spike data or open up a previously saved data file.
	fname_SWs = str( SW_extracted_Dir + 'GLMsim_' + whichCells + '_' + whichPop + '_' + whichGLM + '_' + stim +'_' + str(1+2*SW_bin) + 'msBins.npz' )
	SWs, SWtimes = rc.extract_spikeWords(spikesIn, SW_bin, fname_SWs)
	#
	# 	SWs is a list of lists (of sets of cells in an assembly). 
	# 	Unwrap it into a single list of sets.
	all_SWs = list()
	all_SWtimes = list()
	all_SWtrials = list()
	for i in range(numTrials):
		all_SWs.extend(SWs[i])
		all_SWtimes.extend(SWtimes[i])
		all_SWtrials.extend( i*np.ones_like(SWs[i]) )
	num_SWs	= len(all_SWs)



	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# Sample spike words so that longer ones are sampled earlier. Or dont.
	print( '1st. ',sample_longSWs_1st, ' Sample with a bias for longer spikewords #', rand )
	YlenSrt = rc.sampleSWs_by_length( all_SWs, sample_longSWs_1st )
	#
	all_SWs_srt 		= [all_SWs[s] for s in YlenSrt ]
	all_SWtimes_srt 	= [all_SWtimes[s] for s in YlenSrt ]
	all_SWtrials_srt 	= [all_SWtrials[s] for s in YlenSrt ]
	#
	all_SWs 		= all_SWs_srt
	all_SWtimes 	= all_SWtimes_srt
	all_SWtrials 	= all_SWtrials_srt
	#
	del all_SWs_srt, all_SWtimes_srt, all_SWtrials_srt


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# take only first "maxSamps" samples if the variable is a number. Use them all if it is a nan.
	if not np.isnan( maxSamps ):
		print('2nd. Take only first',maxSamps,'spike words as training data.')
		#
		all_SWs_MS 		= [ all_SWs[s] 		for s in np.arange(int(maxSamps)) ]
		all_SWtimes_MS 	= [ all_SWtimes[s] 	for s in np.arange(int(maxSamps)) ]
		all_SWtrials_MS = [ all_SWtrials[s] for s in np.arange(int(maxSamps)) ]
		#
		all_SWs 	= all_SWs_MS
		all_SWtimes = all_SWtimes_MS
		all_SWtrials = all_SWtrials_MS
		#
		del all_SWs_MS, all_SWtimes_MS, all_SWtrials_MS
		


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# Randomly sample a set of indices of all spike words that pass yMin threshold.
	# The split these into train and test sets for Cross Validation.
	print('3rd. Randomly Sample Training and Test SW datasets for Cross Validation.')
	t0 = time.time()
	#
	indSWs_gt_yMin_train, indSWs_gt_yMin_test = rc.construct_crossValidation_SWs(all_SWs, yMinSW, pct_xVal_train)
	#
	print( 'Training Data = ',len(indSWs_gt_yMin_train) )
	print( 'Test Data = ',len(indSWs_gt_yMin_test) )
	#

	SWs_train 		= [ all_SWs[i] for i in indSWs_gt_yMin_train ]
	SWs_test  		= [ all_SWs[i] for i in indSWs_gt_yMin_test ]
	Z_list_train 	= [ set() for i in range(len(indSWs_gt_yMin_train)) ]
	Z_list_test 	= [ set() for i in range(len(indSWs_gt_yMin_test)) ]

	t1 = time.time()
	print('Done Sampling for Cross Validation: time = ',t1-t0)




	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# Sample spike words so that longer ones are sampled earlier. Or dont.
	print( '4th. ',sample_longSWs_1st, ' Sample with a bias for longer spikewords #', rand )
	YlenSrt = rc.sampleSWs_by_length( SWs_train, sample_longSWs_1st )
	#
	SWs_train_srt 	= [SWs_train[s] for s in YlenSrt ]
	SWs_train 		= SWs_train_srt
	del SWs_train_srt




	# print('SWs train', SWs_train)
	# print('len SWs train', len(SWs_train))
	# print('len SWs test', len(SWs_test))











	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# More Params and directories and junk
	num_EM_samps = int(num_SWs*pct_xVal_train)
	M_mod 				= overcomp*N # # number of Cell Assemblies (length of binary Z-vector)
	C_noise 			= np.array([Z_hot/M_mod, C_noise_ri, C_noise_ria ])		#[q, pi, pia] Mean of parameters for 'NoisyConst' initializations. Scalar between 0 and 1. 1 means almost silent. 0 means almost always firing.
	pjt_tol 			= 10
	numPia_snapshots 	= np.max( [np.round(num_EM_samps/ds_fctr_snapshots).astype(int), 1] )
	samps2snapshot_4Pia = (np.hstack([ 1, np.arange( np.round(num_EM_samps/numPia_snapshots), num_EM_samps, \
								np.round(num_EM_samps/numPia_snapshots)  ), num_EM_samps]) -1).astype(int)

	if not np.isnan( maxSamps ):
		maxSampTag = str( '_'+str( int(maxSamps ) )+'Samps' )
	else:
		maxSampTag = '_allSamps'


	if flg_EgalitarianPrior:	
		priorType = 'EgalQ' 
	else:
		priorType = 'BinomQ'	



	## Set up directory and file name for output data files.
	params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )
	dataModel_dir 	= str( str(cellSubTypes) + '_N' + str(N) + '_M' + str(M_mod) + z0_tag + '_ylims' + str(yLo) + '_' +  \
						str(yHi) + '_yMinSW' + str(yMinSW) + '_' + sample_longSWs_1st + 'Smp1st_' + priorType + '/')

	initLearn_dir 	= str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') + 
								'_LRsc' + str(lRateScale) + '/' )
	#

	if not os.path.exists( str(EM_learning_Dir + initLearn_dir + dataModel_dir) ):
		os.makedirs( str(EM_learning_Dir + initLearn_dir + dataModel_dir) )		
	#


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# (2).  Check if npz datafiles are already there for both A and 
	# 		B models. If so, break without doing anything else.
	#
	fname_EMlrn = str( EM_learning_Dir + initLearn_dir + dataModel_dir + 'LearnedModel_GLM' + whichGLM \
		+ '_' + stim + '_' + str(num_SWs) + 'SWs_' + str(pct_xVal_train).replace('.','pt') + 'trn_'\
		+ str(1+2*SW_bin) + 'msBins'+ maxSampTag + '_rand' + str(rand) + '.npz' )
	#
	fname_EMlrnB = str(fname_EMlrn[:-4]+'B.npz')
	print( fname_EMlrn )



	#flg_checkNPZvars = True # CHECK THAT THE LIST BELOW IS COORECT AND CONTAINS EVERTHING WE ARE SAVING
	# INTO NPZ FILES BEFORE RUNNING THIS FLAG=TRUE. WILL DELETE FILES THAT DO NOT CONTAIN THESE VARIABLES.
	#
	realKeys = ['qp', 'rip', 'riap', 'q_deriv', 'ri_deriv', 'ria_deriv', \
		'ria_snapshots', 'ri_snapshots', 'q_snapshots', 'ds_fctr_snapshots', \
		#
		'Z_inferred_train', 'pyiEq1_gvnZ_train', 'SWs_train', \
		'Z_inferred_test', 'pyiEq1_gvnZ_test', 'SWs_test', \
		#
		'pj_zHyp_train', 'pj_zHyp_test', 'pj_zTru_Batch', 'pj_zTrain_Batch', 'pj_zTest_Batch', \
		'cond_zHyp_train', 'cond_zHyp_test', 'cond_zTru_Batch', 'cond_zTrain_Batch', 'cond_zTest_Batch', \
		#
		'zActivationHistory', 'argsRec']
		


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# 
	# Check that npz file contains the all the right, most current keys.
	# If it doesnt, remove the file.
	fExists = False
	#
	if os.path.isfile(fname_EMlrn):
		fExists=True
		#
		if flg_checkNPZvars: 
			data = np.load(fname_EMlrn)
			if not all (k in data.keys() for k in realKeys):
				print('Old File. Removing it: ',fname_EMlrn)
				os.remove(fname_EMlrn)
				fExists=False
	#
	# #
	#
	if train_2nd_model and fExists and os.path.isfile(fname_EMlrnB):
		fExists=True
		#
		if flg_checkNPZvars: 
			data = np.load(fname_EMlrnB)
			if not all (k in data.keys() for k in realKeys):
				print('Old File. Removing it: ',fname_EMlrnB)
				os.remove(fname_EMlrnB)
				fExists=False
	else:
		fExists = False



	#
	if fExists:
		print('File(s) already exist for this rand', str(rand) ,' and pct_xVal_train', pct_xVal_train ,'. Not replacing em.')
		return
	else:
		print('File(s) do not exist for this rand', str(rand) ,' and pct_xVal_train', pct_xVal_train ,'.')



		print('N',N)
		print('M',M_mod)


	# Run Expectation Maximization Algorithm on Real Data.  If saved file already exists, just load it.
	q = np.zeros(1,) 			# for real data, we dont know model parameters so we set them to uniform, reasonable values.
	ri = np.ones(N,)
	ria = np.ones( (N,M_mod) )
	ria_mod = ria


	# Initialize parameters of model
	q_init, ri_init, ria_init, params_init_param = rc.init_parameters(q, ri, ria_mod, params_init, sig_init, C_noise)
	# qp = q_init
	# rip = ri_init
	# riap = ria_init	


	print('Training model using EM algorithm')
	t0 = time.time()
	#
	qp = q_init
	rip = ri_init
	riap = ria_init	
	#



	qp, rip, riap, Z_inferred_train, Z_inferred_test, pyiEq1_gvnZ_train, pyiEq1_gvnZ_test, \
	ria_snapshots, ri_snapshots, q_snapshots, q_deriv, ri_deriv, ria_deriv, Q_SE, Pi_SE, Pi_AE, \
	pj_zHyp_train, pj_zHyp_test, pj_zTru_Batch, pj_zTrain_Batch, pj_zTest_Batch, \
	cond_zHyp_train, cond_zHyp_test, cond_zTru_Batch, cond_zTrain_Batch, cond_zTest_Batch, zActivationHistory \
	= rc.run_EM_algorithm( qp, rip, riap, q, ri, ria_mod, q_init, ri_init, ria_init, Z_list_train, Z_list_test, \
		SWs_train, SWs_test, xVal_snapshot, xVal_batchSize, samps2snapshot_4Pia, pjt_tol, learning_rate, lRateScale, \
		flg_include_Zeq0_infer, yLo, yHi, flg_EgalitarianPrior, sample_longSWs_1st, verbose_EM)






	#
	# Save the learned model to an npz file
	print('Saving data from EM model learning in ', fname_EMlrn)
	np.savez( fname_EMlrn, qp=qp, rip=rip, riap=riap, Z_inferred_train=Z_inferred_train, Z_inferred_test=Z_inferred_test, \
		pyiEq1_gvnZ_train=pyiEq1_gvnZ_train, pyiEq1_gvnZ_test=pyiEq1_gvnZ_test, SWs_train=SWs_train, SWs_test=SWs_test, \
		q_snapshots=q_snapshots, ri_snapshots=ri_snapshots, ria_snapshots=ria_snapshots, ds_fctr_snapshots=ds_fctr_snapshots,\
		#
		pj_zHyp_train=pj_zHyp_train, pj_zHyp_test=pj_zHyp_test, pj_zTru_Batch=pj_zTru_Batch, pj_zTrain_Batch=pj_zTrain_Batch,\
		pj_zTest_Batch=pj_zTest_Batch, cond_zHyp_train=cond_zHyp_train, cond_zHyp_test=cond_zHyp_test, \
		cond_zTru_Batch=cond_zTru_Batch, cond_zTrain_Batch=cond_zTrain_Batch, cond_zTest_Batch=cond_zTest_Batch, \
		#
		q_deriv=q_deriv, ri_deriv=ri_deriv, ria_deriv=ria_deriv, zActivationHistory=zActivationHistory, argsRec=argsRec )	

	#
	t1 = time.time()
	print('Done Training and saving model: time = ',t1-t0)	









	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	#
	if train_2nd_model:
		# To validate cell assemblies found on real data, we split the data into two non-overlapping sets (called "test"
		# and "train") and train one model on each holding the other one out (can use the other one for cross-validation 
		# too). We can then compare the cell assemblies found by each model. For this "B" model, train using the same
		# noisy random nearly silent initialization and switch test to train and train to test.
		print('Training a 2nd model B on test data using EM algorithm')
		t0 = time.time()
		#
		qp = q_init
		rip = ri_init
		riap = ria_init	
		#


		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		# Sample spike words so that longer ones are sampled earlier. Or dont.
		print( '4th. ',sample_longSWs_1st, ' Sample with a bias for longer spikewords #', rand )
		YlenSrt = rc.sampleSWs_by_length( SWs_test, sample_longSWs_1st )
		#
		SWs_test_srt 	= [SWs_test[s] for s in YlenSrt ]
		SWs_test 		= SWs_test_srt
		del SWs_test_srt


		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		# reshuffle spike words in training data set since we are using it for test. 
		# Test reshuffling happens in sampleSWs_by_length above
		ind = np.random.choice(len(SWs_train), len(SWs_train), replace=False) 
		SWs_train = [ SWs_train[i] for i in list(ind.astype(int)) ] 



		qp, rip, riap, Z_inferred_train, Z_inferred_test, pyiEq1_gvnZ_train, pyiEq1_gvnZ_test, \
		ria_snapshots, ri_snapshots, q_snapshots, q_deriv, ri_deriv, ria_deriv, Q_SE, Pi_SE, Pi_AE, \
		pj_zHyp_train, pj_zHyp_test, pj_zTru_Batch, pj_zTrain_Batch, pj_zTest_Batch, \
		cond_zHyp_train, cond_zHyp_test, cond_zTru_Batch, cond_zTrain_Batch, cond_zTest_Batch, zActivationHistory \
		= rc.run_EM_algorithm( qp, rip, riap, q, ri, ria_mod, q_init, ri_init, ria_init, Z_list_test, Z_list_train, \
			SWs_test, SWs_train, xVal_snapshot, xVal_batchSize, samps2snapshot_4Pia, pjt_tol, learning_rate, lRateScale, \
			flg_include_Zeq0_infer, yLo, yHi, flg_EgalitarianPrior, sample_longSWs_1st, verbose_EM)



		#
		# Save the learned model to an npz file
		print('Saving data from EM model learning in ', str(fname_EMlrn[:-4]+'B.npz'))
		np.savez( str(fname_EMlrn[:-4]+'B.npz'), qp=qp, rip=rip, riap=riap, Z_inferred_train=Z_inferred_train, Z_inferred_test=Z_inferred_test, \
			pyiEq1_gvnZ_train=pyiEq1_gvnZ_train, pyiEq1_gvnZ_test=pyiEq1_gvnZ_test, SWs_train=SWs_train, SWs_test=SWs_test, \
			q_snapshots=q_snapshots, ri_snapshots=ri_snapshots, ria_snapshots=ria_snapshots, ds_fctr_snapshots=ds_fctr_snapshots,\
			#
			pj_zHyp_train=pj_zHyp_train, pj_zHyp_test=pj_zHyp_test, pj_zTru_Batch=pj_zTru_Batch, pj_zTrain_Batch=pj_zTrain_Batch,\
			pj_zTest_Batch=pj_zTest_Batch, cond_zHyp_train=cond_zHyp_train, cond_zHyp_test=cond_zHyp_test, \
			cond_zTru_Batch=cond_zTru_Batch, cond_zTrain_Batch=cond_zTrain_Batch, cond_zTest_Batch=cond_zTest_Batch, \
			#
			q_deriv=q_deriv, ri_deriv=ri_deriv, ria_deriv=ria_deriv, zActivationHistory=zActivationHistory, argsRec=argsRec )	


		t1 = time.time()
		print('Done Training and saving a 2nd model B on test data: time = ',t1-t0)	

		#
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


	
		




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == "__main__":

	#
	# # Set up Parser
	#
	print('Trying to run as a command line function call.')
	parser = argparse.ArgumentParser(description="run pgmCA_realData from command line")
	
	#
	# # Info about which real data to process.
	#
	# cell_type
	parser.add_argument('-ct', '--cell_type', dest='cell_type', type=str, default='allCells', 
		help='What cell type to process from GField retina data. ex.(allCells)')
	# cellSubTypes
	parser.add_argument('-cst', '--cellSubTypes', dest='cellSubTypes', type=str, default=['offBriskTransient','offBriskSustained'], 
		help='Which subtypes of cells to run. ')
	# whichCells
	parser.add_argument('-cll', '--whichCells', dest='whichCells', type=str, default='offBT', 
		help='Which subtypes of cells GLM simulations was run on. Either "offBT" or "offBT_offBS" or "offBT_onBT" or "" if real data.')
	# whichPop
	parser.add_argument('-pop', '--whichPop', dest='whichPop', type=str, default='fullpop',
		help='Which popluation of cells GLM simulation was run on "fullpop" for 137 cells or "subpop" of 8 or 9 cells or "" for real data')
	# whichGLM
	parser.add_argument('-glm', '--whichGLM', dest='whichGLM', type=str, default='ind',
		help='Which type of GLM simulation was run "ind" or "cpl"')
	# stim
	parser.add_argument('-st', '--stim', dest='stim', type=str, default='NatMov',
		help='Which stimuli presentations to process. ex( ''NatMov'' or ''Wnoise'' )')
	# SW_bin
	parser.add_argument('-sw', '--SW_bin', dest='SW_bin', type=int, default=0,
		help='Bin width in ms within which to consider spikes part of a spike word, ex([0,1,2]) --> 1(+/-)bin --> [1,3,5]ms ')
	# yLo
	parser.add_argument('-yLo', '--yLo', dest='yLo', type=int, default=1,
		help='If |y| < yLo, automatically infer z=0 vector.')
	# yHi
	parser.add_argument('-yHi', '--yHi', dest='yHi', type=int, default=9999,
		help='If |y| > yHi, z=0 not allowed for inference. Assume there must be at least one CA on.')
	# yMinSW
	parser.add_argument('-ySW', '--yMinSW', dest='yMinSW', type=int, default=1,
		help='If |y| < yLo, ignore the spike word for learning. Minimum number of active cells in a time bin to be considered a spike word and used to train model.')
	# num_test_samps_4xVal
	parser.add_argument('-nXs', '--num_test_samps_4xVal', dest='num_test_samps_4xVal', type=int, default=1,
		help='Number of test data samples to compute pjoint on and average over to implement cross validation.')
	# rand
	parser.add_argument('-rnd', '--rand', dest='rand', type=int, default=0,
		help='Due to order of spike words sampled, model learned can be different. Model can be learned with a different random sampling of spike words.')
	# pct_xVal_train
	parser.add_argument('-xvp', '--pct_xVal_train', dest='pct_xVal_train', type=float, default=0.8,
		help='Percent of spike words to put into the training data set. Put the rest into test data set to perform cross-validation.')
	# xVal_snapshot
	parser.add_argument('-xVs', '--xVal_snapshot', dest='xVal_snapshot', type=int, default=1,
		help='How often to compute batches of joint and conditional probabilities to average over them.')
	# xVal_batchSize
	parser.add_argument('-xVb', '--xVal_batchSize', dest='xVal_batchSize', type=int, default=1,
		help='How large a batch of data points to compute joint and conditional probabilities to average over them.')

	#
	# # Model Initialization parameters 
	#
	# params_init
	parser.add_argument('-pi', '--params_init', dest='params_init', type=str, default='NoisyConst',
		help='How to initialize the model for EM algorithm. ex(NoisyConst True NoisyTrue RandomUniform NoisyConst) ')
	# sigQ_init
	parser.add_argument('-si_q', '--sigQ_init', dest='sigQ_init', type=float, default=0.01,
		help='Variability / sigma in model params initializations. Distributed around centers defined by [C_noise_ri, C_noise_ria, overcomp, Z_hot') 
	# sigPi_init
	parser.add_argument('-si_ri', '--sigPi_init', dest='sigPi_init', type=float, default=0.05,
		help='Variability / sigma in model params initializations. Distributed around centers defined by [C_noise_ri, C_noise_ria, overcomp, Z_hot') 
	# sigPia_init
	parser.add_argument('-si_ria', '--sigPia_init', dest='sigPia_init', type=float, default=0.05,
		help='Variability / sigma in model params initializations. Distributed around centers defined by [C_noise_ri, C_noise_ria, overcomp, Z_hot') 
	# overcomp
	parser.add_argument('-oc', '--overcomp', dest='overcomp', type=int, default=1,
		help='How over overcomplete model is vs cells. M = (oc)*N')
	# Z_hot
	parser.add_argument('-z1', '--Z_hot', dest='Z_hot', type=int, default=4,
		help='How many 1''s in the sparse z-vector')
	# C_noise_ri
	parser.add_argument('-mi_ri', '--C_noise_ri', dest='C_noise_ri', type=float, default=1.,
		help='Center or mean of distributions on Pi in initializations. ex(=1 means all r_i initialized near silent)')
	# C_noise_ria
	parser.add_argument('-mi_ria', '--C_noise_ria', dest='C_noise_ria', type=float, default=1.,
		help='Center or mean of distributions on Pia in initializations. ex(=1 means all r_ia initialized near silent)')

	#
	# # Learning rate stuff and some options on Inference.
	#
	# learning_rate,
	parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=0.5,
		help='Learning rate in model learning at each iteration. ex(1 works well)')  
	# lRateScale
	parser.add_argument('-LRS', '--lRateScale', dest='lRateScale', type=str, default='1.0,0.1,0.1',
		help='Comma delimited string to be converted into a list of learning rate scales for Pia, Pi and Q.')
	# flg_include_Zeq0_infer
	parser.add_argument('-z0', '--flg_include_Zeq0_infer', dest='flg_include_Zeq0_infer', action='store_true', default=True, 
		help=', Flag to include the z=0''s vector in inference if True. ex( True or False))')
	# sample_longSWs_1st
	parser.add_argument('-smp1st', '--sample_longSWs_1st', dest='sample_longSWs_1st', type=str, default='Dont',
		help="Sample Spikewords for learning randomly, with probability proportional to their length, or sorted by there SW length. Options are: {'Dont', 'Prob', 'Hard'} ")
	# maxSamps
	parser.add_argument('-ms', '--maxSamps', dest='maxSamps', type=float, default=np.nan,
		help='Number of Spikewords to use as samples. A scalar number or NaN to use all available data.') 

	#
	# # Flags to save, for inference, and verbose.
	#
	#
	# ds_fctr_snapshots,
	parser.add_argument('-t_snap', '--ds_fctr_snapshots', dest='ds_fctr_snapshots', type=int, default=100,
		help='How often to take snapshot of model params during learning process.')
	# flg_EgalitarianPrior
	parser.add_argument('-Egal', '--flg_EgalitarianPrior', dest='flg_EgalitarianPrior', action='store_true', default=False,
		help='Flag to use Egalitarian activity based prior on z-vec by Q-vec. Otherwise use Binomial Q-scalar prior.. ex( True or False))')	
	# train_2nd_model
	parser.add_argument('-2nd', '--train_2nd_model', dest='train_2nd_model', action='store_true', default=False,
		help='Flag to train a 2nd model with 50% test data so we can compare CAs found by each. ex( True or False))')	
	# flg_checkNPZvars
	parser.add_argument('-chkNPZ', '--flg_checkNPZvars', dest='flg_checkNPZvars', action='store_true', default=False,
		help='Flag to check keys / variables in the NPZ file and get rid of the file if they dont match expected values.')
	# verbose_EM
	parser.add_argument('-v', '--verbose_EM', dest='verbose_EM', action='store_true', default=False,
		help=', Flag to dislay additional output messages (Sanity Checking). ex( True or False))')	


	#
	# # Get args from Parser.
	#
	args = parser.parse_args()
	pgmCA_realData(args) #args.x, args.y)  
