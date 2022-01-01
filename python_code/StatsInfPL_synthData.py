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


# %pdb - python debug.
#DEFAULTS = dict()


def StatsInfPL_synthData(args): # y=DEFAULTS["y"], ):  #

	print('THIS FUNCTION IS NOT FINISHED !!!')

	print('Running EM algorithm to learn PGM parameters on real data.')

	# Extract variables from args input into function from command line
	argsRec = args

	print(args)
	globals().update(vars(args))





	M_mod 		= overcomp*M
	C_noise 	= np.array([Z_hot/M_mod, C_noise_ri, C_noise_ria ])		# Mean of parameters for 'NoisyConst' initializations. Scalar between 0 and 1. 1 means almost silent. 0 means almost always firing.
	sig_init 	= np.array([sigQ_init, sigPi_init, sigPia_init ])		# STD on parameter initializations from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
	pjt_tol 			= 10
	numPia_snapshots 	= np.round(num_EM_samps/ds_fctr_snapshots).astype(int)
	samps2snapshot_4Pia = (np.hstack([ np.arange( np.round(num_EM_samps/numPia_snapshots), num_EM_samps, 
							np.round(num_EM_samps/numPia_snapshots)  ), num_EM_samps]) - 1).astype(int)


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# # (1). Synthesize model from user input parameters.
	# q = None #ri = None #ria = None #ria_mod = None
	# while q==None: # while statement to resynthesize model if any cells participate in 0 assemblies.
	# # 	q, ri, ria, ria_mod = rc.synthesize_model(N,M,M_mod, C,K, Cmin,Cmax, bernoulli_Pi,mu_Pi,sig_Pi, mu_Pia,sig_Pia, verbose=False)




	# when dealing with real data we dont know the model parameters so we set them to these uniform, reasonable values.
	# When dealing with synthetic model and generated data, these will have been defined above.
	#q_init, ri_init, ria_init, params_init_param = rc.init_parameters(q, ri, ria_mod, params_init, sig_init, C_noise)
	params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )
	#


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# (4).  Directory structure for output figures
	#
	if flg_include_Zeq0_infer:
		z0_tag='_zeq0'
	else:
		z0_tag='_zneq0'

	ModelType = str( 'SyntheticData_N' + str(N) + '_M' + str(M) + '_Mmod' + str(M_mod) + '_K' + str(K) + 
				'_' + str(Kmin) + '_' + str(Kmax) + '_C' + str(C) + '_' + str(Cmin)+ '_' + str(Cmax) + 
				'_mPia' + str(mu_Pia) + '_sPia' + str(sig_Pia) + '_bPi' + str(bernoulli_Pi) + '_mPi' + 
				str(mu_Pi) + '_sPi' + str(sig_Pi) + z0_tag + '_xVal/' )

	InitLrnInf = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') +
				'_LRpi' + str(lRateScale_Pi).replace('.','pt') + '/' )
	#
	dirHome, dirScratch = dm.set_dir_tree()
	#
	EM_data_dir = str( dirScratch + 'data/python_data/PGM_analysis/synthData/Models_learned_EM/' + InitLrnInf + ModelType )
	#	
	SW_extracted_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')
	#
	EM_learnStats_Dir	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/InferStats_from_EM_learning/')
	#
	infer_postLrn_dir = str( dirScratch + 'data/python_data/PGM_analysis/synthData/inference_postLrn/' + InitLrnInf + ModelType )
	if not os.path.exists(infer_postLrn_dir):
		os.makedirs(infer_postLrn_dir)
	




	sig_init 		= np.array([ sigQ_init, sigPi_init, sigPia_init ])
	params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )

	

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # (1).  Set up directory structure and filename. Load it in and extract variables.
	init_dir = str( 'Init' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') + '_LRpi' + str(lRateScale_Pi).replace('.','pt') +'/' )
	#

	# #
	# # Find directory (model_dir) with unknown N and M that matches cell_type and yMin
	# subDirs = os.listdir( str(EM_learning_Dir + init_dir) ) 
	# CST = str(cellSubTypes).replace('\'','').replace(' ','') # convert cell_type list into a string of expected format.
	# model_dir = [s for s in subDirs if CST in s and str( z0_tag + '_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) ) in s ]
	# #print(model_dir)
	# #
	# if len(model_dir) != 1:
	# 	print('I am expecting one matching directory. I have ',len(model_dir))
	# 	print(model_dir)
	# #
	# model_dir = str(model_dir[0]+'/')	
	# print(model_dir)
	# #
	# a = model_dir.find('_N')
	# b = model_dir.find('_M')	
	# c = model_dir.find(z0_tag)
	# #
	# N = int(model_dir[a+2:b])
	# M = int(model_dir[b+2:c])
	# #
	# # #



	# # WORKING HERE TO CONVERT THIS FOR SYNTH DATA !!!
	# Inside dir:  /clusterfs/cortex/scratch/cwarner/Projects/G_Field_Retinal_Data/data/python_data/PGM_analysis/
	#	synthData/Models_learned_EM/Init_DiagonalPia_1hot_mPi1.0_mPia1.0_sI[0.01 0.05 0.05]_LR0pt5_LRpi1pt0/
	#	SyntheticData_N50_M50_Mmod50_K2_0_2_C2_2_6_mPia0.0_sPia0.1_bPi1.0_mPi0.0_sPi0.1_zeq0_xVal_gnu/
	# Looking for file like: EM_model_data_ 50000 [xx,yy]SWsTrnTst_ylims 0 _ 1000 _ 10000 EMsamps_*.npz






	# Find npz file (model_file) inside model_dir with unknown numSWs, numTrain, numTest but matching stim, msBins, EMsamps and rand.
	filesInDir = os.listdir( str(EM_learning_Dir + init_dir + model_dir) ) 
	#
	model_files = [s for s in filesInDir if str('LearnedModel_' + stim) in s and \
			str( 'SWsTstTrn_' + str(msBins) + 'msBins_' + str(num_EM_samps) + 'EMsamps_rand' + str(rand) ) in s ]
	#
	try:
		mfA = [s for s in model_files if str('rand' + str(rand) + '.npz') in s ] 	# Model learned from train data.
		mfB = [s for s in model_files if str('rand' + str(rand) + 'B.npz') in s ] 	# Model learned from test data.
		print(mfA)
		print(mfB)

		if len(mfA>1):
			# # THIS CAN BE USEFUL WHEN WE ARE TRAINING_2ND_MODEL AND WE WANT MODEL NAMES TO MATCH OTHER THAN FOR THE "B"
			mfA = difflib.get_close_matches(mfB[0], mfA)

		if len(mfB>1):
			print('ugh, i dont fuckin know. i quit.')

	except:
		mfA = model_files		
	#
	if train_2nd_model:  
		model_file = str(mfB[0]) 		# Process *B.npz file. Learned on Test data.
	else:
		model_file = str(mfA[0]) 		# Process *.npz file. Learned on Train data.

	a = model_file.find( str(stim+'_') )
	b = model_file.find('[')
	c = model_file.find(',')	
	d = model_file.find(']')
	#
	numSWs 		 = int(model_file[a+1+len(stim):b]) 
	numSWs_train = int(model_file[b+1:c])
	numSWs_test  = int(model_file[c+1:d])

	print(numSWs, numSWs_train, numSWs_test)


	# # # # # # # #  # # # # # # # #  # # # # # # # # # # # # # # # # # # # # # # # # 

	# Make a directories for output plots if they are not already there.
	# if not os.path.exists( str(EM_figs_dir + init_dir + model_dir) ):
	# 	os.makedirs( str(EM_figs_dir + init_dir + model_dir) )
	#
	if not os.path.exists( str(EM_learnStats_Dir + init_dir + model_dir) ):
		os.makedirs( str(EM_learnStats_Dir + init_dir + model_dir) )
	#
	if not os.path.exists( str(EM_learning_Dir + init_dir + model_dir) ):
		os.makedirs( str(EM_learning_Dir + init_dir + model_dir) )	
	#
	if not os.path.exists( str(Infer_postLrn_Dir + init_dir + model_dir) ):
		os.makedirs( str(Infer_postLrn_Dir + init_dir + model_dir) )







	# Extracting number of training and test samples from the file name. 
	fnamesInDir = os.listdir( str(EM_learning_Dir + init_dir + model_dir ) ) 
	fname = [s for s in fnamesInDir if str('LearnedModel_' + stim) in s \
			and str('SWsTstTrn_' + str(msBins) + 'msBins_' + str(num_EM_samps) + 'EMsamps_rand' + str(rand) + Btag +'.npz' ) in s ]
	if len(fname) != 1:
		print('I am expecting one matching file.  I have ',len(fname))
		print('In dir: ',str(EM_learning_Dir + init_dir + model_dir ))
		print(fname)
		# continue

	model_file = fname[0][:-4]






	



	# Set up file names and directories.
	LearnedModel_fname 			= str( EM_learning_Dir   + init_dir + model_dir + model_file + '.npz')	
	StatsDuringLearning_fname 	= str( EM_learnStats_Dir + init_dir + model_dir + model_file.replace('LearnedModel_','InferStats_DuringLearn_') + '.npz')
	StatsPostLearning_fname 	= str( EM_learnStats_Dir + init_dir + model_dir + model_file.replace('LearnedModel_','InferStats_PostLrn_') + '.npz')
	rasterZs_fname 				= str( Infer_postLrn_Dir + init_dir + model_dir + model_file.replace('LearnedModel_','rasterZ_allSWs_') + '.npz')

















	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	print('Loading in data file saved from EM learning algorithm in pgmCA_realData.py')
	print( LearnedModel_fname )
	#
	data = np.load( LearnedModel_fname )
	print(data.keys())
	#
	qp = data['qp']
	rip = data['rip']
	riap = data['riap']
	#
	argsRec = data['argsRec']
	#
	Z_inferred_train = data['Z_inferred_train']
	Z_inferred_test = data['Z_inferred_test']
	Y_inferred_train = data['Y_inferred_train']
	Y_inferred_test = data['Y_inferred_test']
	#
	indSWs_gt_yMin_train = data['indSWs_gt_yMin_train']
	indSWs_gt_yMin_test = data['indSWs_gt_yMin_test']
	#
	smp_train = data['smp_train']
	smp_test = data['smp_test']
	pjoint_train = data['pjoint_train']
	pjoint_test = data['pjoint_test']
	#
	# if flg_Pia_snapshots_gif:
	# 	ria_snapshots = data['ria_snapshots']
	# 	ri_snapshots = data['ri_snapshots']
	# 	q_snapshots = data['q_snapshots']
	#
	q_deriv = data['q_deriv']
	ri_deriv = data['ri_deriv']
	ria_deriv = data['ria_deriv']
	#
	q_MSE = data['q_MSE']
	ri_MSE = data['ri_MSE']
	ria_MSE = data['ria_MSE']
	#
	del data	

	# Used below in plotting and junk
	Q = rc.sig(qp)
	Pi = rc.sig(rip)
	Pia = rc.sig(riap)








	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # (2).  Get extracted spikewords from spike data to look at statistics of spiking activity.
	# Extract spike words from raw data or load in npz file if it has already been done and saved.
	print('Extracting spikewords')
	t0 = time.time()
	fname_SWs = str( SW_extracted_Dir + cellSubTypes + '_' + stim + '_' + str(msBins) + 'msBins.npz' )
	spikesIn = list()
	SWs, SWtimes = rc.extract_spikeWords(spikesIn, msBins, fname_SWs)
	#
	#numTrials = len(SWs)
	t1 = time.time()
	print('Done Extracting spikewords: time = ',t1-t0) # Fast: ~5 seconds.
	





	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	print('Loading in data file saved from post EM learning inference in raster_zs_inferred_allSWs_given_model.py')
	print(  )
	#
	try:	
		data = np.load( rasterZs_fname )
		print(data.keys())
		#
		Z_inferred_allSWs 			= data['Z_inferred_allSWs'] 
		Y_inferred_allSWs 			= data['Y_inferred_allSWs'] 
		pj_inferred_allSWs 			= data['pj_inferred_allSWs'] 

		Ycell_hist_allSWs 			= data['Ycell_hist_allSWs']
		YcellInf_hist_allSWs 		= data['YcellInf_hist_allSWs'] 
		Zassem_hist_allSWs 			= data['Zassem_hist_allSWs']

		nY_allSWs 					= data['nY_allSWs']
		nYinf_allSWs 				= data['nYinf_allSWs'] 
		nZ_allSWs 					= data['nZ_allSWs']

		CA_coactivity_allSWs 		= data['CA_coactivity_allSWs']
		Cell_coactivity_allSWs 		= data['Cell_coactivity_allSWs'] 
		CellInf_coactivity_allSWs 	= data['CellInf_coactivity_allSWs'] 

		argsRecModelLearn 			= data['argsRecModelLearn']
		argsRaster 					= data['argsRaster']
		#
		# try:
		# 	raster_Z_inferred_allSWs 	= data['raster_Z_inferred_allSWs'] 
		# 	raster_Y_inferred_allSWs 	= data['raster_Y_inferred_allSWs']
		# 	raster_allSWs 				= data['raster_allSWs']
		# except:
		# 	print('Converting SWs,Zinf,Yinf into 3D tensor that is CellsCAs x Trials x TimesActive.')
		# 	t0 = time.time()
		# 	#
		# 	print('raster_Z_inferred_allSWs')
		# 	raster_Z_inferred_allSWs = rc.compute_raster_list(SWtimes, Z_inferred_allSWs, M, minTms, maxTms )
		# 	print('raster_Y_inferred_allSWs')
		# 	raster_Y_inferred_allSWs = rc.compute_raster_list(SWtimes, Y_inferred_allSWs, N, minTms, maxTms )
		# 	print('raster_allSWs')
		# 	raster_allSWs 			 = rc.compute_raster_list(SWtimes, SWs, N, minTms, maxTms ) 
			# #
			# t1 = time.time()
			# print('Done Converting SWs,Zinf,Yinf into 3D tensor that is CellsCAs x Trials x TimesActive: time = ',t1-t0)
		#
		del data

	except:
		print( rasterZs_fname )
		print('Inference / raster file not there. Moving on.')
		#continue	
	
	num_SWs = nY_allSWs.size





	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (Stats XX). Some explanation.
	#
	#	
	print('Unwrap SWs, SWtimes, SWtrials, Zinf, Yinf into 1D vector.')
	t0 = time.time()
	#
	# SWs is a list of lists (of sets of cells in an assembly). Unwrap it into a single list of sets.
	all_SWs = list()
	all_SWtimes = list()
	all_SWtrials = list()
	#
	all_Zinf = list()
	all_Yinf = list()
	all_pjs  = list() 
	#
	numTrials = len(SWs)
	for i in range(numTrials):
		#
		#index into spike times that are between minTms and maxTms
		ind = list( np.where(np.bitwise_and(np.array(SWtimes[i])>minTms, np.array(SWtimes[i])<maxTms))[0] )
		#
		# Ind is 1 too long on rare / single occasion.
		if ind[-1] > np.min( [len(SWs[i]),len(Y_inferred_allSWs[i]), len(Z_inferred_allSWs[i])] )-1:
			# print( 'max ind = ', ind[-1] )
			# print( 'len SWs = ', len(SWs[i]) )
			# print( 'len Yinf = ', len(Y_inferred_allSWs[i]) )
			# print( 'len Zinf = ', len(Z_inferred_allSWs[i]) )
			xxx = list(np.where(ind < np.min( [len(SWs[i]),len(Y_inferred_allSWs[i]), len(Z_inferred_allSWs[i])] ) )[0])
			print('Trial #',i,' : cut out ', len(ind) - len(xxx),' from weirdness.')
			print('             : cut out ', len(SWtimes[i]) - len(xxx),'from weirdness and overtime.')
			ind = [ind[x] for x in xxx ]
		#
		all_SWs.extend([SWs[i][x] for x in ind]) #   SWs[i][ind])
		all_SWtimes.extend([SWtimes[i][x] for x in ind])
		all_SWtrials.extend( i*np.ones_like(ind) )
		#
		all_Zinf.extend([Z_inferred_allSWs[i][x] for x in ind])
		all_Yinf.extend([Y_inferred_allSWs[i][x] for x in ind])
		for x in ind:
			try:
				all_pjs.extend([pj_inferred_allSWs[i][x][0]]) # sometimes is array.
			except:
				all_pjs.extend([pj_inferred_allSWs[i][x]]) # sometimes is scalar.

		#
	num_SWsUnWrap = len(all_SWs)	

	# for jj in range(numTrials):
	# 	print( jj, len(SWs[jj]), len(Y_inferred_allSWs[jj]), len(Z_inferred_allSWs[jj]) )
	#	
	t1 = time.time()
	print('Done Unwrapping SWs, SWtimes, SWtrials, Zinf, Yinf into 1D vector: time = ',t1-t0)
		
	#
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
											











	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (Stats XX). Some explanation.
	#
	#	
	if flg_compute_StatsDuringEM:

		try:	
			data = np.load( StatsDuringLearning_fname )
			print(data.keys())
			#
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
			print('Loading in data file saved from inference during EM learning.')
			print( StatsDuringLearning_fname )
			
			Ycell_hist_Samps 				= data['Ycell_hist_Samps']
			nY_Samps 					 	= data['nY_Samps']
			Cell_coactivity_Samps 			= data['Cell_coactivity_Samps']
			#
			Ycell_hist_Infer_train 			= data['Ycell_hist_Infer_train'] 
			Zassem_hist_Infer_train 		= data['Zassem_hist_Infer_train']
			nY_Infer_train 					= data['nY_Infer_train']
			nZ_Infer_train 					= data['nZ_Infer_train']
			CA_coactivity_Infer_train		= data['CA_coactivity_Infer_train']
			Cell_coactivity_Infer_train 	= data['Cell_coactivity_Infer_train']
			#
			Ycell_hist_Infer_test 			= data['Ycell_hist_Infer_test']
			Zassem_hist_Infer_test 			= data['Zassem_hist_Infer_test'] 
			nY_Infer_test 					= data['nY_Infer_test']
			nZ_Infer_test 					= data['nZ_Infer_test']
			CA_coactivity_Infer_test 		= data['CA_coactivity_Infer_test'] 
			Cell_coactivity_Infer_test 		= data['Cell_coactivity_Infer_test']
			#
			Kinf_train 						= data['Kinf_train']
			KinfDiff_train 					= data['KinfDiff_train']
			zCapture_train 					= data['zCapture_train']
			zMissed_train 					= data['zMissed_train']
			zExtra_train 					= data['zExtra_train']
			yCapture_train 					= data['yCapture_train']
			yMissed_train 					= data['yMissed_train']
			yExtra_train 					= data['yExtra_train']
			inferCA_Confusion_train 		= data['inferCA_Confusion_train']
			zInferSampled_train 			= data['zInferSampled_train']
			zInferSampledRaw_train 			= data['zInferSampledRaw_train']
			zInferSampledT_train 			= data['zInferSampledT_train']
			inferCell_Confusion_train 		= data['inferCell_Confusion_train']
			yInferSampled_train 			= data['yInferSampled_train']
			#
			Kinf_test 						= data['Kinf_test']
			KinfDiff_test 					= data['KinfDiff_test']
			zCapture_test 					= data['zCapture_test']
			zMissed_test 					= data['zMissed_test']
			zExtra_test 					= data['zExtra_test']
			yCapture_test 					= data['yCapture_test']
			yMissed_test 					= data['yMissed_test']
			yExtra_test 					= data['yExtra_test']
			inferCA_Confusion_test 			= data['inferCA_Confusion_test']
			zInferSampled_test 				= data['zInferSampled_test']
			zInferSampledRaw_test 			= data['zInferSampledRaw_test']
			zInferSampledT_test 			= data['zInferSampledT_test']
			inferCell_Confusion_test 		= data['inferCell_Confusion_test']
			yInferSampled_test 				= data['yInferSampled_test']

			del data

		except:
			
			# Stats on spike words sampled for EM algorithm to learn model on.
			print('Compute statistics for sampling, train inference and test inference during EM algorithm')
			t0 = time.time()
			smp_SWs = [ all_SWs[i] for i in smp_train ]
			# NOTE: There are no Zs Ground Truth in Real Data.
			Ycell_hist_Samps, _, nY_Samps, _, _, Cell_coactivity_Samps \
				= rc.compute_dataGen_Histograms(smp_SWs, list(np.arange(num_EM_samps)) , M, N) #, samps
			#
			# #
			#
			# Stats on active cells and cell assemblies inferred on training data during EM algorithm
			Ycell_hist_Infer_train, Zassem_hist_Infer_train, nY_Infer_train, nZ_Infer_train,\
				CA_coactivity_Infer_train, Cell_coactivity_Infer_train \
				= rc.compute_dataGen_Histograms(Y_inferred_train, Z_inferred_train, M, N) #, samps
			#
			# #
			#
			# Stats on active cells and cell assemblies inferred on test data during EM algorithm
			Ycell_hist_Infer_test, Zassem_hist_Infer_test, nY_Infer_test, nZ_Infer_test,\
				CA_coactivity_Infer_test, Cell_coactivity_Infer_test \
				= rc.compute_dataGen_Histograms(Y_inferred_test, Z_inferred_test, M, N) #, samps
			#
			t1 = time.time()
			print('Done w/ Computing statistics for sampling, train inference and test inference during EM algorithm : time = ',t1-t0)#
			#
			# # 
			# # #
			# #
			#
			print('Compute  %captured, missed, extra etc statistics for train inference and test inference during EM algorithm')
			t0 = time.time()
			Y_train = [ all_SWs[i] for i in indSWs_gt_yMin_train[smp_train] if i < num_SWsUnWrap ]
			Z_train = [ list() for num in range(len(Y_train)) ]
			Yinf_train = [ Y_inferred_train[i] for i in range(num_EM_samps) if indSWs_gt_yMin_train[smp_train[i]] < num_SWsUnWrap ]
			Zinf_train = [ Z_inferred_train[i] for i in range(num_EM_samps) if indSWs_gt_yMin_train[smp_train[i]] < num_SWsUnWrap ]
			#
			Kinf_train, KinfDiff_train, zCapture_train, zMissed_train, zExtra_train, \
			yCapture_train, yMissed_train, yExtra_train, inferCA_Confusion_train, zInferSampled_train, zInferSampledRaw_train, \
			zInferSampledT_train, inferCell_Confusion_train, yInferSampled_train = rc.compute_inference_statistics_allSamples( \
			len(Y_train), N, M, Z_train, Zinf_train, np.arange(M), np.arange(M), Y_train, Yinf_train, verbose_EM)
			#
			# Computing Inference statistics for test samples.
			Y_test = [ all_SWs[i] for i in indSWs_gt_yMin_test[smp_test]  if i < num_SWsUnWrap]
			Z_test = [ list() for num in range(len(smp_test)) ]
			Yinf_test = [ Y_inferred_test[i] for i in range(num_EM_samps) if indSWs_gt_yMin_test[smp_test[i]] < num_SWsUnWrap ]
			Zinf_test = [ Z_inferred_test[i] for i in range(num_EM_samps) if indSWs_gt_yMin_test[smp_test[i]] < num_SWsUnWrap ]
			#
			Kinf_test, KinfDiff_test, zCapture_test, zMissed_test, zExtra_test, \
			yCapture_test, yMissed_test, yExtra_test, inferCA_Confusion_test, zInferSampled_test, zInferSampledRaw_test, \
			zInferSampledT_test, inferCell_Confusion_test, yInferSampled_test = rc.compute_inference_statistics_allSamples( \
			len(Y_test), N, M, Z_test, Zinf_test, np.arange(M), np.arange(M), Y_test, Yinf_test, verbose_EM)
			#
			t1 = time.time()
			print('Done w/ Computing Statistics on Inference process during EM learning on test data. : time = ',t1-t0)



			# Save the rasters of inferred Zs, Ys and all spike words to an npz file
			np.savez( StatsDuringLearning_fname, 
				Ycell_hist_Samps=Ycell_hist_Samps, nY_Samps=nY_Samps, Cell_coactivity_Samps=Cell_coactivity_Samps,
				#
				Ycell_hist_Infer_train=Ycell_hist_Infer_train, Zassem_hist_Infer_train=Zassem_hist_Infer_train,
				nY_Infer_train=nY_Infer_train, nZ_Infer_train=nZ_Infer_train, CA_coactivity_Infer_train=CA_coactivity_Infer_train,
				Cell_coactivity_Infer_train=Cell_coactivity_Infer_train,
				#
				Ycell_hist_Infer_test=Ycell_hist_Infer_test, Zassem_hist_Infer_test=Zassem_hist_Infer_test, 
				nY_Infer_test=nY_Infer_test, nZ_Infer_test=nZ_Infer_test,
				CA_coactivity_Infer_test=CA_coactivity_Infer_test, Cell_coactivity_Infer_test=Cell_coactivity_Infer_test,
				#
				Kinf_train=Kinf_train, KinfDiff_train=KinfDiff_train, zCapture_train=zCapture_train, zMissed_train=zMissed_train,
				zExtra_train=zExtra_train, yCapture_train=yCapture_train, yMissed_train=yMissed_train, yExtra_train=yExtra_train,
				inferCA_Confusion_train=inferCA_Confusion_train, inferCell_Confusion_train=inferCell_Confusion_train,
				zInferSampledRaw_train=zInferSampledRaw_train, zInferSampledT_train=zInferSampledT_train,
				zInferSampled_train=zInferSampled_train, yInferSampled_train=yInferSampled_train,
				#
				Kinf_test=Kinf_test, KinfDiff_test=KinfDiff_test, zCapture_test=zCapture_test, zMissed_test=zMissed_test,
				zExtra_test=zExtra_test, yCapture_test=yCapture_test, yMissed_test=yMissed_test, yExtra_test=yExtra_test,
				inferCA_Confusion_test=inferCA_Confusion_test, inferCell_Confusion_test=inferCell_Confusion_test,
				zInferSampledRaw_test=zInferSampledRaw_test, zInferSampledT_test=zInferSampledT_test,
				zInferSampled_test=zInferSampled_test, yInferSampled_test=yInferSampled_test)
		






	#
	# #
	# # #
	# #
	#
	if flg_compute_StatsPostLrn:

		try:
			print('Loading in data file saved from inference post EM learning.')
			print( StatsPostLearning_fname )
			data = np.load( StatsPostLearning_fname )
			print(data.keys())
			#
			Ycell_hist_allSWs 			= data['Ycell_hist_allSWs']
			YcellInf_hist_allSWs 		= data['YcellInf_hist_allSWs'] 
			Zassem_hist_allSWs 			= data['Zassem_hist_allSWs']
			#
			nY_allSWs 					= data['nY_allSWs']
			nYinf_allSWs 				= data['nYinf_allSWs'] 
			nZ_allSWs 					= data['nZ_allSWs']
			#
			CA_coactivity_allSWs 		= data['CA_coactivity_allSWs']
			Cell_coactivity_allSWs 		= data['Cell_coactivity_allSWs'] 
			CellInf_coactivity_allSWs 	= data['CellInf_coactivity_allSWs'] 
			#
			Kinf_postLrn 				= data['Kinf_postLrn']
			yCapture_postLrn 			= data['yCapture_postLrn']
			yMissed_postLrn 			= data['yMissed_postLrn']
			yExtra_postLrn 				= data['yExtra_postLrn']
			inferCA_Confusion_postLrn 	= data['inferCA_Confusion_postLrn'] 
			inferCell_Confusion_postLrn = data['inferCell_Confusion_postLrn']
			zInferSampledRaw_postLrn 	= data['zInferSampledRaw_postLrn']
			zInferSampled_postLrn 		= data['zInferSampled_postLrn']
			yInferSampled_postLrn 		= data['yInferSampled_postLrn']

		except:	

			#
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
			print('Loading in data file saved from inference post EM learning.')
			print( rasterZs_fname )
			data = np.load( rasterZs_fname )
			print(data.keys())
			#
			Ycell_hist_allSWs 			= data['Ycell_hist_allSWs']
			YcellInf_hist_allSWs 		= data['YcellInf_hist_allSWs'] 
			Zassem_hist_allSWs 			= data['Zassem_hist_allSWs']
			#
			nY_allSWs 					= data['nY_allSWs']
			nYinf_allSWs 				= data['nYinf_allSWs'] 
			nZ_allSWs 					= data['nZ_allSWs']
			#
			CA_coactivity_allSWs 		= data['CA_coactivity_allSWs']
			Cell_coactivity_allSWs 		= data['Cell_coactivity_allSWs'] 
			CellInf_coactivity_allSWs 	= data['CellInf_coactivity_allSWs'] 
			#
			# #
			#
			all_Zs = [ list() for num in range(num_SWsUnWrap) ] # GT Z's not known for real data..
			# Computing Inference statistics for all spike words after learning.
			print('Computing Statistics inferring using all spikewords after Learning.')
			t0 = time.time()
			#
			Kinf_postLrn, KinfDiff_postLrn, zCapture_postLrn, zMissed_postLrn, zExtra_postLrn, \
			yCapture_postLrn, yMissed_postLrn, yExtra_postLrn, inferCA_Confusion_postLrn, zInferSampled_postLrn, zInferSampledRaw_postLrn, \
			zInferSampledT_postLrn, inferCell_Confusion_postLrn, yInferSampled_postLrn = rc.compute_inference_statistics_allSamples( \
			num_SWsUnWrap, N, M, all_Zs, all_Zinf, np.arange(M), np.arange(M), all_SWs, all_Yinf, verbose_EM)
			#
			t1 = time.time()
			print('Done w/ Computing Statistics inferring using all spikewords after Learning : time = ',t1-t0)


			# Save the rasters of inferred Zs, Ys and all spike words to an npz file
			np.savez( StatsPostLearning_fname, 
				Ycell_hist_allSWs=Ycell_hist_allSWs, YcellInf_hist_allSWs=YcellInf_hist_allSWs, Zassem_hist_allSWs=Zassem_hist_allSWs,
				nY_allSWs=nY_allSWs, nYinf_allSWs=nYinf_allSWs, nZ_allSWs=nZ_allSWs, CA_coactivity_allSWs=CA_coactivity_allSWs, 
				Cell_coactivity_allSWs=Cell_coactivity_allSWs, CellInf_coactivity_allSWs=CellInf_coactivity_allSWs,
				#
				Kinf_postLrn=Kinf_postLrn, yCapture_postLrn=yCapture_postLrn, yMissed_postLrn=yMissed_postLrn, yExtra_postLrn=yExtra_postLrn,
				inferCA_Confusion_postLrn=inferCA_Confusion_postLrn, inferCell_Confusion_postLrn=inferCell_Confusion_postLrn,
				zInferSampledRaw_postLrn=zInferSampledRaw_postLrn, zInferSampled_postLrn=zInferSampled_postLrn, yInferSampled_postLrn=yInferSampled_postLrn)






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
	# # Model Construction / synthesis parameters. These determine structure of ground truth model we will generate data from and try to learn.
	#
	# N
	parser.add_argument('-N', '--N', dest='N', type=int, default=20,
		help='Number of neurons / cells in Synthesized model.')
	# M
	parser.add_argument('-M', '--M', dest='M', type=int, default=40,
		help='Number of cell assemblies in Synthesized model.')
	# K
	parser.add_argument('-K', '--K', dest='K', type=int, default=2,
		help='Number of active cell assemblies in a spike word |z-vec|. Actual sample pulled from a bernoulli distribution w/ p(z_a=1) = K/M.')		
	# Kmin
	parser.add_argument('-Kmin', '--Kmin', dest='Kmin', type=int, default=0,
		help='Minimum number of active cell assemblies in a sampled spike word |z-vec|.')
	# Kmax
	parser.add_argument('-Kmax', '--Kmax', dest='Kmax', type=int, default=2,
		help='Maximum number of active cell assemblies in a sampled spike word |z-vec|.')
	# C
	parser.add_argument('-C', '--C', dest='C', type=int, default=2,
		help='Number of active cells in a given cell assembly. Actual sample pulled from a bernoulli distribution w/ p(z_a=1) = C/N.')		
	# Cmin
	parser.add_argument('-Cmin', '--Cmin', dest='Cmin', type=int, default=2,
		help='Minimum number of active cell in a given cell assembly. Resample if not satisfied.')
	# Cmax
	parser.add_argument('-Cmax', '--Cmax', dest='Cmax', type=int, default=6,
		help='Maximum number of active cell in a given cell assembly. Resample if not satisfied.')
	# yLo
	parser.add_argument('-yLo', '--yLo', dest='yLo', type=int, default=1,
		help='If |y| < yLo, ignore the spike word for learning. Minimum number of active cells in a time bin to be considered a spike word and used to train model.')
	# yHi
	parser.add_argument('-yHi', '--yHi', dest='yHi', type=int, default=9999,
		help='If |y| > yHi, z=0 not allowed for inference. Assume there must be at least one CA on.')
	# mu_Pia
	parser.add_argument('-mu_Pia', '--mu_Pia', dest='mu_Pia', type=float, default=0.,
		help='"Mean" (distance from binary 0 or 1 values) of Pia matrix parameters in GT synthesized model used to construct data')	
	# sig_Pia
	parser.add_argument('-sig_Pia', '--sig_Pia', dest='sig_Pia', type=float, default=0.1,
		help='Spread or STD of Pia matrix parameters in GT synthesized model used to construct data')
	# bernoulli_Pi
	parser.add_argument('-bernoulli_Pi', '--bernoulli_Pi', dest='bernoulli_Pi', type=float, default=1.,
		help='Probability of drawing a 0 (very noisy cell) in Pi vector.')
	# mu_Pi
	parser.add_argument('-mu_Pi', '--mu_Pi', dest='mu_Pi', type=float, default=0.,
		help='"Mean" (distance from binary 0 or 1 values) of Pi vector parameters in GT synthesized model used to construct data')		
	# sig_Pi
	parser.add_argument('-sig_Pi', '--sig_Pi', dest='sig_Pi', type=float, default=0.1,
		help='Spread or STD of Pi vector parameters in GT synthesized model used to construct data')	
				

	
	#
	# # How many spike-words to generate, how many samples to take for EM, how many times to randomly sample SW distribution
	#
	# num_EM_samps
	parser.add_argument('-ns', '--num_EM_samps', dest='num_EM_samps', type=int, default=100,
		help='Number of total EM samples , #iterations - It randomly, uniformly samples from all Spikewords. So it can be >/=/< # SWs.')
	# num_EM_rands
	parser.add_argument('-nr', '--num_EM_rands', dest='num_EM_rands', type=int, default=1,
		help='Due to order of spike words sampled, model learned can be different. How many times to run initialzed model on different sampling of spike words.')
	# num_SWs 
	parser.add_argument('-nsw', '--num_SWs', dest='num_SWs', type=int, default=1000,
		help='Number of spike words to generate from model to be split into training and test data.')
	# pct_xVal_train
	parser.add_argument('-pcttrn', '--pct_xVal_train', dest='pct_xVal_train', type=float, default=0.8,
		help='Percent of spike words to generate from model to use as training data. Rest is test for cross validation...')


	#
	# # Learning rate stuff and some options on Inference.
	#
	# learning_rate,
	parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=1.,
		help='Learning rate in model learning at each iteration. ex(1 works well)')  
	# lRateScale_Pi,
	parser.add_argument('-lrPi', '--lRateScale_Pi', dest='lRateScale_Pi', type=float, default=1.,
		help='An Multiplicative factor on Pi learning rate. Thought <1 was good, but now =1 turns this off. That problem was fixed by flg_include_Zeq0_infer.')
	# flg_include_Zeq0_infer
	parser.add_argument('-z0', '--flg_include_Zeq0_infer', dest='flg_include_Zeq0_infer', action='store_true', default=False,
		help=', Flag to include the z=0''s vector in inference if True. ex( True or False))')




	#
	# # Model Initialization parameters 
	#
	# params_init
	parser.add_argument('-pi', '--params_init', dest='params_init', type=str, default='NoisyConst',
		help='How to initialize the model for EM algorithm. ex(NoisyConst True NoisyTrue RandomUniform NoisyConst)')
	# sigQ_init
	parser.add_argument('-si_q', '--sigQ_init', dest='sigQ_init', type=float, default=0.01,
		help='Variability / sigma in Q initializations. Distributed around center at Z_hot/(overcomp*M_mod)') 
	# sigPi_init
	parser.add_argument('-si_ri', '--sigPi_init', dest='sigPi_init', type=float, default=0.05,
		help='Variability / sigma in Pi initializations. Distributed around centers defined by C_noise_ri') 
	# sigPia_init
	parser.add_argument('-si_ria', '--sigPia_init', dest='sigPia_init', type=float, default=0.05,
		help='Variability / sigma in Pia initializations. Distributed around centers defined by C_noise_ria') 
	# overcomp
	parser.add_argument('-oc', '--overcomp', dest='overcomp', type=int, default=1,
		help='How over overcomplete model is vs cells. M_mod = overcomp*M')
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
	# # Flags to plot and for verbose mode.
	#
	# -MSE -genSmp -derivs -lrndEM -infStats -infTemp -trans -rand
	#
	# ds_fctr_snapshots,
	parser.add_argument('-t_snap', '--ds_fctr_snapshots', dest='ds_fctr_snapshots', type=float, default=1,
		help='How often to take snapshot of learning process. Note r_ia to follow learning for MSE in SynthData.')

	# plt_dataGen_andSampling_hist_flg - (Plot 1). Do multiple comparison of stats to see if different data are biased. {Full vs. Sampled vs. Inferred vs. Train vs. Test}
	parser.add_argument('-genSmp', '--plt_dataGen_andSampling_hist_flg', dest='plt_dataGen_andSampling_hist_flg', action='store_true', default=False,
		help='')
	# verbose_EM
	parser.add_argument('-v', '--verbose_EM', dest='verbose_EM', action='store_true', default=False,
		help=', Flag to dislay additional output messages (Sanity Checking). ex( True or False))')	





	


	#
	# # Get args from Parser.
	#
	args = parser.parse_args()
	StatsInfPL_synthData(args) #args.x, args.y)  
