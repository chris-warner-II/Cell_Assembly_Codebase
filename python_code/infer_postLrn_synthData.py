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



def infer_postLrn_SynthData(args):

	print('Running EM algorithm to learn PGM parameters on synthetic data.')

	print(args)

	# Extract variables from args input into function from command line
	argsRec = args
	globals().update(vars(args))

	# Turn comma delimited strings input into lists.
	lRateScale 	= [float(item) for item in args.lRateScale.split(',')]

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# (1).  Construct params that are built from other input params.
	#
	num_EM_samps = int(num_SWs*pct_xVal_train)
	M_mod 		= int(np.round(overcomp*M))
	C_noise 	= np.array([Z_hot/M_mod, C_noise_ri, C_noise_ria ])		# Mean of parameters for 'NoisyConst' initializations. Scalar between 0 and 1. 1 means almost silent. 0 means almost always firing.
	sig_init 	= np.array([sigQ_init, sigPi_init, sigPia_init ])		# STD on parameter initializations from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
	pjt_tol 			= 10
	numPia_snapshots 	= np.round(num_EM_samps/ds_fctr_snapshots).astype(int)
	samps2snapshot_4Pia = (np.hstack([ np.arange( np.round(num_EM_samps/numPia_snapshots), num_EM_samps, 
							np.round(num_EM_samps/numPia_snapshots)  ), num_EM_samps]) - 1).astype(int)
	#
	params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )
	#
	if flg_include_Zeq0_infer:
		z0_tag='_zeq0'
	else:
		z0_tag='_zneq0'
	#
	if train_2nd_model:
		Btag = 'B'
	else:
		Btag = ''
	#
	# Tag file name pointing at rand0 file we are resampling spikewords from. 
	if resample_available_spikewords and rand>0:
		rsTag = str( '_resampR0trn'+ str(pct_xVal_train_prev).replace('.','pt') )
	else:
		rsTag = '_origModnSWs'

	#
	if flg_EgalitarianPrior:	
		priorType = 'EgalQ' 
	else:
		priorType = 'BinomQ'


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# (2).  Directory structures for input data and output figures
	#
	dirHome, dirScratch = dm.set_dir_tree()
	#
	ModelType = str( 'SyntheticData_N' + str(N) + '_M' + str(M) + '_Mmod' + str(M_mod) + '_K' + str(K) + 
				'_' + str(Kmin) + '_' + str(Kmax) + '_C' + str(C) + '_' + str(Cmin)+ '_' + str(Cmax) + 
				'_mPia' + str(mu_Pia) + '_sPia' + str(sig_Pia) + '_bPi' + str(bernoulli_Pi) + '_mPi' + 
				str(mu_Pi) + '_sPi' + str(sig_Pi) + '_' + sample_longSWs_1st + 'Smp1st_' + priorType + '/')

	InitLrnInf = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') +
				'_LRsc' + str(lRateScale) + '/' )
	#
	EMLrn_data_dir = str( dirScratch + 'data/python_data/PGM_analysis/synthData/Models_learned_EM/' + InitLrnInf + ModelType )
	#	
	infer_postLrn_dir = str( dirScratch + 'data/python_data/PGM_analysis/synthData/inference_postLrn/' + InitLrnInf + ModelType )
	if not os.path.exists(infer_postLrn_dir):
		os.makedirs(infer_postLrn_dir)




	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# (3).  Check if all npz EM_model_data.npz and SWs_inferred_postLrn.npz 
	# 		files are already there for A and B models for all rands. 
	#		If so, break without doing anything else.
	#



	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# Construct file names for EM_model_data.npz file and also B file if we've trained 2nd model.
	fname_EMlrn = str( 'EM_model_data_' + str(num_SWs) + 'SWs_trn' + str(pct_xVal_train).replace('.','pt') + '_xTTT' + str(xTimesThruTrain)
		+ '_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) + '_rand' + str(rand) + rsTag + Btag + '.npz' ) 
	
	fname_inferPL = fname_EMlrn.replace('EM_model_data_','SWs_inferred_postLrn_')



	#flg_checkNPZvars = True # CHECK THAT THE LIST BELOW IS CORRECT AND CONTAINS EVERTHING WE ARE SAVING
	# INTO NPZ FILES BEFORE RUNNING THIS FLAG=TRUE. WILL DELETE FILES THAT DO NOT CONTAIN THESE VARIABLES.
	#
	synthKeys = ['pj_inferred_train_postLrn', 'pj_inferred_test_postLrn', 'cond_inferred_train_postLrn', 'cond_inferred_test_postLrn', \
		#
		'Z_inferred_train_postLrn', 'pyiEq1_gvnZ_train_postLrn', 'Kinf_train_postLrn', 'KinfDiff_train_postLrn', \
		'zCapture_train_postLrn', 'zMissed_train_postLrn', 'zExtra_train_postLrn', 'inferCA_Confusion_train_postLrn', \
		'zInferSampledT_train_postLrn', 'zInferSampled_train_postLrn', 'pyi_gvnZ_stats_train', 'pyi_gvnZ_auc_train', \
		'pyi_gvnZ_ROC_train', 'Z_trainM', 'Z_inferredM_train', 'Z_inferredM_train_postLrn', \
		#
		'Z_inferred_test_postLrn', 'pyiEq1_gvnZ_test_postLrn', 'Kinf_test_postLrn', 'KinfDiff_test_postLrn', \
		'zCapture_test_postLrn', 'zMissed_test_postLrn', 'zExtra_test_postLrn', 'inferCA_Confusion_test_postLrn', \
		'zInferSampled_test_postLrn', 'zInferSampledT_test_postLrn', 'pyi_gvnZ_stats_test', 'pyi_gvnZ_auc_test', \
		'pyi_gvnZ_ROC_test', 'Z_testM', 'Z_inferredM_test_postLrn', 'Z_inferredM_test', 'argsRec' ]


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# 
	# Check that npz file contains the all the right, most current keys.
	# If it doesnt, remove the file.
	fExists = False
	#
	if os.path.isfile(fname_inferPL):
		fExists=True
		#
		if flg_checkNPZvars: 
			data = np.load(fname_inferPL)
			if not all (k in data.keys() for k in synthKeys):
				print('Old File. Removing it: ',fname_inferPL)
				os.remove(fname_inferPL)
				fExists=False
			#	
	else:
		fExists = False

	#
	if fExists:
		print('File(s) already exist for this rand', str(rand) ,' and pct_xVal_train', pct_xVal_train ,'. Not replacing em.')
		return
	else:
		print('File(s) do not exist for this rand', str(rand) ,' and pct_xVal_train', pct_xVal_train ,'.')
	

















	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# If InfPL file already exists dont remake it. Else run inference
	# for all spike words post EM learning.
	#
	if os.path.isfile( str(infer_postLrn_dir+fname_inferPL) ):
			print('infPL file already there. Not remaking it.')
			print( str(infer_postLrn_dir+fname_inferPL) )
			return

	else: # Load fname_EMlrn (and fname_EMlrnB) file

		print('Loading: ', str(EMLrn_data_dir+fname_EMlrn) )
		try:
			data = np.load( str(EMLrn_data_dir+fname_EMlrn) )
		except:
			#delete npz file
			os.remove( str(EMLrn_data_dir+fname_EMlrn) )
			print('Learned Model NPZ file corrupt. Removed!')


		print(data.keys())
		#
		ria 					= data['ria']
		ri 	 					= data['ri']
		q 						= data['q']
		#
		riap 					= data['riap']
		rip 					= data['rip']
		qp 						= data['qp']
		#
		Z_inferred_train 		= data['Z_inferred_train'] 
		pyiEq1_gvnZ_train 		= data['pyiEq1_gvnZ_train'] # p(yi=1|z) during learning
		#
		Z_inferred_test 		= data['Z_inferred_test']
		pyiEq1_gvnZ_test 		= data['pyiEq1_gvnZ_test'] # p(yi=1|z) during learning
		#
		Z_train 				= data['Z_train']
		Y_train 				= data['Y_train'] 
		#
		Z_test 					= data['Z_test']
		Y_test 					= data['Y_test'] 
		#
		ind_matchGT2Mod 		= data['ind_matchGT2Mod']
		cosSim_matchGT2Mod 		= data['cosSim_matchGT2Mod']
		#csNM_matchGT2Mod 		= data['csNM_matchGT2Mod']
		lenDif_matchGT2Mod 		= data['lenDif_matchGT2Mod']
		#ldNM_matchGT2Mod 		= data['ldNM_matchGT2Mod']
		cosSimMat_matchGT2Mod 	= data['cosSimMat_matchGT2Mod']
		lenDifMat_matchGT2Mod 	= data['lenDifMat_matchGT2Mod']
		#
		ind_matchMod2GT 		= data['ind_matchMod2GT']
		cosSim_matchMod2GT 		= data['cosSim_matchMod2GT']
		#csNM_matchMod2GT 		= data['csNM_matchMod2GT']
		lenDif_matchMod2GT 		= data['lenDif_matchMod2GT']
		#ldNM_matchMod2GT 		= data['ldNM_matchMod2GT']
		cosSimMat_matchMod2GT 	= data['cosSimMat_matchMod2GT']
		lenDifMat_matchMod2GT 	= data['lenDifMat_matchMod2GT']
		#

		#
		# 'pj_zTrain_Batch', 
		# 'pj_zTru_Batch', 
		# 'pj_zTest_Batch'
		# 'pj_zHyp_train', 
		# 'pj_zHyp_test',
		# #
		# 'cond_zTru_Batch',
		# 'cond_zTrain_Batch', 
		# 'cond_zTest_Batch',
		# 'cond_zHyp_train',
		# 'cond_zHyp_test', 
		# #
		# 'zActivationHistory', 
		# 'num_dpSig_snaps', 







		


		


		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		# Run inference on all Spike Words / Data (Training and Test)
		# using the fixed model after training.
		#
		print('Inferring all data with post-learning model.')
		t0 = time.time()
		
		print('Inferring ',len(Y_train),' training spike words.')
		Z_inferred_train_postLrn, pyiEq1_gvnZ_train_postLrn, pj_inferred_train_postLrn, cond_inferred_train_postLrn \
		= rc.inferZ_allSWs( [Y_train], [np.ones_like(Y_train)], [Z_train], 1, 0, 2, qp, rip, riap, flg_include_Zeq0_infer, yLo, yHi )

		print('Inferring ',len(Y_test),' test spike words.')
		Z_inferred_test_postLrn, pyiEq1_gvnZ_test_postLrn, pj_inferred_test_postLrn, cond_inferred_test_postLrn \
		 = rc.inferZ_allSWs( [Y_test],  [np.ones_like(Y_test)], [Z_test], 1, 0, 2, qp, rip, riap, flg_include_Zeq0_infer, yLo, yHi )
								#
		t1 = time.time()
		print('Done with inferring all data with post-learning model: time = ',t1-t0)




		print( list(ind_matchMod2GT[0]) )
		print( list(ind_matchGT2Mod[1]) )

		if verbose:
			M_sml = np.min( [M, M_mod] )
			print('------------------------------------------------------')
			print('N',N,' M',M,' M_mod',M_mod)
			print('ind_matchMod2GT', ind_matchMod2GT.shape)
			print('unique begin Mod2GT (GT,Mod) = ', len( np.unique(ind_matchMod2GT[0][:M_sml]) ), len( np.unique(ind_matchMod2GT[1][:M_sml]) ) )
			print('unique whole Mod2GT (GT,Mod) = ', len( np.unique(ind_matchMod2GT[0]) ),len( np.unique(ind_matchMod2GT[1]) )  )
			print('')
			print('ind_matchGT2Mod', ind_matchGT2Mod.shape)
			print('unique begin GT2Mod (GT,Mod) = ', len( np.unique(ind_matchGT2Mod[1][:M_sml]) ), len( np.unique(ind_matchGT2Mod[0][:M_sml]) ) )
			print('unique whole GT2Mod (GT,Mod) = ', len( np.unique(ind_matchGT2Mod[1]) ),len( np.unique(ind_matchGT2Mod[0]) )  )
			print('')
			print('Mod')
			print( list(ind_matchMod2GT[0]) )
			print( list(ind_matchGT2Mod[1]) )
			print('')
			print('GT')
			print( list(ind_matchMod2GT[0]) )
			print( list(ind_matchGT2Mod[1]) )



		# MAKE A FUNCTION?
		#
		# Based on completeness of Model, rename (or reindex) CA's so that they match up as best as possible.
		# 		outputs: indGT_match, indMod_match
		# 		inputs : ind_matchMod2GT, ind_matchGT2Mod, M_mod, M
		if M_mod<=M: 	# undercomplete or complete
			indGT_match 	= list(ind_matchMod2GT[0])
			indMod_match 	= list(ind_matchMod2GT[1])
		else: 			# overcomplete model.
			indGT_match 	= list(ind_matchGT2Mod[1])
			indMod_match 	= list(ind_matchGT2Mod[0])




		#Create Z_trainM and Z_testM that match the learned model using the output from the rc.matchModels_AtoB_cosSim function.
		Z_trainM = list()
		Z_inferredM_train = list()
		Z_inferredM_train_postLrn = list()
		Z_inferredM_train_postLrn.append([]) # this weird preallocation is to have a symmetry between synth and real data.
		#
		for sample in range(len(Z_train)):
			Z_trainM.append( set([ indGT_match.index(z) for z in list(Z_train[sample]) ]) )
		#
		for sample in range(len(Z_inferred_train)):
			Z_inferredM_train.append( set([ indMod_match.index(z) for z in list(Z_inferred_train[sample]) ]) )
		#
		for sample in range(len(Z_inferred_train_postLrn[0])):
			Z_inferredM_train_postLrn[0].append( set([ indMod_match.index(z) for z in list(Z_inferred_train_postLrn[0][sample]) ]) )

		#
		Z_testM = list()
		Z_inferredM_test = list()
		Z_inferredM_test_postLrn = list()
		Z_inferredM_test_postLrn.append([]) # this weird preallocation is to have a symmetry between synth and real data.
		#
		for sample in range(len(Z_test)):
			Z_testM.append( set([ indGT_match.index(z) for z in list(Z_test[sample]) ]) )
		#
		for sample in range(len(Z_inferred_test)):	
			Z_inferredM_test.append( set([ indMod_match.index(z) for z in list(Z_inferred_test[sample]) ]) )	
		#
		for sample in range(len(Z_inferred_test_postLrn[0])):
			Z_inferredM_test_postLrn[0].append( set([ indMod_match.index(z) for z in list(Z_inferred_test_postLrn[0][sample]) ]) )
			





		if verbose:
			print('---------------------------')
			print('')
			print('Z_train: ')
			for i in range(len(Z_trainM)):
				print(i, Z_trainM[i],' --> ', Z_inferredM_train_postLrn[0][i],' :: ',Z_train[i],' -/> ', Z_inferred_train_postLrn[0][i])
				print('')
			print('---------------------------')
			print('---------------------------')
			print('')
			print('Z_test: ')
			for i in range(len(Z_testM)):
				print(i, Z_testM[i],' --> ', Z_inferredM_test_postLrn[0][i],' :: ',Z_test[i],' -/> ', Z_inferred_test_postLrn[0][i])
				print('')
			



			Piap = rc.sig(riap)
			Pia = rc.sig(ria)
			#
			ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat = rc.matchModels_AtoB_cosSim(A=1-Piap, B=1-Pia)
			#
			print('Exactly the same?  ', not np.any( ind[0]-ind_matchMod2GT[0] ),  not np.any( ind[1]-ind_matchMod2GT[1] ) )
			#
			pf.visualize_matchModels_cosSim( A=1-Piap, Atag='Mod', B=1-Pia, Btag='GT', ind=ind, cos_sim=cosSim, len_dif=lenDif, \
							cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir='./', fname_tag=str(rand) )




		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		# Compute p(yi=1|z) and p(yi=0|z) on inferred z-vector and true z-vector. Also, compute the area under the ROC curve.
		#
		print('Compute p(yi={0,1}|z) stats and ROC curve measures. ')
		t0 = time.time()
		
		ROC_THs = np.linspace(0,1,11) 
		#
		pyi_gvnZ_stats_train, pyi_gvnZ_auc_train, pyi_gvnZ_ROC_train = rc.pyi_gvnZ_performance_allSamps( \
			pyiEq1_gvnZ_train_postLrn[0], Y_train, ROC_THs )
		#
		pyi_gvnZ_stats_test, pyi_gvnZ_auc_test, pyi_gvnZ_ROC_test = rc.pyi_gvnZ_performance_allSamps( \
			pyiEq1_gvnZ_test_postLrn[0], Y_test, ROC_THs )

		t1 = time.time()
		print('Done computing p(yi={0,1}|z) stats and ROC curve measures: time = ',t1-t0)


		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		# Compute inference statistics on all inferred Z's and Y's post EM learning
		#
		print('Computing inference statistics on all data with post-learning model.')
		t0 = time.time()
		# COMPUTE INFERENCE STATISTICS AND CONFUSION MATRICES WITH CORRECT / PERMUTED CELL ASSEMBLIES





		Kinf_train_postLrn, KinfDiff_train_postLrn, zCapture_train_postLrn, zMissed_train_postLrn, zExtra_train_postLrn, \
		inferCA_Confusion_train_postLrn, zInferSampled_train_postLrn, zInferSampledT_train_postLrn, \
		= rc.compute_infZstats_allSamples( Z_trainM, Z_inferredM_train_postLrn[0], M, M_mod, verbose) 
		# ind_matchMod2GT, ind_matchGT2Mod, # translate_Lrn2Tru, translate_Tru2Lrn,



		Kinf_test_postLrn, KinfDiff_test_postLrn, zCapture_test_postLrn, zMissed_test_postLrn, zExtra_test_postLrn, \
		inferCA_Confusion_test_postLrn, zInferSampled_test_postLrn, zInferSampledT_test_postLrn, \
		= rc.compute_infZstats_allSamples( Z_testM, Z_inferredM_test_postLrn[0], M, M_mod, verbose) 
		# ind_matchMod2GT, ind_matchGT2Mod, # translate_Lrn2Tru, translate_Tru2Lrn,
		#
		t1 = time.time()
		print('Done computing inference statistics on all data with post-learning model.: time = ',t1-t0)



		# Save the learned model to an npz file
		print('Saving npz file: ',str(infer_postLrn_dir+fname_inferPL) )
		np.savez( str(infer_postLrn_dir+fname_inferPL), pj_inferred_train_postLrn=pj_inferred_train_postLrn, pj_inferred_test_postLrn=pj_inferred_test_postLrn, \
			cond_inferred_train_postLrn=cond_inferred_train_postLrn, cond_inferred_test_postLrn=cond_inferred_test_postLrn, \
			#
			Z_inferred_train_postLrn=Z_inferred_train_postLrn, pyiEq1_gvnZ_train_postLrn=pyiEq1_gvnZ_train_postLrn, Kinf_train_postLrn=Kinf_train_postLrn, \
			KinfDiff_train_postLrn=KinfDiff_train_postLrn, zCapture_train_postLrn=zCapture_train_postLrn, zMissed_train_postLrn=zMissed_train_postLrn, \
			zExtra_train_postLrn=zExtra_train_postLrn, inferCA_Confusion_train_postLrn=inferCA_Confusion_train_postLrn, zInferSampledT_train_postLrn=zInferSampledT_train_postLrn, \
			zInferSampled_train_postLrn=zInferSampled_train_postLrn, pyi_gvnZ_stats_train=pyi_gvnZ_stats_train, pyi_gvnZ_auc_train=pyi_gvnZ_auc_train, \
			pyi_gvnZ_ROC_train=pyi_gvnZ_ROC_train, Z_trainM=Z_trainM, Z_inferredM_train=Z_inferredM_train, Z_inferredM_train_postLrn=Z_inferredM_train_postLrn, \
			#
			Z_inferred_test_postLrn=Z_inferred_test_postLrn, pyiEq1_gvnZ_test_postLrn=pyiEq1_gvnZ_test_postLrn, Kinf_test_postLrn=Kinf_test_postLrn, \
			KinfDiff_test_postLrn=KinfDiff_test_postLrn, zCapture_test_postLrn=zCapture_test_postLrn, zMissed_test_postLrn=zMissed_test_postLrn, \
			zExtra_test_postLrn=zExtra_test_postLrn, inferCA_Confusion_test_postLrn=inferCA_Confusion_test_postLrn, zInferSampled_test_postLrn=zInferSampled_test_postLrn,  \
			zInferSampledT_test_postLrn=zInferSampledT_test_postLrn, pyi_gvnZ_stats_test=pyi_gvnZ_stats_test, pyi_gvnZ_auc_test=pyi_gvnZ_auc_test, \
			pyi_gvnZ_ROC_test=pyi_gvnZ_ROC_test, Z_testM=Z_testM, Z_inferredM_test_postLrn=Z_inferredM_test_postLrn, Z_inferredM_test=Z_inferredM_test, argsRec=argsRec )



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == "__main__":


	#
	# # Set up Parser
	#
	print('Trying to run as a command line function call.')
	parser = argparse.ArgumentParser(description="run infer_postLrn_SynthData from command line")


	#
	# # Model Construction / synthesis parameters. These determine structure of ground truth model we will generate data from and try to learn.
	#
	# N
	parser.add_argument('-N', '--N', dest='N', type=int, default=20,
		help='Number of neurons / cells in Synthesized model.')
	# M
	parser.add_argument('-M', '--M', dest='M', type=int, default=20,
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
	# yMinSW
	parser.add_argument('-yMinSW', '--yMinSW', dest='yMinSW', type=int, default=1,
		help='If |y| < yMinSW, throw it away and dont use it for training model.')
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
	# num_test_samps_4xVal
	parser.add_argument('-nXs', '--num_test_samps_4xVal', dest='num_test_samps_4xVal', type=int, default=1,
		help='Number of test data samples to compute pjoint on and average over to implement cross validation.')# num_EM_rands
	# rand
	parser.add_argument('-rnd', '--rand', dest='rand', type=int, default=0,
		help='Due to order of spike words sampled, model learned can be different. Model can be learned with a different random sampling of spike words.')
	# num_SWs 
	parser.add_argument('-nsw', '--num_SWs', dest='num_SWs', type=int, default=1000,
		help='Number of spike words to generate from model to be split into training and test data.')
	# pct_xVal_train
	parser.add_argument('-pcttrn', '--pct_xVal_train', dest='pct_xVal_train', type=float, default=0.8,
		help='Percent of spike words to generate from model to use as training data. Rest is test for cross validation...')
	# pct_xVal_train_prev
	parser.add_argument('-pctTrnPrv', '--pct_xVal_train_prev', dest='pct_xVal_train_prev', type=float, default=0.5,
	help='If resampling spikewords with different rands, this is value for pct_xVal_train on rand0 file to grab.')
	# xTimesThruTrain
	parser.add_argument('-xTTT', '--xTimesThruTrain', dest='xTimesThruTrain', type=int, default=1,
	help='Number of times to go through training data set during EM algorithm.')


	#
	# # Learning rate stuff and some options on Inference.
	#
	# learning_rate,
	parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=1.,
		help='Learning rate in model learning at each iteration. ex(1 works well)')  
	# lRateScale
	parser.add_argument('-LRS', '--lRateScale', dest='lRateScale', type=str, default='1.0,0.1,0.1',
		help='Comma delimited string to be converted into a list of learning rate scales for Pia, Pi and Q.')
	# flg_include_Zeq0_infer
	parser.add_argument('-z0', '--flg_include_Zeq0_infer', dest='flg_include_Zeq0_infer', action='store_true', default=False,
		help=', Flag to include the z=0''s vector in inference if True. ex( True or False))')
	# sample_longSWs_1st
	parser.add_argument('-smp1st', '--sample_longSWs_1st', dest='sample_longSWs_1st', type=str, default='Dont',
		help="Sample Spikewords for learning randomly, with probability proportional to their length, or sorted by there SW length. Options are: {'Dont', 'Prob', 'Hard'} ")
	# flg_EgalitarianPrior
	parser.add_argument('-Egal', '--flg_EgalitarianPrior', dest='flg_EgalitarianPrior', action='store_true', default=False,
		help='Flag to use Egalitarian activity based prior on z-vec by Q-vec. Otherwise use Binomial Q-scalar prior.. ex( True or False))')	
	



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
	parser.add_argument('-oc', '--overcomp', dest='overcomp', type=float, default=1.,
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
	# train_2nd_model
	parser.add_argument('-2nd', '--train_2nd_model', dest='train_2nd_model', action='store_true', default=False,
		help='Flag to train a 2nd model with 50% test data so we can compare CAs found by each. ex( True or False))')	
	# resample_available_spikewords
	parser.add_argument('-resampSW', '--resample_available_spikewords', dest='resample_available_spikewords', action='store_true', default=False,
		help='Flag that says to load in synthesized model and generated spikewords from rand0 file - from a previous run.')	
	# flg_checkNPZvars
	parser.add_argument('-chkNPZ', '--flg_checkNPZvars', dest='flg_checkNPZvars', action='store_true', default=False,
		help='Flag to check keys / variables in the NPZ file and get rid of the file if they dont match expected values.')
	# verbose
	parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
		help=', Flag to dislay additional output messages (Sanity Checking). ex( True or False))')	




	#
	# # Get args from Parser.
	#
	args = parser.parse_args()
	infer_postLrn_SynthData(args)  

