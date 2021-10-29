import argparse
import numpy as np
import scipy as sp
from scipy import io as io
import matplotlib.pyplot as plt
import time 						# to time operations for code analysis
import os.path 						# to check if a file or directory exists already
import os
import sys

import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.plot_functions as pf
import utils.retina_computation as rc



def pgmCA_synthData(args):

	print('Running EM algorithm to learn PGM on synthetic data.')



	print(args)

	# Extract variables from args input into function from command line
	argsRec = args
	globals().update(vars(args))

	print('Random Sampling of Spike Words #',rand)

	# Turn comma delimited strings input into lists.
	lRateScale 	= [float(item) for item in args.lRateScale.split(',')]

	# result = None
	# while result is None: # errors out sometimes because of nans
		#try:
		#if True:

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # MODEL SYNTHESIS # # # # # # # # # # # #
	# # # # # # # # # # # # # # - & - # # # # # # # # # # # # # # #
	# # # # # # # # # # # # DATA GENERATION # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# (0). Parameters for Data Generation of Spike Words to train 
	#	   Probabalistic Generative Model for Cell Assemblies.


	num_EM_samps = int(num_SWs*pct_xVal_train*xTimesThruTrain)

	M_mod 		= int(np.round(overcomp*M))
	C_noise 	= np.array([Z_hot/M_mod, C_noise_ri, C_noise_ria ])		# Mean of parameters for 'NoisyConst' initializations. Scalar between 0 and 1. 1 means almost silent. 0 means almost always firing.
	sig_init 	= np.array([sigQ_init, sigPi_init, sigPia_init ])		# STD on parameter initializations from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
	pjt_tol 			= 10
	numPia_snapshots 	= np.ceil(num_EM_samps/ds_fctr_snapshots).astype(int)
	

	samps2snapshot_4Pia = (np.hstack([1, np.arange( np.round(num_EM_samps/numPia_snapshots), num_EM_samps, 
							np.round(num_EM_samps/numPia_snapshots)  ), num_EM_samps]) - 1).astype(int)
	#numPia_snapshots 	+=1

	if flg_include_Zeq0_infer:
		z0_tag='_zeq0'
	else:
		z0_tag='_zneq0'

	if flg_EgalitarianPrior:	
		priorType = 'EgalQ' 
	else:
		priorType = 'BinomQ'

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# (1).  Directory structure for output figures
	#
	ModelType = str( 'SyntheticData_N' + str(N) + '_M' + str(M) + '_Mmod' + str(M_mod) + '_K' + str(K) + 
				'_' + str(Kmin) + '_' + str(Kmax) + '_C' + str(C) + '_' + str(Cmin)+ '_' + str(Cmax) + 
				'_mPia' + str(mu_Pia) + '_sPia' + str(sig_Pia) + '_bPi' + str(bernoulli_Pi) + '_mPi' + 
				str(mu_Pi) + '_sPi' + str(sig_Pi) + '_' + sample_longSWs_1st + 'Smp1st_' + priorType + '/')


	#
	params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )
	#
	InitLrnInf = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') +
				'_LRsc' + str(lRateScale) + '/' )
	#
	dirHome, dirScratch = dm.set_dir_tree()
	#
	EM_figs_dir = str( dirScratch + 'figs/PGM_analysis/EM_Algorithm/' + InitLrnInf + ModelType )
	if not os.path.exists(EM_figs_dir):
		os.makedirs(EM_figs_dir)
	#
	EM_data_dir = str( dirScratch + 'data/python_data/PGM_analysis/synthData/Models_learned_EM/' + InitLrnInf + ModelType )
	if not os.path.exists(EM_data_dir):
		os.makedirs(EM_data_dir)




	#flg_checkNPZvars = True # CHECK THAT THE LIST BELOW IS COORECT AND CONTAINS EVERTHING WE ARE SAVING
	# INTO NPZ FILES BEFORE RUNNING THIS FLAG=TRUE. WILL DELETE FILES THAT DO NOT CONTAIN THESE VARIABLES.
	#
	synthKeys = ['q', 'ri', 'ria', 'ria_mod', 'qp', 'rip', 'riap', 'q_init', 'ri_init', 'ria_init', \
		#
		'Q_SE', 'Pi_SE', 'Pi_AE', 'Pia_AE', 'PiaOn_SE', 'PiaOn_AE', 'PiaOff_SE', 'PiaOff_AE', 'num_dpSig_snaps', \
		'q_deriv', 'ri_deriv', 'ria_deriv', 'ria_snapshots', 'ri_snapshots', 'q_snapshots', 'ds_fctr_snapshots', \
		#
		'Z_inferred_train', 'pyiEq1_gvnZ_train', 'Y_train', 'Z_train', \
		'Z_inferred_test', 'pyiEq1_gvnZ_test', 'Y_test', 'Z_test', \
		#
		'pj_zHyp_train', 'pj_zHyp_test', 'pj_zTru_Batch', 'pj_zTrain_Batch', 'pj_zTest_Batch', \
		'cond_zHyp_train', 'cond_zHyp_test', 'cond_zTru_Batch', 'cond_zTrain_Batch', 'cond_zTest_Batch', \
		#
		'ind_matchGT2Mod', 'cosSim_matchGT2Mod', 'lenDif_matchGT2Mod', 'cosSimMat_matchGT2Mod', 'lenDifMat_matchGT2Mod', \
		'csNM_matchGT2Mod', 'ldNM_matchGT2Mod', 'HungRowCol_matchGT2Mod','cos_simHM_matchGT2Mod', \
		'ind_matchMod2GT', 'cosSim_matchMod2GT', 'lenDif_matchMod2GT', 'cosSimMat_matchMod2GT', 'lenDifMat_matchMod2GT', \
		'csNM_matchMod2GT', 'ldNM_matchMod2GT', 'HungRowCol_matchMod2GT', 'cos_simHM_matchMod2GT', \
		#
		'xVal_snapshot', 'xVal_batchSize', 'zActivationHistory', 'argsRec']





	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# (2).  Check if npz datafiles are already there for A and 
	# 		B models for rand0 if we want to resample_available_spikewords.
	# 		Otherwise, synthesize (randomly) new GT model and generate (randomly) spike words data from it.
	#
	# 			PARAMS:
	#			------
	# 			resample_available_spikewords = True
	# 			pct_xVal_train_prev = 0.5
	#
	t0 = time.time()
	print('Generating Data and Sampling for Cross Validation.')



	#
	# #
	#
	# If both (if training_2nd_model) or one file(s) for rand0 already there, then load in model and spike words from that.
	fname_EMlrn_r0 = str( EM_data_dir + 'EM_model_data_' + str(num_SWs) + 'SWs_trn' + str(pct_xVal_train_prev).replace('.','pt') + \
		'_xTTT' + str(xTimesThruTrain) + '_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) + '_rand0_origModnSWs.npz' )
	fname_EMlrnB_r0 = str(fname_EMlrn_r0[:-4] + 'B.npz')
	#





	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# 
	# Check that npz file contains the all the right, most current keys.
	# If it doesnt, remove the file.
	r0_exists = False
	#
	if os.path.isfile(fname_EMlrn_r0):
		fExists=True
		#
		if flg_checkNPZvars: 
			data = np.load(fname_EMlrn_r0)
			if not all (k in data.keys() for k in synthKeys):
				print('Old File. Removing it: ',fname_EMlrn_r0)
				os.remove(fname_EMlrn_r0)
				r0_exists=False
	#
	# #
	#
	if train_2nd_model and r0_exists and os.path.isfile(fname_EMlrnB_r0):
		r0_exists=True
		#
		if flg_checkNPZvars: 
			data = np.load(fname_EMlrnB_r0)
			if not all (k in data.keys() for k in synthKeys):
				print('Old File. Removing it: ',fname_EMlrnB_r0)
				os.remove(fname_EMlrnB_r0)
				r0_exists=False
	else:
		r0_exists = False












	#
	# #
	#
	# Tag file name pointing at rand0 file we are resampling spikewords from. 
	if resample_available_spikewords and r0_exists:
		rsTag = str( '_resampR0trn'+ str(pct_xVal_train_prev).replace('.','pt') )
		if rand==0:
			print('Not tryna remake rand0 file when resampling it, meh. Breaking out.')
			return
	else:
		rsTag = '_origModnSWs'




	#
	# #
	#
	# If both (if training_2nd_model) or one file(s) already there, then dont run for this rand value.
	fname_EMlrn = str( EM_data_dir + 'EM_model_data_' + str(num_SWs) + 'SWs' + '_trn' + str(pct_xVal_train).replace('.','pt') + '_xTTT' \
		 + str(xTimesThruTrain)  + '_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) + '_rand' + str(rand) + rsTag + '.npz' )
	fname_EMlrnB = str(fname_EMlrn[:-4] + 'B.npz')
	#


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
			if not all (k in data.keys() for k in synthKeys):
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
			if not all (k in data.keys() for k in synthKeys):
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
	





	#
	if resample_available_spikewords and r0_exists:
		#
		print('Rand0 file exists. Reusing model params and spike words from it.')
		data = np.load( fname_EMlrn_r0 )
		#print( data.keys() )
		#
		ria 		= data['ria']
		ri			= data['ri']
		q 			= data['q']
		try:
			ria_mod = data['ria_mod']
		except:
			ria_mod = ria_r0
		#
		print('Ztest len= ',)

		Z_all = np.concatenate( (data['Z_train'], data['Z_test']) )
		Y_all = np.concatenate( (data['Y_train'], data['Y_test']) )

		print('num Z_all =', Z_all.shape)
		print('num Y_all =', Y_all.shape)
	else:
		#
		# (3). Synthesize model from user input parameters. and
		# (4). Generate Data Vectors Y & Z using Pia & Pi and some sampling assumptions.
		# 		(a). Sample Z vector. 
		# 		(b). Construct Y vector from Z and model (Pia, Pi).
		#
		print('Rand0 file doesnt exist. Synthesize model and generate spike words.')
		if rand==0:
			q = None #ri = None #ria = None #ria_mod = None
			while q==None: # while statement to resynthesize model if any cells participate in 0 assemblies.
				q, ri, ria, ria_mod = rc.synthesize_model(N,M,M_mod, C,K, Cmin,Cmax, bernoulli_Pi,mu_Pi,sig_Pi, mu_Pia,sig_Pia, verbose=False)
			#
			Y_all, Z_all = rc.generate_data(num_SWs, q, ri, ria, Cmin, Cmax, Kmin, Kmax)
		else:
			print('I dont want to generate rand files > 0 if the rand0 file does not exist yet.')
			print('Because rand essentially means that we are using the same model and resampling test and train sets from same bag of spike words.')
			return


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# (5). Initialize learned model in the non-probability q, ri, ria space and Store the parameter init values so I can compare to them later.
	# 		For NoisyTrue case set sigma_std in probability space and pipe it thru the logistic function sig(sigma_std)
	#		For P = {0,1}, set r = {-b,+b} where b is something reasonably big (but not machine precision). 
	#		Nice Bayesian interpretation that we can never be absolutely certain of cell being active or silent.
	#

	# when dealing with real data we dont know the model parameters so we set them to these uniform, reasonable values.
	# When dealing with synthetic model and generated data, these will have been defined above.
	q_init, ri_init, ria_init, params_init_param = rc.init_parameters(q, ri, ria_mod, params_init, sig_init, C_noise)




	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# (6). RUN FULL EXPECTATION MAXIMIZATION ALGORITHM.
	#		(A). Infer Z-vector from observed Y-vector and current model parameters.
	#		(B). Learn model parameters (via gradient descent) given Y-vector and Z-vector.
	#
	#


	# Randomly sample a set of indices of all spike words that pass yLo threshold.
	# Then split these into train and test sets for Cross Validation.
	indSWs_gt_yLo_train, indSWs_gt_yLo_test = rc.construct_crossValidation_SWs( Y_all, yMinSW, pct_xVal_train )
	num_SWs_xVal = [ len(indSWs_gt_yLo_train), len(indSWs_gt_yLo_test) ]
	#
	Y_train = list( np.array(Y_all)[indSWs_gt_yLo_train] )
	Y_test 	= list( np.array(Y_all)[indSWs_gt_yLo_test] )
	Z_train = list( np.array(Z_all)[indSWs_gt_yLo_train] )
	Z_test 	= list( np.array(Z_all)[indSWs_gt_yLo_test] )
	t1 = time.time()
	print('Done Generating Data and Sampling for Cross Validation: time = ',t1-t0)


	# # NOW RESHUFFLE TEST DATA. SORT TRAIN DATA BY SPIKEWORD LENGTH
	YrndSrt 	= np.random.choice(len(Y_test), len(Y_test), replace=False)
	Y_test_rnd	= [Y_test[s] for s in YrndSrt ]
	Z_test_rnd	= [Z_test[s] for s in YrndSrt ]

	# Sample spike words so that longer ones are sampled earlier. Or dont.
	YlenSrt 	= rc.sampleSWs_by_length( Y_train, sample_longSWs_1st )
	Y_train_srt = [Y_train[s] for s in YlenSrt ]
	Z_train_srt = [Z_train[s] for s in YlenSrt ]




	# Run full EM algorithm randomly sampling from train & test data. 
	# Note: For each 'rand', the train & test samples are also redrawn from the whole data distribution.
	qp = q_init
	rip = ri_init
	riap = ria_init
	#
	xTimesThruTest = xTimesThruTrain*np.round( pct_xVal_train/(1-pct_xVal_train) ).astype(int)
	#


	qp, rip, riap, Z_inferred_train, Z_inferred_test, pyiEq1_gvnZ_train, pyiEq1_gvnZ_test, \
	ria_snapshots, ri_snapshots, q_snapshots, q_deriv, ri_deriv, ria_deriv, Q_SE, Pi_SE, Pi_AE, \
	pj_zHyp_train, pj_zHyp_test, pj_zTru_Batch, pj_zTrain_Batch, pj_zTest_Batch, \
	cond_zHyp_train, cond_zHyp_test, cond_zTru_Batch, cond_zTrain_Batch, cond_zTest_Batch, zActivationHistory \
	= rc.run_EM_algorithm( qp, rip, riap, q, ri, ria_mod, q_init, ri_init, ria_init, \
		Z_train_srt*xTimesThruTrain, Z_test_rnd*xTimesThruTest, Y_train_srt*xTimesThruTrain, Y_test_rnd*xTimesThruTest, \
		xVal_snapshot, xVal_batchSize, samps2snapshot_4Pia, pjt_tol, learning_rate, lRateScale, \
		flg_include_Zeq0_infer, yLo, yHi, flg_EgalitarianPrior, sample_longSWs_1st, verbose_EM) 


	# Permute Piap so that it best matches Pia. But translate should have each entry only 1 time.
	Piap=rc.sig(riap)
	Pia=rc.sig(ria)


	# Find best GT CAs to match each Model CA.
	ind_matchGT2Mod, cosSim_matchGT2Mod, csNM_matchGT2Mod, lenDif_matchGT2Mod, ldNM_matchGT2Mod, \
	cosSimMat_matchGT2Mod, lenDifMat_matchGT2Mod, HungRowCol_matchGT2Mod, cos_simHM_matchGT2Mod \
		= rc.matchModels_AtoB_cosSim(A=1-Pia, B=1-Piap)

	# Find best Model CAs to match each GT CA.
	ind_matchMod2GT, cosSim_matchMod2GT, csNM_matchMod2GT, lenDif_matchMod2GT, ldNM_matchMod2GT, \
	cosSimMat_matchMod2GT, lenDifMat_matchMod2GT, HungRowCol_matchMod2GT, cos_simHM_matchMod2GT \
		= rc.matchModels_AtoB_cosSim(A=1-Piap, B=1-Pia)




	print('ind_matchGT2Mod',ind_matchGT2Mod) # ind_matchGT2Mod.shape,
	print('')
	print('HungRowCol_matchMod2GT',HungRowCol_matchMod2GT) # HungRowCol_matchMod2GT.shape, 
	print('')
	print('cosSim_matchGT2Mod',cosSim_matchGT2Mod) # cosSim_matchGT2Mod.shape,
	print('')
	print('cos_simHM_matchMod2GT',cos_simHM_matchMod2GT) # cos_simHM_matchMod2GT.shape, 
	print('')




	# Compute Absolute Value and Signed Errors of Pia using translations.
	Pia_AE, PiaOn_SE, PiaOn_AE, PiaOff_SE, PiaOff_AE, num_dpSig_snaps = rc.compute_errs_Pia_snapshots( \
							ria_snapshots, ria, ind_matchGT2Mod, cosSim_matchGT2Mod, lenDif_matchGT2Mod ) 
		


	# Save the learned model to an npz file
	print('Saving npz file: ',fname_EMlrn)
	np.savez( fname_EMlrn, q=q, ri=ri, ria=ria, ria_mod=ria_mod, qp=qp, rip=rip, riap=riap, q_init=q_init, ri_init=ri_init, ria_init=ria_init, \
		#
		Q_SE=Q_SE, Pi_SE=Pi_SE, Pi_AE=Pi_AE, Pia_AE=Pia_AE, PiaOn_SE=PiaOn_SE, PiaOn_AE=PiaOn_AE, PiaOff_SE=PiaOff_SE, PiaOff_AE=PiaOff_AE, \
		num_dpSig_snaps=num_dpSig_snaps, q_deriv=q_deriv, ri_deriv=ri_deriv, ria_deriv=ria_deriv, \
		ria_snapshots=ria_snapshots, ri_snapshots=ri_snapshots, q_snapshots=q_snapshots, ds_fctr_snapshots=ds_fctr_snapshots,  \
		#
		Z_inferred_train=Z_inferred_train, pyiEq1_gvnZ_train=pyiEq1_gvnZ_train, Y_train=Y_train_srt, Z_train=Z_train_srt, \
		Z_inferred_test=Z_inferred_test, pyiEq1_gvnZ_test=pyiEq1_gvnZ_test, Y_test=Y_test_rnd, Z_test=Z_test_rnd, \
		#
		pj_zHyp_train=pj_zHyp_train, pj_zHyp_test=pj_zHyp_test, pj_zTru_Batch=pj_zTru_Batch, pj_zTrain_Batch=pj_zTrain_Batch, \
		pj_zTest_Batch=pj_zTest_Batch, cond_zHyp_train=cond_zHyp_train, cond_zHyp_test=cond_zHyp_test, cond_zTru_Batch=cond_zTru_Batch, \
		cond_zTrain_Batch=cond_zTrain_Batch, cond_zTest_Batch=cond_zTest_Batch, \
		#
		ind_matchGT2Mod=ind_matchGT2Mod, cosSim_matchGT2Mod=cosSim_matchGT2Mod, csNM_matchGT2Mod=csNM_matchGT2Mod, \
		lenDif_matchGT2Mod=lenDif_matchGT2Mod, ldNM_matchGT2Mod=ldNM_matchGT2Mod,\
		cosSimMat_matchGT2Mod=cosSimMat_matchGT2Mod, lenDifMat_matchGT2Mod=lenDifMat_matchGT2Mod, \
		HungRowCol_matchGT2Mod=HungRowCol_matchGT2Mod, cos_simHM_matchGT2Mod=cos_simHM_matchGT2Mod, \
		#
		ind_matchMod2GT=ind_matchMod2GT, cosSim_matchMod2GT=cosSim_matchMod2GT, csNM_matchMod2GT=csNM_matchMod2GT, \
		lenDif_matchMod2GT=lenDif_matchMod2GT, ldNM_matchMod2GT=ldNM_matchMod2GT, \
		cosSimMat_matchMod2GT=cosSimMat_matchMod2GT, lenDifMat_matchMod2GT=lenDifMat_matchMod2GT, \
		HungRowCol_matchMod2GT=HungRowCol_matchMod2GT, cos_simHM_matchMod2GT=cos_simHM_matchMod2GT, \
		#
		xVal_snapshot=xVal_snapshot, xVal_batchSize=xVal_batchSize, zActivationHistory=zActivationHistory, argsRec=argsRec )

		



	if train_2nd_model:
		# Train a second model ('B') swapping test for train and train for test.
		#
		qp = q_init
		rip = ri_init
		riap = ria_init

		# # NOW RESHUFFLE TRAINING DATA. SORT TEST DATA BY SPIKEWORD LENGTH
		YrndSrt 	= np.random.choice(len(Y_train), len(Y_train), replace=False)
		Y_train_rnd	= [Y_train[s] for s in YrndSrt ]
		Z_train_rnd	= [Z_train[s] for s in YrndSrt ]

		# Sample spike words so that longer ones are sampled earlier. Or dont.
		YlenSrt 	= rc.sampleSWs_by_length( Y_test, sample_longSWs_1st )
		Y_test_srt 	= [Y_test[s] for s in YlenSrt ]
		Z_test_srt 	= [Z_test[s] for s in YlenSrt ]



		#
		qp, rip, riap, Z_inferred_train, Z_inferred_test, pyiEq1_gvnZ_train, pyiEq1_gvnZ_test, \
		ria_snapshots, ri_snapshots, q_snapshots, q_deriv, ri_deriv, ria_deriv, Q_SE, Pi_SE, Pi_AE, \
		pj_zHyp_train, pj_zHyp_test, pj_zTru_Batch, pj_zTrain_Batch, pj_zTest_Batch, \
		cond_zHyp_train, cond_zHyp_test, cond_zTru_Batch, cond_zTrain_Batch, cond_zTest_Batch, zActivationHistory \
		= rc.run_EM_algorithm( qp, rip, riap, q, ri, ria_mod, q_init, ri_init, ria_init, \
			Z_test*xTimesThruTrain, Z_train*xTimesThruTest, Y_test*xTimesThruTrain, Y_train*xTimesThruTest, \
			xVal_snapshot, xVal_batchSize, samps2snapshot_4Pia, pjt_tol, learning_rate, lRateScale, \
			flg_include_Zeq0_infer, yLo, yHi, flg_EgalitarianPrior, sample_longSWs_1st, verbose_EM) # Pia_AE, PiaOn_SE, PiaOn_AE, PiaOff_SE, PiaOff_AE,


		# Permute Piap so that it best matches Pia. But translate should have each entry only 1 time.
		Piap=rc.sig(riap)
		Pia=rc.sig(ria)


		# Find best GT CAs to match each Model CA.
		ind_matchGT2Mod, cosSim_matchGT2Mod, csNM_matchGT2Mod, lenDif_matchGT2Mod, ldNM_matchGT2Mod, \
		cosSimMat_matchGT2Mod, lenDifMat_matchGT2Mod, HungRowCol_matchGT2Mod, cos_simHM_matchGT2Mod \
			= rc.matchModels_AtoB_cosSim(A=1-Pia, B=1-Piap)

		# Find best Model CAs to match each GT CA.
		ind_matchMod2GT, cosSim_matchMod2GT, csNM_matchMod2GT, lenDif_matchMod2GT, ldNM_matchMod2GT, \
		cosSimMat_matchMod2GT, lenDifMat_matchMod2GT, HungRowCol_matchMod2GT, cos_simHM_matchMod2GT \
			= rc.matchModels_AtoB_cosSim(A=1-Piap, B=1-Pia)



		# Compute Absolute Value and Signed Errors of Pia using translations.
		Pia_AE, PiaOn_SE, PiaOn_AE, PiaOff_SE, PiaOff_AE, num_dpSig_snaps = rc.compute_errs_Pia_snapshots( \
			ria_snapshots, ria, ind_matchGT2Mod, cosSim_matchGT2Mod, lenDif_matchGT2Mod ) 


		# Save the learned model to an npz file
		print('Saving npz file: ',fname_EMlrnB)
		np.savez( fname_EMlrnB, q=q, ri=ri, ria=ria, ria_mod=ria_mod, qp=qp, rip=rip, riap=riap, q_init=q_init, ri_init=ri_init, ria_init=ria_init, \
			#
			Q_SE=Q_SE, Pi_SE=Pi_SE, Pi_AE=Pi_AE, Pia_AE=Pia_AE, PiaOn_SE=PiaOn_SE, PiaOn_AE=PiaOn_AE, PiaOff_SE=PiaOff_SE, PiaOff_AE=PiaOff_AE, \
			num_dpSig_snaps=num_dpSig_snaps, q_deriv=q_deriv, ri_deriv=ri_deriv, ria_deriv=ria_deriv, \
			ria_snapshots=ria_snapshots, ri_snapshots=ri_snapshots, q_snapshots=q_snapshots, ds_fctr_snapshots=ds_fctr_snapshots,  \
			#
			Z_inferred_train=Z_inferred_train, pyiEq1_gvnZ_train=pyiEq1_gvnZ_train, Y_train=Y_train_rnd, Z_train=Z_train_rnd, \
			Z_inferred_test=Z_inferred_test, pyiEq1_gvnZ_test=pyiEq1_gvnZ_test, Y_test=Y_test_srt, Z_test=Z_test_srt, \
			#
			pj_zHyp_train=pj_zHyp_train, pj_zHyp_test=pj_zHyp_test, pj_zTru_Batch=pj_zTru_Batch, pj_zTrain_Batch=pj_zTrain_Batch, \
			pj_zTest_Batch=pj_zTest_Batch, cond_zHyp_train=cond_zHyp_train, cond_zHyp_test=cond_zHyp_test, cond_zTru_Batch=cond_zTru_Batch, \
			cond_zTrain_Batch=cond_zTrain_Batch, cond_zTest_Batch=cond_zTest_Batch, \
			#
			ind_matchGT2Mod=ind_matchGT2Mod, cosSim_matchGT2Mod=cosSim_matchGT2Mod, csNM_matchGT2Mod=csNM_matchGT2Mod, \
			lenDif_matchGT2Mod=lenDif_matchGT2Mod, ldNM_matchGT2Mod=ldNM_matchGT2Mod,\
			cosSimMat_matchGT2Mod=cosSimMat_matchGT2Mod, lenDifMat_matchGT2Mod=lenDifMat_matchGT2Mod, \
			HungRowCol_matchGT2Mod=HungRowCol_matchGT2Mod, cos_simHM_matchGT2Mod=cos_simHM_matchGT2Mod, \
			#
			ind_matchMod2GT=ind_matchMod2GT, cosSim_matchMod2GT=cosSim_matchMod2GT, csNM_matchMod2GT=csNM_matchMod2GT, \
			lenDif_matchMod2GT=lenDif_matchMod2GT, ldNM_matchMod2GT=ldNM_matchMod2GT, \
			cosSimMat_matchMod2GT=cosSimMat_matchMod2GT, lenDifMat_matchMod2GT=lenDifMat_matchMod2GT, \
			HungRowCol_matchMod2GT=HungRowCol_matchMod2GT, cos_simHM_matchMod2GT=cos_simHM_matchMod2GT, \
			#
			xVal_snapshot=xVal_snapshot, xVal_batchSize=xVal_batchSize, zActivationHistory=zActivationHistory, argsRec=argsRec )




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
	# xTimesThruTrain
	parser.add_argument('-TTT', '--xTimesThruTrain', dest='xTimesThruTrain', type=int, default=1,
		help='Number of times to go through the Training data set.')
	# num_test_samps_4xVal
	parser.add_argument('-nXs', '--num_test_samps_4xVal', dest='num_test_samps_4xVal', type=int, default=1,
		help='Number of test data samples to compute pjoint on and average over to implement cross validation.')
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
	# xVal_snapshot
	parser.add_argument('-xVs', '--xVal_snapshot', dest='xVal_snapshot', type=int, default=1,
		help='How often to compute batches of joint and conditional probabilities to average over them.')
	# xVal_batchSize
	parser.add_argument('-xVb', '--xVal_batchSize', dest='xVal_batchSize', type=int, default=1,
		help='How large a batch of data points to compute joint and conditional probabilities to average over them.')

	#
	# # Learning rate stuff and some options on Inference.
	#
	# learning_rate,
	parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=1.,
		help='Learning rate in model learning at each iteration. ex(1 works well)')  
	# # lRateScale_Pi,
	# parser.add_argument('-lrPi', '--lRateScale_Pi', dest='lRateScale_Pi', type=float, default=1.,
	# 	help='An Multiplicative factor on Pi learning rate. Thought <1 was good, but now =1 turns this off. That problem was fixed by flg_include_Zeq0_infer.')
	# lRateScale
	parser.add_argument('-LRS', '--lRateScale', dest='lRateScale', type=str, default='1.0,0.1,0.1',
		help='Comma delimited string to be converted into a list of learning rate scales for Pia, Pi and Q.')
	# flg_include_Zeq0_infer
	parser.add_argument('-z0', '--flg_include_Zeq0_infer', dest='flg_include_Zeq0_infer', action='store_true', default=False,
		help=', Flag to include the z=0''s vector in inference if True. ex( True or False))')
	# sample_longSWs_1st
	parser.add_argument('-smp1st', '--sample_longSWs_1st', dest='sample_longSWs_1st', type=str, default='Dont',
		help="Sample Spikewords for learning randomly, with probability proportional to their length, or sorted by there SW length. Options are: {'Dont', 'Prob', 'Hard'} ")


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
	# 				 -rand -2nd -v 
	#
	# ds_fctr_snapshots,
	parser.add_argument('-t_snap', '--ds_fctr_snapshots', dest='ds_fctr_snapshots', type=float, default=1,
		help='How often to take snapshot of learning process. Note r_ia to follow learning for Errors in SynthData.')
	# flg_recordRandImprove
	parser.add_argument('-randImp', '--flg_recordRandImprove', dest='flg_recordRandImprove', action='store_true', default=False,
		help='Not really reasonable to do this here anymore.  Cood idea to do elsewhere. How to combine diff models learned on same data.')
	# flg_EgalitarianPrior
	parser.add_argument('-Egal', '--flg_EgalitarianPrior', dest='flg_EgalitarianPrior', action='store_true', default=False,
		help='Flag to use Egalitarian activity based prior on z-vec by Q-vec. Otherwise use Binomial Q-scalar prior.. ex( True or False))')	
	# train_2nd_model
	parser.add_argument('-2nd', '--train_2nd_model', dest='train_2nd_model', action='store_true', default=False,
		help='Flag to train a 2nd model with 50% test data so we can compare CAs found by each. ex( True or False))')	
	# resample_available_spikewords
	parser.add_argument('-resampSW', '--resample_available_spikewords', dest='resample_available_spikewords', action='store_true', default=False,
		help='Flag that says to load in synthesized model and generated spikewords from rand0 file - from a previous run.')		
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
	pgmCA_synthData(args)  

