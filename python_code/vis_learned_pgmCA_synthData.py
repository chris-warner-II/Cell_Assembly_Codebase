import numpy as np 
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import os
import time
from scipy import signal as sig
from scipy import stats as st

import utils.data_manipulation as dm
import utils.plot_functions as pf
import utils.retina_computation as rc
from textwrap import wrap

import pandas as pd



# Flags for plotting and saving things
doplts = True
flg_plt_dataGen_andSampling_hist 		= doplts#True # (Plot 1). Do multiple comparison of stats to see if different data are biased. {Full vs. Sampled vs. Inferred vs. Train vs. Test}
flg_plt_EMlearning_Err 					= doplts#True # (Plot 2). Plot Mean Squared Error of Parameters (between Learned and True) during EM iterations.
flg_plt_EMlearning_derivs 				= doplts#True # (Plot 3). Plot derivatives of model parameters during EM iterations.
flg_plt_EMlearning_params_init_n_final 	= doplts#True # (Plot 4). Plot model parameters of true model, learned model and initialized model. 
flg_plt_EMinfer_performance_stats		= doplts#True # (Plot 5). Plot Inference performance stats over whole model learning.  (CA & cell confusion matrices, activation & co-act stats)
flg_plt_matchModels_cosSim 				= doplts#True # (Plot 6). Plots to investigate translation / permutation of Cell Assemblies between Learned ones and Ground Truth ones
flg_plt_temporal_EM_inference_analysis 	= doplts#True # (Plot 7). Plot tracking correctly and incorrectly inferred cell assemblies vs. EM algorithm iteration.
flg_plt_crossValidation_Cond 			= doplts#True
flg_plt_crossValidation_Joint			= doplts#True
flg_plt_dataGen_Samp_sanityCheck		= False# (Plot 8). Distributions of Sampled Spike words and their cardinalities |y| & |z|. 
flg_Pia_snapshots_gif 					= False

flg_compute_StatsPostLrn				= True 	# (Stats 1). Compute statistics on inferred Z & Y for all spike words- inferred from fixed model after EM algorithm.
flg_compute_StatsDuringEM				= False # (Stats 2). Compute statistics on inferred Z & Y for all EM samples, during learning. Can have many more EM samples than spike words.
												# 			 and stats can look shitty because of burn-in / learning phase.
#
flg_plot_train_data						= True 
flg_plot_test_data						= False 
#
flg_write_CSV_stats 					= True



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (0). Set up directories and create dirs if necessary.
dirHome, dirScratch = dm.set_dir_tree()
EM_data_dir 		= str( dirScratch + 'data/python_data/PGM_analysis/synthData/Models_learned_EM/')
infPL_data_dir 		= str( dirScratch + 'data/python_data/PGM_analysis/synthData/inference_postLrn/')
Stats_Inf_data_dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/InferStats_from_EM_learning/')
EM_figs_dir 		= str( dirScratch + 'figs/PGM_analysis/EM_Algorithm/')


# Parameters we can loop over.
num_SWs_tot = [100000]
num_Cells = [55] 		# Looping over N values
num_CAs_true = [55]		# Looping over M values used to build the model
#
overcompletenesses 	= [0.5, 1.0, 1.5] # 

ds_fctr_snapshots 	= 1000 	# Take a sample of model parameters every *ds_fctr* time steps to compute and plot MSE after we determine permutation of learned CA's to True CA's
xVal_snapshot 		= 1000
xVal_batchSize 		= 100


xTimesThruTrain = 1
num_EM_rands	= 3 	# number of times to randomize samples and rerun EM on same data generated from single synthetic model.
#


flg_include_Zeq0_infer  = True
verbose_EM 				= False

resample_available_spikewords = True # ALWAYS TRUE I THINK FROM NOW ON !!!
train_2nd_modelS = [False,True]
pct_xVal_train 	= 0.5 	# percentage of spikewords (total data) to use as training data. Set the rest aside for test data for cross validation.
pct_xVal_train_prev = 0.5	

sample_longSWs_1stS = ['Prob']#,'Dont'] # Options are: {'Dont', 'Prob', 'Hard'}
		# set to False to not specify amount of memory to use.

# Synthetic Model Construction Parameters
Ks 				= [1,2] 		# Number of	cell assemblies active, given a Bernoulli distribution ("hotness" of Z-vector)
Kmins 			= [0,0]		# Max & Min number of cell assemblies active 
Kmaxs 			= [4,4]		# 
#
Cs 				= [6,2]		# number of cells participating in cell assemblies. ("hotness" of each row in Pia)
Cmins 			= [2,2] 		# Max & Min number of cell active to call it a cell assembly
Cmaxs 			= [6,6] 		# 
#
mu_PiaS			= [0.30, 0.55]  	# Binary values in Pia matrix made continuous by sampling from Gaussian centered at 1-mu_Pia or 0+mu_Pia.
sig_PiaS		= [0.10, 0.05] 	# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
#
mu_PiS			= [0.04, 0.04]  	# Binary values in Pi vector made continuous by sampling from Gaussian centered at 1-mu_Pi or 0+mu_Pi.
sig_PiS			= [0.02, 0.02] 	# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
bernoulli_Pi	= 1.0   	# Each value in binary Pi vector randomly sampled from bernoulli with p(Pi=1) = bernoulli_Pi.	

#
yLo_Vals 		= [0] 		# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<=yLo.
yHi_Vals 		= [1000] 	# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution.
yMinSWs 		= [1] #,3] 	# Only grab spike words that are at least this length for training.


num_test_samps_4xValS = [1] 	# Number of test data points to to use to calculate pjoint_test (take their average) for each single train data point.
								# Increasing this should smooth out the pjoint curve and make cross validation easier.

# Parameter initializations for EM algorithm to learn model parameters
params_initS 	= ['NoisyConst'] #['DiagonalPia', 'NoisyConst'] 	# Options: {'True', 'RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise) }
sigQ_init 		= 0.01			# STD on Q initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
sigPi_init 		= 0.05			# STD on Pi initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
sigPia_init 	= 0.05			# STD on Pia initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
Z_hot 			= 5 			# Mean of initialization for Q value (how many 1's expected in binary z-vector)
C_noise_riS 	= [1.0, 1.0]	# Mean of initialization of Pi values (1 means mostly silent) with variability defined by sigPia_init
C_noise_riaS 	= [1.0, 0.9]	# Mean of initialization of Pia values (1 means mostly silent) with variability defined by sigPia_init



# Learning rates for EM algorithm
learning_rates 	= [0.5] #, 0.1]	# Learning rates to loop through
lRateScaleS = [ [1.0, 0.1, 0.1]] #, [1.0, 0.1, 1.0] ]  # Multiplicative scaling to Pia,Pi,Q learning rates. Can set to zero and take Pi taken out of model essentially.



flg_EgalitarianPriorS = [False,True]




sortBy = 'DontSort' #'Full' or 'Infer_postLrn' or 'DontSort' (( NOT THESE: or 'Infer_allSWs' or 'allSWs' ))





# Parameter initializations for learning & EM algorithm
params_initS 	= ['NoisyConst'] #['DiagonalPia', 'NoisyConst'] 	# Options: {'True', 'RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise) }
#
sigQ_init 		= 0.01
sigPi_init 		= 0.05
sigPia_init 	= 0.05
Z_hot 			= 5
C_noise_riS		= [1.0]
C_noise_riaS 	= [1.0]






if flg_write_CSV_stats:

	data_stats_CSV = dict() 									# Make an empty Dictionary
	fname_CSV = str(Stats_Inf_data_dir + 'STATs_synthData.csv')	# Filename to save CSV to
	dfh = 0														# clunky way to make the header only in the first row.



for rand in range(num_EM_rands):
	if resample_available_spikewords:
		if rand==0:
			rsTag = '_origModnSWs'
		else:
			rsTag = str( '_resampR0trn'+ str(pct_xVal_train_prev).replace('.','pt') )
	#
	for train_2nd_model in train_2nd_modelS:
		if train_2nd_model:
			pct_xVal_train = 0.5
			Btag ='B'
		else:
			Btag =''
		#
		for params_init in params_initS:
			#
			for learning_rate in learning_rates:
				#
				for lRateScale in lRateScaleS:
					#
					for abc in range(len(num_Cells)):
						N = num_Cells[abc]
						M = num_CAs_true[abc]
						#
						for flg_EgalitarianPrior in flg_EgalitarianPriorS:
							#
							if flg_EgalitarianPrior:	
								priorType = 'EgalQ' 
							else:
								priorType = 'BinomQ'
							#
							for sample_longSWs_1st in sample_longSWs_1stS:
								#
								for Cn_ind in range(len(C_noise_riaS)):
									C_noise_ria = C_noise_riaS[Cn_ind]
									C_noise_ri = C_noise_riS[Cn_ind]
									#
									for overcomp in overcompletenesses:
										#
										for num_SWs in num_SWs_tot:
											num_EM_samps = int(pct_xVal_train*num_SWs)
											#
											M_mod 		= int(np.round(overcomp*M))
											C_noise 	= np.array([Z_hot/M_mod, C_noise_ri, C_noise_ria ])		# Mean of parameters for 'NoisyConst' initializations. Scalar between 0 and 1. 1 means almost silent. 0 means almost always firing.
											sig_init 	= np.array([sigQ_init, sigPi_init, sigPia_init ])		# STD on parameter initializations from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
											pjt_tol 			= 10
											numPia_snapshots 	= np.round(num_EM_samps/ds_fctr_snapshots).astype(int)
											samps2snapshot_4Pia = (np.hstack([1, np.arange( np.round(num_EM_samps/numPia_snapshots), num_EM_samps, 
																	np.round(num_EM_samps/numPia_snapshots)  ), num_EM_samps]) - 1).astype(int)
											#
											# Loop through different combinations of parameters and generate plots of results.
											for xyz in range( len(Ks) ):
												#
												K 	 = Ks[xyz]
												Kmin = Kmins[xyz]
												Kmax = Kmaxs[xyz]
												#
												C 	 = Cs[xyz]
												Cmin = Cmins[xyz]
												Cmax = Cmaxs[xyz]
												#
												mu_Pia = mu_PiaS[xyz]
												mu_Pi = mu_PiS[xyz]
												sig_Pia = sig_PiaS[xyz]
												sig_Pi = sig_PiS[xyz]
												#
												for yLo in yLo_Vals:
													#
													for yHi in yHi_Vals:
														#
														for yMinSW in yMinSWs:

															if flg_write_CSV_stats:
																data_stats_CSV.update( [  ('N',[N]) , ('M',[M]) , ('K',[K]) , ('Kmin',[Kmin]) , ('Kmax',[Kmax]) , \
																	('C',[C]) , ('Cmin',[Cmin]) , ('Cmax',[Cmax]) , ('yLo',[yLo]), ('yHi',[yHi]), ('# SWs',[num_SWs]) , \
																	('# EM samps',[num_EM_samps]), ('LR',[learning_rate]), ('2nd model',[Btag]), ('init',[params_init]), \
																	('Pi mean init',[C_noise_ri]), ('Pia mean init',[C_noise_ria]), ('Pi std init',[sigPi_init]), \
																	('Pia std init',[sigPia_init]), ('Q std init',[sigQ_init]), ('Zhot init',[Z_hot]), ('LRxPi',[lRateScale]) ] )


															# if True:
															try:

																
																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# Building up directory structure and npz file name to load in.
																#
																if flg_include_Zeq0_infer:
																	z0_tag='_zeq0'
																else:
																	z0_tag='_zneq0'
																#
																params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )	
																#
																InitLrnInf = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') +
																		'_LRsc' + str(lRateScale) + '/' )	
																#
																ModelType = str( 'SyntheticData_N' + str(N) + '_M' + str(M) + '_Mmod' + str(M_mod) + '_K' + str(K) + 
																			'_' + str(Kmin) + '_' + str(Kmax) + '_C' + str(C) + '_' + str(Cmin)+ '_' + str(Cmax) + 
																			'_mPia' + str(mu_Pia) + '_sPia' + str(sig_Pia) + '_bPi' + str(bernoulli_Pi) + '_mPi' + 
																			str(mu_Pi) + '_sPi' + str(sig_Pi) + '_' + sample_longSWs_1st + 'Smp1st_' + priorType + '/')

																



																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																#
																# Construct file names for EM_model_data.npz file and also B file if we've trained 2nd model.
																#
																fname_EMlrn = str( 'EM_model_data_' + str(num_SWs) + 'SWs_trn' + str(pct_xVal_train).replace('.','pt') + '_xTTT' + str(xTimesThruTrain) \
																		+'_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) + '_rand' + str(rand) + rsTag + Btag + '.npz' ) 
																
																fname_inferPL = fname_EMlrn.replace('EM_model_data_','SWs_inferred_postLrn_')




																# Set up file names and directories.
																LearnedModel_fname 			= str( EM_data_dir + InitLrnInf + ModelType + fname_EMlrn )	
																infer_postLrn_fname			= str( infPL_data_dir + InitLrnInf + ModelType + fname_EMlrn.replace('EM_model_data_','SWs_inferred_postLrn_') )
																StatsDuringLearning_fname 	= str( Stats_Inf_data_dir + InitLrnInf + ModelType + fname_EMlrn.replace('EM_model_data_','InferStats_DuringLearn_') )
																StatsPostLearning_fname 	= str( Stats_Inf_data_dir + InitLrnInf + ModelType + fname_EMlrn.replace('EM_model_data_','InferStats_PostLrn_') )
																plt_save_dir 				= str( EM_figs_dir + InitLrnInf + ModelType )

																
																fname_tag = fname_EMlrn.replace('EM_model_data_','').replace('.npz','' )


																# Make a directories for output plots and data files if not already there.
																#
																if not os.path.exists( str(Stats_Inf_data_dir + InitLrnInf + ModelType) ):
																	os.makedirs( str(Stats_Inf_data_dir + InitLrnInf + ModelType) )
																#
																if not os.path.exists( str(infPL_data_dir + InitLrnInf + ModelType) ):
																	os.makedirs( str(infPL_data_dir + InitLrnInf + ModelType) )
																#
																if not os.path.exists( str(EM_data_dir + InitLrnInf + ModelType) ):
																	os.makedirs( str(EM_data_dir + InitLrnInf + ModelType) )
																#
																if not os.path.exists( plt_save_dir ):
																	os.makedirs( plt_save_dir )				



																# # 
																# #
																# # 
																# # NOTE: I WANT A WAY TO CHECK IF ALL THE PLOTS ARE ALREADY THERE. AND IF SO, DONT REMAKE THEM.
																# # 		REINSTATE THIS LATER...
																#
																# all_plots_there_already = True
																# filesInDir = os.listdir( plt_save_dir ) 
																# #
																# if plt_dataGen_andSampling_hist_flg and all_plots_there_already:
																# 	fType = str('DataBiases_' + fname_tag)
																# 	A = any(fType in file for file in filesInDir)
																# 	print(A)
																# 	all_plots_there_already = all_plots_there_already and A
																# 	#
																# if plt_EMlearning_MSE_flg and all_plots_there_already:
																# 	fType = str('ParamsMSE_' + fname_tag)
																# 	A = any(fType in file for file in filesInDir)
																# 	print(A)
																# 	all_plots_there_already = all_plots_there_already and A
																# 	#
																# if plt_EMlearning_derivs_flg and all_plots_there_already:
																# 	fType = str('ParamsDerivatives_' + fname_tag)
																# 	A = any(fType in file for file in filesInDir)
																# 	print(A)
																# 	all_plots_there_already = all_plots_there_already and A
																# 	#
																# if plt_EMlearning_params_init_n_final and all_plots_there_already:
																# 	fType = str('LearnedParams_' + fname_tag)
																# 	A = any(fType in file for file in filesInDir)
																# 	print(A)
																# 	all_plots_there_already = all_plots_there_already and A
																# 	#
																# if plt_EMinfer_performance_stats and all_plots_there_already:
																# 	fType = str('CA_Confus_Overl_Coactiv_1stOrdrInf_' + fname_tag)
																# 	A = any(fType in file for file in filesInDir)
																# 	print(A)
																# 	all_plots_there_already = all_plots_there_already and A
																# 	#
																# if plt_translation_Lrn2Tru and all_plots_there_already:
																# 	fType = str('translatePermute_' + fname_tag)
																# 	A = any(fType in file for file in filesInDir)
																# 	print(A)
																# 	all_plots_there_already = all_plots_there_already and A
																# 	#
																# if plt_temporal_EM_inference_analysis and all_plots_there_already:
																# 	fType = str('InferenceTemporal_1stOrdrInf_' + fname_tag)
																# 	A = any(fType in file for file in filesInDir)
																# 	print(A)
																# 	all_plots_there_already = all_plots_there_already and A
																# 	#
																# if plt_dataGen_Samp_sanityCheck_flg and all_plots_there_already:
																# 	fType = str('Sampling4Learning_' + fname_tag)
																# 	A = any(fType in file for file in filesInDir)
																# 	print(A)
																# 	all_plots_there_already = all_plots_there_already and A
																# 	#
																# if all_plots_there_already:
																# 	break









																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# Loading in npz data file and extracting all variables within it.
																#
																print( str(' Loading in file: ' + LearnedModel_fname ) )
																# try:
																if True:
																	data = np.load( LearnedModel_fname )
																	#
																	print(data.keys())
																	#
																	ria 				= data['ria']
																	ria_mod				= data['ria_mod']
																	ri 					= data['ri']
																	q 					= data['q']
																	#
																	riap 				= data['riap']
																	rip 				= data['rip']
																	qp 					= data['qp']
																	#
																	ria_init 			= data['ria_init']
																	ri_init 			= data['ri_init']
																	q_init 				= data['q_init']
																	#
																	Z_train 			= data['Z_train']
																	Y_train 			= data['Y_train']
																	Z_inferred_train 	= data['Z_inferred_train']
																	pyiEq1_gvnZ_train 	= data['pyiEq1_gvnZ_train'] # p(yi=1|z) during learning
																	#
																	Z_test 				= data['Z_test']
																	Y_test 				= data['Y_test'] 
																	Z_inferred_test 	= data['Z_inferred_test']
																	pyiEq1_gvnZ_test 	= data['pyiEq1_gvnZ_test'] # p(yi=1|z) during learning
										
																	#
																	pj_zHyp_train 		= data['pj_zHyp_train']
																	pj_zHyp_test 		= data['pj_zHyp_test']
																	pj_zTru_Batch 		= data['pj_zTru_Batch'] 
																	pj_zTrain_Batch 	= data['pj_zTrain_Batch'] 
																	pj_zTest_Batch		= data['pj_zTest_Batch']
																	#
																	# Conditional probabilities for Cross Validation
																	cond_zHyp_train 	= data['cond_zHyp_train'] 
																	cond_zHyp_test 		= data['cond_zHyp_test']
																	cond_zTru_Batch 	= data['cond_zTru_Batch'] 
																	cond_zTrain_Batch 	= data['cond_zTrain_Batch'] 
																	cond_zTest_Batch 	= data['cond_zTest_Batch']
																	#
																	# Parameters for batch computation of Joint and conds.
																	xVal_snapshot 		= data['xVal_snapshot']
																	xVal_batchSize 		= data['xVal_batchSize']
																	#
																	ria_snapshots 		= data['ria_snapshots']
																	ri_snapshots 		= data['ri_snapshots']
																	q_snapshots 		= data['q_snapshots']
																	num_dpSig_snaps 	= data['num_dpSig_snaps']
																	#
																	try:
																		ds_fctr_snapshots 	= data['ds_fctr_snapshots']
																	except:
																		print('ds_fctr_snapshots not there yet.')
																	#	

																	ria_deriv 	= data['ria_deriv']
																	ri_deriv 	= data['ri_deriv']
																	q_deriv 	= data['q_deriv']
																	#
																	Q_SE 		= data['Q_SE']
																	Pi_SE 		= data['Pi_SE']
																	Pi_AE 		= data['Pi_AE']
																	Pia_AE 		= data['Pia_AE']
																	PiaOn_SE 	= data['PiaOn_SE']
																	PiaOn_AE 	= data['PiaOn_AE']
																	PiaOff_SE 	= data['PiaOff_SE']
																	PiaOff_AE 	= data['PiaOff_AE']
																	


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
																	argsRec 				= data['argsRec']


																	
																	 



																# except:
																# 	print('File either not there or not done processing')
																# 	print('Trying to remove: ', LearnedModel_fname )
																# 	os.remove( LearnedModel_fname )
																# 	continue


																Q = rc.sig(qp)	
																Pi = rc.sig(rip)	
																Pia = rc.sig(riap)	



																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																#
																#
																print( str(' Loading in file: ' + infer_postLrn_fname ) )
																# try:
																#if True:
																data = np.load( infer_postLrn_fname )
																#
																print(data.keys())

																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# 
																pj_inferred_train_postLrn 			= data['pj_inferred_train_postLrn']
																try:
																	cond_inferred_train_postLrn 	= data['cond_inferred_train_postLrn']
																	cond_inferred_test_postLrn 		= data['cond_inferred_test_postLrn']
																except:
																	print('cond_inferred_train_postLrn in newer files.')
																Z_inferred_train_postLrn 			= data['Z_inferred_train_postLrn']
																pyiEq1_gvnZ_train_postLrn 			= data['pyiEq1_gvnZ_train_postLrn']
																pyi_gvnZ_auc_train 					= data['pyi_gvnZ_auc_train']
																pyi_gvnZ_ROC_train 					= data['pyi_gvnZ_ROC_train']  
																pyi_gvnZ_stats_train 				= data['pyi_gvnZ_stats_train']
																#
																Kinf_train_postLrn 					= data['Kinf_train_postLrn']
																KinfDiff_train_postLrn 				= data['KinfDiff_train_postLrn']
																zCapture_train_postLrn 				= data['zCapture_train_postLrn']
																zMissed_train_postLrn 				= data['zMissed_train_postLrn']
																zExtra_train_postLrn 				= data['zExtra_train_postLrn']
																#
																inferCA_Confusion_train_postLrn 	= data['inferCA_Confusion_train_postLrn']
																zInferSampledT_train_postLrn 		= data['zInferSampledT_train_postLrn']
																zInferSampled_train_postLrn 		= data['zInferSampled_train_postLrn']
																#
																if len(zInferSampledT_train_postLrn)-1==M_mod and \
																	len(zInferSampled_train_postLrn)-1==M and M!=M_mod:
																	#
																	print('NOTE: zInferSampled_train_postLrn and zInferSampledT_train_postLrn are flipped for now \
																		because of a bug in previous code that I have since fixed. This is for old data files.')	
																	zInferSampled_train_postLrn 		= data['zInferSampledT_train_postLrn']
																	zInferSampledT_train_postLrn 		= data['zInferSampled_train_postLrn']
																#
																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# 
																pj_inferred_test_postLrn 			= data['pj_inferred_test_postLrn']
																Z_inferred_test_postLrn 			= data['Z_inferred_test_postLrn']
																pyiEq1_gvnZ_test_postLrn 			= data['pyiEq1_gvnZ_test_postLrn']
																pyi_gvnZ_ROC_test 					= data['pyi_gvnZ_ROC_test']
																pyi_gvnZ_auc_test 					= data['pyi_gvnZ_auc_test']
																pyi_gvnZ_stats_test 				= data['pyi_gvnZ_stats_test']
																#
																Kinf_test_postLrn 					= data['Kinf_test_postLrn']
																KinfDiff_test_postLrn 				= data['KinfDiff_test_postLrn']
																zCapture_test_postLrn 				= data['zCapture_test_postLrn']
																zMissed_test_postLrn 				= data['zMissed_test_postLrn']
																zExtra_test_postLrn 				= data['zExtra_test_postLrn']
																#
																inferCA_Confusion_test_postLrn 		= data['inferCA_Confusion_test_postLrn']
																zInferSampled_test_postLrn 			= data['zInferSampled_test_postLrn']
																zInferSampledT_test_postLrn 		= data['zInferSampledT_test_postLrn']
																#
																Z_trainM 							= data['Z_trainM']
																Z_testM 							= data['Z_testM']
																Z_inferredM_train 					= data['Z_inferredM_train']
																Z_inferredM_test 		 			= data['Z_inferredM_test']
																Z_inferredM_train_postLrn 			= data['Z_inferredM_train_postLrn']
																Z_inferredM_test_postLrn 			= data['Z_inferredM_test_postLrn']
																


																#
																#
																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


																if flg_write_CSV_stats:
																	pj_postLrn = np.concatenate([ np.float64(pj_inferred_train_postLrn[0]), np.float64(pj_inferred_test_postLrn[0]) ])
																	pj_not_nan = np.where( 1-np.isnan(pj_postLrn).astype(int) )[0]
																	#
																	data_stats_CSV.update( [ ('mean pj postLrn',[pj_postLrn[pj_not_nan].mean()]) , \
																							('% pj not nan',[pj_not_nan.size / pj_postLrn.size]) ] )
																		

																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# (Stats 1). Compute overlap or similarity of Cell Assemblies in Ground Truth and Learned Models.
																#
																CA_ovl 			= rc.compute_CA_Overlap(ria)
																CA_ovl_model 	= rc.compute_CA_Overlap(riap)
																#
																if False:
																	CA_ovl_stats = rc.compute_Confusion_Stats(CA_ovl,flgAddDrop=False)
																	CA_ovl_model_stats = rc.compute_Confusion_Stats(CA_ovl_model,flgAddDrop=False)




																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# (Stats 2). Number of CA pairs (true and learned) that pass a dot product threshold from [0:0.1:1]
																#
																# NOT BEING USED RIGHT NOW.
																if False:
																	dp_sig = [ (dot_prod_Lrn2Tru[0]>t).sum() for t in np.linspace(0,1,11) ]




																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																









																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# (Stats 5). Compute Statistics of Sampling and Inference for Training and Test Data Sets/
																#
																if flg_compute_StatsPostLrn:

																	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																	# (Stats 5A). Compute Stats for All Spike Words (Usually much fewer than number of EM samples.)
																	#
																	#
																	try:
																		data = np.load( StatsPostLearning_fname )
																		#
																		Ycell_hist_Full_train 				= data['Ycell_hist_Full_train']
																		Zassem_hist_Full_train 				= data['Zassem_hist_Full_train']
																		nY_Full_train 						= data['nY_Full_train']
																		nZ_Full_train 						= data['nZ_Full_train']
																		CA_coactivity_Full_train 			= data['CA_coactivity_Full_train']
																		Cell_coactivity_Full_train 			= data['Cell_coactivity_Full_train']
																		#
																		Ycell_hist_Full_test 				= data['Ycell_hist_Full_test']
																		Zassem_hist_Full_test 				= data['Zassem_hist_Full_test']
																		nY_Full_test 						= data['nY_Full_test']
																		nZ_Full_test 						= data['nZ_Full_test']
																		CA_coactivity_Full_test 			= data['CA_coactivity_Full_test']
																		Cell_coactivity_Full_test 			= data['Cell_coactivity_Full_test']
																		#
																		Ycell_hist_Infr_train_postLrn 		= data['Ycell_hist_Infr_train_postLrn']
																		Zassem_hist_Infr_train_postLrn		= data['Zassem_hist_Infr_train_postLrn']
																		nY_Infr_train_postLrn				= data['nY_Infr_train_postLrn']
																		nZ_Infr_train_postLrn				= data['nZ_Infr_train_postLrn']
																		CA_coactivity_Infr_train_postLrn	= data['CA_coactivity_Infr_train_postLrn']
																		Cell_coactivity_Infr_train_postLrn	= data['Cell_coactivity_Infr_train_postLrn']
																		#
																		Ycell_hist_Infr_test_postLrn		= data['Ycell_hist_Infr_test_postLrn']
																		Zassem_hist_Infr_test_postLrn		= data['Zassem_hist_Infr_test_postLrn']
																		nY_Infr_test_postLrn				= data['nY_Infr_test_postLrn']
																		nZ_Infr_test_postLrn				= data['nZ_Infr_test_postLrn']
																		CA_coactivity_Infr_test_postLrn		= data['CA_coactivity_Infr_test_postLrn']
																		Cell_coactivity_Infr_test_postLrn	= data['Cell_coactivity_Infr_test_postLrn']									


																	except:

																		# (A1). These are for full training data.
																		print('compute statistics on full training data set post EM learning')
																		t0 = time.time()
																		Ycell_hist_Full_train, Zassem_hist_Full_train, nY_Full_train, nZ_Full_train, CA_coactivity_Full_train, \
																		Cell_coactivity_Full_train = rc.compute_dataGen_Histograms(Y_train, Z_train, M, N)
																		#
																		# (A2). These are for full test data.
																		print('compute statistics on full test data post EM learning')
																		Ycell_hist_Full_test, Zassem_hist_Full_test, nY_Full_test, nZ_Full_test, CA_coactivity_Full_test, \
																		Cell_coactivity_Full_test = rc.compute_dataGen_Histograms(Y_test, Z_test, M, N) 
																		#
																		# # (A3). These are for training data inferred with the fixed model post EM learning.
																		print('Compute statistics on inference results from training data post EM learning')
																		Ycell_hist_Infr_test_postLrn, Zassem_hist_Infr_test_postLrn, nY_Infr_test_postLrn, nZ_Infr_test_postLrn, CA_coactivity_Infr_test_postLrn, \
																		Cell_coactivity_Infr_test_postLrn = rc.compute_dataGen_Histograms(pyiEq1_gvnZ_test_postLrn[0], Z_inferredM_test_postLrn[0], M_mod, N)
																		
																		# # (A4). These are for test data inferred with the fixed model post EM learning.
																		print('Compute statistics on inference results from test data post EM learning')
																		Ycell_hist_Infr_train_postLrn, Zassem_hist_Infr_train_postLrn, nY_Infr_train_postLrn, nZ_Infr_train_postLrn, CA_coactivity_Infr_train_postLrn, \
																		Cell_coactivity_Infr_train_postLrn = rc.compute_dataGen_Histograms(pyiEq1_gvnZ_train_postLrn[0], Z_inferredM_train_postLrn[0], M_mod, N)
																		t1 = time.time()
																		print('Done w/ stats from inference results from full training and testdata sets and inference of both post EM learning : time = ',t1-t0)
																		#
																		np.savez( StatsPostLearning_fname, 
																			Ycell_hist_Full_train=Ycell_hist_Full_train, Zassem_hist_Full_train=Zassem_hist_Full_train, nY_Full_train=nY_Full_train, \
																			nZ_Full_train=nZ_Full_train, CA_coactivity_Full_train=CA_coactivity_Full_train, Cell_coactivity_Full_train=Cell_coactivity_Full_train, \
																			#
																			Ycell_hist_Full_test=Ycell_hist_Full_test, Zassem_hist_Full_test=Zassem_hist_Full_test, nY_Full_test=nY_Full_test, \
																			nZ_Full_test=nZ_Full_test, CA_coactivity_Full_test=CA_coactivity_Full_test, Cell_coactivity_Full_test=Cell_coactivity_Full_test, \
																			#
																			Ycell_hist_Infr_train_postLrn=Ycell_hist_Infr_train_postLrn, Zassem_hist_Infr_train_postLrn=Zassem_hist_Infr_train_postLrn, \
																			nY_Infr_train_postLrn=nY_Infr_train_postLrn, nZ_Infr_train_postLrn=nZ_Infr_train_postLrn, \
																			CA_coactivity_Infr_train_postLrn=CA_coactivity_Infr_train_postLrn, Cell_coactivity_Infr_train_postLrn=Cell_coactivity_Infr_train_postLrn,
																			#
																			Ycell_hist_Infr_test_postLrn=Ycell_hist_Infr_test_postLrn, Zassem_hist_Infr_test_postLrn=Zassem_hist_Infr_test_postLrn, \
																			nY_Infr_test_postLrn=nY_Infr_test_postLrn, nZ_Infr_test_postLrn=nZ_Infr_test_postLrn, \
																			CA_coactivity_Infr_test_postLrn=CA_coactivity_Infr_test_postLrn, Cell_coactivity_Infr_test_postLrn=Cell_coactivity_Infr_test_postLrn)
																

															
																# if flg_compute_StatsDuringEM:
																# 	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# 	# (Stats 5B). Compute Stats for All EM Samples (Usually much more than numSWs.)
																# 	#
																# 	#
																# 	try:

																# 		data = np.load( StatsDuringLearning_fname )
																# 		#
																# 		Ycell_hist_Samp_train 		= data['Ycell_hist_Samp_train'] 
																# 		Zassem_hist_Samp_train 		= data['Zassem_hist_Samp_train'] 
																# 		nY_Samp_train 				= data['nY_Samp_train']
																# 		nZ_Samp_train				= data['nZ_Samp_train']
																# 		CA_coactivity_Samp_train 	= data['CA_coactivity_Samp_train']
																# 		Cell_coactivity_Samp_train 	= data['Cell_coactivity_Samp_train']
																# 		#
																# 		Ycell_hist_Samp_test 		= data['Ycell_hist_Samp_test'] 
																# 		Zassem_hist_Samp_test 		= data['Zassem_hist_Samp_test'] 
																# 		nY_Samp_test 				= data['nY_Samp_test']
																# 		nZ_Samp_test				= data['nZ_Samp_test']
																# 		CA_coactivity_Samp_test 	= data['CA_coactivity_Samp_test']
																# 		Cell_coactivity_Samp_test 	= data['Cell_coactivity_Samp_test']
																# 		#
																# 		Ycell_hist_Infr_train 		= data['Ycell_hist_Infr_train'] 
																# 		Zassem_hist_Infr_train 		= data['Zassem_hist_Infr_train'] 
																# 		nY_Infr_train 				= data['nY_Infr_train']
																# 		nZ_Infr_train				= data['nZ_Infr_train']
																# 		CA_coactivity_Infr_train 	= data['CA_coactivity_Infr_train']
																# 		Cell_coactivity_Infr_train 	= data['Cell_coactivity_Infr_train']
																# 		#
																# 		Ycell_hist_Infr_test 		= data['Ycell_hist_Infr_test'] 
																# 		Zassem_hist_Infr_test 		= data['Zassem_hist_Infr_test'] 
																# 		nY_Infr_test 				= data['nY_Infr_test']
																# 		nZ_Infr_test				= data['nZ_Infr_test']
																# 		CA_coactivity_Infr_test 	= data['CA_coactivity_Infr_test']
																# 		Cell_coactivity_Infr_test 	= data['Cell_coactivity_Infr_test']

																# 	except:
																# 		#
																# 		smp_train = np.arange(len(Y_train))
																# 		Z_smp_train_list = [ Z_train[i] for i in list(smp_train) ]
																# 		Y_smp_train_list = [ Y_train[i] for i in list(smp_train) ]
																# 		#
																# 		smp_test = np.arange(len(Y_test))
																# 		Z_smp_test_list = [ Z_test[i] for i in list(smp_test) ]
																# 		Y_smp_test_list = [ Y_test[i] for i in list(smp_test) ]
																# 		#
																# 		# (B1). These are for training data sampled for EM algorithm.
																# 		print('compute statistics on sampled training data.')
																# 		t0 = time.time()
																# 		Ycell_hist_Samp_train, Zassem_hist_Samp_train, nY_Samp_train, nZ_Samp_train, CA_coactivity_Samp_train, \
																# 		Cell_coactivity_Samp_train = rc.compute_dataGen_Histograms(Y_smp_train_list, Z_smp_train_list, M_mod, N)
																# 		#
																# 		# (B2). These are for test data sampled for EM algorithm.
																# 		print('compute statistics on sampled test data')
																# 		Ycell_hist_Samp_test, Zassem_hist_Samp_test, nY_Samp_test, nZ_Samp_test, CA_coactivity_Samp_test, \
																# 		Cell_coactivity_Samp_test = rc.compute_dataGen_Histograms(Y_smp_test_list, Z_smp_test_list, M_mod, N)
																# 		#
																# 		# # (B3). These are inferred from sampled training data.
																# 		# print('compute statistics on inference results from EM algorithm from sampled training data')
																# 		# Ycell_hist_Infr_train, Zassem_hist_Infr_train, nY_Infr_train, nZ_Infr_train, CA_coactivity_Infr_train, \
																# 		# Cell_coactivity_Infr_train = rc.compute_dataGen_Histograms(Y_inferred_train, Z_inferred_train, M_mod, N)
																# 		# #
																# 		# # (B4). These are inferred from sampled test data.
																# 		# print('compute statistics on inference results from EM algorithm from sampled test data')
																# 		# Ycell_hist_Infr_test, Zassem_hist_Infr_test, nY_Infr_test, nZ_Infr_test, CA_coactivity_Infr_test, \
																# 		# Cell_coactivity_Infr_test = rc.compute_dataGen_Histograms(Y_inferred_test, Z_inferred_test, M_mod, N)
																# 		t1 = time.time()
																# 		print('Done w/ stats from inference results from sampled and inferred test and train data : time = ',t1-t0)
																		

																# 		np.savez( StatsDuringLearning_fname, 
																# 			Ycell_hist_Samp_train=Ycell_hist_Samp_train, Zassem_hist_Samp_train=Zassem_hist_Samp_train, nY_Samp_train=nY_Samp_train,
																# 			nZ_Samp_train=nZ_Samp_train, CA_coactivity_Samp_train=CA_coactivity_Samp_train, Cell_coactivity_Samp_train=Cell_coactivity_Samp_train,
																# 			#
																# 			Ycell_hist_Samp_test=Ycell_hist_Samp_test, Zassem_hist_Samp_test=Zassem_hist_Samp_test, nY_Samp_test=nY_Samp_test,
																# 			nZ_Samp_test=nZ_Samp_test, CA_coactivity_Samp_test=CA_coactivity_Samp_test, Cell_coactivity_Samp_test=Cell_coactivity_Samp_test,
																# 			#
																# 			Ycell_hist_Infr_train=Ycell_hist_Infr_train, Zassem_hist_Infr_train=Zassem_hist_Infr_train, nY_Infr_train=nY_Infr_train,
																# 			nZ_Infr_train=nZ_Infr_train, CA_coactivity_Infr_train=CA_coactivity_Infr_train, Cell_coactivity_Infr_train=Cell_coactivity_Infr_train,
																# 			#
																# 			Ycell_hist_Infr_test=Ycell_hist_Infr_test, Zassem_hist_Infr_test=Zassem_hist_Infr_test, nY_Infr_test=nY_Infr_test,
																# 			nZ_Infr_test=nZ_Infr_test, CA_coactivity_Infr_test=CA_coactivity_Infr_test, Cell_coactivity_Infr_test=Cell_coactivity_Infr_test)
																		













																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# # (Stats 4). Compute Statistics on Zinferred Confusion Matrices.
																# # 			 Boil down to 4 numbers ->[#Right, #MixedUp, #Dropped, #Added]
																# #
																# # # # # # # # # # # # # # # # # # # # #
																#	
																# #
																#
																if flg_write_CSV_stats & flg_compute_StatsPostLrn:

																	zTot = zCapture_train_postLrn.sum()+zMissed_train_postLrn.sum() + \
																			zCapture_test_postLrn.sum()+zMissed_test_postLrn.sum()
																	zCap = (zCapture_train_postLrn.sum()+zCapture_test_postLrn.sum()) / zTot
																	zMis = (zMissed_train_postLrn.sum()+zMissed_test_postLrn.sum()) / zTot
																	zExt = (zExtra_train_postLrn.sum()+zExtra_test_postLrn.sum()) / zTot
																	#
																	data_stats_CSV.update( [ ('# z Total postLrn',[zTot]) , ('% z Captured postLrn',[zCap]) , 
																							 ('% z Missed postLrn',[zMis]) , ('% z Extra postLrn',[zExt]) ] )
																		









																

																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# (Stats 6). Mean & STD of difference between inferred |y|,|z| and observed |y|,|z|.
																#
																if flg_compute_StatsPostLrn:

																	nZ_inf_allSWs 	= np.concatenate( (nZ_Infr_train_postLrn,nZ_Infr_test_postLrn) )
																	nZ_obs_allSWs  	= np.concatenate( (nZ_Full_train,nZ_Full_test) )
																	#
																	nY_obs_allSWs  	= np.concatenate( (nY_Full_train,nY_Full_test) )
																	nY_inf_allSWs 	= np.concatenate( ( pyiEq1_gvnZ_train_postLrn[0].sum(axis=1),
																										pyiEq1_gvnZ_test_postLrn[0].sum(axis=1) ) )
																	#
																	# nZ_all_diff = nZ_inf_allSWs - nZ_obs_allSWs
																	# nY_all_diff = nY_inf_allSWs - nY_obs_allSWs
																	#
																	if flg_write_CSV_stats:
																		data_stats_CSV.update( [ ('|Z| inf mean',[np.mean( nZ_inf_allSWs )]) , \
																								 ('|Z| inf std',[np.std( nZ_inf_allSWs )]) , \
																								 ('|Z| inf skew',[st.skew( nZ_inf_allSWs )]) , \
																								 #
																								 ('|Z| obs mean',[np.mean( nZ_obs_allSWs )]) , \
																								 ('|Z| obs std',[np.std( nZ_obs_allSWs )]) , \
																								 ('|Z| obs skew',[st.skew( nZ_obs_allSWs )]) , \
																								 #
																								 ('|Y| inf mean',[np.mean( nY_inf_allSWs )]) , \
																								 ('|Y| inf std',[np.std( nY_inf_allSWs )]) , \
																								 ('|Y| inf skew',[st.skew( nY_inf_allSWs )]) , \
																								 #
																								 ('|Y| obs mean',[np.mean( nY_obs_allSWs )]) , \
																								 ('|Y| obs std',[np.std( nY_obs_allSWs )]) , \
																								 ('|Y| obs skew',[st.skew( nY_obs_allSWs )]) , \
																								 #
																								 # ('|Z| overinferred mean',[np.mean( nZ_all_diff )]) , \
																								 # ('|Z| overinferred std',[np.std( nZ_all_diff )]) , \
																								 # ('|Z| overinferred skew',[st.skew( nZ_all_diff )]) , \
																								 # #
																								 # ('|Y| overinferred mean',[np.mean( nY_all_diff )]) , \
																								 # ('|Y| overinferred std',[np.std( nY_all_diff )]), \
																								 # ('|Y| overinferred skew',[st.skew( nY_all_diff )]) , \
																								 #
																								 ('%times z=0 inf allSWs',[(nZ_inf_allSWs==0).sum()/nZ_inf_allSWs.size]), \
																								 ('%times y=1 obs allSWs',[(nY_obs_allSWs==1).sum()/nY_obs_allSWs.size]) ] )
																	#
																	if False:
																		print('Amount over inferred on all SWs post-EM-Learning |Z| (mean, std): ', \
																				np.round(np.mean( nZ_all_diff ), 3), \
																				np.round(np.std( nZ_all_diff ), 3) )
																		#
																		#print(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
																		#
																		print('                                              |Y| (mean, std): ', \
																				np.round(np.mean( nY_all_diff ), 3), \
																				np.round(np.std( nY_all_diff ), 3) )
																		#
																		print(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
																		#

																	# Scatter plot of spike word lengths (observed vs. inferred)
																	if False:
																		pf.scatter_lenSW_inf_vs_obs(nY_obs_allSWs, nY_inf_allSWs)
																		pf.scatter_lenSW_inf_vs_obs(nZ_obs_allSWs, nZ_inf_allSWs)


																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# (Stats 6). Sort cell assemblies and cells by how active they are in Training Samples (or Training Inference?).
																# And determine which cells participate in which cell assemblies by thresholding. 
																#
																# NOTE: I DONT THINK I AM USING ANY OF THESE IN HERE AT ALL.



																print('Sorting CAs and Cells by their activity in ',sortBy)

																if sortBy=='Full':
																	sortCells_byActivity 	= np.argsort( (Ycell_hist_Full_train + Ycell_hist_Full_test)[:-1])[::-1]
																	sortCAs_byActivity 		= np.argsort( (Zassem_hist_Full_train + Zassem_hist_Full_test)[:-1])[::-1]
																	#
																# elif sortBy=='Infer_postLrn':
																# 	sortCells_byActivity 	= np.argsort( (Ycell_hist_Infr_train_postLrn + Ycell_hist_Infr_test_postLrn)[:-1])[::-1]
																# 	sortCAs_byActivity 		= np.argsort( (Zassem_hist_Infr_train_postLrn + Zassem_hist_Infr_test_postLrn)[:-1])[::-1]
																	#
																elif sortBy=='DontSort':
																	sortCells_byActivity 	= np.arange(N)
																	sortCAs_byActivity 		= np.arange(M)
																	#
																# elif sortBy=='Infer_allSWs':
																# 	sortCells_byActivity 	= np.argsort( YcellInf_hist_allSWs[:-1])[::-1]
																# 	sortCAs_byActivity 	= np.argsort( (ZassemInf_hist_allSWs)[:-1])[::-1]
																# 	#
																# elif sortBy=='allSWs':
																# 	sortCells_byActivity 	= np.argsort( Ycell_hist_allSWs[:-1])[::-1]
																# 	sortCAs_byActivity 	= np.argsort( (Zassem_hist_allSWs)[:-1])[::-1]
																# elif sortBy=='Samp':
																# 	sortCells_byActivity 	= np.argsort( (Ycell_hist_Samp_train + Ycell_hist_Samp_test)[:-1])[::-1]
																# 	sortCAs_byActivity 		= np.argsort( (Zassem_hist_Samp_train + Zassem_hist_Samp_test)[:-1])[::-1]
																# 	#
																# elif sortBy=='Infer_during':
																# 	sortCells_byActivity 	= np.argsort( (Ycell_hist_Infr_train + Ycell_hist_Infr_test)[:-1])[::-1]
																# 	sortCAs_byActivity 		= np.argsort( (Zassem_hist_Infr_train + Zassem_hist_Infr_test)[:-1])[::-1]
																	#
																else:
																	print('Dont understand how to sort Cells and CAs by their activity. Look at sortBy variable.')



																if True:
																	TH_bounds = np.array([0.3, 0.5, 0.7])


																	# if sortBy == 'Full':
																	# 	sortCAs_byActivity 	  	= np.argsort(Zassem_hist_Full_train[:-1])[::-1] 		# sorting by full training data.
																	# 	sortCells_byActivity 	= np.argsort(Ycell_hist_Full_train[:-1])[::-1]
																	# 	#
																	# elif sortBy == 'Samp':
																	# 	sortCAs_byActivity 	  	= np.argsort(Zassem_hist_Samp_train[:-1])[::-1] 		# sorting by sampled training data.
																	# 	sortCells_byActivity 	= np.argsort(Ycell_hist_Samp_train[:-1])[::-1]
																	# 	#
																	# elif sortBy == 'infer':
																	# 	sortCAs_byActivity 	  	= np.argsort(Zassem_hist_Infr_train[:-1])[::-1] 		# sorting by Inference from sampled training data
																	# 	sortCells_byActivity 	= np.argsort(Ycell_hist_Infr_train[:-1])[::-1]
																	# 	#
																	# elif sortBy == 'postLrn':
																	# 	sortCAs_byActivity 	  	= np.argsort(Zassem_hist_Infr_train_postLrn[:-1])[::-1] # sorting by Inference from training data post EM learning
																	# 	sortCells_byActivity 	= np.argsort(Ycell_hist_Infr_train_postLrn[:-1])[::-1]
																	# 	#
																	# else:
																	# 	print('Sorting CA and Cells by their activity in ', sortBy, 'doesnt make any sense. Look into this. Breaking.')
																	# 	continue
																	#
																	# Ycell_hist_Full_train_sort  		= Ycell_hist_Full_train[sortCells_byActivity]
																	# Ycell_hist_Samp_train_sort			= Ycell_hist_Samp_train[sortCells_byActivity]
																	# Ycell_hist_Infr_train_sort			= Ycell_hist_Infr_train[sortCells_byActivity]
																	# Ycell_hist_Infr_train_postLrn_sort	= Ycell_hist_Infr_train_postLrn[sortCells_byActivity]
																	# #
																	# Ycell_hist_Full_test_sort  			= Ycell_hist_Full_test[sortCells_byActivity]
																	# Ycell_hist_Samp_test_sort			= Ycell_hist_Samp_test[sortCells_byActivity]
																	# Ycell_hist_Infr_test_sort			= Ycell_hist_Infr_test[sortCells_byActivity]
																	# Ycell_hist_Infr_test_postLrn_sort	= Ycell_hist_Infr_test_postLrn[sortCells_byActivity]
																	# #
																	# Zassem_hist_Full_train_sort 		= Zassem_hist_Full_train[sortCAs_byActivity]
																	# Zassem_hist_Samp_train_sort			= Zassem_hist_Samp_train[sortCAs_byActivity]
																	# Zassem_hist_Infr_train_sort			= Zassem_hist_Infr_train[sortCAs_byActivity]
																	# Zassem_hist_Infr_train_postLrn_sort	= Zassem_hist_Infr_train_postLrn[sortCAs_byActivity]
																	# #
																	# Zassem_hist_Full_test_sort  		= Zassem_hist_Full_test[sortCAs_byActivity]
																	# Zassem_hist_Samp_test_sort			= Zassem_hist_Samp_test[sortCAs_byActivity]
																	# Zassem_hist_Infr_test_sort			= Zassem_hist_Infr_test[sortCAs_byActivity]
																	# Zassem_hist_Infr_test_postLrn_sort	= Zassem_hist_Infr_test_postLrn[sortCAs_byActivity]
																	#
																	PiInv 	 	= ( 1-rc.sig(rip) )[sortCells_byActivity]
																	PiaInv 	  	= ( 1-rc.sig(riap) )[sortCells_byActivity,:]  # [np.ix_(sortCells_byActivity,sortCAs_byActivity)]
																	#
																	maxPiQ 		= np.array([ rc.sig(qp),	PiInv.max() ]).max()
																	#	
																	numCAsUpp		= ( PiaInv>TH_bounds[0] ).sum(axis=1)
																	numCAsMid		= ( PiaInv>TH_bounds[1] ).sum(axis=1)
																	numCAsLow		= ( PiaInv>TH_bounds[2] ).sum(axis=1)
																	numCellsUpp		= ( PiaInv>TH_bounds[0] ).sum(axis=0)
																	numCellsMid		= ( PiaInv>TH_bounds[1] ).sum(axis=0)
																	numCellsLow 	= ( PiaInv>TH_bounds[2] ).sum(axis=0)
																	#
																	maxNumCells 	= np.zeros(3)
																	maxNumCells[0] 	= numCellsLow.max()
																	maxNumCells[1] 	= numCellsMid.max()
																	maxNumCells[2] 	= numCellsUpp.max()
																	maxNumCells 	= maxNumCells.astype(int)
																	#
																	maxNumCAs 		= np.zeros(3)
																	maxNumCAs[0]	= numCAsLow.max()
																	maxNumCAs[1]	= numCAsMid.max()
																	maxNumCAs[2]	= numCAsUpp.max()
																	maxNumCAs 		= maxNumCAs.astype(int)
																	#
																	CAsAndCellsUpp 	= np.where(PiaInv>TH_bounds[0])
																	CAsAndCellsMid 	= np.where(PiaInv>TH_bounds[1])
																	CAsAndCellsLow 	= np.where(PiaInv>TH_bounds[2])
																	#
																	whichCellsUpp = list()
																	whichCellsMid = list()
																	whichCellsLow = list()
																	for i in range(M):
																		whichCellsUpp.append( CAsAndCellsUpp[0][CAsAndCellsUpp[1]==i] )
																		whichCellsMid.append( CAsAndCellsMid[0][CAsAndCellsMid[1]==i] )
																		whichCellsLow.append( CAsAndCellsLow[0][CAsAndCellsLow[1]==i] ) # set up a list of empty lists to contain spike times for each za.
																	#
																	whichCAsUpp = list()
																	whichCAsMid = list()
																	whichCAsLow = list()
																	for i in range(N):
																		whichCAsUpp.append( CAsAndCellsUpp[1][CAsAndCellsUpp[0]==i] )
																		whichCAsMid.append( CAsAndCellsMid[1][CAsAndCellsMid[0]==i] )
																		whichCAsLow.append( CAsAndCellsLow[1][CAsAndCellsLow[0]==i] ) # set up a list of empty lists to contain spike times for each za.
																	#


																	print('Dont I want to sort CA_ovl and CA_ovl_model by CA activity? I think so!  Anything else? Also, CA coactivity?  Cell coactivitiy?')
																	print('Do something with: pj_inferred_train_postLrn, pj_inferred_test_postLrn, pjoint_train, pjoint_test. Both statistics and cross-validation.') 



																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																# (Stats 7). Cumulative Sum across samples of Activity of each Cell Assembly
																# 		Step thru Zassem_hist and record sample numbers when each CA is inferred. 
																#
																#
																if False:
																	print('compute Inference temporal for training data')
																	t0 = time.time()
																	samps = len(smp_train)
																	Z_inf_temp_train = np.zeros( (samps+1,M+1) )
																	for samp in range(samps):
																		Z_inf_temp_train[samp+1] = Z_inf_temp_train[samp]
																		Z_inf_temp_train[samp+1, list(Z_inferredM_train[samp])] += 1
																	t1 = time.time()
																	print('Done w/ inference temporal for training data : time = ',t1-t0)
																	#
																	print('compute Inference temporal for test data')
																	t0 = time.time()
																	samps = len(smp_test)
																	Z_inf_temp_test = np.zeros( (samps+1,M+1) )
																	for samp in range(samps):
																		Z_inf_temp_test[samp+1] = Z_inf_temp_test[samp]
																		Z_inf_temp_test[samp+1, list(Z_inferredM_test[samp])] += 1
																	t1 = time.time()
																	print('Done w/ inference temporal for test data : time = ',t1-t0)








																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																#
																# (Stats 8). Collect statistics and info about which model / data was run into a CSV file.
																#
																if flg_write_CSV_stats:
																	#
																	ovl = CA_ovl[np.triu(CA_ovl,1)>0]
																	data_stats_CSV.update( [ 
																					('%|CAs|=0',[ (numCellsMid==0).sum()/M ]), \
																					('%|CAs|=1',[ (numCellsMid==1).sum()/M ]), \
																					('%|CAs|=2',[ (numCellsMid==2).sum()/M ]), \
																					('%|CAs|>2&<6', np.bitwise_and( numCellsMid>2,numCellsMid<6  ).sum() /M ), \
																					('%|CAs|>=6',[ (numCellsMid>=6).sum()/M ]), \
																					('|CAs| max',[ (numCellsMid).max() ]), \
																					('%Pi>0.1',[ ((1-Pi)>0.1).sum()/N ]), \
																					('Q',[ Q ]), \
																					('mean CA ovl',[np.mean(ovl)]), \
																					('std CA ovl',[np.std(ovl)]), \
																					('max CA ovl',[np.max(ovl)]) ] )








																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																#
																# (Plot 1). Do multiple comparison of stats to see if different data are biased. {Full vs. Sampled vs. Inferred vs. Train vs. Test}
																#
																if flg_plt_dataGen_andSampling_hist:
																	#
																	# try:
																	if True:
																		print( 'Plotting Statistics for data generation, sampling, inference, for test and train data' )
																		t0 = time.time()
																		#
																		# # (Plot A). Sampled vs Full Dataset.
																		# if flg_compute_StatsPostLrn and flg_compute_StatsDuringEM:
																		# 	nameIdentifiers = ['Full','Sampled','Train'] # [Diff1, Diff2, Same]Cell_coactivity_Full_test
																		# 	print('A. Compute stats on ', nameIdentifiers)
																		# 	pf.hist_dataGen_andSampling_statistics( \
																		# 		Ycell_hist_Full_train, Zassem_hist_Full_train, nY_Full_train, nZ_Full_train, CA_coactivity_Full_train, Cell_coactivity_Full_train, len(Z_train), \
																		# 		Ycell_hist_Samp_train, Zassem_hist_Samp_train, nY_Samp_train, nZ_Samp_train, CA_coactivity_Samp_train, Cell_coactivity_Samp_train, len(Z_smp_train_list), \
																		# 		CA_ovl, ria, riap, N, M, Kmax, rand, plt_save_dir, fname_tag, nameIdentifiers)
																		# #
																		# # (Plot B). Inferred vs Sampled.
																		# if flg_compute_StatsDuringEM:
																		# 	nameIdentifiers = ['Sampled','Inferred','Train'] # [Diff1, Diff2, Same]
																		# 	print('B. Compute stats on ', nameIdentifiers)
																		# 	pf.hist_dataGen_andSampling_statistics( \
																		# 		Ycell_hist_Samp_train, Zassem_hist_Samp_train, nY_Samp_train, nZ_Samp_train, CA_coactivity_Samp_train, Cell_coactivity_Samp_train, len(Z_smp_train_list), \
																		# 		Ycell_hist_Infr_train, Zassem_hist_Infr_train, nY_Infr_train, nZ_Infr_train, CA_coactivity_Infr_train, Cell_coactivity_Infr_train, len(Z_inferred_train), \
																		# 		CA_ovl, ria, riap, N, M, Kmax, rand, plt_save_dir, fname_tag, nameIdentifiers)
																		#
																		# (Plot C). Train vs. Test
																		if flg_compute_StatsPostLrn and flg_plot_train_data and flg_plot_test_data:
																			nameIdentifiers = ['Train','Test','Full'] # [Diff1, Diff2, Same]
																			print('C. Compute stats on ', nameIdentifiers)
																			pf.hist_dataGen_andSampling_statistics( Ycell_hist_Full_train, Zassem_hist_Full_train, nY_Full_train, nZ_Full_train, CA_coactivity_Full_train, \
																				Cell_coactivity_Full_train, len(Z_train), Ycell_hist_Full_test,  Zassem_hist_Full_test,  nY_Full_test,  nZ_Full_test,  CA_coactivity_Full_test, \
																				Cell_coactivity_Full_test,  len(Z_test), CA_ovl, ria, riap, N, M, Kmax, rand, plt_save_dir, fname_tag, nameIdentifiers)






																		# 
																		# # # (Plot D). During-Learning vs Post-Learning both Inferred and training
																		# if flg_compute_StatsPostLrn and flg_compute_StatsDuringEM:
																		# 	nameIdentifiers = ['While-EM','Post-EM','Inferred Train'] # [Diff1, Diff2, Same]
																		# 	print('D. Compute stats on ', nameIdentifiers)
																		# 	pf.hist_dataGen_andSampling_statistics( \
																		# 		Ycell_hist_Infr_train, Zassem_hist_Infr_train, nY_Infr_train, nZ_Infr_train, CA_coactivity_Infr_train, Cell_coactivity_Infr_train, len(Z_inferred_train), \
																		# 		Ycell_hist_Infr_train_postLrn, Zassem_hist_Infr_train_postLrn, nY_Infr_train_postLrn, nZ_Infr_train_postLrn, CA_coactivity_Infr_train_postLrn, Cell_coactivity_Infr_train_postLrn, len(Z_train), \
																		# 		CA_ovl, ria, riap, N, M, Kmax, rand, plt_save_dir, fname_tag, nameIdentifiers)
																		#
																		# # (Plot E). Inferred post training vs True Full training.
																		if flg_compute_StatsPostLrn and flg_plot_train_data:
																			nameIdentifiers = ['Inf PL','Full','Train'] # [Diff1, Diff2, Same]
																			print('E. Compute stats on ', nameIdentifiers)
																			pf.hist_dataGen_andSampling_statistics( Ycell_hist_Infr_train_postLrn, Zassem_hist_Infr_train_postLrn, nY_Infr_train_postLrn, nZ_Infr_train_postLrn, \
																				CA_coactivity_Infr_train_postLrn, Cell_coactivity_Infr_train_postLrn, len(Z_train), Ycell_hist_Full_train, Zassem_hist_Full_train, nY_Full_train, \
																				nZ_Full_train, CA_coactivity_Full_train, Cell_coactivity_Full_train, len(Z_train), CA_ovl, ria, riap, N, M, Kmax, rand, plt_save_dir, fname_tag, nameIdentifiers)
																		#
																		# # (Plot F). Inferred post test vs True Full test.
																		if flg_compute_StatsPostLrn and flg_plot_test_data:
																			nameIdentifiers = ['Inf PL','Full','Test'] # [Diff1, Diff2, Same]
																			print('F. Compute stats on ', nameIdentifiers)
																			pf.hist_dataGen_andSampling_statistics( Ycell_hist_Infr_test_postLrn, Zassem_hist_Infr_test_postLrn, nY_Infr_test_postLrn, nZ_Infr_test_postLrn, \
																				CA_coactivity_Infr_test_postLrn, Cell_coactivity_Infr_test_postLrn, len(Z_test), Ycell_hist_Full_test, Zassem_hist_Full_test, nY_Full_test, \
																				nZ_Full_test, CA_coactivity_Full_test, Cell_coactivity_Full_test, len(Z_test), CA_ovl, ria, riap, N, M, Kmax, rand, plt_save_dir, fname_tag, nameIdentifiers)
																		#
																		t1 = time.time()
																		print('Done w/ Plotting Statistics for data generation, sampling, inference, for test and train data : time = ',t1-t0)
																	# except:
																	# 	print('problem with plt_dataGen_andSampling_hist_flg')




																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																# 				MAYBE SAME INFO AS IN DATAGEN_HISTOGRAMS.	
																# (Plot XX). Plot single cell and pairwise correlations.
																#
																print('NOTE: This is a nice plot and I should put it in pf. and also save it. To Do. flg_plot_pyiEq1_gvnZ_OPR.')
																flg_plot_pyiEq1_gvnZ_OPR = False
																if flg_plot_pyiEq1_gvnZ_OPR:
																	# Plot of pairwise correlations 
																	# NOTE OPR = Cell_coactivity_...
																	# Y_single = Ycell_hist_...
																	OPR = np.zeros( (N,N) )
																	for s in range(len(Y_list)):
																		OPR += np.outer(Y_list[s],Y_list[s])
																	#compute_dataGen_Histograms
																	Y_single = np.sqrt(OPR.diagonal())
																	i = np.argsort(Y_single)[::-1]
																	OPR = OPR / np.outer( Y_single, Y_single ) # normalize by single cell activity.
																	np.fill_diagonal(OPR,0)	
																	#
																	f,ax = plt.subplots(1,2)
																	im = ax[1].imshow( OPR[np.ix_(i,i)] )
																	ax[1].set_xlabel('Cell id ($y_i$)')
																	ax[1].set_title('Outer Product of $p(y_i=1|z)$. Normalized by $\sum_{SWs} p(y_i=1|z)$. \n (i.e., Pairwise correlation in inferred ys.)')
																	cax = f.add_axes([0.92, 0.2, 0.02, 0.6])
																	f.colorbar(im, cax=cax)
																	#plt.colorbar()
																	#
																	ax[0].plot(Y_single[i])
																	ax[0].set_aspect('auto','box')
																	ax[0].set_xlabel('Cell id ($y_i$)')
																	ax[0].set_ylabel( str('$\sum_{SWs} p(y_i=1|z)$ out of '+str(len(Y_list))+' SWs') )
																	ax[0].set_title('Single Cell Activity in Inferred ys, \n i.e. $\sum_{SWs} p(y_i=1|z)$. sorted.')
																	plt.show()	




																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																# #
																# # (Plot 2). Plot Mean Squared Error of Parameters (between Learned and True) during EM iterations.
																# #
																# if flg_plt_EMlearning_MSE and MSE:
																# 	print( 'Plotting Mean Squared Error vs. EM iteration.' )
																# 	print('THIS SHOULD BE OBSELETE SOON...')
																# 	t0 = time.time()
																# 	if True:
																# 	# try:
																# 		pf.plot_params_MSE_during_learning(q_MSE, ri_MSE, ria_MSE, samps2snapshot_4Pia, num_EM_samps, \
																# 			N, M,learning_rate,lRateScale, params_init, params_init_str, rand, plt_save_dir, fname_tag)
																# 	# except:
																# 	# 	print('problem with plt_EMlearning_MSE_flg')
																# 	t1 = time.time()
																# 	print('Done w/ Plotting Mean Squared Error vs. EM iteration : time = ',t1-t0)


																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																#
																# 					REPLACEMENT !!
																# (Plot 2). Plot Absolute Value and Signed Error of Parameters (between Learned and True) during EM iterations.
																#
																
																if flg_plt_EMlearning_Err:
																	print( 'Plotting Parameter Errors vs. EM iteration.' )
																	t0 = time.time()
																	if True:
																	# try:
																		pf.plot_params_Err_during_learning(Q_SE, Pi_SE, Pi_AE, Pia_AE, PiaOn_SE, PiaOn_AE, PiaOff_SE, PiaOff_AE, samps2snapshot_4Pia, \
																			num_EM_samps, N, M, learning_rate, lRateScale, params_init, params_init_str, rand, plt_save_dir, fname_tag)
																		
																	# except:
																	# 	print('problem with plt_EMlearning_MSE_flg')
																	t1 = time.time()
																	print('Done w/ Plotting Parameter Errors vs. EM iteration : time = ',t1-t0)



																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																#
																# (Plot 3). Plot derivatives of model parameters during EM iterations.
																#
																if flg_plt_EMlearning_derivs:
																	print( 'Plotting Derivatives of Model Parameters vs. EM iteration.' )
																	t0 = time.time()
																	if True:
																	# try:
																		pf.plot_params_derivs_during_learning(q_deriv, ri_deriv, ria_deriv, num_EM_samps, N, M, learning_rate, \
																					lRateScale, ds_fctr_snapshots, params_init, params_init_str, rand, plt_save_dir, fname_tag)
																	# except:
																	# 	print('problem with plt_EMlearning_derivs_flg')
																	t1 = time.time()
																	print('Done w/ Plotting Derivatives of Model Parameters vs. EM iteration. : time = ',t1-t0)	






																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																#
																# (Plot 4). Plot model parameters of true model, learned model and initialized model. 
																#
																if flg_plt_EMlearning_params_init_n_final:
																	print( 'Plotting Model Learned, Initialized and Ground Truth' )
																	t0 = time.time()
																	# try:
																	if True:

																		# MAKE A FUNCTION?
																		#
																		# Based on completeness of Model, rename (or reindex) CA's so that they match up as best as possible.
																		# 		outputs: indGT_match, indMod_match
																		# 		inputs : ind_matchMod2GT, ind_matchGT2Mod, M_mod, M
																		if M_mod<=M: 													# undercomplete or complete
																			indGT_match 	= list(ind_matchMod2GT[0])
																			indMod_match 	= list(ind_matchMod2GT[1])
																		else: 															# overcomplete model.
																			indGT_match 	= list(ind_matchGT2Mod[1])
																			indMod_match 	= list(ind_matchGT2Mod[0])

																		# indxGT 	= [indGT_match.index(i) for i in range(M)]
																		# indxMod = [indMod_match.index(i) for i in range(M_mod)]

																		pf.plot_params_init_n_learned(q, ri, ria[:,indGT_match], qp, rip, riap[:,indMod_match], q_init, ri_init, ria_init[:,indMod_match], \
																			zInferSampled_train_postLrn[indMod_match], num_EM_samps, N, M, M_mod, params_init, params_init_str, rand, plt_save_dir, fname_tag) 
																			# learning_rate, lRateScale, 
																	# except:
																	# 	print('problem with plt_EMlearning_params_init_n_final')
																	t1 = time.time()
																	print('Done w/ Plotting Model Learned, Initialized and Ground Truth : time = ',t1-t0)

																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																#
																# (Plot 5). Plot Inference performance stats over whole model learning.  (CA & cell confusion matrices, activation & co-act stats)
																#
																# Plots: - Can do this individually. for Test and Train data sets.
																# 1. Cells inferred confusion matrix.
																# 2. CAs inferred confusion matrix.
																# 3. CAs truth similarity, overlap, dot-prod.
																# 4. CAs coactivity in sampled data (not full, not inferred)
																# 5. CA activity (single) in full, sampled & inferred data.
																# 6. Number of CAs active in Sampled and Inferred Z's
																#
																#
																# Maybe want to use:
																# Zassem_hist_Full_test Zassem_hist_Samp_test Zassem_hist_Infr_test and Zassem_hist_Samp_test_sort in ax5 !!
																#
																#
																if flg_plt_EMinfer_performance_stats:
																	print( 'Plotting Inference performance statistics for whole EM algorithm.' )
																	t0 = time.time()
																	# try:
																	if True:
																		# if flg_compute_StatsDuringEM:
																		# 	# For Training Data.
																		# 	pltSpecifierTag = 'train'
																		# 	print('A. Plot inf stats for ',pltSpecifierTag, 'during EM')
																		# 	pf.plot_CA_inference_performance(inferCA_Confusion_train, inferCell_Confusion_train, CA_ovl, CA_coactivity_Samp_train, \
																		# 				zInferSampledRaw_train, zInferSampledT_train, Zassem_hist_Samp_train, yInferSampled_train, \
																		# 				Kinf_train, KinfDiff_train, N, M, M_mod, translate_Tru2LrnShuff[0],  translate_Tru2Tru, \
																		# 				translate_Lrn2TruShuff[0], translate_Lrn2Lrn, num_EM_samps, params_init, params_init_str, rand, \
																		# 				plt_save_dir, fname_tag, pltSpecifierTag)
																		# 	#
																		# 	# For Test Data
																		# 	pltSpecifierTag = 'test'
																		# 	print('B. Plot inf stats for ',pltSpecifierTag, 'during EM')
																		# 	pf.plot_CA_inference_performance(inferCA_Confusion_test, inferCell_Confusion_test, CA_ovl, CA_coactivity_Samp_test, \
																		# 				zInferSampledRaw_test, zInferSampledT_test, Zassem_hist_Samp_test, yInferSampled_test, \
																		# 				Kinf_test, KinfDiff_test, N, M, M_mod, translate_Tru2LrnShuff[0], translate_Tru2Tru, \
																		# 				translate_Lrn2TruShuff[0], translate_Lrn2Lrn, num_EM_samps, params_init, params_init_str, rand, \
																		# 				plt_save_dir, fname_tag, pltSpecifierTag)
																		#
																		if flg_compute_StatsPostLrn:
																			# For Training Data Post Learning.
																			pltSpecifierTag = 'train_postLrn'
																			print('C. Plot inf stats for ',pltSpecifierTag)
																			#
																			inferCell_Confusion_train_postLrn = Cell_coactivity_Infr_train_postLrn
																			yInferSampled_train_postLrn = Ycell_hist_Infr_train_postLrn
																			#
																			pf.plot_CA_inference_performance(inferCA_Confusion_train_postLrn, inferCell_Confusion_train_postLrn, CA_ovl, CA_coactivity_Infr_train_postLrn, \
																						zInferSampled_train_postLrn, zInferSampledT_train_postLrn,  Zassem_hist_Infr_train_postLrn, yInferSampled_train_postLrn, \
																						Kinf_train_postLrn, KinfDiff_train_postLrn, N, M, M_mod, num_EM_samps, params_init, params_init_str, rand, plt_save_dir, fname_tag, pltSpecifierTag)
																			#
																			# For Test Data Post Learning
																			if flg_plot_test_data:
																				pltSpecifierTag = 'test_postLrn'
																				print('D. Plot inf stats for ',pltSpecifierTag)
																				#
																				inferCell_Confusion_test_postLrn = Cell_coactivity_Infr_train_postLrn
																				yInferSampled_test_postLrn = Ycell_hist_Infr_train_postLrn
																				#
																				pf.plot_CA_inference_performance(inferCA_Confusion_test_postLrn, inferCell_Confusion_test_postLrn, CA_ovl, CA_coactivity_Infr_test_postLrn, \
																					zInferSampled_test_postLrn, zInferSampledT_test_postLrn, Zassem_hist_Infr_test_postLrn, yInferSampled_test_postLrn, \
																					Kinf_test_postLrn, KinfDiff_test_postLrn, N, M, M_mod, num_EM_samps, params_init, params_init_str, rand, plt_save_dir, fname_tag, pltSpecifierTag)
																	# except:
																	# 	print('problem with plt_EMinfer_performance_stats')
																	t1 = time.time()
																	print('Done w/ Plotting Inference performance statistics for whole EM algorithm. : time = ',t1-t0)	







																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																#
																# (Plot 6). Plots to investigate translation / permutation of Cell Assemblies between Learned ones and Ground Truth ones
																#
																if flg_plt_matchModels_cosSim:
																	print( 'Plotting Translation/Conversion of Cell Assembly ordering from Learned model to Ground Truth Model.' )
																	t0 = time.time()
																	#
																	Pia = rc.sig(ria)
																	Piap = rc.sig(riap)
																	#
																	# try:
																	if False:	# Dont need to do both. Other is better. Permutes learned to fit ground truth
																		pf.visualize_matchModels_cosSim( A=1-Piap, Atag='Learned Piap', B=1-Pia, Btag='GndTruth Pia', ind=ind_matchMod2GT, \
																			cos_sim=cosSim_matchMod2GT, len_dif=lenDif_matchMod2GT, cosSimMat=cosSimMat_matchMod2GT, lenDifMat=lenDifMat_matchMod2GT, \
																			numSamps=num_EM_samps, r=rand, plt_save_dir=plt_save_dir, fname_tag=fname_tag )
																	#
																	if True:
																		pf.visualize_matchModels_cosSim( A=1-Pia, Atag='GndTruth Pia', B=1-Piap, Btag='Learned Piap', ind=ind_matchGT2Mod, \
																			cos_sim=cosSim_matchGT2Mod, len_dif=lenDif_matchGT2Mod, cosSimMat=cosSimMat_matchGT2Mod, lenDifMat=lenDifMat_matchGT2Mod, \
																			numSamps=num_EM_samps, r=rand, plt_save_dir=plt_save_dir, fname_tag=fname_tag )
																	# except:
																	# 	print('problem with plt_translation_Lrn2Tru')		
																	t1 = time.time()
																	print('Done w/ Plotting Translation/Conversion of Cell Assembly ordering from Learned model to Ground Truth Model. : time = ',t1-t0)	
																							




																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																#
																# (Plot 7). Plot tracking correctly and incorrectly inferred cell assemblies vs. EM algorithm iteration.
																# 			QUANTIFY IF INFERENCE IS IMPROVING WITH TIME (AND LEARNING) 
																#
																if flg_plt_temporal_EM_inference_analysis:
																	print( 'Plotting EM inference statistics as a function of time / EM iteration.' )
																	t0 = time.time()	
																	# try:
																	if True:
																		# For Train Data
																		pf.plot_CA_inference_temporal_performance(Z_inferredM_train, Z_trainM, np.arange(len(Z_trainM)-1), \
																				M_mod, M, num_SWs, params_init, params_init_str, rand, plt_save_dir, fname_tag)
																	
																	# except:
																	# 	print('problem with plt_temporal_EM_inference_analysis')
																	t1 = time.time()
																	print('Done w/ Plotting EM inference statistics as a function of time / EM iteration. : time = ',t1-t0)	


																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																#
																# (Plot 8). Sanity Check histograms of Sampling spike words for training and test.
																#
																if flg_plt_dataGen_Samp_sanityCheck:
																	print( 'Plotting Sanity Check histograms of Sampling spike words for training and test.' )
																	t0 = time.time()
																	if True:
																	# try:
																		# (Plot 1D & 1E). Plot which Spike words are sampled and stats on {Cells per CA} and {CAs per Cell}
																		if flg_plot_train_data:
																			pf.hist_SWsampling4learning_stats(smp_train, len(Z_train), ria, ri, C, Cmin, Cmax, rand, plt_save_dir, fname_tag, 'train' )
																		#
																		if flg_plot_test_data:
																			pf.hist_SWsampling4learning_stats(smp_test,  len(Z_test),  ria, ri, C, Cmin, Cmax, rand, plt_save_dir, fname_tag, 'test'  )
																		#
																	# except:
																	# 	print('problem with plt_dataGen_Samp_sanityCheck_flg')
																	t1 = time.time()
																	print('Done w/ Plotting Sanity Check histograms of Sampling spike words for training and test : time = ',t1-t0)
																	

																	

																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																#
																# (Plot 9). Plot Cross validation - For training and test, plot pjoint vs EM iteration #.
																#
																if flg_plt_crossValidation_Joint:
																	print( 'Plotting Cross validation.' )
																	t0 = time.time()

																	kerns = [1] #100, 500] 

																	# OPTIONS: pjoint_zHyp_trainUnBias, pjoint_zTruUnBias, pjoint_zTru, pjoint_train, pjoint_test
																	pj_labels = ['train','test','truth']
																	# try:
																	# 	pJoints = np.vstack( [ pjoint_zHyp_trainUnBias, pjoint_test, pjoint_zTruUnBias] )
																	# 	pf.plot_xValidation(pJoints, pj_labels, plt_save_dir, str('xValJoint_'+fname_tag), kerns)
																	# except:
																	if True:
																		kerns = [1] #10, 50] 
																		pJoints = np.vstack( [ pj_zTrain_Batch[:,0], pj_zTest_Batch[:,0], pj_zTru_Batch[:,0]] )
																		pf.plot_xValidation(pJoints, pj_labels, plt_save_dir, str('xValJoint_'+fname_tag), kerns)

																	t1 = time.time()
																	print('Done w/ Cross validation. : time = ',t1-t0)




																	 # 'pj_zHyp_train', 'pj_zHyp_test', 'pj_zTru_Batch', 'pj_zTrain_Batch', 'pj_zTest_Batch', 
																	 # 'cond_zHyp_train', 'cond_zHyp_test', 'cond_zTru_Batch', 'cond_zTrain_Batch', 'cond_zTest_Batch'




																if flg_plt_crossValidation_Cond: # for conditional probabilities, not joints.
																	print( 'Plotting Cross validation.' )
																	t0 = time.time()

																	kerns = [1] #100, 500] 

																	# OPTIONS: pjoint_zHyp_trainUnBias, pjoint_zTruUnBias, pjoint_zTru, pjoint_train, pjoint_test
																	pc_labels = ['train','test','truth']  #, 'zTru train', 'zHyp test', 'zTru test']
																	# try:
																	# 	pConds = np.vstack( [cond_zHyp_trainUnBias, cond_zHyp_test, cond_zTru] ) #, cond_zTru_train, cond_zHyp_test, cond_zTru_test] )
																	# 	pf.plot_xValidation(pConds, pc_labels, plt_save_dir, str('xValCond_'+fname_tag), kerns)
																	# except:
																	if True:
																		kerns = [1] #10, 50] 
																		pConds = np.vstack( [cond_zTrain_Batch[:,0], cond_zTest_Batch[:,0], cond_zTru_Batch[:,0]] ) #, cond_zTru_train, cond_zHyp_test, cond_zTru_test] )
																		pf.plot_xValidation(pConds, pc_labels, plt_save_dir, str('xValCond_'+fname_tag), kerns)
																		
																	t1 = time.time()
																	print('Done w/ Cross validation. : time = ',t1-t0)




				

																# # # # # # # # # # # # # # # # # 
																if flg_Pia_snapshots_gif:
																	#
																	TH_bounds = np.array([0.5, 0.7]) # upper and lower bounds for threshold to count up CperA and AperC
																	#
																	# yInferSampled_traintrans = yInferSampled_train[sortCells_byActivity]
																	# Ycell_hist_SampsTrans = (Ycell_hist_Samp_train + Ycell_hist_Samp_test)[sortCells_byActivity]

																	numSnaps 		= ria_snapshots.shape[0]
																	maxPiQ 			= np.array([ rc.sig(q_snapshots).max(),	(1-rc.sig(ri_snapshots)).max() ]).max()
																	#
																	maxNumCells 	= np.zeros(2)
																	maxNumCells[0] 	= ( (1-rc.sig(ria_snapshots))>TH_bounds.max()).sum(axis=1).max()
																	maxNumCells[1] 	= ( (1-rc.sig(ria_snapshots))>TH_bounds.min()).sum(axis=1).max()
																	maxNumCells 	= maxNumCells.astype(int)
																	#
																	maxNumCAs 		= np.zeros(2)
																	maxNumCAs[0]	= ( (1-rc.sig(ria_snapshots))>TH_bounds.max()).sum(axis=2).max()
																	maxNumCAs[1]	= ( (1-rc.sig(ria_snapshots))>TH_bounds.min()).sum(axis=2).max()
																	maxNumCAs 		= maxNumCAs.astype(int)

																	plt_snap_dir = str( plt_save_dir + 'Snapshots/' )
																	if not os.path.exists( plt_snap_dir ):
																		os.makedirs( plt_snap_dir )		


																	print('Number of snapshots are ',numSnaps)	
																	snaps = range(0,numSnaps) #range(ria_snapshots.shape[0])
																	for i in snaps:

																		# Stats on active cells and cell assemblies inferred during EM algorithm
																		print(i,' compute Inference statistics')
																		t0 = time.time()
																		sampAtSnap = int( i*num_EM_samps/(numSnaps-1) )
																		Ycell_hist_InferSnap, Zassem_hist_InferSnap, nY_InferSnap, nZ_InferSnap, \
																			CA_coactivity_InferSnap, Cell_coactivity_InferSnap = rc.compute_dataGen_Histograms( \
																			pyiEq1_gvnZ_train[:sampAtSnap], Z_inferred_train[:sampAtSnap], M, N) # sampAtSnap
																		t1 = time.time()
																		print('Done w/ inference stats : time = ',t1-t0) # Fast enough: ~10 seconds
																		#
																		Ycell_hist_InferSnap  = Ycell_hist_InferSnap[sortCells_byActivity]
																		Zassem_hist_InferSnap = Zassem_hist_InferSnap # [sortCAs_byActivity]
																		#
																		QSnap 			= rc.sig(q_snapshots[i])
																		PiSnap 	 	  	= ( 1-rc.sig(ri_snapshots[i]) )[sortCells_byActivity]
																		PiaSnap 	  	= ( 1-rc.sig(ria_snapshots[i]) )[sortCells_byActivity,:] # [np.ix_(sortCells_byActivity,sortCAs_byActivity)]
																		numCAsSnapUB	= ( PiaSnap>TH_bounds.min()).sum(axis=1)
																		numCAsSnapLB	= ( PiaSnap>TH_bounds.max()).sum(axis=1)
																		numCellsSnapUB	= ( PiaSnap>TH_bounds.min()).sum(axis=0)
																		numCellsSnapLB 	= ( PiaSnap>TH_bounds.max()).sum(axis=0)

																		plt_title = str('Learned Model ' + ModelType + ' Params w/ LR =' + str(learning_rate) + \
																		'LRsc =' + str(lRateScale) + ' :: ' + str(num_SWs) + ' SW data & ' + str(sampAtSnap) + ' EM samples' )

																		plt_save_tag = str( fname_EMlrn[:-4].replace('EM_model_data_','SnapShotsReal_') + '_snap' + str(i) )
																		#
																		pf.plot_learned_model(PiSnap, PiaSnap, QSnap, numCAsSnapUB, numCAsSnapLB, numCellsSnapUB, numCellsSnapLB, 
																			Zassem_hist_InferSnap, Ycell_hist_InferSnap, Ycell_hist_Full_train, 
																			TH_bounds, maxNumCells, maxNumCAs, maxPiQ, nY_Full_train, nY_InferSnap, nZ_InferSnap, \
																			sampAtSnap, num_EM_samps, plt_snap_dir, plt_save_tag, plt_title)
																			# replaced Ycell_hist_SampsTrans with Ycell_hist_Full_train, nY_Samp_train with nY_Full_train



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

																		# indxGT 	= [indGT_match.index(i) for i in range(M)]
																		# indxMod = [indMod_match.index(i) for i in range(M_mod)]


																		plt_save_tag2 = str( fname_EMlrn[:-4].replace('EM_model_data_','SnaphotsSynth_') + '_snap' + str(i) )
																		pf.plot_params_init_n_learned(q, ri, ria[:,indGT_match], q_snapshots[i], ri_snapshots[i], ria_snapshots[i][:,indMod_match], q_init, ri_init, ria_init[:,indMod_match], \
																										zInferSampled_train_postLrn[indMod_match], num_EM_samps, N, M, M_mod, learning_rate, params_init_str, rand, plt_snap_dir, plt_save_tag2)

																		












																if flg_write_CSV_stats:
																	df = pd.DataFrame(data=data_stats_CSV)
																	df.to_csv(fname_CSV, mode='a', header=(dfh==0))
																	dfh+=1	



																# bins = np.linspace(0, np.ceil( np.max( [ (1-Pia).sum(axis=0).max(), (1-Piap).sum(axis=0).max() ] ) ), 10)
																# CperA_gt = np.histogram( (1-Pia).sum(axis=0), bins=bins )
																# CperA_md = np.histogram( (1-Piap).sum(axis=0), bins=bins )
																# #
																# plt.plot( CperA_gt[1][1:], CperA_gt[0], 'b-', label='Pia' )
																# plt.plot( CperA_md[1][1:], CperA_md[0], 'r-', label='Piap' )
																# #
																# plt.xlabel('$\sum_i P_{ia}$ ')
																# plt.ylabel( str('Counts / '+str(M)+' CAs') )
																# plt.legend()
																# plt.title('Histogram Cells per CA ')
																# #
																# plt.show()
																


																print('SYNTH SEXY TIME !!!!')
																# sexytime


															except:
																print('Something not working!!')
																print( params_init, learning_rate, M, num_SWs, num_EM_samps, M_mod, \
																		 C_noise, sig_init, K, Kmin, Kmax, C, Cmin, Cmax, yLo, yHi, rand )