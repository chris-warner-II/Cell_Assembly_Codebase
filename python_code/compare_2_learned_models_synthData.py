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



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (0). Set up directories and create dirs if necessary.
dirHome, dirScratch = dm.set_dir_tree()
EM_learning_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/Models_learned_EM/')
Infer_postLrn_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/inference_postLrn/')
EM_learnStats_Dir	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/inferStats_from_EM_learning/')
EM_figs_dir 		= str( dirScratch + 'figs/PGM_analysis/EM_Algorithm/')






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (1). Load in npz file and extract data from it.

# Parameters we can loop over.
num_SWs_tot = [1000] #[100000] 				# number of spike words total in the data corpus (training and test.)
num_Cells = [55] 					# Looping over N values
num_CAs_true = [55]					# Looping over M values used to build the model
model_CA_overcompleteness =[1] #[1.5, 1., 0.5]	# how many times more cell assemblies the model assumes than are in true model (1 means complete - M_mod=M, 2 means 2x overcomplete)


# Learning rates for EM algorithm
learning_rates 	= [0.5]#, 0.1] 	 	# Learning rates to loop through
lRateScaleS = [ [1., 0.1, 0.1]]#, [1., 0.1, 1.] ]  # Multiplicative scaling to Pia,Pi,Q learning rates. Can set to zero and take Pi taken out of model essentially.


#
ds_fctr_snapshots 	= 1000 			# Take a sample of model parameters every *ds_fctr* time steps to compute and plot MSE after we determine permutation of learned CA's to True CA's
pct_xVal_train 		= 0.5 			# percentage of spikewords (total data) to use as training data. Set the rest aside for test data for cross validation.
pct_xVal_train_prev	= 0.5			# For resampling previously constructed spike words to check repeatability of parameters in learned models.
num_EM_rands		= 1 			# number of times to randomize samples and rerun EM on same data generated from single synthetic model.
#
# train_2nd_modelS = [True,False] # I am comparing the two models trained on same data so this is a given.


# Plotting flags
flg_plot_indiv_model_pairs 		= True # plot translatePermute figure comparing different pairs of models and comparing individual models to ground truth
flg_directCompareCAs 			= False # plot match between model pairs vs match between each model and GT.
#
flg_plot_our_translate 			= False # Make plots using our greedy heuristic column resorting (CA matching) method
flg_plot_Hungarian_translate	= True # Make plots using optimal Hungarian method for column resorting (CA matching)
figure_file_type = 'png' # 'pdf' for higher resolution or 'png' for smaller file



# Synthetic Model Construction Parameters
Ks 				= [1]#,2] 	# Number of	cell assemblies active, given a Bernoulli distribution ("hotness" of Z-vector)
Kmins 			= [0]#,0] 	# Max & Min number of cell assemblies active 
Kmaxs 			= [4]#,4] 	# 
#
Cs 				= [6]#,2] 	# number of cells participating in cell assemblies. ("hotness" of each row in Pia)
Cmins 			= [2]#,2] 	# Max & Min number of cell active to call it a cell assembly
Cmaxs 			= [6]#,6] 	# 
#
mu_PiaS			= [0.30]#, 0.55]	# Binary values in Pia matrix made continuous by sampling from Gaussian centered at 1-mu_Pia or 0+mu_Pia.
sig_PiaS		= [0.10]#, 0.05]	# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
#
mu_PiS			= [0.04]#, 0.04]	# Binary values in Pi vector made continuous by sampling from Gaussian centered at 1-mu_Pi or 0+mu_Pi.
sig_PiS			= [0.02]#, 0.02] 	# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
bernoulli_Pi	= 1.0   		# Each value in binary Pi vector randomly sampled from bernoulli with p(Pi=1) = bernoulli_Pi.	

#
yLo_Vals 		= [0] 		# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<=yLo.
yHi_Vals 		= [1000] 	# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution.
yMinSWs 		= [1] # [1,2,3]

sample_longSWs_1st = 'Prob' # Options are: {'Dont', 'Prob', 'Hard'}
flg_EgalitarianPriorS = [False]#,True] # True means Perrinet thing and False means Binomial prior on p(za)=Q




# Parameter initializations for EM algorithm to learn model parameters
params_initS 	= ['NoisyConst'] 	# Options: {'True', 'RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise) }
sigQ_init 		= 0.01			# STD on Q initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
sigPi_init 		= 0.05			# STD on Pi initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
sigPia_init 	= 0.05			# STD on Pia initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
Z_hot 			= 5 			# Mean of initialization for Q value (how many 1's expected in binary z-vector)
C_noise_ri 		= 1.0			# Mean of initialization of Pi values (1 means mostly silent) with variability defined by sigPia_init
C_noise_ria 	= 1.0			# Mean of initialization of Pia values (1 means mostly silent) with variability defined by sigPia_init

sig_init 		= np.array([ sigQ_init, sigPi_init, sigPia_init ])
params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )



GTtag_plot_pairwise_model_CA_match = 'GT'



num_test_samps_4xValS 	= [1] 	# Think not using either
xTimesThruTrain 		= 1		# of these anymore.
#
# Flags for the EM (inference & learning) algorithm.
flg_include_Zeq0_infer = True # NOT USING THIS ANYMORE. ALWAYS INCLUDING Z=0 SOLUTION VECTOR.
if flg_include_Zeq0_infer:
	z0_tag='_zeq0'
else:
	z0_tag='_zneq0'

	


for num_SWs in num_SWs_tot:
	#
	for flg_EgalitarianPrior in flg_EgalitarianPriorS:
		#
		if flg_EgalitarianPrior:	
			priorType = 'EgalQ' 
		else:
			priorType = 'BinomQ'
		#	
		for num_test_samps_4xVal in num_test_samps_4xValS:
			#
			for params_init in params_initS:
				#
				for learning_rate in learning_rates:
					#
					for lRateScale in lRateScaleS:
						#
						for xyz in range( len(Ks) ):
							#
							K 	 	= Ks[xyz]
							Kmin 	= Kmins[xyz]
							Kmax 	= Kmaxs[xyz]
							#
							C 	 	= Cs[xyz]
							Cmin 	= Cmins[xyz]
							Cmax 	= Cmaxs[xyz]
							#
							mu_Pia 	= mu_PiaS[xyz]
							sig_Pia = sig_PiaS[xyz]
							#
							mu_Pi 	= mu_PiS[xyz]
							sig_Pi 	= sig_PiS[xyz]
							#
							for abc in range(len(num_Cells)):
								N = num_Cells[abc]
								M = num_CAs_true[abc]
								#
								for yMinSW in yMinSWs:
										#
										for yLo in yLo_Vals:
											#
											for yHi in yHi_Vals:
												#
												for overcomp in model_CA_overcompleteness:
													M_mod = int( np.round(overcomp*M) )
													#
													data 	= list() # To collect up model data and filenames over 
													fnames 	= list() # different rands and A,B model combinations.
													fparams = list() 
													#
													for rand in range(num_EM_rands): #Must be inner most for loop.

														print('Random Sampling of Spike Words #',rand)
														if rand==0:
															rsTag = '_origModnSWs'
														else:
															rsTag = str( '_resampR0trn'+ str(pct_xVal_train_prev).replace('.','pt') )
														#

														# try:
														if True:

															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # (1).  Set up directory structure and filename. Load it in and extract variables.
															init_dir = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') \
																	+ '_LRsc' + str(lRateScale) + '/' )
															#
															model_dir = str( 'SyntheticData_N' + str(N) + '_M' + str(M) + '_Mmod' + str(M_mod) + '_K' + str(K) + 
																	'_' + str(Kmin) + '_' + str(Kmax) + '_C' + str(C) + '_' + str(Cmin)+ '_' + str(Cmax) + 
																	'_mPia' + str(mu_Pia) + '_sPia' + str(sig_Pia) + '_bPi' + str(bernoulli_Pi) + '_mPi' + 
																	str(mu_Pi) + '_sPi' + str(sig_Pi) + '_' + sample_longSWs_1st + 'Smp1st_' + priorType + '/')



															plt_save_dir = str(EM_figs_dir + init_dir + model_dir)
															#fname_save = str( fnames[0][:fnames[0].find('_rand')].replace('EM_model_data_','CA_match_matrix_' ) + '_' + str(len(fnames)) + 'files' )
													

															# Make a directories for output plots if they are not already there.
															if not os.path.exists( plt_save_dir ):
																os.makedirs( plt_save_dir )	

															# Directory for CA_matches
															saveDir_CAmatch = str( plt_save_dir + 'CA_model_matches/' )
															if not os.path.exists( saveDir_CAmatch ):
																os.makedirs( saveDir_CAmatch )	




															# Build up file name and load in npz file.
															fname_EMlrn = str('EM_model_data_' + str(num_SWs) + 'SWs_trn' + str(pct_xVal_train).replace('.','pt') \
																		+ '_xTTT' + str(xTimesThruTrain) + '_ylims' + str(yLo) + '_' + str(yHi) \
																		+ '_yMinSW' + str(yMinSW) + '_rand' + str(rand) + rsTag + '.npz' )
															#
															print('Loading in matching data files saved from EM learning algorithm in pgmCA_realData.py')
															print( fname_EMlrn )
															#
															data1 = np.load( str(EM_learning_Dir + init_dir + model_dir +fname_EMlrn) )
															data.append( data1 )
															fnames.append( str(fname_EMlrn) )
															#
															# #parsing 'argsRec' variable in data file.
															# xx = str(data1['argsRec']).replace('Namespace(','').replace(')','').split(',')
															# CnK_info 	= str( xx[0]+' '+xx[3]+' '+xx[4]+' '+xx[5]+' '+xx[6]+' '+xx[7]+' '    )
															# muNsig_info = str( xx[20]+' '+xx[34]+' '+xx[21]+' '+xx[35]+' '    )
															#
															fparams.append( str('rand'+str(rand)) ) # 'oc'+str(overcomp)+
															


															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															#
															# Try it also for the B file.
															try:
																fname_EMlrnB = str(fname_EMlrn[:-4]+'B.npz')
																#
																data2 = np.load( str(EM_learning_Dir + init_dir + model_dir +fname_EMlrnB) )
																data.append(data2)
																fnames.append( fname_EMlrnB )
																fparams.append( str('rand'+str(rand)+'B') ) # oc'+str(overcomp)+'_
																
																BfileThere = True
															except:
																BfileThere = False
																



															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															#
															if False: #flg_plot_indiv_model_pairs:

																#






																# NOTE: FOR TROUBLESHOOTING, CAN BUILD A MATRIX VERY CLOSE TO PIAGT1.
																# xxx = np.maximum( np.minimum( ( PiaGT1[:, np.random.choice(M,M,replace=False)] + 0.4 ) \
																# 					+ np.random.normal(loc=0., scale=0.10, size=PiaGT1.shape) , 1), 0)
																# print('max & min: ',np.min(xxx), np.max(xxx))					
																#str( str(rand) + Btag )

																#
																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																#
																# (1a). Find GT CAs that best match each Model CA.
																#
																Pia1 	= rc.sig(data1['riap'])
																PiaGT1 	= rc.sig(data1['ria'])
																#del data1
																#
																ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-PiaGT1, B=1-Pia1)
																#
																if flg_plot_our_translate:
																	pf.visualize_matchModels_cosSim( A=1-PiaGT1, Atag='GT1', B=1-Pia1, Btag='Mod1', ind=ind, cos_sim=cosSim, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)), figSaveFileType=figure_file_type )
																#
																if flg_plot_Hungarian_translate:
																	pf.visualize_matchModels_cosSim( A=1-PiaGT1, Atag='GT1', B=1-Pia1, Btag='Mod1', ind=HungRowCol, cos_sim=cos_simHM, len_dif=lenDif, \
																				cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)+'_HungMethd'), figSaveFileType=figure_file_type )
																
																#
																# (1b). Find Model CAs that best match each GT CA.
																ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-Pia1, B=1-PiaGT1)
																#
																if flg_plot_our_translate:
																	pf.visualize_matchModels_cosSim( A=1-Pia1, Atag='Mod1', B=1-PiaGT1, Btag='GT1', ind=ind, cos_sim=cosSim, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)), figSaveFileType=figure_file_type )
																#
																if flg_plot_Hungarian_translate:
																	pf.visualize_matchModels_cosSim( A=1-Pia1, Atag='Mod1', B=1-PiaGT1, Btag='GT1', ind=HungRowCol, cos_sim=cos_simHM, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)+'_HungMethd'), figSaveFileType=figure_file_type )

																
																
																if BfileThere:

																	Pia2 = rc.sig(data2['riap'])
																	PiaGT2 	= rc.sig(data2['ria'])
																	del data2
																	#
																	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																	#
																	# (2a). Find GT1 CAs that best match each GT2 CA.
																	ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-PiaGT2, B=1-Pia2)
																	#
																	if flg_plot_our_translate:
																		pf.visualize_matchModels_cosSim( A=1-PiaGT2, Atag='GT2', B=1-Pia2, Btag='Mod2', ind=ind, cos_sim=cosSim, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)), figSaveFileType=figure_file_type )
																	#
																	if flg_plot_Hungarian_translate:
																		pf.visualize_matchModels_cosSim( A=1-PiaGT2, Atag='GT2', B=1-Pia2, Btag='Mod2', ind=HungRowCol, cos_sim=cos_simHM, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)+'_HungMethd'), figSaveFileType=figure_file_type )
																	#
																	# (2b). Find GT1 CAs that best match each GT2 CA.
																	ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-Pia2, B=1-PiaGT2)
																	#
																	if flg_plot_our_translate:
																		pf.visualize_matchModels_cosSim( A=1-Pia2, Atag='Mod2', B=1-PiaGT2, Btag='GT2', ind=ind, cos_sim=cosSim, len_dif=lenDif, \
																				cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)), figSaveFileType=figure_file_type )
																	#
																	if flg_plot_Hungarian_translate:
																		pf.visualize_matchModels_cosSim( A=1-Pia2, Atag='Mod2', B=1-PiaGT2, Btag='GT2', ind=HungRowCol, cos_sim=cos_simHM, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)+'_HungMethd'), figSaveFileType=figure_file_type )
																	#
																	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																	#
																	# (3a). Find GT1 CAs that best match each GT2 CA.
																	#																	
																	ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-PiaGT1, B=1-PiaGT2)
																	#
																	if flg_plot_our_translate:
																		pf.visualize_matchModels_cosSim( A=1-PiaGT1, Atag='GT1', B=1-PiaGT2, Btag='GT2', ind=ind, cos_sim=cosSim, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)), figSaveFileType=figure_file_type )
																	#
																	if flg_plot_Hungarian_translate:
																		pf.visualize_matchModels_cosSim( A=1-PiaGT1, Atag='GT1', B=1-PiaGT2, Btag='GT2', ind=HungRowCol, cos_sim=cos_simHM, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)+'_HungMethd'), figSaveFileType=figure_file_type )

																	#
																	# (3b). Find GT1 CAs that best match each GT2 CA.
																	ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-PiaGT2, B=1-PiaGT1)
																	#
																	if flg_plot_our_translate:
																		pf.visualize_matchModels_cosSim( A=1-PiaGT2, Atag='GT2', B=1-PiaGT1, Btag='GT1', ind=ind, cos_sim=cosSim, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)), figSaveFileType=figure_file_type )
																	#
																	if flg_plot_Hungarian_translate:
																		pf.visualize_matchModels_cosSim( A=1-PiaGT2, Atag='GT2', B=1-PiaGT1, Btag='GT1', ind=HungRowCol, cos_sim=cos_simHM, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)+'_HungMethd'), figSaveFileType=figure_file_type )
																	



												# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
												#
												# Compare GT-Mod1, GT-Mod2 & Mod1-Mod2 for each CA pair individually.
												# HERE, plotting CA-by-CA the cosSim between Mod1-Mod2 and the cosSim between each Mod-GT.
												if flg_plot_indiv_model_pairs:


													print(data) 
													print(fnames)
													print(fparams)

													for a in range(len(data)):
														for b in range(len(data)):
															if b < a:

																#
																print('Loading in model with:',fnames[a], fparams[a])
																Pia1 	= rc.sig(data[a]['riap'])
																Pia1_init 	= rc.sig(data[a]['ria_init'])
																#PiaGT 	= rc.sig(data[a]['ria'])
																print('Loading in model with:',fnames[b], fparams[b])
																Pia2 	= rc.sig(data[b]['riap'])
																Pia2_init 	= rc.sig(data[b]['ria_init'])
																#PiaGT2 	= rc.sig(data[b]['ria'])





																# Get rid of / ignore CAs that have not changed significantly enough from their init.
																ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-Pia1, B=1-Pia1_init)
																x1 = Pia1[:,np.where(csNM<0.9)[0]]
																#
																pf.visualize_matchModels_cosSim( A=1-Pia1, Atag=fparams[a].replace('rand',''), B=1-Pia1_init, Btag=fparams[a].replace('rand','init'), ind=HungRowCol, cos_sim=cos_simHM, \
																		 len_dif=lenDif, cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag='HungMethd', figSaveFileType=figure_file_type )



																ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-Pia2, B=1-Pia2_init)
																x2 = Pia2[:,np.where(csNM<0.9)[0]]
																#
																pf.visualize_matchModels_cosSim( A=1-Pia2, Atag=fparams[b].replace('rand',''), B=1-Pia2_init, Btag=fparams[b].replace('rand','init'), ind=HungRowCol, cos_sim=cos_simHM, \
																		 len_dif=lenDif, cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag='HungMethd_init', figSaveFileType=figure_file_type )




																#
																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																#
																# (4a). Find Model1 CAs that best match each Model2 CA.
																#
																ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-x1, B=1-x2)
																#
																if flg_plot_our_translate:
																	pf.visualize_matchModels_cosSim( A=1-Pia1, Atag=fparams[a].replace('rand',''), B=1-Pia2, Btag=fparams[b].replace('rand',''), ind=ind, cos_sim=cosSim, \
																		 len_dif=lenDif, cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag='', figSaveFileType=figure_file_type )
																#
																if flg_plot_Hungarian_translate:
																	pf.visualize_matchModels_cosSim( A=1-x1, Atag=fparams[a].replace('rand',''), B=1-x2, Btag=fparams[b].replace('rand',''), ind=HungRowCol, cos_sim=cos_simHM, \
																		 len_dif=lenDif, cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag='HungMethd_i', figSaveFileType=figure_file_type )
																#
																# (4b). Find Model2 CAs that best match each Model1 CA.
																ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-x2, B=1-x1)
																#
																if flg_plot_our_translate:
																	pf.visualize_matchModels_cosSim( A=1-Pia2, Atag=fparams[b].replace('rand',''), B=1-Pia1, Btag=fparams[a].replace('rand',''), ind=ind, cos_sim=cosSim,  \
																		 len_dif=lenDif, cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag='', figSaveFileType=figure_file_type )
																#
																if flg_plot_Hungarian_translate:
																	pf.visualize_matchModels_cosSim( A=1-x2, Atag=fparams[b].replace('rand',''), B=1-x1, Btag=fparams[a].replace('rand',''), ind=HungRowCol, cos_sim=cos_simHM,  \
																		 len_dif=lenDif, cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag='HungMethd_i', figSaveFileType=figure_file_type )










																# # Cosine similarity take higher values for cell assemblies that were 
																# # initialized the same in rand0 and rand0B and never or rarely inferred.
																# # This is why null model is quite high for lots of values.
																# if 'B.npz' in fnames[a]:
																# 	ZinfA = data[a]['Z_inferred_test']
																# else:
																# 	ZinfA = data[a]['Z_inferred_train']
																# ZinfHistA = np.zeros(M)
																# for Zset in ZinfA:
																# 	ZinfHistA[list(Zset)] += 1
																# #
																# # #
																# #
																# if 'B.npz' in fnames[b]:
																# 	ZinfB = data[b]['Z_inferred_test']
																# else:
																# 	ZinfB = data[b]['Z_inferred_train']
																# ZinfHistB = np.zeros(M)
																# for Zset in ZinfB:
																# 	ZinfHistB[list(Zset)] += 1
																# #
																# pctInferLrn = (ZinfHistA + ZinfHistB) / ( len(ZinfA) + len(ZinfB) )
																# nmSort = np.argsort(pctInferLrn)[::-1]
																
																# xxx = np.vstack( (csNM[nmSort].round(3), (pctInferLrn[nmSort]/pctInferLrn.max()).round(3) ) )
																# #print( xxx[:,:] )

																# # plt.plot(xxx.T)
																# # plt.show()
																# inferNorm = pctInferLrn/pctInferLrn.max()






																#
																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
																#
																# (5a). Find match between Ground Truth and Average of 2 models, Pia1 and Pia2.
																#
																if False:
																	ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-Pia1, B=1-Pia2)
																	#
																	Ay = Pia1[:,ind[1]] # 
																	Be = Pia2[:,ind[0]]
																	PiaAvg = (Ay+Be)/2
																	#
																	ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-PiaAvg, B=1-PiaGT1)
																	#
																	if flg_plot_our_translate:
																		pf.visualize_matchModels_cosSim( A=1-PiaAvg, Atag='ModAvg', B=1-PiaGT1, Btag='GT1', ind=ind, cos_sim=cosSim, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)), figSaveFileType=figure_file_type )
																	#
																	if flg_plot_Hungarian_translate:
																		pf.visualize_matchModels_cosSim( A=1-PiaAvg, Atag='ModAvg', B=1-PiaGT1, Btag='GT1', ind=HungRowCol, cos_sim=cos_simHM, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)+'_HungMethd'), figSaveFileType=figure_file_type )
																	#
																	# (5b). Find Model2 CAs that best match each Model1 CA.
																	ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-PiaGT1, B=1-PiaAvg)
																	#
																	if flg_plot_our_translate:
																		pf.visualize_matchModels_cosSim( A=1-PiaGT1, Atag='GT1', B=1-PiaAvg, Btag='ModAvg', ind=ind, cos_sim=cosSim, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)), figSaveFileType=figure_file_type )
																	#
																	if flg_plot_Hungarian_translate:
																		pf.visualize_matchModels_cosSim( A=1-PiaGT1, Atag='GT1', B=1-PiaAvg, Btag='ModAvg',  ind=HungRowCol, cos_sim=cos_simHM, len_dif=lenDif, \
																					cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir_CAmatch, fname_tag=str('rand'+str(rand)+'_HungMethd'), figSaveFileType=figure_file_type )





												# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
												#
												# Compare GT-Mod1, GT-Mod2 & Mod1-Mod2 for each CA pair individually.
												# HERE, plotting CA-by-CA the cosSim between Mod1-Mod2 and the cosSim between each Mod-GT.
												if flg_directCompareCAs:
													for a in range(len(data)):
														for b in range(len(data)):
															if b < a:

																#
																print('Loading in model with:',fnames[a], fparams[a])
																Pia1 	= rc.sig(data[a]['riap'])
																PiaGT 	= rc.sig(data[a]['ria'])
																print('Loading in model with:',fnames[b], fparams[b])
																Pia2 	= rc.sig(data[b]['riap'])
																PiaGT2 	= rc.sig(data[b]['ria'])
																

																if np.all(PiaGT.shape==PiaGT2.shape) and np.any(PiaGT-PiaGT2):
																	print('problem in directCompareCAs: Ground truths arent the same..')
																#
																#	
																indGreedy12, cosSimGreedy12, csNM12, lenDif12, ldNM12, cosSimMat12, lenDifMat12, HungRowCol12, cos_simHM_12 = rc.matchModels_AtoB_cosSim(A=1-Pia1, B=1-Pia2)
																indGreedyG1, cosSimGreedyG1, csNMG1, lenDifG1, ldNMG1, cosSimMatG1, lenDifMatG1, HungRowColG1, cos_simHM_G1 = rc.matchModels_AtoB_cosSim(A=1-PiaGT, B=1-Pia1)
																indGreedyG2, cosSimGreedyG2, csNMG2, lenDifG2, ldNMG2, cosSimMatG2, lenDifMatG2, HungRowColG2, cos_simHM_G2 = rc.matchModels_AtoB_cosSim(A=1-PiaGT, B=1-Pia2)


																if True:

																	ind12 = HungRowCol12
																	indG1 = HungRowColG1
																	indG2 = HungRowColG2
																	cosSim12 = np.vstack([cos_simHM_12, np.zeros_like(cos_simHM_12)])
																	cosSimG1 = np.vstack([cos_simHM_G1, np.zeros_like(cos_simHM_G1)])
																	cosSimG2 = np.vstack([cos_simHM_G2, np.zeros_like(cos_simHM_G2)])
																	hungTag = '_HungMethd'
																else:
																	ind12 = indGreedy12
																	indG1 = indGreedyG1
																	indG2 = indGreedyG2
																	cosSim12 = cosSimGreedy12
																	cosSimG1 = cosSimGreedyG1
																	cosSimG2 = cosSimGreedyG2
																	hungTag = ''




																M = len(ind12[0])


																# Find GT CA that corresponds with CA in model1 or 2 but with ordering that matches
																# the descending ordering of cosSim in the match between models 1 & 2.
																indIntoGw1 = [list( indG1[0] ).index( ind12[1][i] ) for i in range(M)]
																indIntoGw2 = [list( indG2[0] ).index( ind12[0][i] ) for i in range(M)]




																# Do the two models agree on the same groundtruth to represent the CAs which are most similar in the two models.
																xx = list(indG1[1][indIntoGw1])
																yy = list(indG2[1][indIntoGw2])
																zz = [xx[i]==yy[i] for i in range(len(xx))]

																print('For models',a,'and',b,': Number of GTs they agree on  =',np.array(zz).sum())
																print('CosSim M1 :: M2: ',cosSim12[0][zz].mean().round(3), cosSim12[0][zz].std().round(3) )
																#print( cosSim12[0][zz].round(3)  )
																print(' ')
																CSagg1 = [ cosSimG1[0][indIntoGw1[i]] for i in np.where(zz)[0] ]
																CSagg2 = [ cosSimG2[0][indIntoGw2[i]] for i in np.where(zz)[0] ]


																print('CosSim GT :: M1: ',np.array(CSagg1).mean().round(3), np.array(CSagg1).std().round(3) )
																#print( np.array(CSagg1).round(3) )
																print(' ')
																print('CosSim GT :: M2: ',np.array(CSagg2).mean().round(3), np.array(CSagg2).std().round(3) )
																#print( np.array(CSagg2).round(3) )
																print(' ')
																print('-----------------------------------------------------------------------------------')



																# # THIS SECTION IS GOOD FOR INTUITION. IS DONE USING LIST COMPREHENSION BELOW. BUT NICE HERE.
																# # 
																# #	NOTE: np.all( cosSim12[0]== cosSimMat12[ ind12[0], ind12[1] ] )
																# # 										"Model2", "Model1"
																# # and
																# #	NOTE: np.all( cosSimG1[0]== cosSimMatG1[ indG1[0], indG1[1] ] )
																# # 										"Model1", "GndTru" 
																# # and
																# #	NOTE: np.all( cosSimG2[0]== cosSimMatG2[ indG2[0], indG2[1] ] )
																# # 										"Model2", "GndTru"




																

																# (4). 
																if True:
																	# pf.scatter_cosSim_v_lenDif_indivCAs( ind12, indG1, indG2, cosSim12, cosSimG1, cosSimG2, lenDif12, lenDifG1, lenDifG2 )
																	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																																		#
																	# define measure to use (cosSim, lenDif, cosSim*lenDif)
																	#	
																	measure_str = 'Cosine Similarity' # * lenDif'
																	meas12 = cosSim12#*lenDif12
																	measG1 = cosSimG1#*lenDifG1
																	measG2 = cosSimG2#*lenDifG2
																	#
																	# Find index into each groundtruth 
																	#
																	verbose=False
																	if verbose:
																		for i in range(M):
																			print( i,': cs12-',cosSim12[0][i].round(3),': M1-',ind12[1][i],'& M2-',ind12[0][i],\
																				'--> GTw1-',indG1[1][indIntoGw1[i]],'(',cosSimG1[0][indIntoGw1[i]].round(3),')' \
																				'--> GTw2-',indG2[1][indIntoGw2[i]],'(',cosSimG2[0][indIntoGw2[i]].round(3),')' )
																		

																	#
																	# Compute histogram of distance from unity line for each of the 3 groups of scatter points.
																	#
																	distrib_M1 		= np.histogram( measG1[0][indIntoGw1] - meas12[0], bins=21, range=(-1,1) )
																	distrib_M2 		= np.histogram( measG2[0][indIntoGw2] - meas12[0], bins=21, range=(-1,1) )
																	distrib_Mmin 	= np.histogram( np.minimum(measG1[0][indIntoGw1],measG2[0][indIntoGw2]) - meas12[0], bins=21, range=(-1,1) )
																	#
																	# dist_LD_M1 		= np.histogram( lenDif12[0] - lenDifG1[0][indIntoGw1], bins=21, range=(-1,1) )
																	# dist_LD_M2 		= np.histogram( lenDif12[0] - lenDifG2[0][indIntoGw2], bins=21, range=(-1,1) )
																	# dist_LD_Mmin 	= np.histogram( lenDif12[0] - np.minimum(lenDifG1[0][indIntoGw1],lenDifG2[0][indIntoGw2]), bins=21, range=(-1,1) )
																	#
																	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
																	#
																	# Scatter measure Model1vs2 against cosSim ModelvsGT.
																	#

																	#
																	f = plt.figure( figsize=(20,10) ) # size units in inches
																	plt.rc('font', weight='bold', size=20)
																	plt.rc('text', usetex=True)
																	plt.rcParams['text.latex.preamble'] = [r'\boldmath']
																	#
																	ax1 = plt.subplot2grid( (2,2),(0,0) )
																	ax1.scatter( cosSim12[0][zz], CSagg1, s=60, c='k', marker='d', alpha=0.6, label='agg1')
																	ax1.scatter( cosSim12[0][zz], CSagg2, s=60, c='k', marker='d', alpha=0.6, label='agg1')
																	ax1.scatter( meas12[0], np.minimum(measG1[0][indIntoGw1],measG2[0][indIntoGw2]), s=50, c='r', marker='o', alpha=0.4)#, label='min 1&2')
																	ax1.scatter( meas12[0], measG1[0][indIntoGw1], s=10, c='g', marker='x')#, label='Model 1' )
																	ax1.scatter( meas12[0], measG2[0][indIntoGw2], s=10, c='b', marker='x')#, label='Model 2' )
																	#
																	ax1.errorbar( (cosSim12[0][zz]).mean(), np.array(CSagg1).mean(), (cosSim12[0][zz]).std(), np.array(CSagg1).std(), color='g', capsize=10, elinewidth=2, marker='s' ) 
																	ax1.errorbar( (cosSim12[0][zz]).mean(), np.array(CSagg2).mean(), (cosSim12[0][zz]).std(), np.array(CSagg2).std(), color='b', capsize=10, elinewidth=2, marker='s' ) 
																	for i in range(M):
																		ax1.plot( np.array([meas12[0][i],meas12[0][i]]), np.array([measG1[0][indIntoGw1[i]],measG2[0][indIntoGw2[i]]]),'k-',alpha=0.1  )
																	ax1.plot(np.array([0,1]),np.array([0,1]),'k--')
																	ax1.set_aspect('equal')
																	ax1.set_ylabel( 'GT :: Mod' )
																	ax1.set_xlabel( 'Mod1 :: Mod2' )
																	ax1.grid()
																	ax1.set_title(measure_str)
																	ax1.set_xlim(-0.1,1.1)
																	ax1.set_ylim(-0.1,1.1)
																	ax1.set_xticks([0, 0.5, 1.0])
																	ax1.set_yticks([0, 0.5, 1.0])

																	#
																	# Compute Pearson Correlation coefficients between GT and model matches.
																	r1=st.pearsonr(meas12[0],measG1[0][indIntoGw1])	
																	r2=st.pearsonr(meas12[0],measG2[0][indIntoGw2])
																	rm=st.pearsonr(meas12[0],np.minimum(measG1[0][indIntoGw1],measG2[0][indIntoGw2]))
																	print('pearsonr = ',r1,r2,rm)
																	#
																	ax2 = plt.subplot2grid( (2,2), (0,1) )
																	ax2.plot(distrib_M1[1][:-1]-0.01,distrib_M1[0], 'gx-', alpha=0.6, linewidth=2, label=str( fparams[a].replace('rand','Mod') + ' r=' + str(r1[0].round(2) ) ) )
																	ax2.plot(distrib_M2[1][:-1]+0.00,distrib_M2[0], 'bx-', alpha=0.6, linewidth=2, label=str( fparams[b].replace('rand','Mod') + ' r=' + str(r2[0].round(2) ) ) )
																	ax2.plot(distrib_Mmin[1][:-1]+0.01,distrib_Mmin[0], 'ro-', alpha=0.6, linewidth=2, label=str( 'Mod min r=' + str(rm[0].round(2) ) ) )
																	#
																	ax2.text( -1, np.max(np.vstack([distrib_M1[0],distrib_M2[0],distrib_Mmin[0]])),str(r'$\diamond$ - Models agree on '+str( np.array(zz).sum() )+' CAs in GT'), ha='left',va='top',fontsize=16 )
																	# ax2.text( -1, 0.85*np.max(np.vstack([distrib_M1[0],distrib_M2[0],distrib_Mmin[0]])),str('r1='+str(r1[0].round(3))), ha='left',va='top',color='green' )
																	# ax2.text( -1, 0.75*np.max(np.vstack([distrib_M1[0],distrib_M2[0],distrib_Mmin[0]])),str('r2='+str(r2[0].round(3))), ha='left',va='top',color='blue' )
																	# ax2.text( -1, 0.65*np.max(np.vstack([distrib_M1[0],distrib_M2[0],distrib_Mmin[0]])),str('rmin='+str(rm[0].round(3))), ha='left',va='top',color='red' )
																	ax2.set_title(r' $\Leftarrow$ Distribution of distance from Unity (y=x) in plot to left.')
																	ax2.set_ylabel(str('Counts /'+str(M)+' CA pairs') )
																	ax2.set_xlabel(r'Below Diag. $\leftrightarrow$ Above Diag.')
																	ax2.set_aspect('auto')
																	ax2.grid()
																	ax2.legend(fontsize=16)
																	#
																	# # # # # # 
																	#
																	ax3 = plt.subplot2grid( (2,3), (1,0) )
																	#ax3.imshow( cosSimMatG1[ np.ix_(indG1[0], indG1[1]) ].T,vmin=0,vmax=1 )
																	ax3.imshow( cosSimMatG1[ np.ix_(indG1[0][indIntoGw1], indG1[1][indIntoGw1])].T,vmin=0,vmax=1 )
																	ax3.set_yticks(np.arange(M))
																	ax3.set_yticklabels(indG1[1],fontsize=6)
																	ax3.set_ylabel('CA in GT')
																	ax3.set_xticks(np.arange(M))
																	ax3.set_xticklabels(indG1[0],fontsize=6,rotation=90)
																	ax3.set_xlabel( str('CA in ' + fparams[a].replace('rand','Mod') ) )
																	ax3.xaxis.label.set_color('green')
																	#ax3.set_title('cosSim between Model1-GT')
																	#
																	ax4 = plt.subplot2grid( (2,3), (1,1) )
																	ax4.imshow( cosSimMat12[ np.ix_(ind12[0], ind12[1]) ],vmin=0,vmax=1 )
																	ax4.set_xticks(np.arange(M))
																	ax4.set_xticklabels(ind12[1],fontsize=6,rotation=90)
																	ax4.set_xlabel( str('CA in ' + fparams[a].replace('rand','Mod') ) )
																	ax4.xaxis.label.set_color('green')
																	ax4.set_yticks(np.arange(M))
																	ax4.set_yticklabels(ind12[0],fontsize=6)
																	ax4.set_ylabel( str('CA in ' + fparams[b].replace('rand','Mod') ) )
																	ax4.yaxis.label.set_color('blue')
																	ax4.set_title('Cos Sim')# between Models1-2')
																	#
																	ax5 = plt.subplot2grid( (2,3), (1,2) )
																	im=ax5.imshow( cosSimMatG2[ np.ix_(indG2[0][indIntoGw2], indG2[1][indIntoGw2]) ],vmin=0,vmax=1 )
																	#ax5.imshow( cosSimMatG2[ np.ix_(indG2[0], indG2[1]) ],vmin=0,vmax=1 )
																	ax5.set_xticks(np.arange(M))
																	ax5.set_xticklabels(indG2[1],fontsize=6,rotation=90)
																	ax5.set_xlabel('CA in GT')
																	ax5.set_yticks(np.arange(M))
																	ax5.set_yticklabels(indG2[0],fontsize=6)
																	ax5.set_ylabel( str('CA in ' + fparams[b].replace('rand','Mod') ) )
																	ax5.yaxis.label.set_color('blue')
																	#ax5.set_title('cosSim between Model2-GT')
																	#plt.show()
																	#
																	# #
																	#
																	cax = f.add_axes([0.92, 0.1, 0.025, 0.35])
																	f.colorbar(im, cax=cax)
																	cax.set_title( str(r'$ \frac{(A.B)}{ \vert A \vert \vert B \vert }$'), fontsize=22, fontweight='bold', ha="center", va="bottom" )
																	# #


																	# plt.suptitle(   str( '\n'.join(wrap( str( str(data[a]['argsRec']) + ' AND ' + str(data[b]['argsRec']) ).replace('_','') ,350)) )   , fontsize=8 )   
																	#
																	plt.tight_layout()

																	print('Saving figure to:', plt_save_dir)
																	fname_save = str('singleCAscatter_'+fparams[a].replace('rand','')+'_'+fparams[b].replace('rand',''))
																	if not os.path.exists( str(plt_save_dir + 'CA_model_matches/') ):
																		os.makedirs( str(plt_save_dir + 'CA_model_matches/') )
																	#
																	plt.savefig( str(plt_save_dir + 'CA_model_matches/' + fname_save + '_cosSim' + hungTag + '.' + figure_file_type ) )
																	plt.close() 


																	# COMPARE TO NULL MODEL:  Plot to compare cosSim to csNM. And Maybe lenDif to ldNM.
																	if False: # A GOOD ONE.
																		f = plt.figure( figsize=(20,5) ) # size units in inches
																		plt.rc('font', weight='bold', size=16)
																		#
																		ax0 = plt.subplot2grid((1,3),(0,0))
																		ax0.plot(np.sort(cosSimG1[0][:M])[::-1], 'r-', label='model')
																		ax0.plot(np.sort(csNMG1)[::-1],'b--', label='null')
																		ax0.legend()
																		ax0.set_title('Model1 :: GT')
																		ax0.set_xlabel('CA id')
																		ax0.set_ylabel('Cosine Similarity')
																		#
																		ax1 = plt.subplot2grid((1,3),(0,1))
																		ax1.plot(np.sort(cosSim12[0][:M])[::-1],'r-')
																		ax1.plot(np.sort(csNM12)[::-1],'b--')
																		ax1.set_title('Model1 :: Model2')

																		#
																		ax2 = plt.subplot2grid((1,3),(0,2))
																		ax2.plot(np.sort(cosSimG2[0][:M])[::-1],'r-')
																		ax2.plot(np.sort(csNMG2)[::-1],'b--')
																		ax2.set_title('Model2 :: GT')
																		#
																		#plt.suptitle( 'Cosine Similarity', fontsize=24, fontweight='bold' ) 
																		plt.show()
																


																	# Plot individual cell assemblies in 2 models and 2 matching GTs with cosSim measure
																	if False: # GOOD ONE.
																		for i in range(10): #M):
																			f = plt.figure( figsize=(20,10) ) # size units in inches
																			plt.rc('font', weight='bold', size=16)
																			plt.plot( 1-Pia1[:,ind12[1][i]], label=str('M1-'+str(ind12[1][i])) )
																			plt.plot( 1-Pia2[:,ind12[0][i]], label=str('M2-'+str(ind12[0][i])) )
																			plt.plot( 1-PiaGT[:,indG1[1][indIntoGw1[i]]], label=str('GT1-'+str(indG1[1][indIntoGw1[i]])) )
																			plt.plot( 1-PiaGT[:,indG2[1][indIntoGw2[i]]], label=str('GT2-'+str(indG2[1][indIntoGw2[i]])) )
																			plt.title( str( str(i)+' cs12 '+str(cosSim12[0][i].round(3))+' - :: - csG1 '+str(cosSimG1[0][[indIntoGw1[i]]].round(3))+', csG2 '+str(cosSimG2[0][[indIntoGw2[i]]].round(3)) ) )
																			plt.legend()
																			plt.show()




																	







												# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
												#
												# Collect up matrices of how many matching CAs models have (on main diagonal)
												# and share (on off diagonals) across A and B models and different rands.
												#
												# NOTE: Should be lined up with the rand for loop and overcomp for loop
												matchMod1toMod2_cosSim, matchMod1toMod2_lenDif, matchMod1toMod2_csHM,\
												matchMod2toMod1_cosSim, matchMod2toMod1_lenDif, matchMod2toMod1_csHM,\
												matchMod1toGT_cosSim, 	matchMod1toGT_lenDif, 	matchMod1toGT_csHM,\
												matchGTtoMod1_cosSim,	matchGTtoMod1_lenDif, 	matchGTtoMod1_csHM,\
												matchMod1toMod2_csNM, 	matchMod1toMod2_ldNM, 	matchMod2toMod1_csNM, 	matchMod2toMod1_ldNM, \
												matchMod1toGT_csNM, 	matchMod1toGT_ldNM, 	matchGTtoMod1_csNM, 	matchGTtoMod1_ldNM \
												= rc.compute_pairwise_model_CA_match(data)


												#
												# Plot cosine similarity for learned model and null model. (essentially learned model without matching up CAs.)
												fname_save = str( fnames[0][:fnames[0].find('_rand')].replace('EM_model_data_','CA_match_cosSim_' ) + '_' + str(len(fnames)) + 'files' )
												if flg_plot_our_translate:
													pf.plot_pairwise_model_CA_match(matchMod1toMod2_cosSim, matchMod1toMod2_csNM, \
														matchMod1toGT_cosSim, matchMod1toGT_csNM, GTtag_plot_pairwise_model_CA_match, fnames, fparams, plt_save_dir, fname_save, 'Cosine Similarity', figure_file_type)
														# can use: matchMod1toMod2_cosSim or matchMod1toMod2_csHM.
												#	
												if flg_plot_Hungarian_translate:
													pf.plot_pairwise_model_CA_match(matchMod1toMod2_csHM, matchMod1toMod2_csNM, \
														matchMod1toGT_csHM, matchMod1toGT_csNM, GTtag_plot_pairwise_model_CA_match, fnames, fparams, plt_save_dir, str(fname_save+'_HungMethd'), 'Cosine Similarity Hungarian Sort', figure_file_type )
												
												#
												# Plot vector length difference for learned model and null model. (essentially learned model without matching up CAs.)
												if False:
													fname_save = str( fnames[0][:fnames[0].find('_rand')].replace('EM_model_data_','CA_match_LenDif_' ) + '_' + str(len(fnames)) + 'files' )
													pf.plot_pairwise_model_CA_match(matchMod1toMod2_lenDif, matchMod1toMod2_ldNM, \
														matchMod1toGT_lenDif, matchMod1toGT_ldNM, GTtag_plot_pairwise_model_CA_match, fnames, fparams, plt_save_dir, fname_save, 'Vector Length Similarity', figure_file_type)


												# # UPDATED FROM REAL DATA COMPARE
												# pf.plot_pairwise_model_CA_match(matchMod1toMod2_cosSim, matchMod1toMod2_csNM, delFromInit, np.zeros((len(data),1)), meanLogCond, \
												# 		GTtag_plot_pairwise_model_CA_match, fnames, fparams, plt_save_dir, fname_save, 'Cosine Similarity', figSaveFileType)

												
												


											

												self_love = True
												print('Self Love = ',self_love)



