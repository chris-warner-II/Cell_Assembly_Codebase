import argparse
import numpy as np 
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import os
import time

import utils.data_manipulation as dm
import utils.plot_functions as pf
import utils.retina_computation as rc
from textwrap import wrap

from scipy import signal as sig
from scipy import stats as st

#import difflib


#
dirHomeLoc, dirScratch = dm.set_dir_tree()


# 
# Synthetic Model Construction Parameters we can loop over.
Ks 				= [2] #[2, 2, 2, 2] 		# Number of	cell assemblies active, given a Bernoulli distribution ("hotness" of Z-vector)
Kmins 			= [0] #[0, 0, 1, 1]		# Max & Min number of cell assemblies active 
Kmaxs 			= [2] #[2, 2, 3, 3]		# 
#
Cs 				= [2] #[2, 3, 2, 3]			# number of cells participating in cell assemblies. ("hotness" of each row in Pia)
Cmins 			= [2] #[2, 2, 2, 2] 		# Max & Min number of cell active to call it a cell assembly
Cmaxs 			= [6] #[6, 6, 6, 6] 		# 
#
yLo_Vals 		= [0] 		# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<=yLo.
yHi_Vals 		= [1000] 	# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution.
yMinSWs 		= [1] #,3]
#
mu_Pia			= 0.0   		# Binary values in Pia matrix made continuous by sampling from Gaussian centered at 1-mu_Pia or 0+mu_Pia.
sig_Pia			= 0.1 		# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
#
bernoulli_Pi	= 1.0   		# Each value in binary Pi vector randomly sampled from bernoulli with p(Pi=1) = bernoulli_Pi.	
mu_Pi			= 0.0   		# Binary values in Pi vector made continuous by sampling from Gaussian centered at 1-mu_Pi or 0+mu_Pi.
sig_Pi			= 0.05 		# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.

num_SWs_tot = [20003] #[50000, 100000] # [10000, 50000, 100000]
#pctSWs_EM_Samples 	= [1/2] #[1/5, 1/2, 1, 2] #  # number of steps to run full EM algorithm - alternating Inference and Learning.
num_Cells = [55] #[50, 100] 					# Looping over N values
num_CAs_true = [55] #[50, 100] #,200				# Looping over M values used to build the model
model_CA_overcompleteness = [1] #,2		# how many times more cell assemblies the model assumes than are in true model (1 means complete - M_mod=M, 2 means 2x overcomplete)
learning_rates = [0.5, 0.1] #, 0.5, 1.0] #[0.1, 0.5, 1.0] 	# Learning rates to loop through


# Parameter initializations for EM algorithm to learn model parameters
params_initS 	= ['NoisyConst'] # 'DiagonalPia', 'NoisyConst'] 	# Options: {'True', 'RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise) }
sigQ_init 		= 0.01	# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigPi_init 		= 0.05	# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigPia_init 	= 0.05	# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.

# Parameter initializations for EM algorithm to learn model parameters
Z_hot 				= 10 	# initialization for Q value (how many 1's expected in binary z-vector)
C_noise_ri 			= 1.0
C_noise_ria 		= 1.0
lRateScale_Pi 		= 0.1	# Multiplicative scaling to Pi learning rate. If set to zero, Pi taken out of model essentially.

flg_include_Zeq0_infer  	= True
verbose_EM 				 	= False

if flg_include_Zeq0_infer:
	z0_tag='_zeq0'
else:
	z0_tag='_zneq0'


#Init_NoisyConst_10hot_mPi1.0_mPia1.0_sI[0.01 0.05 0.05]_LR0pt1_LRpiq0pt1/SyntheticData_N20_M20_Mmod20_K2_0_2_C2_2_6_mPia0.4_sPia0.1_bPi1.0_mPi0.0_sPi0.05_ProbSmp1st/SWs_inferred_postLrn_2000SWs_trn0pt9_xTTT1_ylims0_1000_yMinSW1_rand0_origModnSWs.npz



sample_longSWs_1st = 'Prob' # Options are: {'Dont', 'Prob', 'Hard'}


ds_fctr_snapshots = 10 	# Take a sample of model parameters every *ds_fctr* time steps to compute and plot MSE after we determine permutation of learned CA's to True CA's
num_EM_rands = 1 #3 
pct_xVal_train = 0.9

xTimesThruTrain = 1

train_2nd_modelS = [False] #[True, False] # True to run B.npz files and False to run others.


resample_available_spikewords = True # ALWAYS TRUE I THINK FROM NOW ON !!!
pct_xVal_train_prev = 0.5


flg_includeInfer 			= False
flg_plot_snaps 				= True # hardcoded below that only plotting first and last few if this is True.

flg_plot_crossValidation 	= True
flg_plot_PiaStreaks 		= False
flg_plot_compare2init 		= False

flg_plot_compare_models_AnB = False



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (1). Load in npz file and extract data from it.
sig_init 		= np.array([ sigQ_init, sigPi_init, sigPia_init ])
params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (0). Set up directories and create dirs if necessary.
dirHome, dirScratch = dm.set_dir_tree()
EM_learning_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/Models_learned_EM/')
#SW_extracted_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/G_Field/SpikeWordsExtracted/')
inferPL_data_dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/inference_postLrn/')
figs_save_dir 		= str( dirScratch + 'figs/PGM_analysis/EM_algorithm/')








# # ACTUALLY NOT LOOPING OVER THESE BELOW. THEY ARE FIXED.
# #
# for params_init in params_initS:
# #
# for yLo in yLo_Vals:
# #
# for yHi in yHi_Vals:
# #
# for overcomp in model_CA_overcompleteness:
# #
params_init = params_initS[0]
yLo = yLo_Vals[0]
yHi = yHi_Vals[0]
overcomp = model_CA_overcompleteness[0]
# # ACTUALLY NOT LOOPING OVER THESE ABOVE. THEY ARE FIXED.




for num_SWs in num_SWs_tot:
	num_EM_samps = int(pct_xVal_train*num_SWs)
	#
	for learning_rate in learning_rates:
		#
		for xyz in range( len(Ks) ):
			K 	 = Ks[xyz]
			Kmin = Kmins[xyz]
			Kmax = Kmaxs[xyz]
			C 	 = Cs[xyz]
			Cmin = Cmins[xyz]
			Cmax = Cmaxs[xyz]
			#
			for abc in range(len(num_Cells)):
				N = num_Cells[abc]
				M = num_CAs_true[abc]
				M_mod = M
				#
				for yMinSW in yMinSWs:
					#
					for rand in range(num_EM_rands):
						#
						if resample_available_spikewords:
							print('Random Sampling of Spike Words #',rand)
							if rand==0:
								rsTag = '_origModnSWs'
							else:
								rsTag = str( '_resampR0trn'+ str(pct_xVal_train_prev).replace('.','pt') )
						#
						for train_2nd_model in train_2nd_modelS:
							#
							if train_2nd_model:
								Btag ='B'
							else:
								Btag =''
							#
							

							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
							# # (1).  Set up directory structure and filename. Load it in and extract variables.
							#
							init_dir = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') + \
									'_LRpiq' + str(lRateScale_Pi).replace('.','pt') +'/' )

							model_dir = str( 'SyntheticData_N' + str(N) + '_M' + str(M) + '_Mmod' + str(M_mod) + '_K' + str(K) + 
										'_' + str(Kmin) + '_' + str(Kmax) + '_C' + str(C) + '_' + str(Cmin)+ '_' + str(Cmax) + 
										'_mPia' + str(mu_Pia) + '_sPia' + str(sig_Pia) + '_bPi' + str(bernoulli_Pi) + '_mPi' + 
										str(mu_Pi) + '_sPi' + str(sig_Pi) + '_' + sample_longSWs_1st + 'Smp1st/' )

							plt_save_dir = str( figs_save_dir + init_dir + model_dir )


							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
							# # (2). Construct file names for EM_model_data.npz file and also B file if we've trained 2nd model.
							#
							fname_EMlrn = str( 'EM_model_data_' + str(num_SWs) + 'SWs_trn' + str(pct_xVal_train).replace('.','pt') + '_xTTT' + str(xTimesThruTrain) \
		 										+ '_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) + '_rand' + str(rand) + rsTag + Btag + '.npz' ) 
							
							fname_inferPL = fname_EMlrn.replace('EM_model_data_','SWs_inferred_postLrn_')
							#
							# fname_EMlrn = str('EM_model_data_' + str(num_SWs) + 'SWs_' + str(pct_xVal_train).replace('.','pt') + 'trn_ylims' \
							# 	+ str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) + '_rand' + str(rand) + '.npz' )
							# #
							# if train_2nd_model:
							# 	fname_EMlrn = str(fname_EMlrn[:-4] + 'B.npz')

							

							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
							# # (3).  Load in model from saved npz file.
							print( str(EM_learning_Dir + init_dir + model_dir + fname_EMlrn) )
							try:
								data = np.load( str(EM_learning_Dir + init_dir + model_dir + fname_EMlrn) )
							except:
								print('NPZ File not there. Moving on.')
								break
							print('Data keys from EM model learning npz data file.')
							print(data.keys())
							q 					= data['q']
							ri  				= data['ri'] 
							ria 		  		= data['ria']
							#
							qp 					= data['qp']
							rip  				= data['rip'] 
							riap 		  		= data['riap']
							#  
							q_init 				= data['q_init']
							ri_init  			= data['ri_init'] 
							ria_init  			= data['ria_init']
							# 
							q_snapshots 		= data['q_snapshots']
							ri_snapshots  		= data['ri_snapshots'] 
							ria_snapshots  		= data['ria_snapshots'] 
							try:
								ds_fctr_snapshots 	= data['ds_fctr_snapshots']
							except:
								print('ds_fctr_snapshots not there yet.')
							#
							pjoint_train  		= data['pjoint_train'] 
							pjoint_test   		= data['pjoint_test'] 
							#
							Z_inferred_train 			= data['Z_inferred_train']
							#Y_inferred_train 			= data['Y_inferred_train']
							pyiEq1_gvnZ_train 			= data['pyiEq1_gvnZ_train'] # p(yi=1|z) during learning
							#
							Z_inferred_test 			= data['Z_inferred_test']
							#Y_inferred_test 			= data['Y_inferred_test']
							pyiEq1_gvnZ_test 			= data['pyiEq1_gvnZ_test'] # p(yi=1|z) during learning
														#


							#
							# if this variable hasnt been saved (it takes a lot of time to compute in 
							# pgmCA_synthData and isnt that useful I dont think), set it to be all 1's
							try:
								zInferSampledRaw_train 	= data['zInferSampledRaw_train']
							except:
								zInferSampledRaw_train = np.ones(M)
							#
							argsRecModelLearn 		= data['argsRec']
							#
							translate_Tru2LrnShuff 	= data['translate_Tru2LrnShuff']
							translate_Tru2Tru 		= data['translate_Tru2Tru']
							translate_Lrn2TruShuff	= data['translate_Lrn2TruShuff']
							translate_Lrn2Lrn 		= data['translate_Lrn2Lrn']
							#
							del data


							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
							# # (4).  Load in inference-post-learn data file.
							#
							if flg_includeInfer:
								data = np.load( str( inferPL_data_dir + init_dir + model_dir + fname_inferPL) ) 
								print('Data keys from inferred allSWs post-learn npz data file.')
								print(data.keys())
								#
								Ycell_hist_allSWs 		= data['yInferSampled_train_postLrn'] + data['yInferSampled_test_postLrn']
								Zassem_hist_allSWs 		= data['zInferSampledT_train_postLrn'] + data['zInferSampledT_test_postLrn']
								#
								sortCAs_byActivity 	 	= np.argsort(Zassem_hist_allSWs[:-1])[::-1]
								sortCells_byActivity 	= np.argsort(Ycell_hist_allSWs[:-1])[::-1]
								#
								sortCells_byActivity 	= np.arange(N) # over riding.
								sortCAs_byActivity 		= np.arange(M)
							else:
								sortCells_byActivity 	= np.arange(N)
								sortCAs_byActivity 		= np.arange(M)
							#
							TH_bounds = np.array([0.5, 0.7]) # upper and lower bounds for threshold to count up CperA and AperC
							#
							numSnaps 		= ria_snapshots.shape[0]
							snaps 			= range(1,numSnaps) # range(numSnaps-1,numSnaps)
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

							

							

							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
							# # (5).  Plot snapshot of model during learning and model at init against ground truth model.
							#
							if flg_plot_snaps:
								#
								plt_snap_dir = str( plt_save_dir + 'Snapshots/' )
								if not os.path.exists( plt_snap_dir ):
									os.makedirs( plt_snap_dir )		
								#
								print('Number of snapshots are ',numSnaps)	
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
									Zassem_hist_InferSnap = Zassem_hist_InferSnap[sortCAs_byActivity]
									#
									QSnap 			= rc.sig(q_snapshots[i])
									PiSnap 	 	  	= ( 1-rc.sig(ri_snapshots[i]) )[sortCells_byActivity]
									PiaSnap 	  	= ( 1-rc.sig(ria_snapshots[i]) )[np.ix_(sortCells_byActivity,sortCAs_byActivity)]
									numCAsSnapUB	= ( PiaSnap>TH_bounds.min()).sum(axis=1)
									numCAsSnapLB	= ( PiaSnap>TH_bounds.max()).sum(axis=1)
									numCellsSnapUB	= ( PiaSnap>TH_bounds.min()).sum(axis=0)
									numCellsSnapLB 	= ( PiaSnap>TH_bounds.max()).sum(axis=0)

									plt_title = str( 'Add something here --- snapshot '+ str(i) )
									# str('Learned Model ' + ModelType + ' Params w/ LR =' + str(learning_rate) + \
									# 'LRpi =' + str(lRateScale_Pi) + ' :: ' + str(num_SWs) + ' SW data & ' + str(sampAtSnap) + ' EM samples' )

									if flg_includeInfer:
										plt_save_tag1 = str( fname_EMlrn[:-4].replace('EM_model_data_','SnaphotsReal_') + '_snap' + str(i) )
										pf.plot_learned_model(PiSnap, PiaSnap, QSnap, numCAsSnapUB, numCAsSnapLB, numCellsSnapUB, numCellsSnapLB, 
											Zassem_hist_InferSnap, Ycell_hist_InferSnap, Ycell_hist_allSWs, 
											TH_bounds, maxNumCells, maxNumCAs, maxPiQ, np.ones_like(Ycell_hist_allSWs), nY_InferSnap, nZ_InferSnap, \
											sampAtSnap, num_EM_samps, plt_snap_dir, plt_save_tag1, plt_title)



									plt_save_tag2 = str( fname_EMlrn[:-4].replace('EM_model_data_','SnaphotsSynth_') + '_snap' + str(i) )
									pf.plot_params_init_n_learned(q, ri, ria, q_snapshots[i], ri_snapshots[i], ria_snapshots[i], q_init, ri_init, ria_init, \
										translate_Tru2LrnShuff[0], translate_Tru2Tru, translate_Lrn2TruShuff[0],translate_Lrn2Lrn, zInferSampledRaw_train, \
										num_EM_samps, N, M, M_mod, learning_rate, lRateScale_Pi, params_init, params_init_str, rand, plt_snap_dir, plt_save_tag2)
						


									#'Ycell_hist_allSWs', 'YcellInf_hist_allSWs',





							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
							# # (6).  Compare init model to learned model at a snapshot.
							# This is to see what model learns when there is no cell assembly
							# structure in the data and all observed spike words are just the 
							# result of individual Pi's. I accomplish this by setting K = {0,0,0}
							#
							if flg_plot_compare2init and (K==0 and Kmin==0 and Kmax==0):
								#
								for i in snaps:
									plt.rc('font', weight='bold', size=10)
									f,ax = plt.subplots(1,3)
									ax[0].imshow( 1-rc.sig(ria_init), vmin=0, vmax=1 )
									ax[0].set_title('$P_{ia}$ init')
									#
									ax[1].imshow( 1-rc.sig(ria_snapshots[i]), vmin=0, vmax=1  )
									ax[1].set_title('$P_{ia}$ learned')
									#
									ax[2].plot([0,1],[0,1],'k--')
									ax[2].scatter( 1-rc.sig(ri), 1-rc.sig(ri_init), s=20, color='black',label='init',alpha=0.1 )
									ax[2].scatter( 1-rc.sig(ri), 1-rc.sig(ri_snapshots[i]), s=50, color='blue',label='lrnd',alpha=0.5 )
									ax[2].scatter( 1-rc.sig(q), 1-rc.sig(q_init), s=20, color='black',alpha=0.1 )# label='init'
									ax[2].scatter( 1-rc.sig(q), 1-rc.sig(q_snapshots[i]), s=50, color='green',label='q',alpha=0.5 )
									ax[2].set_title('$P_i and Q$')
									ax[2].set_xlabel('true value')
									ax[2].set_aspect('equal',adjustable='box')
									ax[2].legend()
									plt.show()

									# ax[1][1].plot([vmin2,vmax2],[vmin2,vmax2],'k--')
									# ax[1][1].scatter(Pi,Pi_init,s=20, color='black',label='init',alpha=0.1)
									# ax[1][1].scatter(Q,Q_init,s=20,color='black',alpha=0.1)
									# ax[1][1].scatter(Pi,Pip,s=80,color='blue',label='$r_i$ learned',alpha=0.5)



							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
							# # (7).  Plot horizontal streaks in Pia matrix vs Pi-Pip. Are cells that are very active
							#  		  in many CAs basically just underestimated in the model, i.e.,  Pi-Pip is large.?
							#
							print('Makes sense! Streaks in Pia are where Pip underestimates Pi.')
							#
							if flg_plot_PiaStreaks:
								Pi = 1 - rc.sig(ri)
								Pip = 1 - rc.sig(ri_snapshots[-1])
								Piap = 1 - rc.sig(ria_snapshots[-1])
								#plt.rc('font', weight='bold', size=14)
								plt.plot(Pi - Pip, color='black',linewidth=3)
								#
								for i in range(N):
									quietOnes = Piap[i]<0.5
									plt.scatter(i*np.ones(quietOnes.sum()), Piap[i][quietOnes])
								#
								#plt.legend()
								plt.xlabel('Cell id N')
								plt.ylabel('$Pi - Pip$')
								plt.title('Streaks in Pia and Pi-Pip discrepancies post learning')
								plt.show()






							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
							#
							# (Plot 8). Plot Cross validation - For training and test, plot pjoint vs EM iteration #.
							#
							if flg_plot_crossValidation:
								# try:
								if False: # This is old. Get it in vis_learned_pgmCA
									# (Plot 1D & 1E). 
									fname_xVal = fname_EMlrn[:-4].replace('EM_model_data_','CrossValidation_')
									pf.plot_xValidation(Z_inferred_train, pjoint_train, pjoint_test, plt_save_dir, fname_xVal, kerns=[10,100])
									#

								# except:
								# 	print('problem with Cross validation.')
								



							# Plot of Pia and Pi for models A & B (one learned on 50% training and one on 50% test data)
							if flg_plot_compare_models_AnB and not train_2nd_model:
								#
								data = np.load( str(EM_learning_Dir + init_dir + model_dir + fname_EMlrn[:-4] + 'B.npz') )
								# 
								qB 					= data['q']
								riB  				= data['ri'] 
								riaB 		  		= data['ria']
								#
								qpB 				= data['qp']
								ripB  				= data['rip'] 
								riapB 		  		= data['riap']
								# 
								q_snapshotsB 		= data['q_snapshots']
								ri_snapshotsB  		= data['ri_snapshots'] 
								ria_snapshotsB  	= data['ria_snapshots'] 
								argsRecModelLearnB 	= data['argsRec']
								del data
								#
								# #
								#
								#for i in snaps:
								f,ax = plt.subplots(2,2)
								#
								ax[0][1].scatter( 1-rc.sig(rip), range(N), s=15, marker='.', color='blue' )
								ax[0][1].scatter( 1-rc.sig(ri), range(N), s=15, marker='.', color='black', alpha=0.3 )
								ax[0][1].scatter( rc.sig(qp), np.round(N/2), marker='x', color='red' )
								ax[0][1].scatter( rc.sig(q), np.round(N/2), marker='+', color='green' )
								ax[0][1].set_title('Pi, Pip, Q, Qp')
								ax[0][1].invert_yaxis()
								ax[0][1].grid()
								ax[0][1].set_aspect('auto')
								#
								ax[0][0].imshow(1-rc.sig(riap),vmin=0,vmax=1)
								ax[0][0].set_ylabel('Model 1A')
								#
								ax[1][1].scatter( 1-rc.sig(ripB), range(N), s=15, marker='.', color='blue'  )
								ax[1][1].scatter( 1-rc.sig(riB) , range(N), s=15, marker='.', color='black', alpha=0.3 )
								ax[1][1].scatter( rc.sig(qpB), np.round(N/2), marker='x', color='red' )
								ax[1][1].scatter( rc.sig(qB), np.round(N/2), marker='+', color='green' )
								ax[1][1].invert_yaxis()
								ax[1][1].grid()
								ax[1][1].set_aspect('auto')
								#
								ax[1][0].imshow(1-rc.sig(riapB),vmin=0,vmax=1)
								ax[1][0].set_ylabel('Model 1B')
								#
								plt.suptitle( str('Compare models') )
								plt.show()




								# translate_Lrn2TruShuff,dot_prod_Lrn2Tru,translate_Lrn2Tru, translate_Lrn2Lrn, Perm_Lrn2Tru, dropWarn_Lrn2Tru = \
								# rc.translate_CAs_LrnAndTru(A=1-Pia1, Atag='model1', B=1-Pia2, Btag='model2', verbose=True)


								# pf.visualize_translation_of_CAs( A=(1-Pia1), Atag='model1', B=(1-Pia2), Btag='model2', 
								# translate=translate_Lrn2TruShuff[0], translate2=translate_Lrn2TruShuff[1], 
								# dot_prod=dot_prod_Lrn2Tru[0], dot_prod2=dot_prod_Lrn2Tru[1], trans_preShuff=translate_Lrn2Tru, ind=translate_Lrn2Lrn, 
								# Perm=Perm_Lrn2Tru, dropWarn=dropWarn_Lrn2Tru, numSamps=0, r=0, plt_save_dir='./', fname_tag='test' )



								# # Piap_snap = rc.sig(ria_snapshots[i])
								# # dp_significant = (dot_prod_Lrn2Tru[0]>0.1)[None,:]
								# # x = Pia_mod[:,translate_Lrn2TruShuff[0]]*dp_significant
								# # y = Piap_snap[:,translate_Lrn2Lrn]*dp_significant
								# # MSE = ( x - y )**2
								# # ria_MSE[i,:] = np.array([MSE.mean(), MSE.std(), MSE.max()])

								# Ay = 1-Pia1[:,translate_Lrn2TruShuff[0]]
								# Be = 1-Pia2[:,translate_Lrn2Lrn]

								# Agree = np.bitwise_and( (Ay>0.5), (Be>0.5) ) 										# where they agree when thresholded.
								# AB_match_inds = np.where( (Ay>0.5).sum(axis=0)+(Be>0.5).sum(axis=0) == 2*Agree.sum(axis=0) )[0] 	# where they agree one-for-one.
								

								# (Agree.sum(axis=0)[AB_match_inds]>1).sum() 											# number that agree that have more than one cell.	


								# print('Number of CAs that match with |CA|=2, 2<|CA|<6')
								# print( np.where( (Agree.sum(axis=0)[AB_match_inds]==2) )[0].size, np.where( np.bitwise_and( (Agree.sum(axis=0)[AB_match_inds]>2), (Agree.sum(axis=0)[AB_match_inds]<6) ) )[0].size )
								# print('Number of CAs in 1st model with |CA|=2, 2<|CA|<6')
								# print( np.where( (Ay>0.5).sum(axis=0)==2 )[0].size, np.where( np.bitwise_and( (Ay>0.5).sum(axis=0)>2, (Ay>0.5).sum(axis=0)<6 ) )[0].size )
								# print('Number of CAs in 2nd model with |CA|=2, 2<|CA|<6')
								# print( np.where( (Be>0.5).sum(axis=0)==2 )[0].size, np.where( np.bitwise_and( (Be>0.5).sum(axis=0)>2, (Be>0.5).sum(axis=0)<6 ) )[0].size )




								# f,ax = plt.subplots(2,2)
								# ax[0][0].imshow(Ay,vmin=0,vmax=1, cmap='viridis' )
								# ax[0][0].set_title('model1')
								# #
								# ax[1][0].imshow(Be,vmin=0,vmax=1, cmap='viridis' )
								# ax[1][0].set_title('reordered model2')
								# #
								# ax[0][1].imshow( np.abs( Ay - Be ),vmin=0, vmax=1, cmap='viridis' )
								# ax[0][1].set_title('difference |M1-M2| ')
								# #
								# #ax[1][1].scatter( np.tile( np.arange(M), (M,1)), np.abs( (Ay>0.5) - (Be>0.5) ),s = 20, alpha=0.05, label='col sums diff' )
								# ax[1][1].plot(dot_prod_Lrn2Tru[0],label='dot prod', color='red')
								# ax[1][1].set_title('')

								# plt.show()

								# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
								# plt.scatter( np.where(Ay>0.5)[1], np.where(Ay>0.5)[0], alpha=0.5, marker='o', color='blue' )	
								# plt.scatter( np.where(Be>0.5)[1], np.where(Be>0.5)[0], alpha=0.5, marker='s', color='yellow' )
								# #plt.scatter( np.where(Agree)[1],  np.where(Agree)[0], alpha=1.0, marker='x', color='red' )	
								# for ii in range(len(AB_match_inds)):
								# 	plt.text( AB_match_inds[ii], M, str(Agree.sum(axis=0)[AB_match_inds[ii]]), color='green', fontsize=6 )	
								# #
								# for ii in range(M):
								# 	plt.text( ii, M+2, str( (Ay>0.5).sum(axis=0)[ii] ), color='blue', fontsize=6 )
								# #
								# for ii in range(M):
								# 	plt.text( ii, M+4, str( (Be>0.5).sum(axis=0)[ii] ), color='black', fontsize=6 )
								# #
								# plt.axis([0, M, 0, N+10])
								# plt.xlabel('CA id')
								# plt.ylabel('Cell id')
								# plt.title('Compare 2 models learned w/ 50/50 test train data split.')
								# plt.show()


										







