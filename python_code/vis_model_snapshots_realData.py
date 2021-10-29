import argparse
import numpy as np 
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import os
import time

import utils.data_manipulation as dm
import utils.plot_functions as pf
import utils.retina_computation as rc
import utils.sbatch_scripts as ss

from textwrap import wrap

import difflib







# In Greg Field's data :::
#        Cell Types : Num Cells
# ------------------------------
# offBriskTransient : 55 cells
# offBriskSustained : 43 cells
#  onBriskTransient : 39 cells
#      offExpanding : 13 cells
#      offTransient :  4 cells
#  onBriskSustained :  6 cells
#       onTransient :  7 cells
#       dsOnoffDown :  7 cells
#      dsOnoffRight :  3 cells
#       dsOnoffLeft :  3 cells
#         dsOnoffUp :  2 cells


#
dirHomeLoc, dirScratch = dm.set_dir_tree()

# Parameters we can loop over.
stims = ['NatMov','Wnoise']
cellSubTypeCombinations = [ ['offBriskTransient','offBriskSustained'], ['offBriskTransient','onBriskTransient'] ]

							# ['offBriskTransient'], ['offBriskSustained'], ['onBriskTransient'], \
							# ['offBriskTransient','offBriskSustained'], ['offBriskTransient','onBriskTransient'] ] # a list of lists. Each internal list is a combination of cell sub types to consider as a group to find Cell Assemblies within them.

#num_EM_Samples 	= [50000] #, 500000] #[50000, 100000, 500000] # [5000, 10000, 50000, 100000, 500000, 1000000] # number of steps to run full EM algorithm - alternating Inference and Learning.
#cell_types = ['allCells'] 

model_CA_overcompleteness = [1] 		# [1,2] 	# how many times more cell assemblies we have than cells (1 means complete - N=M, 2 means 2x overcomplete)
SW_bins = [0, 1, 2] 					# ms. Build up spikewords from groups of cells in the same trial that fire within SW_bins of eachother.
learning_rates = [0.1, 0.5, 1.0]


yLo_Vals 		= [0] #[1] 		# If |y|<=yLo, then we force the z=0 inference solution and change Pi. This defines cells assemblies to be more than 1 cell.
yHi_Vals 		= [300] 		# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution and change Pia.
yMinSWs 		= [1,3] #[1,2,3]			# DOING BELOW THING WITH YYY inside pgm functions. --> (set to 0, so does nothing) 
								# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<yLo.
#

train_2nd_model = True
pct_xVal_train = 0.5



num_EM_rands = 3 
ds_fctr_snapshots = 1000 	# Take a sample of model parameters every *ds_fctr* time steps to compute and plot MSE after we determine permutation of learned CA's to True CA's


# NEEDED FOR THE rasZ and statI
#N 	 	= 55 # number of cells in real data
maxTms 	= 6000 # ms 
minTms 	= 0 # ms 



								# number of times to randomize samples and rerun EM on same data generated from single synthetic model.
params_init 	= 'DiagonalPia' 						# Options: {'True', RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise), 'DiagonalPia' (w/ sig_init & C_noise) }
#sig_init 		= np.array([ 0.01, 0.05, 0.05 ])	# STD on parameter initializations from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigQ_init 		= 0.01	# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigPi_init 		= 0.05	# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigPia_init 	= 0.05	# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.

# Parameter initializations for EM algorithm to learn model parameters
Z_hot 				= 2 	# initialization for Q value (how many 1's expected in binary z-vector)
C_noise_ri 			= 1.
C_noise_ria 		= 1.
lRateScale_Pi 		= 1.0	# Multiplicative scaling to Pi learning rate. If set to zero, Pi taken out of model essentially.

# Flags for the EM (inference & learning) algorithm.
flg_include_Zeq0_inferS  = [True] # [True, False]
verbose_EM 				 = False



TH_bounds = np.array([0.5, 0.7]) # upper and lower bounds for threshold to count up CperA and AperC for plotting in compute_dataGen_Histograms


run_raster 	= False
plt_xVal 	= True
plt_modelPair = True



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (1). Load in npz file and extract data from it.
sig_init 		= np.array([ sigQ_init, sigPi_init, sigPia_init ])
params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (0). Set up directories and create dirs if necessary.
dirHome, dirScratch = dm.set_dir_tree()
EM_learning_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Models_learned_EM/')
SW_extracted_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')
rasterZ_data_dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Rasters_trial_vs_time_zInferred/')
figs_save_dir 		= str( dirScratch + 'figs/PGM_analysis/EM_algorithm/Greg_retinal_data/')


#for num_EM_samps in num_EM_Samples:
	#
#for cell_type in cell_types:
	#	



for rand in range(num_EM_rands):
	#	
	for cellSubTypes in cellSubTypeCombinations:
		#
		for k,stim in enumerate(stims):
			#
			for learning_rate in learning_rates:
				#
				for SW_bin in SW_bins:
					#
					for yMinSW in yMinSWs:
						#
						#
						#
						# # # NOTE: THESE BELOW DONT CHANGE !!! # # # # # # # # # # # # # #
						for yLo in yLo_Vals:
							#
							for yHi in yHi_Vals:
								#
								for overcomp in model_CA_overcompleteness:
									#
									for flg_include_Zeq0_infer in flg_include_Zeq0_inferS:
										# # # NOTE: THESE ABOVE DONT CHANGE !!! # # # # # # # # # # # # # #
										#
										#
										#
										msBins = 1+2*SW_bin

										if flg_include_Zeq0_infer:
											z0_tag='_zeq0'
										else:
											z0_tag='_zneq0'


										B_file_there = True

										try:
										# if True:

											# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
											# # (1).  Set up directory structure and filename. Load it in and extract variables.
											init_dir = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') + \
													'_LRpi' + str(lRateScale_Pi).replace('.','pt') +'/' )

											#
											# #
											# Find directory (model_dir) with unknown N and M that matches cell_type and yMin
											CST = str(cellSubTypes).replace('\'','').replace(' ','') # convert cell_type list into a string of expected format.
											#
											subDirs = os.listdir( str(EM_learning_Dir + init_dir) ) 

											model_dir = [s for s in subDirs if CST in s and str( z0_tag + '_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) ) in s ]
											#print(model_dir)
											#
											if len(model_dir) != 1:
												print('I am expecting one matching directory. I have ',len(model_dir))
												print(model_dir)
											#
											model_dir = str(model_dir[0]+'/')	
											print(model_dir)
											#
											a = model_dir.find('_N')
											b = model_dir.find('_M')	
											c = model_dir.find(z0_tag)
											#
											N = int(model_dir[a+2:b])
											M = int(model_dir[b+2:c])
											#
											# Find npz file (model_file) inside model_dir with unknown numSWs, numTrain, numTest but matching stim, msBins, EMsamps and rand.
											filesInDir = os.listdir( str(EM_learning_Dir + init_dir + model_dir) ) 
											#
											model_files = [s for s in filesInDir if str('LearnedModel_' + stim) in s and \
													str( 'SWs_' + str(pct_xVal_train).replace('.','pt') + 'trn_' + str(msBins) + 'msBins_rand' + str(rand) ) in s ]
											#
											mfA = [s for s in model_files if str('rand' + str(rand) + '.npz') in s ] 	# Model learned from train data.
											mfB = [s for s in model_files if str('rand' + str(rand) + 'B.npz') in s ] 	# Model learned from test data.
											print(mfA)
											print(mfB)

											if not mfB:
												B_file_there = False


											# Load in model from saved npz file.
											dataA = np.load( str(EM_learning_Dir + init_dir + model_dir + mfA[0]) )
											print(dataA.keys())
											q_snapshotsA 		= dataA['q_snapshots']
											ri_snapshotsA  		= dataA['ri_snapshots'] 
											ria_snapshotsA  	= dataA['ria_snapshots'] 
											Y_inferred_trainA 	= dataA['Y_inferred_train']
											Z_inferred_trainA  	= dataA['Z_inferred_train']
											pjoint_trainA 		= dataA['pjoint_train']
											pjoint_testA 		= dataA['pjoint_test']
											argsRecModelLearnA 	= dataA['argsRec']
											#
											numSnaps 		= ria_snapshotsA.shape[0]
											maxPiQA 		= np.array([ rc.sig(q_snapshotsA).max(),	(1-rc.sig(ri_snapshotsA)).max() ]).max()
											#
											maxNumCellsA 	= np.zeros(2)
											maxNumCellsA[0] = ( (1-rc.sig(ria_snapshotsA))>TH_bounds.max()).sum(axis=1).max()
											maxNumCellsA[1] = ( (1-rc.sig(ria_snapshotsA))>TH_bounds.min()).sum(axis=1).max()
											maxNumCellsA 	= maxNumCellsA.astype(int)
											#
											maxNumCAsA 		= np.zeros(2)
											maxNumCAsA[0]	= ( (1-rc.sig(ria_snapshotsA))>TH_bounds.max()).sum(axis=2).max()
											maxNumCAsA[1]	= ( (1-rc.sig(ria_snapshotsA))>TH_bounds.min()).sum(axis=2).max()
											maxNumCAsA 		= maxNumCAsA.astype(int)
											
											
											if train_2nd_model and B_file_there:
												#
												dataB = np.load( str(EM_learning_Dir + init_dir + model_dir + mfB[0]) )
												q_snapshotsB 		= dataB['q_snapshots']
												ri_snapshotsB  		= dataB['ri_snapshots'] 
												ria_snapshotsB  	= dataB['ria_snapshots'] 
												Y_inferred_trainB 	= dataB['Y_inferred_train']
												Z_inferred_trainB  	= dataB['Z_inferred_train']
												pjoint_trainB 		= dataB['pjoint_train']
												pjoint_testB 		= dataB['pjoint_test']
												argsRecModelLearnB 	= dataB['argsRec']
												#
												#numSnaps 		= ria_snapshotsA.shape[0]
												maxPiQB 		= np.array([ rc.sig(q_snapshotsB).max(),	(1-rc.sig(ri_snapshotsB)).max() ]).max()
												#
												maxNumCellsB 	= np.zeros(2)
												maxNumCellsB[0] = ( (1-rc.sig(ria_snapshotsB))>TH_bounds.max()).sum(axis=1).max()
												maxNumCellsB[1] = ( (1-rc.sig(ria_snapshotsB))>TH_bounds.min()).sum(axis=1).max()
												maxNumCellsB 	= maxNumCellsB.astype(int)
												#
												maxNumCAsB 		= np.zeros(2)
												maxNumCAsB[0]	= ( (1-rc.sig(ria_snapshotsB))>TH_bounds.max()).sum(axis=2).max()
												maxNumCAsB[1]	= ( (1-rc.sig(ria_snapshotsB))>TH_bounds.min()).sum(axis=2).max()
												maxNumCAsB 		= maxNumCAsA.astype(int)
											


											plt_save_dir = str( figs_save_dir + init_dir + model_dir )
											plt_snap_dir = str( plt_save_dir + 'Snapshots/' )
											if not os.path.exists( plt_snap_dir ):
												os.makedirs( plt_snap_dir )		



											if run_raster:
												# Load in raster data file of inferred spike words on all data made after the entire model was learned.
												dataRA = np.load( str( rasterZ_data_dir + init_dir + model_dir + mfA[0].replace('LearnedModel_','rasterZ_allSWs_') ) )
												print(dataRA.keys())
												#
												Ycell_hist_allSWsA 		= dataRA['Ycell_hist_allSWs']
												Zassem_hist_allSWsA 	= dataRA['Zassem_hist_allSWs']
												#
												nY_allSWsA 				= dataRA['nY_allSWs']
												nZ_allSWsA 				= dataRA['nZ_allSWs']
												#
												num_EM_samps 			= len(nY_allSWsA)
												#
												sortCAs_byActivityA   = np.argsort(Zassem_hist_allSWsA[:-1])[::-1]
												sortCells_byActivityA = np.argsort(Ycell_hist_allSWsA[:-1])[::-1]


												if train_2nd_model and B_file_there:
													# Load in raster data file of inferred spike words on all data made after the entire model was learned.
													dataRB = np.load( str( rasterZ_data_dir + init_dir + model_dir + mfB[0].replace('LearnedModel_','rasterZ_allSWs_') ) )
													print(dataRB.keys())
													#
													Ycell_hist_allSWsB 		= dataRB['Ycell_hist_allSWs']
													Zassem_hist_allSWsB 	= dataRB['Zassem_hist_allSWs']
													#
													nY_allSWsB 				= dataRB['nY_allSWs']
													nZ_allSWsB 				= dataRB['nZ_allSWs']
													#
													num_EM_sampsB = len(nY_allSWsB)
													#
													sortCAs_byActivityB 	= np.argsort(Zassem_hist_allSWsB[:-1])[::-1]
													sortCells_byActivityB 	= np.argsort(Ycell_hist_allSWsB[:-1])[::-1]


												# UNCOMMENT THIS IF YOU DONT WANT TO SORT !
												sortCells_byActivityA = np.arange(N)
												sortCAs_byActivityA 	= np.arange(M)
												sortCells_byActivityB = np.arange(N)
												sortCAs_byActivityB 	= np.arange(M)

											else:
												#
												# Get num_EM_samps from file name.
												a = mfA[0].find(stim)
												b = mfA[0].find( str('SWs_' + str(pct_xVal_train).replace('.','pt') + 'trn_') )	
												num_EM_samps = int(mfA[0][a+1+len(stim):b])
												#
												sortCells_byActivityA	= np.arange(N)
												sortCAs_byActivityA 	= np.arange(M)

												Ycell_hist_allSWsA 		= np.zeros(N)
												Zassem_hist_allSWsA 	= np.zeros(M)
												#
												nY_allSWsA 				= np.zeros(num_EM_samps)
												nZ_allSWsA 				= np.zeros(num_EM_samps)
												#
												if train_2nd_model and B_file_there:
													#
													sortCells_byActivityB	= np.arange(N)
													sortCAs_byActivityB 	= np.arange(M)

													Ycell_hist_allSWsB 		= np.zeros(N)
													Zassem_hist_allSWsB 	= np.zeros(M)
													#
													nY_allSWsB 				= np.zeros(num_EM_samps)
													nZ_allSWsB 				= np.zeros(num_EM_samps)
													#
													# FOR NOW I AM ASSUMING THERE ARE SAME NUMBER OF EM SAMPLES IN A AND B FILES.
													# # Get num_EM_samps from file name.
													# a = mfB[0].find(stim)
													# b = mfB[0].find( str('SWs_' + str(pct_xVal_train).replace('.','pt') + 'trn_') )	
													# num_EM_sampsB = int(mfB[0][a+1+len(stim):b])


											print('Number of snapshots are ',numSnaps)	
											snaps = range(numSnaps-1,numSnaps) #range(ria_snapshots.shape[0])
											#
											for i in snaps:
												#
												sampAtSnap = int( i*num_EM_samps/(numSnaps-1) )
												#
												if run_raster:
													# Stats on active cells and cell assemblies inferred during EM algorithm
													print(i,' compute Inference statistics')
													t0 = time.time()
													#
													Ycell_hist_InferSnapA, Zassem_hist_InferSnapA, nY_InferSnapA, nZ_InferSnapA, \
														CA_coactivity_InferSnapA, Cell_coactivity_InferSnapA = rc.compute_dataGen_Histograms( \
														Y_inferred_trainA[:sampAtSnap], Z_inferred_trainA[:sampAtSnap], M, N)
													#
													if train_2nd_model:
														Ycell_hist_InferSnapB, Zassem_hist_InferSnapB, nY_InferSnapB, nZ_InferSnapB, \
															CA_coactivity_InferSnapB, Cell_coactivity_InferSnapB = rc.compute_dataGen_Histograms( \
															Y_inferred_trainB[:sampAtSnap], Z_inferred_trainB[:sampAtSnap], M, N)
													#
													t1 = time.time()
													print('Done w/ inference stats : time = ',t1-t0) # Fast enough: ~10 seconds

												else:

													Ycell_hist_InferSnapA=np.zeros(N+1)
													Zassem_hist_InferSnapA=np.zeros(M+1)
													nY_InferSnapA = [len(xx) for xx in Y_inferred_trainA]
													nZ_InferSnapA = [len(xx) for xx in Z_inferred_trainA]
													CA_coactivity_InferSnapA=0
													Cell_coactivity_InferSnapA=0
													#
													if train_2nd_model and B_file_there:
														#
														Ycell_hist_InferSnapB=np.zeros(N+1)
														Zassem_hist_InferSnapB=np.zeros(M+1)
														nY_InferSnapB = [len(xx) for xx in Y_inferred_trainB]
														nZ_InferSnapB = [len(xx) for xx in Z_inferred_trainB]
														CA_coactivity_InferSnapB=0
														Cell_coactivity_InferSnapB=0
												# # # # #
												#
												# #
												# # #
												#
												Ycell_hist_InferSnapA  = Ycell_hist_InferSnapA[sortCells_byActivityA]
												Zassem_hist_InferSnapA = Zassem_hist_InferSnapA[sortCAs_byActivityA]
												#
												QSnapA 			= rc.sig(q_snapshotsA[i])
												PiSnapA 	 	= ( 1-rc.sig(ri_snapshotsA[i]) )[sortCells_byActivityA]
												PiaSnapA 	  	= ( 1-rc.sig(ria_snapshotsA[i]) )[np.ix_(sortCells_byActivityA,sortCAs_byActivityA)]
												numCAsSnapUbA	= ( PiaSnapA>TH_bounds.min()).sum(axis=1)
												numCAsSnapLbA	= ( PiaSnapA>TH_bounds.max()).sum(axis=1)
												numCellsSnapUbA	= ( PiaSnapA>TH_bounds.min()).sum(axis=0)
												numCellsSnapLbA = ( PiaSnapA>TH_bounds.max()).sum(axis=0)
												#
												plt_titleA = str( 'Add something here --- snapshot '+ str(i) )
												plt_save_tagA = str( mfA[0][:-4] + '_snap' + str(i) )
												#
												pf.plot_learned_model(PiSnapA, PiaSnapA, QSnapA, numCAsSnapUbA, numCAsSnapLbA, numCellsSnapUbA, \
													numCellsSnapLbA, Zassem_hist_InferSnapA, Ycell_hist_InferSnapA, Ycell_hist_allSWsA, \
													TH_bounds, maxNumCellsA, maxNumCAsA, maxPiQA, nY_allSWsA, nY_InferSnapA, nZ_InferSnapA, \
													sampAtSnap, num_EM_samps, plt_snap_dir, plt_save_tagA, plt_titleA)

												
												if train_2nd_model and B_file_there:
													Ycell_hist_InferSnapB  = Ycell_hist_InferSnapB[sortCells_byActivityB]
													Zassem_hist_InferSnapB = Zassem_hist_InferSnapB[sortCAs_byActivityB]
													#
													QSnapB 			= rc.sig(q_snapshotsB[i])
													PiSnapB 	 	= ( 1-rc.sig(ri_snapshotsB[i]) )[sortCells_byActivityB]
													PiaSnapB 	  	= ( 1-rc.sig(ria_snapshotsB[i]) )[np.ix_(sortCells_byActivityB,sortCAs_byActivityB)]
													numCAsSnapUbB	= ( PiaSnapB>TH_bounds.min()).sum(axis=1)
													numCAsSnapLbB	= ( PiaSnapB>TH_bounds.max()).sum(axis=1)
													numCellsSnapUbB	= ( PiaSnapB>TH_bounds.min()).sum(axis=0)
													numCellsSnapLbB = ( PiaSnapB>TH_bounds.max()).sum(axis=0)
													#
													plt_titleB = str( 'Add something here --- snapshot '+ str(i) )
													plt_save_tagB = str( mfB[0][:-4] + '_snap' + str(i) )
													#
													pf.plot_learned_model(PiSnapB, PiaSnapB, QSnapB, numCAsSnapUbB, numCAsSnapLbB, numCellsSnapUbB, numCellsSnapLbB, 
														Zassem_hist_InferSnapB, Ycell_hist_InferSnapB, Ycell_hist_allSWsB, 
														TH_bounds, maxNumCellsB, maxNumCAsB, maxPiQB, nY_allSWsB, nY_InferSnapB, nZ_InferSnapB, \
														sampAtSnap, num_EM_samps, plt_snap_dir, plt_save_tagB, plt_titleB)



												# Plot two models learned on 50/50 split test and train side by side.
													if plt_modelPair:
														translate_Lrn2TruShuff,dot_prod_Lrn2Tru,translate_Lrn2Tru, translate_Lrn2Lrn, Perm_Lrn2Tru, dropWarn_Lrn2Tru = \
															rc.translate_CAs_LrnAndTru( A=PiaSnapA, Atag='A', B=PiaSnapB, Btag='B', verbose=False )
														#
														plt.figure( figsize=(20,10) ) # size units in inches
														plt.rc('font', weight='bold', size=8)
														f,ax = plt.subplots(3,2)
														#
														ax[0][0].imshow(PiaSnapA[:,translate_Lrn2Lrn],vmin=0,vmax=1,aspect='auto')
														ax[0][0].set_title('Model 1')
														ax[0][1].imshow(PiaSnapB[:,translate_Lrn2TruShuff[0]],vmin=0,vmax=1,aspect='auto')
														ax[0][1].set_title('Model 2')
														ax[0][0].set_ylabel('Pia')	
														#
														ax[1][0].scatter( range(M), (PiaSnapA[:,translate_Lrn2Lrn]>0.5).sum(axis=0), s=5 )
														ax[1][0].text(0.6*M,1.2, str('#|CA|>1: '+str( ( (PiaSnapA[:,translate_Lrn2Lrn]>0.5).sum(axis=0)>1 ).sum() ) ) )
														ax[1][1].scatter( range(M), (PiaSnapB[:,translate_Lrn2TruShuff[0]]>0.5).sum(axis=0), s=5 )
														ax[1][1].text(0.6*M,1.2, str('#|CA|>1: '+str( ( (PiaSnapB[:,translate_Lrn2TruShuff[0]]>0.5).sum(axis=0)>1 ).sum() ) ) )
														ax[1][0].set_ylabel('Pia Col sums')	
														#
														ax[2][0].scatter( range(N), PiSnapA, s=5 )
														ax[2][0].scatter( np.round(N/2), QSnapA, s=5, marker='x', color='red' )
														ax[2][1].scatter( range(N), PiSnapB, s=5 )
														ax[2][1].scatter( np.round(N/2), QSnapB, s=5, marker='x', color='red' )
														ax[2][0].set_ylabel('Pi and Q')	
														#
														plt.suptitle( str('Snapshot '+str(i)) )
														
														if not os.path.exists( str(plt_save_dir+ 'SnapModelPairs/') ):
															os.makedirs( str(plt_save_dir+ 'SnapModelPairs/') )
														plt.savefig( str(plt_save_dir + 'SnapModelPairs/' + str( mfA[0][:-4] + '_snapPair' + str(i) ) + '.png') )
														plt.close() 



											# # OUTSIDE SNAPSHOTS LOOP.
											#
											# Plot Pjoints for cross validation
											if plt_xVal:

												plt_xval_dir = str( figs_save_dir + init_dir + model_dir)
												fname_xVal = mfA[0][:-4].replace('LearnedModel_','CrossValidation_')
												pf.plot_xValidation(Z_inferred_trainA, pjoint_trainA, pjoint_testA, plt_save_dir, fname_xVal, kerns=[1000,5000])
										
												if train_2nd_model and B_file_there:
													#
													fname_xVal = mfB[0][:-4].replace('LearnedModel_','CrossValidation_')
													pf.plot_xValidation(Z_inferred_trainB, pjoint_trainB, pjoint_testB, plt_save_dir, fname_xVal, kerns=[1000,5000])

										except:
											print('Skip!')






