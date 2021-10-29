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




# Flags for plotting and saving things
doPlts = True
flg_derivs_during_learning_plot		= doPlts
flg_temporal_inference_plot			= doPlts
flg_plt_crossValidation_Joint		= False
flg_plt_crossValidation_Cond		= False
#
flg_learned_model_plot				= True # nice diagnostic plot
flg_histograms_learned_model		= False # maybe the only one i need !
flg_Pia_snapshots_gif 				= False # flashy thing to show Pia as it is learned thru EM.
#
flg_plot_PSTH_numCellsYmin 			= False # good one
flg_plot_PSTH_cellsAndCAs			= False # good one

#flg_dataGen_andSampling_statistics	= True # This takes a looong time with many EM_samples (NOT REALLY GOOD ANYMORE)
flg_2Dhist_pctYCapVsLenSW			= False # NOT VERY INFORMATIVE OR INTERESTING.
flg_2Dhist_numCAvsLenSW				= False # NOT VERY INFORMATIVE OR INTERESTING.
#
flg_compute_StatsPostLrn			= False 	# (Stats 1). Compute statistics on inferred Z & Y for all spike words- inferred from fixed model after EM algorithm.
flg_compute_StatsDuringEM			= False 	# (Stats 2). Compute statistics on inferred Z & Y for all EM samples, during learning. Can have many more EM samples than spike words.
											# 			 and stats can look shitty because of burn-in / learning phase.
flg_write_CSV_stats					= False				



figSaveFileType = 'png' # 'png' or 'pdf'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (0). Set up directories and create dirs if necessary.
dirHome, dirScratch = dm.set_dir_tree()
EM_learning_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Models_learned_EM/')
Infer_postLrn_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Rasters_trial_vs_time_zInferred/')
EM_learnStats_Dir	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/InferStats_from_EM_learning/')

EM_figs_dir 		= str( dirScratch + 'figs/PGM_analysis/EM_Algorithm/Greg_retinal_data/')
SW_extracted_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')
#
# Make directories for output data if  not already there.
if not os.path.exists( SW_extracted_Dir ):
	os.makedirs( SW_extracted_Dir )





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (1). Load in npz file and extract data from it.

# Parameters we can loop over.
cell_types = ['[offBriskTransient]'] #,'[onBriskTransient]','[offBriskSustained]',\
	#'[offBriskTransient,offBriskSustained]','[offBriskTransient,onBriskTransient]']
Ns = [55] #, 39, 43, 98, 94]


# cell_types = [ '[offBriskTransient,offBriskSustained]' ]
# Ns = [98]

# cell_types = [ '[offBriskTransient,onBriskTransient]' ]
# Ns = [94]

stims = ['NatMov','Wnoise']


train_2nd_modelS = [False] #[True,False] # False if there is no B file, True to run only B file, [True,False] to run both.

model_CA_overcompleteness = [1] #[1,2] 	# how many times more cell assemblies we have than cells (1 means complete - N=M, 2 means 2x overcomplete)
SW_bins = [2]#,1]#,0]			# ms. Build up spikewords from groups of cells in the same trial that fire within SW_bins of eachother.

yLo_Vals 		= [0] #[1,4] 		# If |y|<=yLo, then we force the z=0 inference solution and change Pi. This defines cells assemblies to be more than 1 cell.
yHi_Vals 		= [1000] 		# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution and change Pia.
yMinSWs 		= [1]#[1,2,3]			# DOING BELOW THING WITH YYY inside pgm functions. --> (set to 0, so does nothing) 
								# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<yLo.
#
flg_EgalitarianPriorS = [True] # False,

learning_rates = [0.5] #[1.0, 0.5, 0.1]
lRateScaleS = [[1., 0.1, 0.1]]# , [1., 0.1, 1.] ]  # Multiplicative scaling to Pia,Pi,Q learning rates. Can set to zero and take Pi taken out of model essentially.


num_train_rands = 1

sample_longSWs_1stS = ['Dont']#,'Prob'] #,'Dont'] # Options are: {'Dont', 'Prob', 'Hard'}

maxSamps = np.nan # np.nan if you want to use all the SWs for samples or a scalar to use only a certain value

maxRasTrials = np.nan # np.nan if you want to use all the SWs for samples or a scalar to use only a certain value



# Parameter initializations for EM algorithm to learn model parameters
params_init 	= 'NoisyConst'	# Options: {'True', 'RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise) }
#
Z_hot 			= 5 			# initialization for Q value (how many 1's expected in binary z-vector)
C_noise_ri 		= 1.
C_noise_ria 	= 1.
#
sigQ_init 		= 0.01			# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigPi_init 		= 0.05			# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigPia_init 	= 0.05			# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sig_init 		= np.array([ sigQ_init, sigPi_init, sigPia_init ])
params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )



# Flags for the EM (inference & learning) algorithm.
flgs_include_Zeq0_infer  = [True] #[False, True]
verbose  				 = False





minTms = 0 
maxTms = 6000




if not np.isnan( maxSamps ):
	maxSampTag = str( '_'+str( int(maxSamps ) )+'Samps' )
else:
	maxSampTag = '_allSamps'


if not np.isnan( maxRasTrials ):
	maxTrialTag = str( '_'+str( int(maxRasTrials) )+'Trials' )
else:
	maxTrialTag = '_allTrials'	


if flg_plot_PSTH_numCellsYmin:
	#f_PSTH_yMin = plt.figure() # make a figure
	hist_vs_yMin = list()


if flg_write_CSV_stats:
	data_stats_CSV = dict() 									# Make an empty Dictionary
	fname_CSV = str(EM_learnStats_Dir + 'STATs_realData.csv')	# Filename to save CSV to
	dfh = 0														# clunky way to make the header only in the first row.





# Loop through different combinations of parameters and generate plots of results.
for ct,cell_type in enumerate(cell_types):
	N = Ns[ct]
	#print(cell_type, N)
	#
	for flg_include_Zeq0_infer in flgs_include_Zeq0_infer:
		#
		if flg_include_Zeq0_infer:
			z0_tag='_zeq0'
		else:
			z0_tag='_zneq0'
		#
		for i in range(len(stims)):
			stim=stims[i]
			#num_SWs=num_dataGen_Samples[i]
			#
			for sample_longSWs_1st in sample_longSWs_1stS:
				#
				for flg_EgalitarianPrior in flg_EgalitarianPriorS:
					#
					if flg_EgalitarianPrior:	
						priorType = 'EgalQ' 
					else:
						priorType = 'BinomQ'
					#
					for overcomp in model_CA_overcompleteness:
						M = overcomp*N
						#
						for yLo in yLo_Vals:
							#
							for yHi in yHi_Vals:
								#
								for yMinSW in yMinSWs:
									#
									for SW_bin in SW_bins:
										msBins = 1+2*SW_bin
										#
										for learning_rate in learning_rates:
											#
											for lRateScale in lRateScaleS:
												#
												for rand in range(num_train_rands):
													#
													for train_2nd_model in train_2nd_modelS:
														#
														if train_2nd_model:
															Btag = 'B'
														else:
															Btag = ''
														#
														if train_2nd_model:
															pct_xVal_train = 0.5
														else:
															pct_xVal_train = 0.9	

														# try:
														if True:


															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # (1).  Set up directory structure and filename. Load it in and extract variables.
															init_dir = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') + \
																	'_LRsc' + str(lRateScale) + '/' )

															#
															model_dir = str( cell_type + '_N' + str(N) +'_M' + str(M) + z0_tag +  '_ylims' + str(yLo) + '_' + str(yHi) + \
																'_yMinSW' + str(yMinSW) + '_' + sample_longSWs_1st + 'Smp1st_' + priorType + '/')
															#




															# Make a directories for output plots if they are not already there.
															if not os.path.exists( str(EM_figs_dir + init_dir + model_dir) ):
																os.makedirs( str(EM_figs_dir + init_dir + model_dir) )
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
																	and str(  'trn_' + str(msBins) + 'msBins' + maxSampTag + '_rand' + str(rand) + Btag +'.npz' ) in s ]
															if len(fname) != 1:
																print('I am expecting one matching file.  I have ',len(fname))
																print('In dir: ',str(EM_learning_Dir + init_dir + model_dir ))
																print(fname)
																continue
															#
															model_file = fname[0][:-4]
															
															#
															# Set up file names and directories.
															LearnedModel_fname 			= str( EM_learning_Dir + init_dir + model_dir + model_file + '.npz')	
															StatsDuringLearning_fname 	= str( EM_learnStats_Dir + init_dir + model_dir + model_file.replace('LearnedModel_','InferStats_DuringLearn_') + '.npz')
															StatsPostLearning_fname 	= str( EM_learnStats_Dir + init_dir + model_dir + model_file.replace('LearnedModel_','InferStats_PostLrn_') + '.npz')
															rasterZs_fname 				= str( Infer_postLrn_Dir + init_dir + model_dir + model_file.replace('LearnedModel_', str('rasterZ_allSWs'+maxTrialTag+'_') ) + '.npz')




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
															##
															SWs_train = data['SWs_train'] 
															SWs_test = data['SWs_test']
															#
															Z_inferred_train = data['Z_inferred_train']
															Z_inferred_test = data['Z_inferred_test']
															pyiEq1_gvnZ_train = data['pyiEq1_gvnZ_train']
															pyiEq1_gvnZ_test = data['pyiEq1_gvnZ_test']
															#
															pj_zHyp_train = data['pj_zHyp_train'] 
															pj_zHyp_test = data['pj_zHyp_test']
															pj_zTru_Batch = data['pj_zTru_Batch'] 
															pj_zTrain_Batch = data['pj_zTrain_Batch'] 
															pj_zTest_Batch = data['pj_zTest_Batch']
															#
															cond_zHyp_train = data['cond_zHyp_train'] 
															cond_zHyp_test = data['cond_zHyp_test']
															cond_zTru_Batch = data['cond_zTru_Batch'] 
															cond_zTrain_Batch = data['cond_zTrain_Batch'] 
															cond_zTest_Batch = data['cond_zTest_Batch']
															#
															if flg_Pia_snapshots_gif:
																ria_snapshots = data['ria_snapshots']
																ri_snapshots = data['ri_snapshots']
																q_snapshots = data['q_snapshots']
															ds_fctr_snapshots = data['ds_fctr_snapshots']
															#
															q_deriv = data['q_deriv']
															ri_deriv = data['ri_deriv']
															ria_deriv = data['ria_deriv']
															#
															zActivationHistory = data['zActivationHistory']
															argsRec = data['argsRec']

															del data	




															if np.any( np.isnan(rip) ) or np.isnan(qp) or np.any( np.isnan(riap) ):
																farrg

															# Used below in plotting and junk
															Q = rc.sig(qp)
															Pi = rc.sig(rip)
															Pia = rc.sig(riap)




															# # HERE PLOT WHOLE MODEL OR SOMETHING...
															if False:
																import matplotlib.gridspec as gsp
																#
																# (1). Compute how active each CA is and sort by that.
																zAct = np.zeros(M)
																for z in Z_inferred_train:
																	zAct[list(z)] += 1 
																indZ = np.argsort(zAct)[::-1]
																#
																# (2). Compute how active each Cell is inferred to be and sort by that.
																avg_pyGvnZ = pyiEq1_gvnZ_train.mean(axis=0) # NOTE: could also use SWs to sort by
																indY = np.argsort(avg_pyGvnZ)[::-1]			# actual (not inferred) cell activity
																#
																# #
																#
																# (3). Plot em

																plt.rc('font', weight='bold', size=8)
																f = plt.figure()
																f.set_size_inches( (15,10) )
																# 
																GS = gsp.GridSpec(3, 3) #, wspace=0.7)
																ax1 = plt.subplot(GS[2,:2])
																#ax1 = plt.subplot2grid( (3,3),(2,0), colspan=2 )
																ax1.scatter( np.arange(M), zAct[indZ], s=10, c='g', marker='o' )
																ax1.invert_yaxis()
																ax1.grid()
																ax1.set_ylabel("# Z inf'rd")
																ax1.set_xlabel('CA id ($z_a$)')
																#
																
																ax2 = plt.subplot2grid( (3,3),(0,2), rowspan=2 )
																ax2.scatter( avg_pyGvnZ[indY], np.arange(N),  s=10, c='r', marker='o', label='$p(y_i=1|\\vec{z})$' )
																ax2.scatter( 1-Pi[indY], np.arange(N), s=10, c='b', marker='x', label='Pi' )
																ax2.legend()
																ax2.invert_yaxis()
																ax2.grid()
																#
																ax3 = plt.subplot2grid( (3,3),(0,0), rowspan=2, colspan=2 )
																im1=ax3.imshow(1-Pia[np.ix_(indY,indZ)],vmin=0,vmax=1, cmap='gist_gray_r')
																ax3.set_aspect('auto')
																ax3.grid()
																ax3.set_title('$P_{ia}$')
																ax3.set_ylabel('cell id ($y_i$)')
																#
																cax1 = f.add_axes([0.65, 0.15, 0.25, 0.08])
																f.colorbar(im1, cax=cax1, orientation='horizontal')
																cax1.set_title('$p(y_i=1|z_a=1)$')
																#
																plt.show()







															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # (2).  Get extracted spikewords from spike data to look at statistics of spiking activity.
															# Extract spike words from raw data or load in npz file if it has already been done and saved.
															print('Extracting spikewords')
															t0 = time.time()
															fname_SWs = str( SW_extracted_Dir + cell_type + '_' + stim + '_' + str(msBins) + 'msBins.npz' )
															spikesIn = list()
															SWs, SWtimes = rc.extract_spikeWords(spikesIn, msBins, fname_SWs)
															#
															#numTrials = len(SWs)
															t1 = time.time()
															print('Done Extracting spikewords: time = ',t1-t0) # Fast: ~5 seconds.
															





															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															print('Loading in data file saved from post EM learning inference in raster_zs_inferred_allSWs_given_model.py')
															print( rasterZs_fname )
															#
															# try:
															if False: #True:	
																data = np.load( rasterZs_fname )
																print(data.keys())
																#
																Z_inferred_allSWs 			= data['Z_inferred_allSWs'] 
																pyiEq1_gvnZ_allSWs 			= data['pyiEq1_gvnZ_allSWs'] 
																pj_inferred_allSWs 			= data['pj_inferred_allSWs'] 
																# try:
																cond_inferred_allSWs		= data['cond_inferred_allSWs'] 
																# except:
																# 	print('cond_inferred_allSWs only in new files.')
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
																argsRecModelLearn 			= data['argsRecModelLearn']
																argsRaster 					= data['argsRaster']
																#
																try:
																	raster_Z_inferred_allSWs 	= data['raster_Z_inferred_allSWs'] 
																	raster_Y_inferred_allSWs 	= data['raster_Y_inferred_allSWs']
																	raster_allSWs 				= data['raster_allSWs']
																except:

																	raster_Z_inferred_allSWs 	= 'meh'
																	raster_Y_inferred_allSWs 	= 'meh'
																	raster_allSWs 				= 'meh'

																	if False:
																		print('Converting SWs,Zinf,Yinf into 3D tensor that is CellsCAs x Trials x TimesActive.')
																		t0 = time.time()
																		#
																		print('raster_Z_inferred_allSWs')
																		raster_Z_inferred_allSWs = rc.compute_raster_list(SWtimes, Z_inferred_allSWs, pj_inferred_allSWs, M, minTms, maxTms )
																		#
																		print('raster_Y_inferred_allSWs')
																		print('I could do this by thresholding pyiEq1_gvnZ_allSWs, but not worth it right now. ')
																		if False:
																			raster_Y_inferred_allSWs = rc.compute_raster_list(SWtimes, Y_inferred_allSWs, pj_inferred_allSWs, N, minTms, maxTms )
																		#
																		print('raster_allSWs')
																		raster_allSWs = rc.compute_raster_list(SWtimes, SWs, pj_inferred_allSWs, N, minTms, maxTms ) 
																		Xtimes, X, pX, numX, minTms, maxTms
																		#
																		t1 = time.time()
																		print('Done Converting SWs,Zinf,Yinf into 3D tensor that is CellsCAs x Trials x TimesActive: time = ',t1-t0)
																#
																del data

															# except:
															# 	print( rasterZs_fname )
															# 	print('Inference / raster file not there. Making up some shit.')
															# # 	continue

															# 	Z_inferred_allSWs = list()
															# 	Z_inferred_allSWs.append(list())
															# 	Z_inferred_allSWs[0].append( Z_inferred_train )
															# 	#
															# 	pyiEq1_gvnZ_allSWs = list()
															# 	pyiEq1_gvnZ_allSWs.append(list())
															# 	pyiEq1_gvnZ_allSWs[0].append(pyiEq1_gvnZ_train)
															# 	#
															# 	pj_inferred_allSWs = list()
															# 	pj_inferred_allSWs.append(list())
															# 	pj_inferred_allSWs[0].append( pj_zHyp_train )
															# 	#
															# 	cond_inferred_allSWs = list()
															# 	cond_inferred_allSWs.append(list())
															# 	cond_inferred_allSWs[0].append( cond_zHyp_train ) 
															# 	# 
															# 	Ycell_hist_allSWs 			= np.zeros(N)
															# 	YcellInf_hist_allSWs 		= np.zeros(N) 
															# 	Zassem_hist_allSWs 			= np.zeros(M) 
															# 	#
															# 	nY_allSWs 					= np.array([ len(w) for w in SWs_train ]) 
															# 	nYinf_allSWs 				= np.array([ len(w) for w in SWs_train ])
															# 	nZ_allSWs 					= np.array([ len(w) for w in SWs_train ]) 
															# 	#
															# 	CA_coactivity_allSWs 		= np.zeros( (M,M) ) 
															# 	Cell_coactivity_allSWs 		= np.zeros( (N,N) ) 
															# 	CellInf_coactivity_allSWs 	= np.zeros( (N,N) ) 
															# 	#
															# 	argsRecModelLearn 			= 'meh'
															# 	argsRaster 					= 'meh'
															# 	#
															# 	raster_Z_inferred_allSWs 	= 'meh'
															# 	raster_Y_inferred_allSWs 	= 'meh'
															# 	raster_allSWs 				= 'meh'
# 







															
															num_SWs = nY_allSWs.size

															if np.isnan(maxSamps):
																maxSamps = num_SWs

													
															
															if False:
																# Compute spike rates for all cells from raster vectors
																spkRate = np.zeros(N)
																for y in range(N):
																	spkRate[y] = np.array([ len(raster_allSWs[T][y]) for T in range(numTrials) ]).sum()
																spkRate = spkRate/(numTrials*(maxTms-minTms))*1000 # spike rate in units of spikes/second.	
																#
																# # Stats about spike rate we could throw in the CSV file.
																# spkRate.mean()
																# spkRate.std()
																# spkRate.max()
																# spkRate.min()
																#
																# # Plots comparing stats (# CAs for each cell and Pi values) of learned model to spike rates.
																f,ax = plt.subplots(1,2)
																ax[0].scatter( spkRate/spkRate.max(),(1-Pi)/(1-Pi).max() ) 
																ax[0].set(aspect='equal',xlim=[-0.05,1.05],ylim=[-0.05,1.05])
																ax[0].set_xlabel('Normalized spike rate')	
																ax[0].set_ylabel('Normalized (1-Pi)')	

																ax[1].scatter( spkRate/spkRate.max(),numCAsMid/numCAsMid.max() )
																ax[1].set(aspect='equal',xlim=[-0.05,1.05],ylim=[-0.05,1.05])
																ax[1].set_xlabel('Normalized spike rate')	
																ax[1].set_ylabel('Normalized numCas / cell.')	
																plt.show()








															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															#
															# Unwrapping Spike words and all else into 1D list from list that is has #Trials dimensions.
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
															all_pYgvnZ = list()
															all_pjs  = list() 
															all_conds  = list() 
															#
															print('Fix num trials in 470ish')
															numTrials = 1 #len(SWs)
															for i in range(numTrials):
																#
																#index into spike times that are between minTms and maxTms
																ind = list( np.where(np.bitwise_and(np.array(SWtimes[i])>minTms, np.array(SWtimes[i])<maxTms))[0] )
																#
																# Ind is 1 too long on rare / single occasion.
																if ind[-1] > np.min( [len(SWs[i]), len(Z_inferred_allSWs[i])] )-1:
																	# print( 'max ind = ', ind[-1] )
																	# print( 'len SWs = ', len(SWs[i]) )
																	# print( 'len Yinf = ', len(Y_inferred_allSWs[i]) )
																	# print( 'len Zinf = ', len(Z_inferred_allSWs[i]) )
																	xxx = list(np.where(ind < np.min( [len(SWs[i]), len(Z_inferred_allSWs[i])] ) )[0])
																	print('Trial #',i,' : cut out ', len(ind) - len(xxx),' from weirdness.')
																	print('             : cut out ', len(SWtimes[i]) - len(xxx),'from weirdness and overtime.')
																	ind = [ind[x] for x in xxx ]
																#
																all_SWs.extend([SWs[i][x] for x in ind]) #   SWs[i][ind])
																all_SWtimes.extend([SWtimes[i][x] for x in ind])
																all_SWtrials.extend( i*np.ones_like(ind) )
																#
																all_Zinf.extend([Z_inferred_allSWs[i][x] for x in ind])
																all_pYgvnZ.extend([pyiEq1_gvnZ_allSWs[i][x] for x in ind])
																for x in ind:
																	try:
																		all_pjs.extend([pj_inferred_allSWs[i][x][0]]) # sometimes is array.
																		# try:
																		all_conds.extend([cond_inferred_allSWs[i][x][0]]) # sometimes is array.
																			# print('get rid of try statement.')
																		# except:
																		# 	print('cond_inferred_allSWs only in new files.')
																	except:
																		all_pjs.extend([pj_inferred_allSWs[i][x]]) # sometimes is scalar.
																		# try:
																		all_conds.extend([cond_inferred_allSWs[i][x]]) # sometimes is scalar.
																		# 	print('get rid of try statement.')
																		# except:
																		# 	print('cond_inferred_allSWs only in new files.')
																		
																#
															   #
															  #
															 #	
															#	
															num_SWsUnWrap = len(all_SWs)	

															# for jj in range(numTrials):
															# 	print( jj, len(SWs[jj]), len(Y_inferred_allSWs[jj]), len(Z_inferred_allSWs[jj]) )
															#	
															t1 = time.time()
															print('Done Unwrapping SWs, SWtimes, SWtrials, Zinf, Yinf into 1D vector: time = ',t1-t0)
																
															#
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
														



															





															if flg_write_CSV_stats:
																data_stats_CSV.update( [ ('stim',[stim]) , ('cell type',[cell_type]) , ('N',[N]) , ('M',[M]) , \
																	('msBins',[msBins]) , ('yLo',[yLo]), ('yHi',[yHi]), ('yMinSW',[yMinSW]), ('# SWs',[num_SWs]) , ('maxSamps',[maxSamps]), \
																	('2nd model',[Btag]), ('init',[params_init]), ('LR',[learning_rate]), ('Pi mean init',[C_noise_ri]), \
																	('Pia mean init',[C_noise_ria]), ('Pi std init',[sigPi_init]), ('Pia std init',[sigPia_init]), \
																	('Q std init',[sigQ_init]), ('Zhot init',[Z_hot]), ('LRsc',[lRateScale]) ] )
						
																
																

															

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
																		= rc.compute_dataGen_Histograms(smp_SWs, list(np.arange(maxSamps)) , M, N)
																	#
																	# #
																	#
																	# Stats on active cells and cell assemblies inferred on training data during EM algorithm
																	Ycell_hist_Infer_train, Zassem_hist_Infer_train, nY_Infer_train, nZ_Infer_train,\
																		CA_coactivity_Infer_train, Cell_coactivity_Infer_train \
																		= rc.compute_dataGen_Histograms(Y_inferred_train, Z_inferred_train, M, N)
																	#
																	# #
																	#
																	# Stats on active cells and cell assemblies inferred on test data during EM algorithm
																	Ycell_hist_Infer_test, Zassem_hist_Infer_test, nY_Infer_test, nZ_Infer_test,\
																		CA_coactivity_Infer_test, Cell_coactivity_Infer_test \
																		= rc.compute_dataGen_Histograms(Y_inferred_test, Z_inferred_test, M, N)
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
																	Yinf_train = [ Y_inferred_train[i] for i in range(maxSamps) if indSWs_gt_yMin_train[smp_train[i]] < num_SWsUnWrap ]
																	Zinf_train = [ Z_inferred_train[i] for i in range(maxSamps) if indSWs_gt_yMin_train[smp_train[i]] < num_SWsUnWrap ]
																	#
																	Kinf_train, KinfDiff_train, zCapture_train, zMissed_train, zExtra_train, \
																	yCapture_train, yMissed_train, yExtra_train, inferCA_Confusion_train, zInferSampled_train, zInferSampledRaw_train, \
																	zInferSampledT_train, inferCell_Confusion_train, yInferSampled_train = rc.compute_inference_statistics_allSamples( \
																	len(Y_train), N, M, Z_train, Zinf_train, np.arange(M), np.arange(M), Y_train, Yinf_train, verbose)
																	#
																	# Computing Inference statistics for test samples.
																	Y_test = [ all_SWs[i] for i in indSWs_gt_yMin_test[smp_test]  if i < num_SWsUnWrap]
																	Z_test = [ list() for num in range(len(smp_test)) ]
																	Yinf_test = [ Y_inferred_test[i] for i in range(maxSamps) if indSWs_gt_yMin_test[smp_test[i]] < num_SWsUnWrap ]
																	Zinf_test = [ Z_inferred_test[i] for i in range(maxSamps) if indSWs_gt_yMin_test[smp_test[i]] < num_SWsUnWrap ]
																	#
																	Kinf_test, KinfDiff_test, zCapture_test, zMissed_test, zExtra_test, \
																	yCapture_test, yMissed_test, yExtra_test, inferCA_Confusion_test, zInferSampled_test, zInferSampledRaw_test, \
																	zInferSampledT_test, inferCell_Confusion_test, yInferSampled_test = rc.compute_inference_statistics_allSamples( \
																	len(Y_test), N, M, Z_test, Zinf_test, np.arange(M), np.arange(M), Y_test, Yinf_test, verbose)
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
																		Ycell_hist_Infer_test=Ycell_hist_Infer_test,Zassem_hist_Infer_test=Zassem_hist_Infer_test, 
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
																	zInferSampledT_postLrn, inferCell_Confusion_postLrn, yInferSampled_postLrn = rc.compute_infZstats_allSamples( \
																	num_SWsUnWrap, N, M, all_Zs, all_Zinf, np.arange(M), np.arange(M), all_SWs, all_Yinf, verbose)
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



															# xyzab



															if flg_write_CSV_stats and flg_compute_StatsPostLrn:

																nY_all_diff = nYinf_allSWs - nY_allSWs
																#
																if flg_write_CSV_stats:
																	data_stats_CSV.update( [ ('|Y| overinferred mean',[np.mean( nYinf_allSWs - nY_allSWs )]) , \
																							 ('|Y| overinferred std',[np.std( nYinf_allSWs - nY_allSWs )]), \
																							 ('|Y| overinferred skew',[st.skew( nY_all_diff )]) , \
																							 #
																							 ('|Z| inf mean',[np.mean( nZ_allSWs )]) , \
																							 ('|Z| inf std',[np.std( nZ_allSWs )]) , \
																							 ('|Z| inf skew',[st.skew( nZ_allSWs )]) , \
																							 #
																							 ('|Y| inf mean',[np.mean( nYinf_allSWs )]) , \
																							 ('|Y| inf std',[np.std( nYinf_allSWs )]) , \
																							 ('|Y| inf skew',[st.skew( nYinf_allSWs )]) , \
																							 #
																							 ('|Y| obs mean',[np.mean( nY_allSWs )]) , \
																							 ('|Y| obs std',[np.std( nY_allSWs )]) , \
																							 ('|Y| obs skew',[st.skew( nY_allSWs )]) , \
																							 #
																							 ('%times z=0 inf train',[(nZ_Infer_train==0).sum()/nZ_Infer_train.size]), \
																							 ('%times y=1 obs train',[(nY_Samps==1).sum()/nY_Samps.size]), \
																							 ('%times z=0 inf allSWs',[(nZ_allSWs==0).sum()/nZ_allSWs.size]), \
																							 ('%times y=1 obs allSWs',[(nY_allSWs==1).sum()/nY_allSWs.size]) ] ) 





															# # # # # # # # # # # # # # # # # 
															# Sort cell assemblies and cells by how active they are 
															TH_bounds = np.array([0.3, 0.5, 0.7])

															# sortCA_byActivity 	  = np.argsort(Zassem_hist_Infer_test[:-1])[::-1]
															# sortCA_byActivity 	  = np.argsort(Zassem_hist_Infer_train[:-1])[::-1]
										
															# sortCells_byActivity = np.argsort(YcellInf_hist_allSWs[:-1])[::-1]
															# sortCells_byActivity = np.argsort(Ycell_hist_Samps[:-1])[::-1]
															# sortCells_byActivity = np.argsort(Ycell_hist_Infer_train[:-1])[::-1]
															# sortCells_byActivity = np.argsort(Ycell_hist_Infer_test[:-1])[::-1]

															sortCA_byActivity 	  = np.argsort(Zassem_hist_allSWs[:-1])[::-1]
															sortCells_byActivity = np.argsort(Ycell_hist_allSWs[:-1])[::-1]

															# sortCA_byActivity = np.arange(M)
															# sortCells_byActivity = np.arange(N)


															PiInv 	 	= ( 1-rc.sig(rip) )[sortCells_byActivity]
															PiaInv 	  	= ( 1-rc.sig(riap) )[np.ix_(sortCells_byActivity,sortCA_byActivity)]

															maxPiQ 		= np.nanmax( np.array([ rc.sig(qp),	np.nanmax(PiInv) ]) )
																

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






															# NOTE: THIS IS HAPPENING DURING EM LEARNING. COULD ALSO DO FOR TEST DATA.
															flg_compute_spkTimes_during_EM = False
															if flg_compute_spkTimes_during_EM:
																print('What Am I doing?')
																t0 = time.time()
																#
																# Compute a PSTH for Latent Variables (za's) by noting SWtimes when each za is inferred.
																Z_spkTimes = list()
																for i in range(M):
																	Z_spkTimes.append( [] ) # set up a list of empty lists to contain spike times for each za.
																#
																for i,s in enumerate(smp_train): 
																	#print(i,s)	
																	if not len(Z_inferred_train[i]) is 0:
																		if all_SWtimes[s] < maxTms:
																			for j in list(Z_inferred_train[i]):
																				Z_spkTimes[j].append( all_SWtimes[s] ) # fill in sample times when each za is active.
																#
																# Compute a PSTH for Observed Variables (yi's) by noting SWtimes when each za is inferred.
																Y_spkTimes= list()
																for i in range(N):
																	Y_spkTimes.append( [] ) # set up a list of empty lists to contain spike times for each za.
																#
																for i,s in enumerate(smp_train): 
																	#print(i,s)	
																	if not len(all_SWs[i]) is 0:
																		if all_SWtimes[s] < maxTms:
																			for j in list(all_SWs[i]):
																				Y_spkTimes[j].append( all_SWtimes[s] ) # fill in sample times when each za is active.
																#
																t1 = time.time()
																print('Done w/ that : time = ',t1-t0) 



																# SUBPLOTS FOR MOST INFERRED SINGLE CELL ASSEMBLIES OF PSTH OF CA IN BLACK AND CELLS IN COLOR.
																if False:
																	print('What Am I doing?')
																	t0 = time.time()
																	#
																	t1 = time.time()
																	print('Done w/ that : time = ',t1-t0)
																	xxx = np.zeros(M).astype(bool)
																	for i in range(M):
																		if len(whichCellsMid[i])>1 and len(whichCellsMid[i])<6:
																			xxx[i] = True
																	indd = np.where(xxx)[0]	
																	#
																	numCAs=5
																	indd = indd[:numCAs]
																	f, ax = plt.subplots(len(indd))
																	for b,j in enumerate(indd):
																		colors =  plt.cm.Set1( np.arange(len(whichCellsMid[j])) )
																		for a,i in enumerate(whichCellsMid[j]):
																			print(a,i)
																			sTs = np.array(Y_spkTimes[i])
																			y,x = np.histogram(sTs[sTs>500], bins=1000, density=True) # NOT CLEAR IF NORMED IS BEST OR NOT. DENSITY=TRUE OR FALSE.
																			ind = np.where(y!=0)[0]
																			ax[b].plot(x[ind],y[ind], linewidth=0.4, color=colors[a], marker='.', markersize=4, alpha=0.6, label=str('Y' + str(i) +' #=' + str(sTs.size) ) )
																		#
																		sTs = np.array(Z_spkTimes[sortCA_byActivity[j]])
																		y,x = np.histogram(sTs[sTs>500], bins=1000, density=True)
																		ind = np.where(y!=0)[0]

																		print(b)

																		print('median = ',np.median(y[ind]))
																		print('mean = ',np.mean(y[ind]))
																		print('std = ',np.std(y[ind]))

																		ax[b].plot(x[ind],y[ind], linewidth=.5, color='black', marker='.', markersize=4, linestyle='--', alpha=1) #, label=str('Z' + str(sortCA_byActivity[j]) + ' #=' + str(sTs.size) ) )
																		ax[b].legend(fontsize=6, bbox_to_anchor=(1, 1), loc=2)
																		ax[b].set_ylabel( str('Z' + str(j) + ' \n #=' + str(sTs.size) ), rotation=0 )
																		ax[b].set_xlim(0,5500)
																		#ax[b].set_yticklabels([])
																		if b<len(indd)-1:	
																			ax[b].set_xticklabels([])
																	#
																	ax[b].set_xlabel( 'Time (in ms)' )
																	plt.show()

																# 
																print('What Am I doing?')
																t0 = time.time()
																#
																t1 = time.time()
																print('Done w/ that : time = ',t1-t0)
																numBins = 5500
																tixx = np.linspace(0,numBins-1,11).astype(int)
																PSTH_cells 	= np.zeros( (N,numBins-1) )
																#
																for i in range(N):
																	sTs = np.array(Y_spkTimes[i])
																	y,x = np.histogram(sTs, bins=np.arange(numBins), density=False) # NOT CLEAR IF NORMED IS BEST OR NOT. DENSITY=TRUE OR FALSE.
																	PSTH_cells[i] = y



																# CREATE PSTH FOR ALL CELLS as an IMSHOW plot
																if False and flg_plot_PSTH_numCellsYmin:
																	print('What Am I doing?')
																	t0 = time.time()
																	#
																	t1 = time.time()
																	print('Done w/ that : time = ',t1-t0)
																	f=plt.figure( figsize=(20,6) ) # size units in inches
																	plt.rc('font', weight='bold', size=16)
																	plt.imshow(np.log10(PSTH_cells), aspect='auto')
																	cb = plt.colorbar( ticks=np.linspace( 0, np.log10(PSTH_cells.max()), 5 ) )
																	cb.ax.set_yticklabels(np.round(np.linspace( 0, PSTH_cells.max(), 5 )).astype(int))
																	cb.set_label( str('Counts in ' + str(SWs.size) + ' trials ($log_{10}$ scale)') )
																	plt.xticks( tixx, x[tixx].astype(int)+1 )
																	plt.xlim(0,5500)
																	plt.yticks( np.arange(0,N,5) )
																	plt.grid()
																	plt.xlabel('Time (in ms)')
																	plt.ylabel('Cell id')
																	plt.title( str('PSTH for ' + cell_type + ' Cells shown ' + stim + ' with $|y|_{lims}$ = ' + str(yLo) + 'and' + str(yHi) + ', SW bin = ' + str(2*SW_bin+1) + 'ms and ' + str(maxSamps) + ' samples') )
																	#       
																	plt_save_dir = str(EM_figs_dir + '../../PSTH_' + cell_type + '_N' + str(N) + '/')
																	if not os.path.exists(plt_save_dir):
																		os.makedirs(plt_save_dir)
																	plt.savefig( str(plt_save_dir + 'PSTH_' + stim + '_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) + '_' + str(2*SW_bin+1) + 'msBins_' + str(maxSamps) +  'Samps_rand' + str(rand) + '.pdf' ) )
																	plt.close() 





																# PLOT A HISTOGRAM (PSTH-LIKE) OF TIME BINS WHEN NUMBER OF CELLS FIRING IN A TRIAL REACHES YMIN THRESHOLD.
																if False and flg_plot_PSTH_numCellsYmin:
																	print('What Am I doing?')
																	t0 = time.time()
																	#
																	t1 = time.time()
																	print('Done w/ that : time = ',t1-t0)
																	# minTms = 0  # ms
																	# maxTms = 6000 # ms

																	# Loop through some parameters plotting results on same figure.
																	sTs = np.array(all_SWtimes)[smp_train]
																	sTs = sTs[ np.bitwise_and(sTs>minTms,sTs<maxTms)]
																	y,x = np.histogram(sTs, bins=np.arange(minTms,maxTms), density=False) # NOT CLEAR IF NORMED IS BEST OR NOT. DENSITY=TRUE OR FALSE.
																	ind = np.where(y!=0)[0]
																	#
																	# plt.figure(f_PSTH_yMin.number) # recall a figure
																	# plt.scatter(x[ind],y[ind], s=3, label=str(yMin) ) #, linewidth=0.4, marker='.', markersize=4, alpha=1. )
																	
																	hist_vs_yMin.append( [x[ind],y[ind]] )

															






																#
																# CREATE PSTH FOR ALL CELLS AND SCATTER PLOT OF CELL ACTIVITY THAT IS EXPLAINED BY CELL ASSEMBLY ACTIVITY.
																if False and flg_plot_PSTH_cellsAndCAs:
																	print('What Am I doing?')
																	t0 = time.time()
																	#
																	t1 = time.time()
																	print('Done w/ that : time = ',t1-t0)
																	plt.imshow(PSTH_cells, aspect='auto', alpha=0.5)
																	#
																	xxx = np.zeros(M).astype(bool)
																	for i in range(M):
																		if len(whichCellsMid[i])>1 and len(whichCellsMid[i])<6:
																			xxx[i] = True
																	indd = np.where(xxx)[0]	
																	#
																	numCAs=10
																	indd = indd[:numCAs]
																	colors =  plt.cm.Set1( np.arange(numCAs) )
																	for b,j in enumerate(indd):
																		
																		sTs = np.array(Z_spkTimes[sortCA_byActivity[j]])
																		y,x = np.histogram(sTs[sTs>500], bins=np.arange(numBins), density=True)
																		ind = np.where(y!=0)[0]
																		ind2 = np.where( y>np.mean(y[ind]) )[0] # +np.std(y[ind])

																		plt.scatter( np.tile(x[ind2],(1,whichCellsMid[j].size) ), np.tile(whichCellsMid[j],(1,ind2.size) ), s=20, marker='s', color=colors[b], \
																			edgecolors='black', linewidths=1, label=str('z '+str(j) ) ) # + ' (|' + str(whichCellsMid[j].size) +'|) (#=' + str(Zassem_hist_Infer_sort[j])+ ')' ) ) #, linewidth=.5, color='black', marker='.', markersize=4, linestyle='--', alpha=1) #, label=str('Z' + str(sortCA_byActivity[j]) + ' #=' + str(sTs.size) ) )

																	plt.legend( bbox_to_anchor=(1, 0), loc='lower right' )
																	#
																	plt.xticks( tixx, x[tixx].astype(int)+1 )
																	plt.yticks( np.arange(0,N,5) )
																	plt.xlim(500,5500)
																	plt.ylim(0,N)
																	plt.grid()
																	plt.gca().invert_yaxis()
																	plt.xlabel('Time (in ms)')
																	plt.ylabel('Cell id')
																	plt.title( str('Cell activity contained in Cell Assemblies') )
																	plt.show()
																		







															# Compute overlap or similarity of learned cell assemblies.
															CA_ovl = rc.compute_CA_Overlap(riap)




															if flg_write_CSV_stats:
																	#
																	ovl = CA_ovl[np.triu(CA_ovl,1)>0]
																	data_stats_CSV.update( [ 
																		('%|CAs|=0',[ (numCellsMid==0).sum()/M ]), \
																		('%|CAs|=1',[ (numCellsMid==1).sum()/M ]), \
																		('%|CAs|=2',[ (numCellsMid==2).sum()/M ]), \
																		('%|CAs|>2&<6', np.bitwise_and( numCellsMid>2,numCellsMid<6 ).sum() /M ), \
																		('%|CAs|>=6',[ (numCellsMid>=6).sum()/M ]), \
																		('|CAs| max',[ (numCellsMid).max() ]), \
																		('%Pi>0.1',[ ((1-Pi)>0.1).sum()/N ]), \
																		('Q',[ Q[0] ]), \
																		('mean CA ovl',[np.mean(ovl)]), \
																		('std CA ovl',[np.std(ovl)]), \
																		('max CA ovl',[np.max(ovl)]) ] )
															

												

															# Step thru Z_inferred_train and record sample numbers when each CA is inferred. 
															if flg_temporal_inference_plot:
																print('compute Inference temporal for training data during EM learning')
																t0 = time.time()
																train_samps = len(Z_inferred_train) # could also do something like pct_xVal_train*maxSamps
																Z_inf_temp = np.zeros( (train_samps+1,M+1) )
																for samp in range(train_samps):
																	Z_inf_temp[samp+1] = Z_inf_temp[samp]
																	Z_inf_temp[samp+1, list(Z_inferred_train[samp])] += 1
																t1 = time.time()
																print('Done w/ inference temporal during EM learning : time = ',t1-t0) # Fast enough: ~10 seconds
																#






															#
															# if flg_compute_StatsDuringEM:
															# 	# Computing Inference statistics for training samples.
															# 	print('Computing Statistics on Inference process during EM learning on training data.')
															# 	t0 = time.time()
															# 	#
															# 	Y_train = [ all_SWs[i] for i in indSWs_gt_yMin_train[smp_train] if i < num_SWsUnWrap ]
															# 	Z_train = [ list() for num in range(len(Y_train)) ]
															# 	Yinf_train = [ Y_inferred_train[i] for i in range(maxSamps) if indSWs_gt_yMin_train[smp_train[i]] < num_SWsUnWrap ]
															# 	Zinf_train = [ Z_inferred_train[i] for i in range(maxSamps) if indSWs_gt_yMin_train[smp_train[i]] < num_SWsUnWrap ]
															# 	#
															# 	Kinf_train, KinfDiff_train, zCapture_train, zMissed_train, zExtra_train, \
															# 	yCapture_train, yMissed_train, yExtra_train, inferCA_Confusion_train, zInferSampled_train, zInferSampledRaw_train, \
															# 	zInferSampledT_train, inferCell_Confusion_train, yInferSampled_train = rc.compute_inference_statistics_allSamples( \
															# 	len(Y_train), N, M, Z_train, Zinf_train, np.arange(M), np.arange(M), Y_train, Yinf_train, verbose)
															# 	#
															# 	t1 = time.time()
															# 	print('Done w/ Computing Statistics on Inference process during EM learning on training data. : time = ',t1-t0)
															# 	#
															# 	# Computing Inference statistics for test samples.
															# 	print('Computing Statistics on Inference process during EM learning on test data.')
															# 	t0 = time.time()
															# 	#
															# 	Y_test = [ all_SWs[i] for i in indSWs_gt_yMin_test[smp_test]  if i < num_SWsUnWrap]
															# 	Z_test = [ list() for num in range(len(smp_test)) ]
															# 	Yinf_test = [ Y_inferred_test[i] for i in range(maxSamps) if indSWs_gt_yMin_test[smp_test[i]] < num_SWsUnWrap ]
															# 	Zinf_test = [ Z_inferred_test[i] for i in range(maxSamps) if indSWs_gt_yMin_test[smp_test[i]] < num_SWsUnWrap ]
															# 	#
															# 	Kinf_test, KinfDiff_test, zCapture_test, zMissed_test, zExtra_test, \
															# 	yCapture_test, yMissed_test, yExtra_test, inferCA_Confusion_test, zInferSampled_test, zInferSampledRaw_test, \
															# 	zInferSampledT_test, inferCell_Confusion_test, yInferSampled_test = rc.compute_inference_statistics_allSamples( \
															# 	len(Y_test), N, M, Z_test, Zinf_test, np.arange(M), np.arange(M), Y_test, Yinf_test, verbose)
															# 	#
															# 	t1 = time.time()
															# 	print('Done w/ Computing Statistics on Inference process during EM learning on test data. : time = ',t1-t0)
															#
															# #
															#	
															# if flg_compute_StatsPostLrn:
															# 	all_Zs = [ list() for num in range(num_SWsUnWrap) ] # GT Z's not know for real data..
															# 	# Computing Inference statistics for all spike words after learning.
															# 	print('Computing Statistics inferring using all spikewords after Learning.')
															# 	t0 = time.time()
															# 	#
															# 	Kinf_postLrn, KinfDiff_postLrn, zCapture_postLrn, zMissed_postLrn, zExtra_postLrn, \
															# 	yCapture_postLrn, yMissed_postLrn, yExtra_postLrn, inferCA_Confusion_postLrn, zInferSampled_postLrn, zInferSampledRaw_postLrn, \
															# 	zInferSampledT_postLrn, inferCell_Confusion_postLrn, yInferSampled_postLrn = rc.compute_inference_statistics_allSamples( \
															# 	num_SWsUnWrap, N, M, all_Zs, all_Zinf, np.arange(M), np.arange(M), all_SWs, all_Yinf, verbose)
															# 	#
															# 	t1 = time.time()
															# 	print('Done w/ Computing Statistics inferring using all spikewords after Learning : time = ',t1-t0)


															

															if flg_write_CSV_stats and flg_compute_StatsPostLrn:
																pj_not_nan = list( np.where( 1-np.isnan(all_pjs).astype(int) )[0] )
																#
																data_stats_CSV.update( [ ('mean pj postLrn',[np.array([ all_pjs[x] for x in pj_not_nan ]).mean()]) , \
																		('% pj not nan',[ len(pj_not_nan) / len(all_pjs) ]) ] )
																#
																# #
																#
																yTot = yCapture_postLrn.sum()+yMissed_postLrn.sum()
																yCap = yCapture_postLrn.sum() / yTot
																yMis = yMissed_postLrn.sum() / yTot
																yExt = yExtra_postLrn.sum() / yTot
																#
																data_stats_CSV.update( [ ('# y Total postLrn',[yTot]) , ('% y Captured postLrn',[yCap]) , \
																							 ('% y Missed postLrn',[yMis]) , ('% y Extra postLrn',[yExt]) ] )


																if flg_write_CSV_stats and flg_compute_StatsPostLrn:
																	#
																	yTot = yCapture_train.sum()+yMissed_train.sum()
																	yCap = yCapture_train.sum() / yTot
																	yMis = yMissed_train.sum() / yTot
																	yExt = yExtra_train.sum() / yTot
																	#
																	data_stats_CSV.update( [ ('# y Total Train',[yTot]) , ('% y Captured Train',[yCap]) , \
																							 ('% y Missed Train',[yMis]) , ('% y Extra Train',[yExt]) ] )
																	#
																	yTot = yCapture_test.sum()+yMissed_test.sum()
																	yCap = yCapture_test.sum() / yTot
																	yMis = yMissed_test.sum() / yTot
																	yExt = yExtra_test.sum() / yTot
																	#
																	data_stats_CSV.update( [ ('# y Total Test',[yTot]) , ('% y Captured Test',[yCap]) , \
																							 ('% y Missed Test',[yMis]) , ('% y Extra Test',[yExt]) ] )
															
															





															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# (Plot 1): Plot Derivatives of model parameters during model learning.
															if flg_derivs_during_learning_plot:
																print('Plotting derivatives or parameters during model learning process.')
																t0 = time.time()
																pf.plot_params_derivs_during_learning(q_deriv, ri_deriv, ria_deriv, maxSamps, N, M, learning_rate, lRateScale, ds_fctr_snapshots, 
																			params_init, params_init_str, rand, str( EM_figs_dir + init_dir + model_dir), model_file.replace('LearnedModel_',''), figSaveFileType )
																t1 = time.time()
																print('Done with derivative plots: time = ',t1-t0) 





															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# (Plot 2): Plot learned Model post-EM-algorithm and some statistics about model and inference performance.
															if flg_learned_model_plot:
																print('Plotting learned model.')
																t0 = time.time()
																#
																plt_title = str('Learned Model ' + cell_type + ' ' + stim + ' Params w/ LR =' + str(learning_rate) + \
																	'LRsc =' + str(lRateScale) + ' :: ' + str(num_SWs) + ' SW data & ' + maxSampTag )
																#
																save_dir = str(EM_figs_dir + init_dir + model_dir)
																#
																pf.plot_learned_model(PiInv, PiaInv, Q, numCAsUpp, numCAsLow, numCellsUpp, numCellsLow,
																	Zassem_hist_allSWs[sortCA_byActivity], YcellInf_hist_allSWs[sortCells_byActivity], 
																	Ycell_hist_allSWs[sortCells_byActivity], TH_bounds, maxNumCells, maxNumCAs, maxPiQ, 
																	nY_allSWs, nYinf_allSWs, nZ_allSWs, num_SWs, num_SWs, save_dir, model_file, plt_title, figSaveFileType)
																#
																t1 = time.time()
																print('Done with plotting learned model: time = ',t1-t0) 


															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# (Plot 3): Stats on learned model and inference performance: 
															# 			Cardinality distributions (|y|,|z|).
															# 			Cells per CA and CAs per Cell.
															if flg_histograms_learned_model:

																plt.figure(figsize=(15,10))
																LW = 3 			# Line Widths
																MS = 8	 		# Marker Sizes
																FSsml = 10	 	# Smaller Font Size
																FSlrg = 14	 	# Larger Font Size
																AL = 0.5 		# Alpha values
																nYT = 6			# Number of bins for cardinality histograms
																plt.rc('font', weight='bold', size=FSsml)
																
																											#
																# #
																#
																# Compute and plot histograms of entries from learned model parameters (Pi and Pia)
																ax1b = plt.subplot2grid((2,3),(0,0))
																ax1b.plot(100*Ycell_hist_allSWs[sortCells_byActivity]/num_SWs, 'go-', linewidth=LW, markersize=MS, alpha=AL, label=str('observed $y_i$'))
																ax1b.plot(100*YcellInf_hist_allSWs[sortCells_byActivity]/num_SWs, 'rx--', linewidth=LW, markersize=MS, alpha=AL, label=str('inferred $y_i$') )
																ax1b.set_title('Cell Activity (Y)', fontsize=FSlrg)
																ax1b.set_ylabel('% Data', fontsize=FSsml)
																ax1b.set_xlabel('cell id', fontsize=FSsml)
																ax1b.set_xticks(np.floor(np.linspace(0,N,5)).astype(int))
																ax1b.grid()
																ax1b.legend(fontsize=FSsml)

																#
																# #
																#
																# Compute and plot histograms from [nYinf_allSWs, nY_allSWs, nZ_allSWs]
																binns = np.arange( np.concatenate([nYinf_allSWs, nY_allSWs, nZ_allSWs]).min(), \
																				   np.concatenate([nYinf_allSWs, nY_allSWs, nZ_allSWs]).max()+2 )
																nYh_Infer = np.histogram(nYinf_allSWs,bins=binns)
																nYh_Samps = np.histogram(nY_allSWs,bins=binns)
																nZh_Infer = np.histogram(nZ_allSWs,bins=binns) 
																#
																ax1 = plt.subplot2grid((2,3),(0,2))
																ind = np.where(nYh_Samps[0]!=0)[0]
																ax1.plot(nYh_Samps[1][ind],    np.log10(nYh_Samps[0][ind]), 'go-', linewidth=LW+1, markersize=MS, alpha=AL, label='sampled |y|')
																ind = np.where(nYh_Infer[0]!=0)[0]
																ax1.plot(nYh_Infer[1][ind],    np.log10(nYh_Infer[0][ind]), 'ro-', linewidth=LW, markersize=MS, alpha=AL, label='inferred |y|')
																ind = np.where(nZh_Infer[0]!=0)[0]
																ax1.plot(nZh_Infer[1][ind],    np.log10(nZh_Infer[0][ind]), 'ko--', linewidth=LW-1, markersize=MS, alpha=AL, label='inferred |z|')
																ax1.set_title( str('cardinality (#samp=' + str(num_SWs) + ')'), fontsize=FSlrg)
																ax1.set_ylabel('counts (log scale)', fontsize=FSsml)
																ax1.set_yticks( np.linspace(0,np.log10(num_SWs),nYT) )
																ax1.set_yticklabels( np.round(np.logspace(0,np.log10(maxSamps),nYT) ).astype(int) ) # NOTE: finish up pf.human_readable function.
																ax1.set_xticks( np.sort( np.concatenate( [ np.array([nZ_allSWs.max(),nYinf_allSWs.max(), nY_allSWs.max()]), \
																			np.round(np.linspace(0,binns.max(),nYT)).astype(int) ] )  ) )
																ax1.set_ylim(0,np.log10(num_SWs))
																ax1.tick_params(axis='both', labelsize=FSsml)
																ax1.grid()
																ax1.legend(fontsize=FSsml)
																
																#
																# #
																#
																# Compute and plot histograms of numer of CAs per Cell (Row Sums of PiaInv)
																ax4 = plt.subplot2grid((2,3),(0,1))
																x1=np.histogram(numCAsUpp,bins=np.arange(maxNumCAs[2]+2))
																ind1=np.where(x1[0]==1)[0]
																indG1=np.where(x1[0]>1)[0]
																ax4.plot(x1[1][indG1], np.log10(x1[0][indG1]), 'y^--', linewidth=LW+1, markersize=MS, alpha=AL, label=str('$\Theta_{UB}='+str(TH_bounds[0])+'$ - ' + str(x1[1][ind1])) ) 
																#
																x2=np.histogram(numCAsMid,bins=np.arange(maxNumCAs[1]+2))
																ind1=np.where(x2[0]==1)[0]
																indG2=np.where(x2[0]>1)[0]
																ax4.plot(x2[1][indG2], np.log10(x2[0][indG2]), 'cs--', linewidth=LW, markersize=MS, alpha=AL, label=str('$\Theta_{Mid}='+str(TH_bounds[1])+'$ - ' + str(x2[1][ind1])) )
																#
																x3=np.histogram(numCAsLow,bins=np.arange(maxNumCAs[0]+2))
																ind1=np.where(x3[0]==1)[0]
																indG3=np.where(x3[0]>1)[0]
																ax4.plot(x3[1][indG3], np.log10(x3[0][indG3]), 'mv--', linewidth=LW-1, markersize=MS, alpha=AL, label=str('$\Theta_{LB}='+str(TH_bounds[2])+'$ - ' + str(x3[1][ind1])) )
																
																ax4.set_yticks(  np.linspace(0,np.log10(N),nYT) )
																ax4.set_yticklabels( np.round(np.logspace(0,np.log10(N),nYT) ).astype(int) )
																ax4.set_title( str('CAs per Cell (N=' + str(N) + ')'), fontsize=FSlrg )
																ax4.set_ylabel('counts', fontsize=FSsml)
																ax4.grid()
																ax4.tick_params(axis='both', labelsize=FSsml)
																#xx = np.where( np.concatenate( [x1[0],x2[0], x3[0]] )!=0)[0]
																binticks = np.unique( np.concatenate( [x1[1][indG1], x2[1][indG2], x3[1][indG3]] ) )#[:-1]
																ax4.set_xticks(binticks)
																ax4.set_ylim(0,np.log10(N))
																ax4.legend(fontsize=FSsml)

																#
																# #
																#
																# Compute and plot histograms of entries from learned model parameters (Pi and Pia)
																x4=np.histogram(PiInv)
																x5=np.histogram(PiaInv)
																#
																ax2 = plt.subplot2grid((2,3),(1,2))
																ax2.plot(x4[1][:-1], np.log10(x4[0]), 'b^-', linewidth=LW, markersize=MS, alpha=AL, label=str('Pi ( N='+str(N)+')')  )
																ax2.plot(x5[1][:-1], np.log10(x5[0]), 'k^-', linewidth=LW, markersize=MS, alpha=AL, label=str('Pia( N*M='+str(N*M)+')')  )
																ax2.set_xlabel( str('Value in 1- Pi|Pia.   -   i.e., $p(y_i)=1$'), fontsize=FSlrg )
																ax2.set_ylabel('counts', fontsize=FSsml)
																ax2.grid()
																ax2.tick_params(axis='both', labelsize=FSsml)
																ax2.set_yticks( np.linspace(0,np.log10(N*M),nYT) )
																ax2.set_yticklabels( np.round(np.logspace(0,np.log10(N*M),nYT) ).astype(int) )
																ax2.set_ylim(0,np.log10(N*M))
																ax2.set_xlim(0,1)
																ax2.legend(fontsize=FSsml)
																#
																# #
																#
																# Compute and plot histograms of Number of Cells per CA (Column Sums of PiaInv)
																ax3 = plt.subplot2grid((2,3),(1,0))
																y1=np.histogram(numCellsUpp,bins=np.arange(maxNumCells[2]+2))
																ind1=np.where(y1[0]==1)[0]
																indG1=np.where(y1[0]>1)[0]
																ax3.plot(y1[1][indG1], np.log10(y1[0][indG1]), 'y^--', linewidth=LW+1, markersize=MS, alpha=AL, label=str('$\Theta_{UB}='+str(TH_bounds[0])+'$ - ' + str(y1[1][ind1])) )
																#ax[1][0].text( , 0.5*np.log10(y1[0]).max(), )
																#
																y2=np.histogram(numCellsMid,bins=np.arange(maxNumCells[1]+2))
																ind1=np.where(y2[0]==1)[0]
																indG2=np.where(y2[0]>1)[0]
																ax3.plot(y2[1][indG2], np.log10(y2[0][indG2]), 'cs--', linewidth=LW, markersize=MS, alpha=AL, label=str('$\Theta_{Mid}='+str(TH_bounds[1])+'$ - ' + str(y2[1][ind1])) )
																#
																y3=np.histogram(numCellsLow,bins=np.arange(maxNumCells[0]+2))
																ind1=np.where(y3[0]==1)[0]
																indG3=np.where(y3[0]>1)[0]
																ax3.plot(y3[1][indG3], np.log10(y3[0][indG3]), 'mv--', linewidth=LW-1, markersize=MS, alpha=AL, label=str('$\Theta_{LB}='+str(TH_bounds[2])+'$ - ' + str(y3[1][ind1])) )
																#
																ax3.set_xlabel( str('Cells per CA (M=' + str(M) + ')'), fontsize=FSlrg )
																ax3.set_ylabel('counts', fontsize=FSsml)
																ax3.set_yticks( np.linspace(0,np.log10(M),nYT) ) 
																ax3.set_yticklabels( np.round(np.logspace(0,np.log10(M),nYT) ).astype(int) )
																ax3.tick_params(axis='both', labelsize=FSsml)
																#xx = np.where( np.concatenate( [y1[0],y2[0], y3[0]] )!=0)[0]
																binticks = np.unique( np.concatenate( [y1[1][indG1], y2[1][indG2], y3[1][indG3]] ) ) #[:-1]
																ax3.set_xticks(binticks)
																ax3.set_ylim(0,np.log10(M))
																ax3.grid()
																ax3.legend(fontsize=FSsml)
																#
																# #
																#
																# Imshow PiaInv
																ax6 = plt.subplot2grid((2,3),(1,1))
																ax6.imshow(PiaInv, cmap='viridis') # Note: These are already sorted by Cell and CA activity.
																ax6.set_xlabel('CA id', fontsize=FSsml)
																ax6.set_ylabel('cell id', fontsize=FSsml)
																ax6.set_title('1 - $P_{ia}$', fontsize=FSlrg)

																#
																plt.suptitle( str( '\n'.join(wrap(str(argsRec),140)) ),fontsize=FSsml)
																plt.savefig( str(EM_figs_dir + init_dir + model_dir + 'Hist_' + model_file + '.pdf' ) )
																plt.close()
								
															

															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# (Plot 4): 
															if flg_temporal_inference_plot:


																print('Can U even use pf.plot_CA_inference_temporal_performance?')
																# pf.plot_CA_inference_temporal_performance(Z_inferred_list, Z_list, smp_EM, M_mod, M, \
	            			# 											numSWs, params_init, params_init_param, r, plt_save_dir, fname_save_tag	

																if np.any(Z_inf_temp):

																	Zlen_inferred = [len(i) for i in Z_inferred_train]
																	# Ylen_inferred = [len(i) for i in Y_inferred_train]
																	Z_inf_plt = Z_inf_temp[:,sortCA_byActivity]
																	activeZs  = np.where(Z_inf_plt.sum(axis=0)!=0)[0]
																	colors = plt.cm.jet_r(np.linspace(0,1,activeZs.size))

																	t0 = time.time()
																	f=plt.figure( figsize=(20,10) ) # size units in inches
																	plt.set_cmap('jet')
																	plt.rc('font', weight='bold', size=16)
																	#
																	ax = f.subplots(3)
																	f.suptitle(str('Learned Model ' + cell_type + ' ' + stim + ' Params w/ LR =' + str(learning_rate) + 'LRsc =' + 
																			str(lRateScale) + ' :: ' + str(num_SWs) + ' SW data & ' + str(maxSamps) + ' EM samples' ))
																	
																	#
																	# Plot cumulative sum of number of times inferred vs iteration # for all active CAs
																	ax[0].plot(Z_inf_plt[:,activeZs[1]]/maxSamps, linewidth=3, color=colors[1], label=str('CA#'+str(activeZs[1])) )
																	for i in range(len(activeZs)):
																		ax[0].plot(Z_inf_plt[:,activeZs[i]]/maxSamps, linewidth=3, color=colors[i])
																	ax[0].plot(Z_inf_plt[:,activeZs[-1]]/maxSamps, linewidth=3, color=colors[-1], label=str('CA#'+str(activeZs[-1])))	
																	ax[0].set_xlabel('EM iteration #')
																	ax[0].set_ylabel('cum % times CA inferred')	
																	ax[0].legend(loc='upper left', fontsize=14)								
																	
																	#
																	# Plot cumulative sum of stats showing how well inference is capturing sampled |y|'s 
																	if flg_compute_StatsDuringEM:
																		ax[1].plot( np.log10( (yCapture_train + yMissed_train).cumsum() ), linewidth=3, color='black', label='Total' )
																		ax[1].plot( np.log10( (yMissed_train).cumsum() ), linewidth=3, color='red', label='Missed' )
																		ax[1].plot( np.log10( (yCapture_train).cumsum() ), linewidth=3, color='green', label='Captured' )
																		ax[1].plot( np.log10( (yExtra_train).cumsum() ), linewidth=3, color='blue', label='Extra' )
																		ax[1].legend(fontsize=12)
																		ax[1].set_ylabel('$log_{10}$ cum # cells')	
																	
																	#
																	# Plot the |z| and |y| inferred and the difference between them. Expect |z| < |y| if we are learning cell assemblies larger than single cells.
																	#ax[2].plot( np.log10( (yCapture_train + yMissed_train).cumsum() ), linewidth=3, linestyle='--', color='red', label='|y| sampled' )
																	ax[2].plot( np.log10( np.cumsum(Zlen_inferred) ), linewidth=3, linestyle='--', color='green', label='|z| inferred' )
																	# ax[2].plot( np.log10( np.cumsum(Ylen_inferred) ), linewidth=3, linestyle=':', color='blue', label='|y| inferred' )           
																	# ax[2].plot( np.log10( np.abs( np.cumsum(Ylen_inferred) - np.cumsum(Zlen_inferred) ) ), linewidth=3, linestyle='-', color='red', label='|y|-|z| inferred' )
																	ax[2].legend(loc='lower right', fontsize=14)
																	
																	#
																	#
																	plt.savefig( str(EM_figs_dir + init_dir + model_dir + model_file.replace('LearnedModel_','TemporalInference_') + '.pdf' ) )
																	plt.close() 
																	t1 = time.time()
																	print('Done with Temporal Inference plots: time = ',t1-t0) 

																else:
																	print('No Zs inferred so cant plot temporal inference.')

															
													

															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															#
															# (Plot 9). Plot Cross validation - For training and test, plot pjoint vs EM iteration #.
															#
															plt_save_dir = str(EM_figs_dir + init_dir + model_dir)
															fname_tag = str(EM_figs_dir + init_dir + model_dir + model_file.replace('LearnedModel_',''))

															if flg_plt_crossValidation_Joint:
																# 'pj_zHyp_train', 'pj_zHyp_test', 'pj_zTru_Batch', 'pj_zTrain_Batch', 'pj_zTest_Batch', 
																print( 'Plotting Cross validation.' )
																t0 = time.time()

																kerns = [1] #100, 500] 
																pj_labels = ['train','test','truth']
																pJoints = np.vstack( [ pj_zTrain_Batch[:,0], pj_zTest_Batch[:,0], pj_zTru_Batch[:,0]] )
																pf.plot_xValidation(pJoints, pj_labels, plt_save_dir, str('xValJoint_'+fname_tag), kerns)

																t1 = time.time()
																print('Done w/ Cross validation. : time = ',t1-t0)


															if flg_plt_crossValidation_Cond: # for conditional probabilities, not joints.
																# 'cond_zHyp_train', 'cond_zHyp_test', 'cond_zTru_Batch', 'cond_zTrain_Batch', 'cond_zTest_Batch'
																print( 'Plotting Cross validation.' )
																t0 = time.time()
																#
																kerns = [1] #100, 500] 
																pc_labels = ['train','test','truth']
																pConds = np.vstack( [cond_zTrain_Batch[:,0], cond_zTest_Batch[:,0], cond_zTru_Batch[:,0]] ) #, cond_zTru_train, cond_zHyp_test, cond_zTru_test] )
																pf.plot_xValidation(pConds, pc_labels, plt_save_dir, str('xValCond_'+fname_tag), kerns)
																#
																t1 = time.time()
																print('Done w/ Cross validation. : time = ',t1-t0)		



															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															



															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# (Plot 6): NOT SO USEFUL: 2D histogram of % of observed y's Captured by inferred y vs number of active y's observed in spike word.
															if flg_2Dhist_pctYCapVsLenSW:
																if flg_compute_StatsDuringEM:
																	print('Plotting  2D Histogram % Y captured vs Len SW plots during EM-learning')
																	t0 = time.time()
																	maxSW = (yCapture_train + yMissed_train).max()
																	x=np.histogram2d(  (yCapture_train + yMissed_train), (yCapture_train)/(yCapture_train + yMissed_train), \
																		bins=[np.linspace(1,maxSW,maxSW).astype(int), np.around(np.linspace(0,1,11),1) ] )
																	#
																	f=plt.figure( figsize=(15,10) ) # size units in inches
																	plt.rc('font', weight='bold', size=16)
																	plt.imshow( np.log10(x[0].T) )
																	plt.gca().set_xticks( range(x[1].size) )
																	plt.gca().set_xticklabels( x[1].astype(int) )
																	plt.gca().set_yticks( range(x[2].size) )
																	plt.gca().set_yticklabels( x[2] )
																	plt.gca().invert_yaxis()
																	cbar=plt.colorbar() 
																	cbar.ax.set_title( str('log counts / '+str(maxSamps)), fontsize=12 )

																	plt.xlabel('# of cells in SW ')
																	plt.ylabel('% of cells in SW correctly inferred')
																	plt.title('Inference performance')
																	#           
																	plt.savefig( str(EM_figs_dir + init_dir + model_dir + model_file.replace('LearnedModel_','2Dhist_pctYCapVsLenSW_') + '_train.pdf' ) )
																	plt.close() 
																	t1 = time.time()
																	print('Done with 2D Histogram % Y captured vs Len SW plots during EM-learning: time = ',t1-t0) 
																#
																if flg_compute_StatsPostLrn:
																	print('Plotting  2D Histogram % Y captured vs Len SW plots post EM-learning')
																	t0 = time.time()
																	maxSW = (yCapture_postLrn + yMissed_postLrn).max()
																	x=np.histogram2d(  (yCapture_postLrn + yMissed_postLrn), (yCapture_postLrn)/(yCapture_postLrn + yMissed_postLrn), \
																		bins=[np.linspace(1,maxSW,maxSW).astype(int), np.around(np.linspace(0,1,11),1) ] )
																	#
																	f=plt.figure( figsize=(15,10) ) # size units in inches
																	plt.rc('font', weight='bold', size=16)
																	plt.imshow( np.log10(x[0].T) )
																	plt.gca().set_xticks( range(x[1].size) )
																	plt.gca().set_xticklabels( x[1].astype(int) )
																	plt.gca().set_yticks( range(x[2].size) )
																	plt.gca().set_yticklabels( x[2] )
																	plt.gca().invert_yaxis()
																	cbar=plt.colorbar() 
																	cbar.ax.set_title( str('log counts / '+str(num_SWsUnWrap)), fontsize=12 )

																	plt.xlabel('# of cells in SW ')
																	plt.ylabel('% of cells in SW correctly inferred')
																	plt.title('Inference performance')
																	#           
																	plt.savefig( str(EM_figs_dir + init_dir + model_dir + model_file.replace('LearnedModel_','2Dhist_pctYCapVsLenSW_') + '_postLrn.pdf' ) )
																	plt.close() 
																	t1 = time.time()
																	print('Done with 2D Histogram % Y captured vs Len SW plots post EM-learning: time = ',t1-t0) 	



															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# (Plot 7): NOT SO USEFUL: 2D histogram of |z| vs |y_obs|, that is # of CAs inferred vs # of active cells in spikeword
															if flg_2Dhist_numCAvsLenSW:
																if flg_compute_StatsDuringEM:
																	print('Plotting  2D Histogram % num CAs vs Len SW plots during EM-learning')
																	t0 = time.time()
																	maxSW = (yCapture_train + yMissed_train).max()
																	x=np.histogram2d(  (yCapture_train + yMissed_train), Kinf_train, \
																		bins=[np.linspace(1,maxSW,maxSW).astype(int), np.linspace(1,Kinf_train.max(),Kinf_train.max()).astype(int) ] )#, color='green', s=10, alpha=0.01, label='Captured' )
																	#
																	f=plt.figure( figsize=(15,10) ) # size units in inches
																	plt.rc('font', weight='bold', size=16)
																	plt.imshow( np.log10(x[0].T) )
																	plt.gca().set_xticks( range(x[1].size) )
																	plt.gca().set_xticklabels( x[1].astype(int) )
																	plt.gca().set_yticks( range(x[2].size) )
																	plt.gca().set_yticklabels( x[2].astype(int) )
																	plt.gca().invert_yaxis()
																	cbar=plt.colorbar() #ax.scatter(  (yCapture_train + yMissed_train), (yMissed_train)/(yCapture_train + yMissed_train), color='red', s=10, alpha=0.01, label='Missed' )
																	cbar.ax.set_title( str('$log_{10}$ counts / '+str(maxSamps)), fontsize=12 )

																	plt.xlabel('# of cells in SW ')
																	plt.ylabel('# of CAs inferred')
																	plt.title('How Many Cell Assemblies are used for longer spike words?')
																	#           
																	plt.savefig( str(EM_figs_dir + init_dir + model_dir + model_file.replace('LearnedModel_','2Dhist_numCAvsLenSW_') + '_train.pdf' ) )
																	plt.close() 
																	t1 = time.time()
																	print('Done with 2D Histogram # CAs active vs Len SW plots during EM-learning: time = ',t1-t0) 
																	
																if flg_compute_StatsPostLrn:
																	print('Plotting 2D Histogram % num CAs vs Len SW plots post EM-learning')
																	t0 = time.time()
																	maxSW = (yCapture_postLrn + yMissed_postLrn).max()
																	x=np.histogram2d(  (yCapture_postLrn + yMissed_postLrn), Kinf_postLrn, \
																		bins=[np.linspace(1,maxSW,maxSW).astype(int), np.linspace(1,Kinf_postLrn.max(),Kinf_postLrn.max()).astype(int) ] )#, color='green', s=10, alpha=0.01, label='Captured' )
																	#
																	f=plt.figure( figsize=(15,10) ) # size units in inches
																	plt.rc('font', weight='bold', size=16)
																	plt.imshow( np.log10(x[0].T) )
																	plt.gca().set_xticks( range(x[1].size) )
																	plt.gca().set_xticklabels( x[1].astype(int) )
																	plt.gca().set_yticks( range(x[2].size) )
																	plt.gca().set_yticklabels( x[2].astype(int) )
																	plt.gca().invert_yaxis()
																	cbar=plt.colorbar() #ax.scatter(  (yCapture_train + yMissed_train), (yMissed_train)/(yCapture_train + yMissed_train), color='red', s=10, alpha=0.01, label='Missed' )
																	cbar.ax.set_title( str('$log_{10}$ counts / '+str(num_SWsUnWrap)), fontsize=12 )

																	plt.xlabel('# of cells in SW ')
																	plt.ylabel('# of CAs inferred')
																	plt.title('How Many Cell Assemblies are used for longer spike words?')
																	#           
																	plt.savefig( str(EM_figs_dir + init_dir + model_dir + model_file.replace('LearnedModel_','2Dhist_numCAvsLenSW_') + '_postLrn.pdf' ) )
																	plt.close() 
																	t1 = time.time()
																	print('Done with 2D Histogram # CAs active vs Len SW plots post EM-learning: time = ',t1-t0) 





															# Plot confusion matrix for Cells.
															if False:
																#
																plt.imshow( np.log10(inferCell_Confusion_train) )
																plt.colorbar()
																plt.show()	



															# Plot cumulative % of observed y's in spikeword captured by inferred y as a function of EM training iteration
															if False:
																plt.scatter( np.arange(maxSamps), yCapture_train/(yCapture_train + yMissed_train)  )
																plt.ylabel('% of active cells captured')
																plt.xlabel('iteration #')
																plt.show()



															# # # # # # # # # # # # # # # # # 
															if flg_Pia_snapshots_gif:

																TH_bounds = np.array([0.3, 0.6]) # upper and lower bounds for threshold to count up CperA and AperC

																# Sort cell assemblies and cells by how active they are in inference procedure
																sortCA_byActivity 	  = np.argsort(Zassem_hist_allSWs[:-1])[::-1]
																sortCells_byActivity = np.argsort(YcellInf_hist_allSWs[:-1])[::-1]
																#sortCells_byActivity = np.argsort(Ycell_hist_Samps[:-1])[::-1]
																#yInferSampled_traintrans = yInferSampled_train[sortCells_byActivity]
																Ycell_hist_SampsTrans = Ycell_hist_Samps[sortCells_byActivity]

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


																print('Number of snapshots are ',numSnaps)	
																snaps = range(1,numSnaps) #range(ria_snapshots.shape[0])
																for i in snaps:

																	# Stats on active cells and cell assemblies inferred during EM algorithm
																	print('compute Inference statistics')
																	t0 = time.time()
																	sampAtSnap = int( i*maxSamps/(numSnaps-1) )
																	Ycell_hist_InferSnap, Zassem_hist_InferSnap, nY_InferSnap, nZ_InferSnap, \
																		CA_coactivity_InferSnap, Cell_coactivity_InferSnap = rc.compute_dataGen_Histograms( \
																		Y_inferred_train[:sampAtSnap], Z_inferred_train[:sampAtSnap], M, N) # , sampAtSnap
																	t1 = time.time()
																	print('Done w/ inference stats : time = ',t1-t0) # Fast enough: ~10 seconds
																	#
																	Ycell_hist_InferSnap  = Ycell_hist_InferSnap[sortCells_byActivity]
																	Zassem_hist_InferSnap = Zassem_hist_InferSnap[sortCA_byActivity]
																	#
																	QSnap 			= rc.sig(q_snapshots[i])
																	PiSnap 	 	  	= ( 1-rc.sig(ri_snapshots[i]) )[sortCells_byActivity]
																	PiaSnap 	  	= ( 1-rc.sig(ria_snapshots[i]) )[np.ix_(sortCells_byActivity,sortCA_byActivity)]
																	numCAsSnapUB	= ( PiaSnap>TH_bounds.min()).sum(axis=1)
																	numCAsSnapLB	= ( PiaSnap>TH_bounds.max()).sum(axis=1)
																	numCellsSnapUB	= ( PiaSnap>TH_bounds.min()).sum(axis=0)
																	numCellsSnapLB 	= ( PiaSnap>TH_bounds.max()).sum(axis=0)

																	plt_title = str('Learned Model ' + cell_type + ' ' + stim + ' Params w/ LR =' + str(learning_rate) + \
																	'LRpiq =' + str(lRateScale_PiQ) + ' :: ' + str(num_SWs) + ' SW data & ' + str(sampAtSnap) + ' EM samples' )


																	plt_save_tag = str( 'SnapShots' + str(i) )
																	#
																	pf.plot_learned_model(PiSnap, PiaSnap, QSnap, numCAsSnapUB, numCAsSnapLB, numCellsSnapUB, numCellsSnapLB, Zassem_hist_InferSnap,\
																		Ycell_hist_InferSnap, Ycell_hist_SampsTrans, TH_bounds, maxNumCells, maxNumCAs, maxPiQ, nY_Samps, nY_InferSnap, nZ_InferSnap, \
																		sampAtSnap, maxSamps, EM_figs_dir, plt_save_tag, plt_title)



															if flg_write_CSV_stats:
																df = pd.DataFrame(data=data_stats_CSV)
																df.to_csv(fname_CSV, mode='a', header=(dfh==0))
																dfh+=1	


													
															print('REAL SEXY TIME !!!!')
															# sexytime


														# except:
														# 	print('Something not working!!')
														# 	print( cell_type, N, z0_tag, stim, maxSamps, overcomp, yLo, yHi, msBins, learning_rate, rand)





# 			# NOTE THAT THIS IS AT THE LEVEL OF DIFFERENT YMIN VALUES. 
# 			# MAKE A PSTH OF NUMBER OF TRIALS WHERE AT LEAST A CERTAIN NUMBER OF CELL SPIKE IN A 1MS BIN
# 			minTms=0
# 			maxTms=6000
# 			binnedHists = rc.compute_PSTH_trial_vs_time( SWs, SWtimes, numTrials, minTms, maxTms )


# 			f=plt.figure( figsize=(20,5) ) # size units in inches
# 			plt.rc('font', weight='bold', size=14)	
# 			for ym in yMins:					
# 				x = np.array(binnedHists>ym)
# 				plt.plot(x.sum(axis=0), label=ym, linewidth=1 )
# 			#
# 			plt.plot( (numTrials/binnedHists.sum(axis=0).max())*binnedHists.sum(axis=0), color='black', label='all', linewidth=1, linestyle='--')
# 			#
# 			plt.title( str('PSTH thresholded for number of cells varying yMin: for SWbins ' + str(2*SW_bin+1) + 'ms, ' + stim + ' stimulus and ' + cell_type + ' cells') )
# 			plt.xlabel('Time (binned in ms)')
# 			plt.ylabel( str('Number of trials out of ' + str(numTrials) ) )
# 			#	
# 			plt.legend()
# 			plt.grid()
# 			#    
# 			plt_save_dir = str(EM_figs_dir + '../../PSTH_' + cell_type + '_N' + str(N) + '/')
# 			if not os.path.exists(plt_save_dir):
# 				os.makedirs(plt_save_dir)
# 			plt.savefig( str(plt_save_dir + 'PSTH_yMins_' + model_file + '.pdf' ) )




# 			# MAKE RASTER PLOT TRIAL VS TIME WHERE SPIKEWORD > YMIN WAS ACTIVE.
# 			f=plt.figure( figsize=(20,10) ) # size units in inches
# 			plt.rc('font', weight='bold', size=14)	
# 			for ym in yMins:	
# 				plt.imshow(binnedHists>ym, aspect='auto', cmap='bone_r')
# 				plt.title( str('PSTH when spike word > yMin' + str(ym) +  ': for SWbins ' + str(2*SW_bin+1) + 'ms, ' + stim + ' stimulus and ' + cell_type + ' cells') )
# 				plt.xlabel('Time (binned in ms)')
# 				plt.ylabel( str('Number of trials out of ' + str(numTrials) ) )
# 				plt.savefig( str(plt_save_dir + 'raster_' + model_file + 'yMin' + str(ym) + '.pdf' ) )



# if flg_plot_PSTH_numCellsYmin:
# 	print('Go')
# 	# f_PSTH_yMin.show() # show a figure.
# 	#plt.figure() # recall a figure
# 	for ind,xxx in enumerate(yMins):
# 		#plt.scatter(hist_vs_yMin[ind][0],hist_vs_yMin[ind][1], s=1, label=xxx ) #, linewidth=0.4, marker='.', markersize=4, alpha=1. )
# 		plt.plot(hist_vs_yMin[ind][0],hist_vs_yMin[ind][1], label=xxx, linewidth=0.2 )
# 		plt.title( str('Biased sampling with varying yMin: ' + stim + 'stimulus and ' + cell_type + ' cells') )
# 		plt.xlabel('Time (binned in ms)')
# 		plt.ylabel( str('Number of samples out of ' + str(maxSamps) ) )
# 	plt.legend()
# 	plt.show()
# 	plt.close()







					




