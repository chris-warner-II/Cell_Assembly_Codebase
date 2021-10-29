import argparse
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.signal as sig
import os
import time
from scipy import io as io

import h5py
import neo
import elephant as el

from quantities import Hz, s, ms, V

import utils.data_manipulation as dm
import utils.plot_functions as pf
import utils.retina_computation as rc
import utils.my_el_sta as sta
#
from sklearn.metrics.pairwise import cosine_similarity 
import scipy.sparse as spsp
from scipy.sparse.csgraph import reverse_cuthill_mckee
#
from textwrap import wrap


# Colors for raster and PSTH plots below.
colsDark = ['green','blue','black','cyan','darkorchid','saddlebrown','orange','beige']
colsBright = ['chartreuse','red','cyan','gold','royalblue','orange','beige','lime'] # 'fuscia',



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (0). Flags for what parts to run and plots to make.
#
#
flg_save_rasters4MAT 			= False 	# Save a .mat file with raster info to be ported into MATLAB. Not using anymore I dont think.
#
flg_compute_CATA				= False 	# Cell Assembly Triggered Average. Not informative. Stimulus too short.
#
flg_raster_Y_andItsZs 			= False 	# Plot single cell raster and rasters of all CAs it participates in. Old. Must edit / update to reinstate.
#	
flg_PSTH_Y_andItsZs 			= False 	# Not really using these. PSTHs instead of rasters above. Meh.
flg_PSTH_Z_andItsYs 			= False 	# Not using.
# (1). Look for periodicity by looking at CA ISIs in raster plots.
flg_compute_CA_ISIs 			= False 
# flg_plot_pOfy_underNull 				= True 	# Diagnostic plots for single models learned. Scatter plots of p(y)_{GLM} and <cosSim>_{learnedModels} for each CA.

# # # NOT REALLY USING OPTIONS ABOVE. BUT KEEP THEM AND MAYBE REINSTATE. MAY BECOME USEFUL AGAIN. # # # # # # # #
#
#
#
#
#
#
#
#
flg_plt_learned_model_realData 			= False 	# Nice visualization of Pia, Pi, Q, inference of CAs and activity of Cells in one figure.
#
flg_plot_CA_metricsCorrs 				= False 
#
flg_raster_Z_andItsYs 					= False 	# Plot single Cell assembly raster and rasters of all cells in it. Plot spatial layout on retina. And CAs in other learned models that match it.
#
flg_plot_crispRobust_RFs_and_PSTH 		= False

flg_plot_tempCoactiveCAs_RFs_and_PSTH 	= False

flg_show_stim_at_high_PSTH 				= True 	# Show stim at and a little before CA activity.

flg_compare_to_GLM_null 				= True

flg_doPlotGLMcomp 						= False

flg_plt_model_CAs_coactivity 			= False

flg_plt_model_CAs_replacements 			= False

#
# TO DO: (6/20/19)

#
# (2). How similar are CA's in their temporal profiles. 
# 			(across model learned for matching ones. Like cosSim for time.)
# 			(and across CAs in single model to get at coactivation matrix.)
flg_match_CAs_temporal 			= False
#
# (X). Compute and make a nice plot displaying stats computed about CAs for each model.
flg_compute_CA_stats 			= True 
flg_plot_CA_stats 				= True 
#
#
# (3). How to and is it worth characterizing shape of cell RFs in CA? 
#		Linear ones or wave front ones or blobby ones or spread out nonsense diffuse ones?
# NOT DOING YET.
# 




figSaveFileType = 'png' # 'png' or 'pdf'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (1). User specified parameters for which things to run.
#
#
cellSubTypesS =  ['[offBriskTransient,onBriskTransient]'] #['[offBriskTransient]']#,'[offBriskTransient,offBriskSustained]'] #['[offBriskTransient,onBriskTransient]']#
Ns 		=  [94]#, 98]#, 94] # 55 # 
Nsplit = 55 # number of cells in 1st population for plotting RFs in different colors in raster_YorZ function below.

stims 	 = ['NatMov']#,'Wnoise']
whichSim = 'GLM' # 'GLM' or 'LNP'


model_CA_overcompleteness = [1] #[1,2] 	# how many times more cell assemblies we have than cells (1 means complete - N=M, 2 means 2x overcomplete)
SW_bins = [2]#, 1, 0]			# ms. Build up spikewords from groups of cells in the same trial that fire within SW_bins of eachother.

yLo_Vals 		= [0] #[1,4] 	# If |y|<=yLo, then we force the z=0 inference solution and change Pi. This defines cells assemblies to be more than 1 cell.
yHi_Vals 		= [1000] 		# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution and change Pia.
yMinSWs 		= [1]#[1,2,3]	# DOING BELOW THING WITH YYY inside pgm functions. --> (set to 0, so does nothing) 
								# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<yLo.



bsPSTH 	= 50 # time in ms of one PSTH bin. For computing temporal cosine similarity of CA activations.


learning_rates = [0.5] #[1.0, 0.5, 0.1]
lRateScaleS = [[1., 0.1, 0.1]]# , [1., 0.1, 1.] ]  # Multiplicative scaling to Pia,Pi,Q learning rates. Can set to zero and take Pi taken out of model essentially.

pct_xVal_train = 0.5

which_train_rand = [0, 1, 2] # Vector of which rand values to use. Choose a "good" model. 
train_2nd_modelS = [False, True]# False if there is no B file, True to run only B file, [True,False] to run both.

flg_EgalitarianPriorS = [ False ] # True,False] # 
sample_longSWs_1stS = ['Dont']#,'Prob'] # Options are: {'Dont', 'Prob', 'Hard'}

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
flgs_include_Zeq0_infer  = [True] # ALWAYS TRUE 
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



xVal_or_all = 'allSWs' # 'xValSWs' or 'allSWs' when computing rasters. # NOTE: DOES NOT WORK WITH xValSWs

# pjt_tol=10

# 	Parameters that I want to look into and see if I can rethink how I determine if a cell is active.
TH 		= 0.3 	# Binarizing Pia: threshold for 1-Pia to pass to say a "cell participates" in a CA.
minMemb = 2 	# minimum number of cells per CA or CAs per cell to call CA or Cell "good" or "interesting"
maxMemb = 8 	# maximum number of cells per CA or CAs per cell to call CA or Cell "good" or "interesting"
				# maxMemb limited by colors above. Add more to look at larger CAs. But rasters will also get cluttered.



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (2). Set up directories and create dirs if necessary.
#
#
dirHome, dirScratch = dm.set_dir_tree()
EM_learning_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Models_learned_EM/')
# EM_figs_dir 		= str( dirScratch + 'figs/PGM_analysis/EM_Algorithm/Greg_retinal_data/')
SW_extracted_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')
PSTH_data_dir  		= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Rasters_trial_vs_time_zInferred/')
rastersMATdir 		= str( dirScratch + 'data/matlab_data/CA_raster_times/')
GLM_pofy_dir 		= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/GLM_p_of_y/')

if not os.path.exists( GLM_pofy_dir ):
	os.makedirs( GLM_pofy_dir )	





# Name of data files saved from my preprocessing in MATLAB of raw datafiles from Greg Field's Lab.
GField_spikeData_File = str('allCells_spikeTrains_CellXTrial.mat')
#
if whichSim == 'GLM':
	GField_GLMsimData_File = str('GLM_cat_sim_fullpop.mat') 	# Independent GLM with spike history.
elif whichSim == 'LNP':
	GField_GLMsimData_File = str('LNP_cat_sim_fullpop.mat')	# LNP model without spike history.
else:
	print('I dont understand whichSim',whichSim)




# Load in STRF mat file
data = spio.loadmat( str( dirScratch + 'data/matlab_data/allCells_STRF_fits_329cells.mat') )
STRF_gaussD = data['STRF_GaussParamsD'] # % parameters are: [Amplitude, x0, sigmax, y0, sigmay, angle(in rad)]
STRF_gaussL = data['STRF_GaussParamsL'] # L = Light Responses, D = Dark Responses, M = Mean Responses.
STRF_timeD = data['STRF_TimeParamsD']
STRF_timeL = data['STRF_TimeParamsL']
del data

# Load in original data file to regrab cell types and cell type IDs
spikes = spio.loadmat( str(dirScratch + 'data/matlab_data/allCells_spikeTrains_CellXTrial.mat') ) # forPython.mat') )
allCellTypes = spikes['allCellTypes']
cellTypeIDs = spikes['cellTypeIDs']
del spikes




for ct,cellSubTypes in enumerate(cellSubTypesS):
	N = Ns[ct]


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # (2). Clean up cell type and cell id data passed in from MATLAB and grab cells from certain subtypes to look for Cell Assemblies in.
	#
	#
	# (a). Covert allCellTypes ndarray of ndarrays weird format passed from matlab to a list.
	cellTypesList = [ allCellTypes[i][0][0] for i in range( np.size(allCellTypes) ) ]
	#print(cellTypesList)
	#
	# (b). Grab entries in cellTypesList that are in particular cellSubTypes we want to compare.
	indx_cellSubTypes = [ allCellTypes[i][0][0] in cellSubTypes for i in range( np.size(allCellTypes) ) ]
	#print(indx_cellSubTypes)
	#
	# (c). Extract number of cells of each cell Type.
	num_cells_SubType = [ np.sum(cellTypeIDs[i]>0)  for i in range( np.size(allCellTypes) ) ]
	#print(num_cells_SubType)
	#
	# (d). Index into cellTypeIDs to grab cell IDs that belong to the cellSubTypes we want to compare.
	cellSubTypeIDs = cellTypeIDs[indx_cellSubTypes].flatten()
	cellSubTypeIDs = cellSubTypeIDs[ cellSubTypeIDs>0 ] - 1 # !!! This -1 converts from Matlab indexing to Python indexing
	#print( cellSubTypeIDs )


	csts = [ cellTypesList[ x ] for x in list(np.where(indx_cellSubTypes)[0]) ]


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# Sort which Receptive fields to use for different cell types. The algorithm to find them in
	# MATLAB is kinda fragile and sometimes the Dark Response yeilds better RFs and sometimes the 
	# Light Response is better. Maybe go back and fix it at some point. Is fine for now with the 
	# cells and celltypes we are currently looking at.
	#
	# For {offBriskTransient,onBriskTransient}, need to combine {STRF_gaussD,STRF_gaussL}
	STRF_gauss = STRF_gaussD # set to all Dark responses at first.
	#
	for i in range(len(indx_cellSubTypes)):
		if indx_cellSubTypes[i] and 'on' in cellTypesList[i]: # replace Dark with Light responses for ON cells.
			#print(cellTypesList[i])
			indsIntoOn = cellTypeIDs[i][cellTypeIDs[i]>0] - 1
			#print(indsIntoOn)
			STRF_gauss[indsIntoOn] = STRF_gaussL[indsIntoOn]







	



	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# Visualize Stimulus at CA activity and slightly before.
	#
	if flg_show_stim_at_high_PSTH: 
		print('Loading in and Unravelling the stimulus movie for STA.')
		# data = h5py.File( str( dirScratch + 'data/GField_data/all_cells_2017_01_16_0.mat'), 'r') # This is right for White Noise, incorrectly shifted for NatMov. (need hp5y because its an old MAT file)
		# Mov_stim = data['movie_wnrep']
		# #
		data = spio.loadmat( str( dirScratch + 'data/GField_data/correct_movie_catrep.mat') ) # This is right for Natural Movie.
		Mov_stim = data['movie_catrep'] 	# Natural Movie Stim: 600 yPix X 795 xPix X 300 timeBins. Time per stim frame is 16 & 2/3 ms.
		del data	

		timeBins 	= Mov_stim.shape[2]
		xPix 		= Mov_stim.shape[1]
		yPix 		= Mov_stim.shape[0]
		

		# There is a border where no stim is shown in the catrep stim. Cut it off to get rid of dims for STA to run quicker.
		crop_mov = True
		if crop_mov:
			yLims 	= np.array([59,539]) # These values were pulled out by eye from visualizing a frame of the stil.
			xLims 	= np.array([0,639])
			Mov_stim = Mov_stim[ yLims[0]:yLims[1], xLims[0]:xLims[1], :  ]
			yPix 	= yLims[1] - yLims[0]
			xPix 	= yLims[1] - yLims[0]
		else:
			yLims = np.array([0,yPix])
			xLims = np.array([0,xPix])

		
		# Mov_stim_unrav = np.zeros( (timeBins, xPix*yPix) ) 
		# for i in range(timeBins):
		# 	print('Time bin',i)
		# 	Mov_stim_unrav[i] = Mov_stim[:,:,i].ravel()


		# print('Unravelled movie stim is size ',Mov_stim_unrav.shape )








	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	#
	# Do global shifts in stimulus correlate with CA activity?
	#
	# Maybe come back to this.
	# 
	
	if False:

		plt.imshow(Mov_stim[:,:,3]-Mov_stim[:,:,4])
		plt.colorbar()
		plt.show()

		Mov_flow = np.diff(Mov_stim,axis=2)
		mnMovFlow = np.zeros(Mov_flow.shape[2])

		for xx in range(Mov_flow.shape[2]):
			mnMovFlow[xx] = np.abs(Mov_flow[:,:,xx]).mean()


		plt.plot(mnMovFlow)
		plt.show()	


		# Q: Why is every second movie frame identical to previous one???
		plt.imshow(Mov_stim[:,:,1]-Mov_stim[:,:,2])
		plt.colorbar()
		plt.show()
		#
		plt.imshow(Mov_stim[:,:,3]-Mov_stim[:,:,4])
		plt.colorbar()
		plt.show()


		plt.plot(2*np.arange(mnMovFlow[::2].size), mnMovFlow[::2])
		plt.show()







	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # (1). Load in npz file and extract data from it.
	#
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

						CAS_to_show_stim_at_high_PSTH =  [5, 19, 68, 78] #np.arange(M) #
						colors = plt.cm.jet(np.linspace(0,1,M)) # plt.cm.get_cmap('jet',M) 
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
												init_dir = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') + '_LRsc' + str(lRateScale) +'/' )
												#
												# LOAD IN TEMPORAL PSTH FILES SO THAT WE CAN MATCH UP CA ACTIVATIONS IN TIME FOR CA'S THAT HAVE
												# BEEN MATCHED UP SPATIALLY USING COS-SIM.
												#
												psthZ_allMods = list()
												psthZ_fnames  = list()
												#
												for rand in which_train_rand:
													#
													for train_2nd_model in train_2nd_modelS:
														#
														if train_2nd_model:
															Btag = 'B'
														else:
															Btag = ''
														#
														try:
															#
															temporalCSofCAs_dir 	= str( EM_learning_Dir + init_dir + 'CA_temporalCosSim/')
															temporalCSofCAs_fname 	= str( cellSubTypes + '_' + stim + '_' + str(msBins) + 'msBins_' + \
																	 str(bsPSTH) + 'msPSTHbins_'+ sample_longSWs_1st + 'Smp1st_' + priorType + '_rand' + str(rand) + Btag + '.npz')
															tPSTH = np.load( str(temporalCSofCAs_dir+temporalCSofCAs_fname) )
															#
															psthZ_accum = tPSTH['psthZ_accum']
															binsPSTH 	= tPSTH['binsPSTH']
															TcosSimMat 	= tPSTH['TcosSimMat'] 
															cmk_resort 	= tPSTH['cmk_resort']

															psthZ_accum.shape
															psthZ_allMods.append( psthZ_accum )
															psthZ_fnames.append( str(str(rand) + Btag) )

														except:
															print('Missing PSTH files for ', str(rand) + Btag)
												#			
												# #
												#
												for rand in which_train_rand:
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


														if True:
														# try:
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# # (2).  Set up directory structure and filename. Load it in and extract variables.
															#
															# Find directory (model_dir) with unknown N and M that matches cell_type and yMin.
															searchDir 	= str(EM_learning_Dir + init_dir)
															specifier1 	= cellSubTypes # str(cellSubTypes).replace('/','').replace(' ','')
															specifier2 	= str( z0_tag + '_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) + '_' + sample_longSWs_1st + 'Smp1st_' + priorType)
															#
															N, M, model_dir = dm.find_M_N(searchDir, specifier1, specifier2, z0_tag)
															model_dir = str(model_dir + '/')


															#
															# #
															# Find npz file (model_file) inside model_dir with unknown numSWs, numTrain, numTest but matching stim, msBins, EMsamps and rand.
															searchDir 	= str(EM_learning_Dir + init_dir + model_dir)
															specifier1 	= str('LearnedModel_' + stim)
															specifier2 	= str(  str(pct_xVal_train).replace('.','pt') + 'trn_' + str(msBins) + 'msBins' + str(maxSampTag) + '_rand' + str(rand) + Btag + '.npz' )
															#
															numSWs, model_file = dm.find_numSWs(searchDir, specifier1, specifier2, stim)



															output_figs_dir = str( dirScratch + 'figs/PGM_analysis/PSTH_and_Rasters/' + init_dir + model_dir)
															if not os.path.exists( output_figs_dir ): # Make directories for output data if not already there.
																os.makedirs( output_figs_dir )



															# For plot titles and file names below...
															fname = str( cellSubTypes + '_' + stim + '_' + str(msBins) + 'msBins_' + sample_longSWs_1st + 'Smp1st_' + priorType + '_rand' + str(rand) + Btag )

															print('fname:', fname)

															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
															#
															# Load in Learned Model Data file
															#
															data = np.load(str(EM_learning_Dir + init_dir + model_dir + model_file) )
															#
															qp = data['qp']
															rip = data['rip']
															riap = data['riap']
															#
															Pia = rc.sig(riap)
															Pi  = rc.sig(rip)
															Q 	= rc.sig(qp)
															# 
															del data
															# 

															


															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
															#
															# Load in PSTH data file for latent Z variables.
															#
															data = np.load( str(PSTH_data_dir + init_dir + model_dir + model_file.replace('LearnedModel_',str('rasterZ_'+xVal_or_all+maxTrialTag+'_')  ) ) )
															#
															Z_inferred_allSWs 			= data['Z_inferred_allSWs'] 
															#pyiEq1_gvnZ_allSWs 			= data['pyiEq1_gvnZ_allSWs'] 
															#pj_inferred_allSWs 			= data['pj_inferred_allSWs'] 
															#cond_inferred_allSWs		= data['cond_inferred_allSWs']
															#
															#Ycell_hist_allSWs 			= data['Ycell_hist_allSWs']
															#YcellInf_hist_allSWs 		= data['YcellInf_hist_allSWs'] 
															#Zassem_hist_allSWs 			= data['Zassem_hist_allSWs']

															#nY_allSWs 					= data['nY_allSWs']
															#nYinf_allSWs 				= data['nYinf_allSWs'] 
															#nZ_allSWs 					= data['nZ_allSWs']

															#CA_coactivity_allSWs 		= data['CA_coactivity_allSWs']
															#Cell_coactivity_allSWs 		= data['Cell_coactivity_allSWs'] 
															#CellInf_coactivity_allSWs 	= data['CellInf_coactivity_allSWs'] 

															#argsRecModelLearn 			= data['argsRecModelLearn']
															#argsRaster 					= data['argsRaster']
															#
															del data





															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
															#
															# Load in data file with p(y) under the GLM (and LNP) null models.
															#
															data = np.load( str(GLM_pofy_dir + cellSubTypes + '_' + stim + '_' + whichSim + '_' + str(msBins) + 'msBins_stAt333ms.npz') )
															pSW_nullGLM = data['pSW_nullGLM']
															# del data

															

															



															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															#
															# (2).  Get extracted spikewords from spike data to look at statistics of spiking activity.
															#
															# Extract spike words from raw data or load in npz file if it has already been done and saved.
															if not os.path.exists( SW_extracted_Dir ): # Make directories for output data if  not already there.
																os.makedirs( SW_extracted_Dir )
															#
															print('Extracting spikewords')
															t0 = time.time()
															fname_SWs = str( SW_extracted_Dir + cellSubTypes + '_' + stim + '_' + str(msBins) + 'msBins.npz' )
															spikesIn = list()
															SWs, SWtimes = rc.extract_spikeWords(spikesIn, msBins, fname_SWs)
															numTrials = len(SWs)
															t1 = time.time()
															print('Done Extracting spikewords: time = ',t1-t0)

															# print(numTrials)
															# print(len(SWs[0]))
															# print(SWs[0][:10])





															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															#
															# (2).  Get average cosine similarity for matching CAs in all other models for each CA in each model
															#
															#
															meanCSofCAs_dir 	= str( EM_learning_Dir + init_dir + 'CA_meanCosSim/')
															meanCSofCAs_fname 	= str( cellSubTypes + '_' + stim + '_' + str(msBins) + \
																'msBins_' + sample_longSWs_1st + 'Smp1st_' + priorType + '_' + str(6) + 'rands.npz')
															mnCS = np.load( str(meanCSofCAs_dir + meanCSofCAs_fname) )

															mean_CS_accModels = mnCS['mean_CS_accModels']
															ZmatchCS_accModels = mnCS['ZmatchCS_accModels']
															PiaBox = mnCS['PiaBox']
															fparams = mnCS['fparams']
															fnameM = mnCS['fnames']
															fnameR = mnCS['fnameR']
															#
															del mnCS

															i_mdl = np.where( fparams==str( str(rand) + Btag ) )[0][0] # index into the model for this particular rand and Btag value.












															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
															#
															# Build Pia_gather and PSTHz_gather: 
															#
															# A cell assembly in this model and all cell assemblies it was matched with in other models
															# This is used below in the raster_YorZ_andItsMembers function. Also, I can compute
															# mean_CST_accModels (the average temporal cosine similarity between CA activation) to use that
															# in the flg_plot_pOfy_underNull section.
															#


															# Bug Fixed. Convert index into model (PiaBox) to an index into PSTH.
															# They can be misalligned because different ordering of going thru rand and Btag above. Ugh, Bugs.
															for a, fn in enumerate(psthZ_fnames):
																if fn == fparams[i_mdl]:
																	i_ras = a


															numMods = mean_CS_accModels.shape[0]
															#
															Pia_gather = list()	# Cell assemblies that match in other models 
															PSTHz_gather = list()	# CA activations for CAs that match spatially across models.
															Zmatch_gather = list()	# list of strings M x nMods. Contains other model and other CA this one is matched to.

															for iii in range(M): # index into 1st model 'i_mdl' CA

																Pia_gather.append( np.zeros( (N,numMods) )	) # Cell assemblies that match in other models
																PSTHz_gather.append( np.zeros( (len(binsPSTH)-1, numMods) )	) # Activations of Cell assemblies that match in other models
																Zmatch_gather.append( list() ) #( (2,numMods) ) # For other matching CAs, this contains (model, z)
																#
																#print( mean_CS_accModels[i_mdl,iii] )
																ggg = 0
																Pia_gather[iii][:,ggg] = 1 - PiaBox[i_mdl,:,iii]
																try:
																	PSTHz_gather[iii][:,ggg] = psthZ_allMods[i_ras][iii] 
																except:
																	aaa=0 #print('Must be a missing raster file for ',Zmatch_gather[iii][ggg])
																Zmatch_gather[iii].append( ( fparams[i_mdl],str(iii) ) ) #str(fparams[i_mdl]+' z'+str(iii)) )
																#
																for bbb in range( numMods ): # model 2nd
																	if not bbb==i_mdl:
																		ggg += 1
																		jjj = int(ZmatchCS_accModels[i_mdl,bbb,iii]) # index into model bbb CA
																		Pia_gather[iii][:,ggg] = 1 - PiaBox[bbb,:,jjj]
																		Zmatch_gather[iii].append( ( fparams[bbb], str(jjj) ) ) #str(fparams[bbb]+' z'+str(jjj)) )
																		

																		# Again, convert index into model (PiaBox) to an index into PSTH.
																		for a, fn in enumerate(psthZ_fnames):
																			if fn == fparams[bbb]:
																				b_ras = a
																		#
																		# Check that psthZ file name for 2nd model matches and then grab time course for matching CA in there.
																		try:
																			PSTHz_gather[iii][:,ggg] = psthZ_allMods[b_ras][jjj] # 
																		except:
																			aaa=0 #print('Must be a missing raster file for ',Zmatch_gather[iii][ggg])





															# Compute average temporal cosine similarity between za activations in this model and other best matching zs in other models. 
															rasFiles = np.where( np.any(PSTHz_gather[0],axis=0) )[0] # if rasterfile wasnt there, values will be all zeros
															mean_CST_accModels = np.zeros( len(PSTHz_gather) ) 
															for i in range(len(PSTHz_gather)):
																csT = cosine_similarity(PSTHz_gather[i].T,PSTHz_gather[i].T) 
																mean_CST_accModels[i] = csT[0,rasFiles[1:]].mean() # [1:] so you dont grab cosSim with itself. CA itself is in 0 position.


															

															# For Diagnostics: Plot each CA and matching CAs in other models side by side.
															if False:
																for iii in range(M): # index into 1st model 'i_mdl' CA
																	f,ax = plt.subplots(1,2)
																	ax[0].imshow(Pia_gather[iii], vmin=0, vmax=1)
																	ax[0].set_title('Pia')
																	ax[0].set_xlabel('model')
																	ax[0].set_ylabel('cell $y_i$')
																	ax[0].set_aspect('auto')
																	ax[1].imshow(PSTHz_gather[iii])
																	ax[1].set_title('PSTH')
																	ax[0].set_xlabel('model')
																	ax[0].set_ylabel('time bins')
																	ax[1].set_aspect('auto')
																	plt.suptitle( str('M#'+str(i_mdl)+', z#'+str(iii)+', $<cs>_{X}$ = '+str(mean_CS_accModels[i_mdl,iii].round(2))+', $<cs>_{T}$ = '+str(mean_CST_accModels[iii].round(2)) ) )
																	plt.show()


															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# (3). Converting SWs & Zinf (which are lists of sets of which cells or CAs fired) in a given ms time bin of
															# 		of a given trial) into a 3D list of [list [of list]]. 1st set of lists indicates trial. 2nd set of lists
															#  		indicates cell or CA. Then within that inner list is the time bins (in ms) when the cell or CA was active 
															# 		in that trial.
															#
															if True:
																print('Converting SWs,Zinf,Yinf into 3D tensor that is Trials x Cells|CAs x Times(ms)Active.')
																t0 = time.time()
																print('raster_Z_inferred_allSWs')
																raster_Z_inferred_allSWs, pSW_forZ_GLM 	= rc.compute_raster_list(SWtimes, Z_inferred_allSWs, pSW_nullGLM, M, minTms, maxTms )
																#_, pSW_forZ_cdl 	= rc.compute_raster_list(SWtimes, Z_inferred_allSWs, cond_inferred_allSWs, M, minTms, maxTms )
																#_, pSW_forZ_jnt 						= rc.compute_raster_list(SWtimes, Z_inferred_allSWs, pj_inferred_allSWs, M, minTms, maxTms )
																#
																print('raster_allSWs')
																raster_allSWs, pSW_allSWs_GLM 			= rc.compute_raster_list(SWtimes, SWs, pSW_nullGLM, N, minTms, maxTms )
																t1 = time.time()
																#
																# if False:
																# 	print('raster_Y_inferred_allSWs')
																# 	raster_Y_inferred_allSWs 	= rc.compute_raster_list(SWtimes, Y_inferred_allSWs, pSW_nullGLM, N, minTms, maxTms )
																#
																print('Done Converting SWs,Zinf,Yinf into 3D tensor that is CellsCAs x Trials x TimesActive: time = ',t1-t0)










															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# 
															# Thinking about filtering out of further analysis the CA's that only have psthZ responses 
															# before 333 ms. ie at stim onset. Are they preferentially larger CAs?
															#
															if False:
																psthZ_sum = psthZ_accum.sum(axis=1)
																psthZ_normed = psthZ_accum/psthZ_sum[:,None]
																cutOnset = np.max( np.where( binsPSTH < 333 )[0] )
																psth_normed_cumsum = np.cumsum(psthZ_normed,axis=1)
																stimOnCAs = np.where(psth_normed_cumsum[:,cutOnset]>0.9)[0]
																#
																print('Number of onset CAs: ', len(stimOnCAs))
																print(stimOnCAs)
																#
																if False:
																	plt.plot(binsPSTH[:-1], np.cumsum(psthZ_normed,axis=1).T)
																	plt.plot([333,333],[0,1],'k--')
																	plt.grid()
																	plt.show()

















															# # # # # # # #	# # # # # # # #	# # # # # # # #	# # # # # # # #		
															#
															# Sort Pia columns and PSTHs to get spatial and temporal extent / coverage of cell assemblies
															#
															if flg_compute_CA_stats:	
																print('compute_CA_stats')
																#
																cellsInStats = np.zeros( (M,5) )
																cellsOutStats = np.zeros( (M,5) )
																#
																if N > Nsplit:
																	cellInType1Stats = np.zeros( (M,5) )
																	cellOutType1Stats = np.zeros( (M,5) )
																	cellAllType1Stats = np.zeros( (M,5) )
																	#
																	cellInType2Stats = np.zeros( (M,5) )
																	cellOutType2Stats = np.zeros( (M,5) )
																	cellAllType2Stats = np.zeros( (M,5) )
																#		

																cellsIn_collect = list()
																for z in range(M):
																	#print(z)
																	srtPia = np.argsort(1-Pia[:,z])
																	mnPia = np.mean(1-Pia[:,z])
																	stdPia = np.std(1-Pia[:,z])
																	#
																	srtDeriv = np.diff( 1-Pia[srtPia,z] )
																	mnDrv = np.mean(srtDeriv)
																	stdDrv = np.std(srtDeriv)
																	#
																	chz1= np.where(1-Pia[srtPia,z][:-1]>mnPia+stdPia)[0]
																	chz2 = np.where(srtDeriv>mnDrv+stdDrv)[0] + 1 # +1 to shift it to grab points after large deriv jump.
																	
																	if chz1.size>0 and chz2.size>0:
																		chzn = np.max( [chz1.min(),chz2.min()] )
																	elif chz1.size==0:
																		chzn = chz2.min()
																	elif chz2.size==0:
																		chzn = chz1.min()
																	else:
																		chzn = N

																	cellsIn = srtPia[chzn:]
																	cellsOut = srtPia[:chzn]
																	#
																	cellsIn_collect.append(cellsIn)
																	#
																	# [number of cells, meanPia, stdPia, maxPia, minPia]
																	if cellsIn.size>0:
																		cellsInStats[z] = [len(cellsIn), np.mean(1-Pia[cellsIn,z]), np.std(1-Pia[cellsIn,z]), np.max(1-Pia[cellsIn,z]), np.min(1-Pia[cellsIn,z]) ]
																	if cellsOut.size>0:
																		cellsOutStats[z] = [len(cellsOut), np.mean(1-Pia[cellsOut,z]), np.std(1-Pia[cellsOut,z]), np.max(1-Pia[cellsOut,z]), np.min(1-Pia[cellsOut,z]) ]
																	
																	# Maybe index into to get RF centers in x0 and y0
																	# STRF_gauss # N x 6 [Amplitude, x0, sigmax, y0, sigmay, angle(in rad)]


																	# If there are two cell types, compute stats for each cell typ seperately too.
																	if N > Nsplit:
																		#
																		cellsInType1 = cellsIn[cellsIn<Nsplit]
																		cellsInType2 = cellsIn[cellsIn>=Nsplit]
																		#
																		cellsOutType1 = cellsOut[cellsOut<Nsplit]
																		cellsOutType2 = cellsOut[cellsOut>=Nsplit]
																		#
																		cellsAllType1 = srtPia[srtPia<Nsplit]
																		cellsAllType2 = srtPia[srtPia>=Nsplit]
																		#
																		# Stats include: [number of cells, meanPia, stdPia, maxPia, minPia]
																		if cellsInType1.size>0:
																			cellInType1Stats[z] = [len(cellsInType1), np.mean(1-Pia[cellsInType1,z]), np.std(1-Pia[cellsInType1,z]), np.max(1-Pia[cellsInType1,z]), np.min(1-Pia[cellsInType1,z]) ]
																		if cellsOutType1.size>0:
																			cellOutType1Stats[z] = [len(cellsOutType1), np.mean(1-Pia[cellsOutType1,z]), np.std(1-Pia[cellsOutType1,z]), np.max(1-Pia[cellsOutType1,z]), np.min(1-Pia[cellsOutType1,z]) ]
																		#
																		if cellsInType2.size>0:
																			cellInType2Stats[z] = [len(cellsInType2), np.mean(1-Pia[cellsInType2,z]), np.std(1-Pia[cellsInType2,z]), np.max(1-Pia[cellsInType2,z]), np.min(1-Pia[cellsInType2,z]) ]
																		if cellsOutType2.size>0:
																			cellOutType2Stats[z] = [len(cellsOutType2), np.mean(1-Pia[cellsOutType2,z]), np.std(1-Pia[cellsOutType2,z]), np.max(1-Pia[cellsOutType2,z]), np.min(1-Pia[cellsOutType2,z]) ]
																		#
																		if cellsAllType1.size>0:
																			cellAllType1Stats[z] = [len(cellsAllType1), np.mean(1-Pia[cellsAllType1,z]), np.std(1-Pia[cellsAllType1,z]), np.max(1-Pia[cellsAllType1,z]), np.min(1-Pia[cellsAllType1,z]) ]
																		if cellsAllType2.size>0:
																			cellAllType2Stats[z] = [len(cellsAllType2), np.mean(1-Pia[cellsAllType2,z]), np.std(1-Pia[cellsAllType2,z]), np.max(1-Pia[cellsAllType2,z]), np.min(1-Pia[cellsAllType2,z]) ]	





																	# # # # # # # #	# # # # # # # #	# # # # # # # #	# # # # # # # #		
																	#
																	# As part of the development process, sort each column in Pia, compute derivatives, 
																	# set mean and std thresholds and display entries that pass thresholds
																	if False:

																		print('Pia TH', chz1)
																		print('Deriv TH',chz2)
																		print('max(mins)',chzn)
																		print('#cells', N-chzn)

																		f,ax = plt.subplots(2,1)
																		
																		ax[0].plot( 1-Pia[srtPia,z], color='blue', marker='o',linewidth=2, label='sorted' )
																		ax[0].plot(srtDeriv, color='green', linewidth=2, marker='o',label='$\Delta$ sorted'   )
																		ax[0].plot( np.arange(chzn,N), 1-Pia[cellsIn,z], marker='x',color='red', linewidth=2, label='chosen' )
																		#
																		ax[0].plot( [0,N], [ mnPia, mnPia], 'b--' )
																		ax[0].plot( [0,N], [ mnPia+stdPia, mnPia+stdPia], 'b:' )
																		#ax[0].plot( [0,N], [ mnPia+2*stdPia, mnPia+2*stdPia], 'k--' )
																		ax[0].plot( [0,N], [ mnDrv, mnDrv], 'g--' )
																		ax[0].plot( [0,N], [ mnDrv+stdDrv, mnDrv+stdDrv], 'g:' )
																		#ax[0].plot( [0,N], [ mnDrv+2*stdDrv, mnDrv+2*stdDrv], 'g--' )
																		#
																		ax[0].legend()
																		ax[0].grid()
																		ax[0].set_title(str('z#'+str(z)))
																		#
																		ax[1].plot( 1 - Pia[:,z], color='blue', linewidth=2, label='unsrt' )
																		ax[1].scatter( cellsIn, 1-Pia[cellsIn,z], linewidth=5, color='red', marker='x', label='chosen' )
																		ax[1].grid()
																		ax[1].legend()
																		#
																		plt.show()







																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																#
																# How to sort CAs for plots later on.
																srtAUC 		= np.argsort(cellsInStats[:,0]) #np.argsort(auc)
																srtCAbySize = np.argsort(cellsInStats[:,0])#[::-1]
																# NOTE THESE ARE THE SAME. SORTING ALL BY CA SIZE.




																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																#
																# 								Metrics to use for plots.
																#								-------------------------
																#
																#
																# (1). Define Spatial Crispness metric. How deterministic or smeared out Pia is. C_X =  d'= mu_in - mu_out / sqrt( sig_in^2 + sig_out^2 )
																CrispX = cellsInStats[:,1] - cellsOutStats[:,1] / np.sqrt( 0.5*(cellsInStats[:,1]**2 + cellsOutStats[:,1]**2) )		
																#
																# (2). Define Heterogeneity metric for how cells in CA are shared between cell types.
																if N > Nsplit:
																	lct1 = cellInType1Stats[:,0] 
																	lct2 = cellInType2Stats[:,0]
																	Hetero = np.min([lct1,lct2],axis=0)  / np.mean([lct1,lct2],axis=0) 	
																else:
																	Hetero = np.zeros(M) 	# Define Heter=0 for only 1 celltype.
																#
																# (3). Define Robustness across models as combination of temporal and spatial cosSims.
																Robust = np.sqrt( mean_CS_accModels[i_mdl]*mean_CST_accModels ) 
																#
																# (4). CrispT = # Define temporal crispness as ... ?
																#
																# #
																#
																# 					Things measured directly which are useful as metrics.
																#					----------------------------------------------------
																#
																# (1). CAsize: 				cellsInStats[:,0]
																# (2). #timesInferred:		psthZ_sum
																# (3). p(y)_{null}:			mnn



															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
															# #
															# # Look at distributions of p(y) under GLM model for spikewords observed when each Za is active 
															# #
															# if flg_plot_pOfy_underNull:
															# 	print('DOES NOTHIGN ACTUALYY: plot_pOfy_underNull')




															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
															#
															# (2). Load in GLM simulation results of RGCs responding to natural movie stimuli
															#
															if flg_compare_to_GLM_null:
																print('compare_to_GLM_null')

																GLM = io.loadmat( str(dirScratch + 'data/GField_data/' + GField_GLMsimData_File) ) 
																# 	In this file, according to Kiersten Ruda,
																# 	The first 55 cells are the Off Brisk Transient ganglion cells, 
																# 	the next 43 are the Off Brisk Sustained cells, 
																# 	and the last 39 are the On Brisk Transient cells.
																#
																#	It starts at the beginning of the stimulus, You may want to skip the first 20 movie frames 
																#	(or 400 finely sampled frames) to account for the history dependence of the receptive fields. 
																#	The units of the finely sampled firing rate are spikes/s. 
																#
																print('GLM.keys()', GLM.keys())
																#
																if whichSim == 'GLM':
																	instFR = GLM['cat_indpt_inst_fr'][0] # Instantaneous firing rate from GLM simulation in ms bins of all 137 offBT, onBT and offBS cells.
																	nBins = instFR[0].shape[1]
																elif whichSim == 'LNP':
																	instFR = GLM['cat_LNPindpt_inst_fr'] # Instantaneous firing rate from LNP simulation in ms bins of all 137 offBT, onBT and offBS cells.
																	nBins = instFR.shape[1]

																del GLM		

																tBins_FR = 5 / nBins # bin size for finely binned spike rate vectors (units = sec/bin). 5 seconds for full stim.

																#
																# Convert cell id in spike-word to index in instFR array knowing that the order
																# of cells is: 137 cells = {55 offBT, 43 offBS, 39 onBT}.
																if cellSubTypes == '[offBriskTransient]':
																	N = 55
																	FR = instFR[:N] # offBT
																	cellSet = set(range(N))
																elif cellSubTypes == '[offBriskTransient,offBriskSustained]':	
																	N = 55+43
																	FR = instFR[:N] # offBT and offBS
																	cellSet = set(range(N))
																elif cellSubTypes == '[offBriskTransient,onBriskTransient]':
																	N = 55+39
																	if whichSim == 'GLM':
																		xx = instFR[:55] # offBT
																		yy = instFR[-39:] # onBT
																	elif whichSim == 'LNP':
																		xx = instFR[:55].T # offBT
																		yy = instFR[-39:].T # onBT
																	FR = np.hstack( (xx,yy) ).T
																	del xx, yy
																	cellSet = set(range(N))
																else:
																	print('Dont understand cell type combination ',cellSubTypes)
																	instFR = None
																	#
																del instFR	






																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																#
																# Convert spike rates in finely sampled bins (~0.83ms) to probability of firing
																# at least one spike in a larger time bin (~5ms).
																#
																# FR is Cell x Trial x TimeBin
																# OR average across trial to make it just Cell x TimeBin?
																#
																# # SPIKE RATE VARIES FROM TRIAL TO TRIAL. WHY?
																# # THIS IS BECAUSE KIERSTEN RUNS THE INDEPENDENT GLM 
																# # WITH SPIKE HISTORY. FOR NOW JUST AVERAGE ACROSS TRIAL. 
																#
																# # KIERSTEN IS RUNNING THE LNP MODEL WITHOUT SPIKE HISTORY.
																#
																#
																# Account for 5ms binning for SWs. Determine how many finely sampled GLM bins to look forward and back.
																#
																secBins = msBins/1000 	# convert ms to sec for spike-word binning in spike train.
																b = secBins/tBins_FR	# number of fine-time GLM bins in one of our larger SW bins.
																#
																bBck = np.int( np.floor( (b-1)/2 ) ) 
																bFwd = np.int( np.ceil( (b-1)/2 ) )
																#
																# Set time bounds to only look at spike words that occur between these times.
																tiFin = 5000 # ms.
																tiBeg = int( np.round(400*tBins_FR*1000) ) 	 # ms. # Kiersten suggests this because of GLM history dependence. int( np.round(400*tBins_FR*1000) )
																#
																#

																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																#
																# Load innpz file with saved variables or else regenerate them and save them.
																#		
																GLMpy_compare_fname = str(GLM_pofy_dir+'compareNull'+whichSim+'timing_'+fname+'.npz')
																													
																try:
																	pyGLMcs = np.load( GLMpy_compare_fname )
																	#
																	GLM_cosSim_PSTH_bins	= pyGLMcs['GLM_cosSim_PSTH_bins']
																	cs_ZwGLMpy				= pyGLMcs['cs_ZwGLMpy']
																	swsObsWz_Hist 			= pyGLMcs['swsObsWz_Hist']
																	CAsCoactHist 			= pyGLMcs['CAsCoactHist']
																	allZ_times 				= pyGLMcs['allZ_times']
																	pYavg_rs 				= pyGLMcs['pYavg_rs']
																	D_KL 					= pyGLMcs['D_KL']
																	swsObsWz_LenHist 		= pyGLMcs['swsObsWz_LenHist'] 
																	swsObsAll_LenHist 		= pyGLMcs['swsObsAll_LenHist'] 
																	del pyGLMcs

																except:

																	print('File,', GLMpy_compare_fname, ' not there. Make it.')
																	print('Its gonna take a minute, but you only gotta do it once.')


																	# Convert firing rate into probability of fewer than 1 spike within 0.83ms interval, i.e. p(yi=0)
																	if whichSim == 'GLM':
																		xx = np.zeros( (N,nBins) )
																		for i in range(N):
																			xx[i] = FR[i].mean(axis=0)
																		pyiEq0 = np.exp( -xx*tBins_FR ) # is N x tBins_FR
																		del xx
																	elif whichSim == 'LNP':
																		pyiEq0 = np.exp( -FR*tBins_FR ) 
																	else:
																		print('Dont understand whichSim = ',whichSim)
																


																	if False:
																		# Plot sorta average firing rate of all cells at each time point according to the GLM null.
																		cumsums = np.zeros(6000)
																		for i in range(N):
																			cumsums += FR[i].sum(axis=0)
																		plt.plot(tBins_FR*np.arange(6000), cumsums)
																		plt.show()

																	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																	#
																	# Compute p(y) under null model for a particular observed spike-word
																	# for ALL TIMES IN THE STIMULUS. This is to see if there is a systematic offset
																	# between when a synchronous firing pattern is most likely under the GLM and
																	# when we observe it using our model.
																	#
																	pyiEq0_Big =np.zeros( (pyiEq0.shape[0], pyiEq0.shape[1], bFwd+bBck) )
																	#
																	shifts = np.arange( -bBck,bFwd )
																	for h,s in enumerate(shifts):
																		#print(h,s)
																		pyiEq0_Big[:,:,h] = np.roll(pyiEq0,s,axis=1) # Adds a 3rd dimension to array and puts a time shifted version of pyiEq0.

														

																	# Preallocate memory for variables to save in npz file.
																	GLM_cosSim_PSTH_bins = [1, 10, 50, 100] # bin size for p(y) from GLM and psthZ for computing their cosine similarity.
																	#
																	swsObsWz_Hist = np.zeros( (len(CAS_to_show_stim_at_high_PSTH),N) )
																	CAsCoactHist = np.zeros( (len(CAS_to_show_stim_at_high_PSTH),M) ) # TO DO. include this.
																	allZ_times = np.zeros( (len(CAS_to_show_stim_at_high_PSTH),5000) )
																	pYavg_rs = np.zeros( (len(CAS_to_show_stim_at_high_PSTH),5000) )
																	D_KL = np.zeros( (len(CAS_to_show_stim_at_high_PSTH),6000) )
																	cs_ZwGLMpy = np.zeros( (len(CAS_to_show_stim_at_high_PSTH), len(GLM_cosSim_PSTH_bins) ) )
																	swsObsWz_LenHist = np.zeros( (len(CAS_to_show_stim_at_high_PSTH),N, 5000) )
																	


																	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																	#
																	# Histogram of observed spike-word lengths across trials at each ms instant?
																	#
																	swsObsAll_LenHist = np.zeros( (N,5000) )	# capture SWs length of every spike word and its time bin.
																	#
																	for tr in range(numTrials): # Loop through trials
																		yCard = [ len(sw) for sw in SWs[tr] ]
																		xxx,_,_ = np.histogram2d(SWtimes[tr],yCard,bins=[np.arange(5000+1),np.arange(N+1)])
																		swsObsAll_LenHist += xxx.T


																	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																	#
																	# Loop through all trials and find when CA is active. Then get p(y) at all times for that spikeword.
																	#
																	for zi,whichZ in enumerate(CAS_to_show_stim_at_high_PSTH): # loop over a vector of CAs of interest.

																		print('z#',whichZ,'CA members (TH=0.3):',np.where(1-Pia[:,whichZ]>0.3))
																		print(cellsIn_collect[whichZ])

																		# swsObsWz_LenHist = np.zeros( (N,5000) ) 	# capture SWs length of spike words observed w/ z and its time bin.
																		# swsObsAll_LenHist = np.zeros( (N,5000) )	# capture SWs length of every spike word and its time bin.
																		pY_avg = np.zeros(6000)						# capture p(y) under GLM at every moment in time for every SW observed with z active and average them.
																		
																		for tr in range(numTrials): # Loop through trials
																			# print('z#',whichZ, ' in trial ',tr)
																			#
																			if not len(Z_inferred_allSWs[tr]) == len(SWtimes[tr]):
																				print('			!! PROBLEM: length Zinf not eq length SWtimes. Skip trial',tr)
																				continue

																			# Find index and time points and spike words where za is inferred in this trial.
																			indZ = [ ind for ind,ti in enumerate(SWtimes[tr]) \
																					if whichZ in Z_inferred_allSWs[tr][ind]] 		# index
																			whenZ = [ SWtimes[tr][ii] for ii in indZ ]				# stim time
																			SWatZ = [ SWs[tr][ii] for ii in indZ ]					# observed spike word


																			for t in range(len(indZ)): 	# loop thru points when za is active.
																				ti = whenZ[t]			# time of a spike word.
																				sw = SWs[tr][indZ[t]] 		# spike word itself.

																				if (ti<(tiFin-bFwd)) and (ti>(tiBeg+bBck)): # 5 seconds. GLM doesnt have results past this.
																				#
																					# print(tr, t, ti, list(sw) )
																					#
																					inn = list( sw )									# cells in the spike-word.
																					out = list( cellSet.difference(sw) )				# cells not in the spike-word.
																					#
																					pOff = pyiEq0_Big[out].prod(axis=2)		# prob of each cell that is off to be off
																					pOn = 1 - pyiEq0_Big[inn].prod(axis=2)	# prob of all cell that is on to be on
																					pY = np.vstack([pOn,pOff]).prod(axis=0)	# prob of whole spike word.
																					
																					pY_avg += pY # average p(y) at all time points for SWs observed when z active

																					allZ_times[zi][ti] += 1 # histogram up times points when z activity.

																					swsObsWz_Hist[zi][inn] += 1 # histogram cells active during z activity. Compare to Pia probabilities

																					CAsCoactHist[zi, list(Z_inferred_allSWs[tr][indZ[t]]) ] += 1 # histogram up other CAs that are coactive with current one.

																					swLen = len(inn) - 1 # -1 because we are using this as an index into swsObsLenHist
																					swsObsWz_LenHist[zi,swLen,ti] += 1 # 2D histogram of lenSW vs stimTime.


																		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																		#
																		# Compute KL divergence between p(y) under GLM and p(y|z_a) for our model.
																		#
																		pyiEq0_GLM = pyiEq0_Big.prod(axis=2)
																		pyiEq0_PGM = Pia[:,whichZ]*Pi**(1-1/M)
																		D_KL[zi] = ( pyiEq0_GLM*np.log(pyiEq0_GLM/pyiEq0_PGM[:,None]) + \
																					(1-pyiEq0_GLM)*np.log((1-pyiEq0_GLM)/(1-pyiEq0_PGM)[:,None]) ).sum(axis=0)


																		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																		#
																		# Compute cosine similarity between pY_avg and allZ_times at different binnings.
																		#
																		pYavg_rs[zi] = sig.resample(pY_avg,5000) # change it from 6000 time points to 5000.
																		#
																		for bi,bn in enumerate(GLM_cosSim_PSTH_bins):
																			#
																			pY_bin = np.convolve(pYavg_rs[zi],np.ones(bn),'same').reshape(-1,1).T
																			allZ_bin = np.convolve(allZ_times[zi],np.ones(bn),'same').reshape(-1,1).T
																			#
																			cs_ZwGLMpy[zi][bi] = cosine_similarity( pY_bin, allZ_bin )

																	# TO SAVE NPZ DATA FILE:
																	np.savez( GLMpy_compare_fname, GLM_cosSim_PSTH_bins=GLM_cosSim_PSTH_bins, swsObsWz_Hist=swsObsWz_Hist, swsObsWz_LenHist=swsObsWz_LenHist, 
																		swsObsAll_LenHist=swsObsAll_LenHist, CAsCoactHist=CAsCoactHist, allZ_times=allZ_times, pYavg_rs=pYavg_rs, D_KL=D_KL, cs_ZwGLMpy=cs_ZwGLMpy )
														
																	del pyiEq0_GLM, pyiEq0_PGM, pyiEq0, pyiEq0_Big, pOn, pOff, pY_avg


																cosdif_ZwGLMpy = 1-cs_ZwGLMpy 			# @ 1ms,  Difference between z activations and p(y) predictions temporally.
																cosdif_binGLM = cosdif_ZwGLMpy[:,-1]	# @ 100ms ( cosdif_ZwGLMpy.max(axis=1) - cosdif_ZwGLMpy.min(axis=1) ) /cosdif_ZwGLMpy.max(axis=1) 	# How diff between z and p(y) change with binning.






															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
															#
															# Making a plot with CA activations and p(y) under null predictions.
															#
															#
															if flg_doPlotGLMcomp:

																print('Making plots of CA activations and p(y) under null predictions.')

																#
																for zi,whichZ in enumerate(CAS_to_show_stim_at_high_PSTH): # loop over a vector of CAs of interest.

																	print('z',whichZ,'CA members (TH=0.3):',np.where(1-Pia[:,whichZ]>0.3))
																	print(cellsIn_collect[whichZ])


																	plt.rc('font', weight='bold', size=12)
																	f = plt.figure(figsize=(20,10))
																	#
																	ax0 = plt.subplot2grid( (4,1),(0,0) )
																	ax0.plot(pYavg_rs[zi], alpha=0.5, color='green')
																	ax0.set_ylabel( str('p(y) obs GLM ) '+whichSim), color='green' )
																	ax0.tick_params(axis='y', labelcolor='green')
																	ax0.grid(True)
																	#
																	ax0c = ax0.twinx()  # instantiate a second axes that shares the same x-axis
																	ax0c.plot(tBins_FR*np.arange(6000)*1000, D_KL[zi], alpha=0.5, color='blue')
																	ax0c.set_title(str(r'z'+str(whichZ)+'= y'+str(cellsIn_collect[whichZ])+' $\Delta$Py='+str(cosdif_ZwGLMpy[zi].round(3)) ) ) #+' in trial'+str(tr) ))
																	#ax0c.grid(True)
																	ax0c.set_ylim( 0, np.nanmax(D_KL[zi]) )
																	ax0c.set_xlim(0,5000)
																	ax0c.set_ylabel( str(r'$D_{KL}('+whichSim+',z_a=1)$'), color='blue' ) # log p(p(y)|za)
																	ax0c.tick_params(axis='y', labelcolor='blue')
																	#ax0c.spines["right"].set_position(("axes", 1.1))
																	# ax0c.set_frame_on(True)
																	# ax0c.patch.set_visible(False)
																	# for sp in ax0c.spines.values():
																	# 	sp.set_visible(False)
																	#ax0c.spines["right"].set_visible(True)

																	#
																	ax0b = ax0.twinx()  # instantiate a second axes that shares the same x-axis
																	ax0b.set_ylabel('psth z', color='red')  # we already handled the x-label with ax1
																	ax0b.plot(allZ_times[zi],color='red', alpha=0.5)
																	ax0b.tick_params(axis='y', labelcolor='red')
																	ax0b.text(5000,allZ_times[zi].max()-2,str('tot z counts = '+str( int(allZ_times[zi].sum()))  ),ha='right',va='top')
																	#ax0b.grid(True)

																	# # # # # 
																	#
																	ymax = 0
																	for y in range(N):
																		if np.any(swsObsWz_LenHist[zi,y]):
																			ymax = y
																	#
																	ax1 = plt.subplot2grid( (4,1),(1,0) )
																	ax1.imshow( np.log(swsObsWz_LenHist[zi]+1),cmap='Greys')
																	#ax1.set_yticks( np.linspace(0,N-1,5) )
																	#ax1.set_yticklabels( np.floor(np.linspace(0,N-1,5)+1).astype(int) )
																	ax1.set_ylabel(r'$\vert \vec{y} \vert$ when z=1')
																	ax1.set_xlim(0,5000)
																	ax1.set_aspect('auto')
																	ax1.set_ylim(0,ymax)
																	#ax1.invert_yaxis()
																	ax1.grid()
																	ax1.text(5000,ymax-2,str('max counts = '+str( int(swsObsWz_LenHist[zi].max())) +  \
																		'\n tot counts = '+str( int(swsObsWz_LenHist[zi].sum()))  ),ha='right',va='top')

																
																	# # # # # 
																	#	
																	ymax = 0
																	for y in range(N):
																		if np.any(swsObsAll_LenHist[y]):
																			ymax = y															
																	ax2 = plt.subplot2grid( (4,1),(2,0) )
																	ax2.imshow( np.log(swsObsAll_LenHist+1), cmap='Greys' )
																	ax2.text(5000,ymax-2,str('max counts = '+str( int(swsObsAll_LenHist.max()))  +  \
																		'\n tot counts = '+str( int(swsObsAll_LenHist.sum()))  ),ha='right',va='top')
																	ax2.set_aspect('auto')
																	ax2.set_ylabel(r'$\vert \vec{y} \vert$ all y')
																	ax2.set_ylim(0,ymax)
																	ax2.set_xlim(0,5000)	
																	#ax2.invert_yaxis()
																	ax2.grid()

																	# TO DO 2: Plot swsObsAll_LenHist - swsObsWz_LenHist to show spike word lengths NOT " (partially)explained" by CAs.


																	# # # # # 
																	#
																	ax3 = plt.subplot2grid( (4,1),(3,0) )
																	srt = np.argsort( Pia[:,whichZ] ) 				# sorting by Pia value #
																	ax3.plot( 1-Pia[srt,whichZ], color='blue' )
																	ax3.plot( [len(cellsIn_collect[whichZ]),len(cellsIn_collect[whichZ])],[0,1], 'k--' )
																	ax3.set_ylim(0,1)
																	ax3.set_ylabel( str('Pia' ), color='blue' )
																	ax3.tick_params(axis='y', labelcolor='blue')
																	ax3.set_xlabel( str('Cell (sorted by Pia)' ) )
																	ax3.grid()
																	# TO DO 1: Mark which cells are "in" by maximizing crispness measure.


																	#
																	ax3b = ax3.twinx()  # instantiate a second axes that shares the same x-axis
																	ax3b.set_ylabel('times obs in z', color='red')  # we already handled the x-label with ax1
																	ax3b.tick_params(axis='y', labelcolor='red')
																	ax3b.plot( swsObsWz_Hist[zi,srt], color='red' )


																	
																	#
																	#
																	new_save_dir = str( output_figs_dir+'../CA_vs_null_'+whichSim+'_timing/')
																	if not os.path.exists( str(new_save_dir) ):
																		os.makedirs( str(new_save_dir) )
																	plt.savefig( str(new_save_dir+fname+'_z'+str(whichZ)+'.png' ) ) 
																	plt.close()	
																
																


																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																#
																# Plot Coactivity of CAs within a model. 1 plot for entire model.
																#
																if flg_plt_model_CAs_coactivity:

																	CAact = np.copy(CAsCoactHist.diagonal())
																	coacts = np.tril(CAsCoactHist,-1)
																	num_coacts = 5
																	top_coacts = np.zeros( (num_coacts,3) ).astype(int)
																	for jj in range(num_coacts):
																		x = np.where(coacts==coacts.max())
																		z1 = int(x[0][0])
																		z2 = int(x[1][0])
																		top_coacts[jj] = [ z1, z2, int(coacts[z1,z2]) ] 
																		coacts[z1,z2] = 0

																	
																	f, ax = plt.subplots(2,1)
																	#	
																	s = ax[0].imshow(np.tril(CAsCoactHist,-1),cmap='Greys')
																	#
																	ax[0].text(M, 0, str('z1 : z2 : #co-'), ha='right', va='top', fontsize=8 )
																	for jj in range(num_coacts):
																		ax[0].text(M, 7*(jj+1), str( str(top_coacts[jj][0]) + ' : ' + str(top_coacts[jj][1]) + ' : ' + str(top_coacts[jj][2]) ), ha='right', va='top', fontsize=8 )
																		ax[0].scatter(top_coacts[jj][1],top_coacts[jj][0],100,marker='o',facecolors='none', edgecolors='r')
																	#
																	ax[0].set_aspect('auto')
																	ax[0].grid()
																	ax[0].set_ylabel('CA id')
																	ax[0].set_xlim(0,M)
																	ax[0].set_ylim(0,M)
																	ax[0].plot([0,M],[0,M],'k--')
																	ax[0].invert_yaxis()
																	cbar_ax = f.add_axes([0.91, 0.53, 0.02, 0.35])
																	cb1=f.colorbar(s, cax=cbar_ax)
																	cb1.ax.set_title('#co-active',fontsize=8)
																	#b1.ax.set_yticklabels([str( str(vmin)+' : active'), str(vmid), str( str(vmax)+' : silent') ],fontsize=10)
																	#
																	ax[1].scatter(np.arange(M),CAact,color='blue',label='#active')
																	ax[1].scatter(np.arange(M),CAsCoactHist.sum(axis=0)-CAact,color='red',label='#co-active')
																	ax[1].legend(fontsize=8)
																	ax[1].grid()
																	ax[1].set_xlabel('CA id')
																	ax[1].set_xlim(0,M)

																	# SAVE PLOT.													# Info for saving the plot.
																	CA_coact_figs_save_dir	= str( dirScratch + 'figs/PGM_analysis/PSTH_and_Rasters/' + init_dir + 'CA_model_coactivity/')
																	#
																	if not os.path.exists( str(CA_coact_figs_save_dir) ):
																		os.makedirs( str(CA_coact_figs_save_dir) )

																	plt.savefig( str(CA_coact_figs_save_dir + fname + '.' + figSaveFileType ) ) 
																	#plt.show()
																	plt.close() 






																# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																#
																# Plot CAs which are very spatially similar. That one could stand in for the other.
																#
																# GOAL: FOR EACH CA, FIND OTHERS THAT ARE MOST SPATIALLY SIMILAR. ARE THEY ALSO TEMPORALLY SIMILAR?
																# IF NOT, THEY ARE REPLACING ONE ANOTHER IN THE REPRESENTATION. THAT COMPLICATES THINGS A LITTLE.
																#
																if flg_plt_model_CAs_replacements:

																	XcosSimMat = cosine_similarity(1-Pia.T,1-Pia.T)	# spatial cosine similairty - within one model.

																	np.fill_diagonal(XcosSimMat,0)
																	#np.fill_diagonal(TcosSimMat,0)

																	for m in range(M):
																		xxx = np.argsort(XcosSimMat[m])[::-1]
																		plt.plot(XcosSimMat[m][xxx],alpha=0.3)
																	plt.grid()
																	plt.ylim(0,1)
																	plt.show()	


																	# f, ax = plt.subplots(1,2)
																	# ax[0].imshow(XcosSimMat,vmin=0,vmax=1,cmap='Greys')
																	# ax[1].imshow(TcosSimMat,vmin=0,vmax=1,cmap='Greys')
																	# plt.show()


																	# # SAVE PLOT.													# Info for saving the plot.
																	# CA_coact_figs_save_dir	= str( dirScratch + 'figs/PGM_analysis/PSTH_and_Rasters/' + init_dir + 'CA_coactivity/')
																	# #
																	# if not os.path.exists( str(CA_coact_figs_save_dir) ):
																	# 	os.makedirs( str(CA_coact_figs_save_dir) )

																	# plt.savefig( str(CA_coact_figs_save_dir + fname + '.' + figSaveFileType ) ) 
																	# #plt.show()
																	# plt.close() 
																



															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
															#
															# Plot some metric comparisons.
															if flg_compute_CA_stats and flg_plot_CA_metricsCorrs:
																print('compute_CA_metricsCorrs')

																plt.figure( figsize=(20,10) ) # size units in inches
																plt.rc('font', weight='bold', size=8)
																#
																# (1). Scatter Heterogeneity vs. Robustness. Points colored by CA size.
																ax0 = plt.subplot2grid( (3,4), (0,0) )
																ax0.scatter( Hetero[srtCAbySize], Robust[srtCAbySize], c=np.arange(M), s=25, alpha=0.3, cmap='jet')
																ax0.set_xlabel('$H$')
																ax0.set_ylabel('$R$')
																ax0.set_xlim(0,1)
																ax0.set_ylim(0,1)
																ax0.set_aspect('auto')
																ax0.grid()
																#
																# (2). Scatter CrispnessX vs. Robustness. Points colored by CA size.
																ax1 = plt.subplot2grid( (3,4), (1,0) )
																ax1.scatter( CrispX[srtCAbySize], Robust[srtCAbySize], c=np.arange(M), s=25, alpha=0.3, cmap='jet')
																ax1.set_xlabel('$C_X \sim d\'$')
																ax1.set_ylabel('$R$')
																ax1.set_xlim(0,np.max([1,CrispX.max()]))
																ax1.set_ylim(0,1)
																ax1.set_aspect('auto')
																ax1.grid()
																#
																# (2). Scatter Heterogeneity vs. CrispnessX. Points colored by CA size.
																ax2 = plt.subplot2grid( (3,4), (2,0) )
																ax2.scatter( Hetero[srtCAbySize], CrispX[srtCAbySize], c=np.arange(M), s=25, alpha=0.3, cmap='jet')
																ax2.set_xlabel('$H$')
																ax2.set_ylabel('$C_X \sim d\'$')
																ax2.set_xlim(0,1)
																ax2.set_ylim(0,np.max([1,CrispX.max()]))
																ax2.set_aspect('auto')
																ax2.grid()
																#
																# #
																#															
																# (1). Scatter Heterogeneity vs p(y)_null. Points colored by CA size.
																ax3 = plt.subplot2grid( (3,4), (0,1) ) 
																ax3.scatter(Hetero[srtCAbySize], cosdif_ZwGLMpy[srtCAbySize,0], c=np.arange(M), s=25, alpha=0.3, cmap='jet')
																ax3.set_xlabel('$H$')
																ax3.set_ylabel(r'$\Delta p(y)_{null}$')
																ax3.set_xlim(0,1)
																ax3.set_aspect('auto')
																ax3.grid()
																#
																# (2). Scatter Robustness vs p(y)_null. Points colored by CA size.
																ax4 = plt.subplot2grid( (3,4), (1,1) )
																ax4.scatter( Robust[srtCAbySize], cosdif_ZwGLMpy[srtCAbySize,0], c=np.arange(M), s=25, alpha=0.3, cmap='jet')
																ax4.set_xlabel('$R$')
																ax4.set_ylabel(r'$\Delta p(y)_{null}$')
																ax4.set_xlim(0,1)
																ax4.set_aspect('auto')
																ax4.grid()
																#
																# (2). Scatter CrispnessX vs p(y)_null. Points colored by CA size.
																ax5 = plt.subplot2grid( (3,4), (2,1) )
																ax5.scatter( CrispX[srtCAbySize], cosdif_ZwGLMpy[srtCAbySize,0], c=np.arange(M), s=25, alpha=0.3, cmap='jet')
																ax5.set_xlabel('$C_X \sim d\'$')
																ax5.set_ylabel(r'$\Delta p(y)_{null}$')
																ax5.set_xlim(0,1)
																ax5.set_aspect('auto')
																ax5.grid()
																#

																# Very Nice plot: Hetero vs. CA size.
																ax6 = plt.subplot2grid( (3,4), (0,2) )
																ax6.scatter( Hetero[srtCAbySize], cellsInStats[srtCAbySize,0], c=np.arange(M), s=25, alpha=0.3, cmap='jet')
																ax6.set_xlabel('$H$')
																ax6.set_ylabel('CA size')
																ax6.set_xlim(0,1)
																ax6.set_aspect('auto')
																ax6.grid()
																#
																if N > Nsplit:
																	ax7 = plt.subplot2grid( (3,4), (1,2) )
																	ax7.scatter( np.min( [cellInType1Stats[srtCAbySize,0], cellInType2Stats[srtCAbySize,0]], axis=0), cellsInStats[srtCAbySize,0], c=np.arange(M), s=25, alpha=0.3, cmap='jet')
																	ax7.set_xlabel('size of smaller CA')
																	ax7.set_ylabel('CA size')
																	ax7.set_aspect('auto')
																	ax7.grid()
																#
																# Histogram of Hetero for all CA's. Good way to compare across different models and cell-types esp.
																ax8 = plt.subplot2grid( (3,4), (2,2) )
																ax8.hist(Hetero)
																ax8.set_title( str('Hist of Hetero. #nonZero = '+str( np.sum(Hetero>0) )+'/'+str(M) ) )
																#
																# Scatter CA-size vs. p(y)_null.
																ax9 = plt.subplot2grid( (3,4), (0,3) )
																ax9.scatter( cellsInStats[srtCAbySize,0], cosdif_ZwGLMpy[srtCAbySize,0], c=np.arange(M), s=25, alpha=0.3, cmap='jet')
																ax9.set_ylabel('$p(y)$')
																ax9.set_xlabel('CAsz')
																ax9.set_title('$p(y) \propto CA size ^{-1}$')
																ax9.set_aspect('auto')
																ax9.grid()
																#
																# Scatter CA-size vs. Robustness.
																ax10 = plt.subplot2grid( (3,4), (1,3) )
																ax10.scatter( cellsInStats[srtCAbySize,0], Robust[srtCAbySize], c=np.arange(M), s=25, alpha=0.3, cmap='jet')
																ax10.set_xlabel('CAsz')
																ax10.set_ylabel('R')
																ax10.set_title('$R loosely \propto CA size$')
																ax10.set_aspect('auto')
																ax10.grid()
																#
																plt.suptitle( str( fname.replace('_',' ') ) )
																#
																# SAVE PLOT.													# Info for saving the plot.
																CA_metricCorrs_figs_save_dir	= str( dirScratch + 'figs/PGM_analysis/PSTH_and_Rasters/' + init_dir + 'CA_metCorrs/')
																#
																if not os.path.exists( str(CA_metricCorrs_figs_save_dir) ):
																	os.makedirs( str(CA_metricCorrs_figs_save_dir) )

																plt.savefig( str(CA_metricCorrs_figs_save_dir + fname + '.' + figSaveFileType ) ) 
																#plt.show()
																plt.close() 



															# # # # # # # #	# # # # # # # #	# # # # # # # #	# # # # # # # #		
															#
															# Plot statistics of Cells in vs out of CAs. 
															#	Stats are [#cells, meanPia, stdPia, maxPia, minPia]
															# 	We have it for allCellsIn, allCellsOut.
															#	CellsType1In, CellsType1Out, CellsType2In, CellsType2Out if N > Nsplit.
															#
															if flg_plot_CA_stats:
																xt = np.arange( 0, len(binsPSTH), 10 )

																# Add an extra row to the plot if there are 2 cell types to show how they interact.
																if N>Nsplit:
																	verts=3
																else:
																	verts=2

																# Plot Statistics from temporal activations of CAs
																#
																f = plt.figure(figsize=(20,10))
																plt.rc('font', weight='bold', size=10)
																#
																# (1). Plot PSTH of CA activities
																ax0 = plt.subplot2grid( (verts,3), (0,0) )
																ax0.imshow(psthZ_accum[srtAUC]) #, origin='lower')
																ax0.set_ylabel('$z_a$')
																ax0.set_title('PSTH')	
																ax0.set_xlabel('time(sec)')
																ax0.set_yticks(np.arange(M))
																ax0.set_yticklabels(srtAUC,fontsize=8)
																ax0.set_xticks( xt )
																ax0.set_xticklabels( (binsPSTH[xt]/1000).round(1) ) #, fontsize=10 )
																ax0.set_aspect('auto')
																for ytick, color in zip(ax0.get_yticklabels(), colors): # colorcode CAs on yticks by srtAUC for visual clarity.
																	ytick.set_color(color)
																#
																# (2). Plot number of cells determined to be participating in CAs
																ax3 = plt.subplot2grid( (verts,3), (0,1) )
																ax3.plot( cellsInStats[srtCAbySize,0] )
																ax3.scatter( np.arange(M), cellsInStats[srtCAbySize,0], c=np.arange(M), s=50, alpha=0.3, cmap='jet')
																ax3.set_title('CA ID ($z_a$) Sorted by Cell Assembly Size')
																#ax3.set_xlabel('CA ID ($z_a$)')
																ax3.set_ylabel( str('# cells (/'+str(N)+') in CA') )
																ax3.grid()
																#ax3.set_xticks(np.arange(M))
																#ax3.set_xticklabels(srtCAbySize, fontsize=6, rotation=90)	
																
																#
																# (3). Plot cell assemblies
																axb = plt.subplot2grid( (verts,3), (0,2) )
																s = axb.imshow(1-Pia[:,srtAUC], vmin=0, vmax=1) #, origin='lower')
																axb.plot([0, M],[Nsplit, Nsplit],'w--')
																axb.set_ylabel('$y_i$')
																axb.set_title('$P_{ia}$')	
																axb.set_xlabel('$z_a$')
																axb.set_xticks(np.arange(M))
																axb.set_xticklabels(srtAUC,fontsize=8)
																axb.set_xlim(0,M-1)
																axb.set_ylim(0,N-1)
																axb.set_aspect('auto')
																for xtick, color in zip(axb.get_xticklabels(), colors): # colorcode CAs on yticks by srtAUC for visual clarity.
																	xtick.set_color(color)
																#
																cbar_ax = f.add_axes([0.91, 0.55, 0.02, 0.30])
																cb1=f.colorbar(s, cax=cbar_ax, ticks=[0, 0.5, 1])# )
																#cb1.ax.set_title('$p(y_i=0|z_a=1)$',fontsize=14)
																#cb1.ax.set_yticklabels([str( str(vmin)+' : active'), str(vmid), str( str(vmax)+' : silent') ],fontsize=10)
																
																
																# 
																# (1). Scatter CrispnessX vs. Robustness. Points colored by CA size.
																ax2 = plt.subplot2grid( (verts,3), (1,1) )
																ax2.scatter( Robust[srtCAbySize], CrispX[srtCAbySize], c=np.arange(M), s=50, alpha=0.3, cmap='jet')
																ax2.errorbar( Robust.mean(), CrispX.mean(),  yerr=CrispX.std(), xerr=Robust.std() )
																ax2.set_ylabel('Crispness $P_{ia}$')
																ax2.set_xlabel('Robustness')
																ax2.set_xlim(0,1) #np.max([1,CrispX.max()]))
																ax2.set_ylim(0,1)
																ax2.set_aspect('equal')
																ax2.grid()

																#
																# (2). Scatter cosSim at 1ms vs. difference that binning at {1,10,50,100}ms makes.
																#
																# cosdif_binGLM = (cosdif_ZwGLMpy).mean(axis=1)/(cosdif_ZwGLMpy).max(axis=1)
																#
																ax1 = plt.subplot2grid( (verts,3), (1,2) )
																ax1.scatter( cosdif_ZwGLMpy[srtCAbySize,0], cosdif_binGLM[srtCAbySize],  c=np.arange(M), s=50, alpha=0.5, cmap='jet' )
																ax1.set_ylabel('binning $\mu$/max')
																ax1.set_xlabel('$\Delta p(y)_{null}$')
																ax1.set_xlim(0,1)
																ax1.set_ylim(0,1)
																ax1.set_aspect('equal')
																ax1.grid()



																# # TO DO: Not plotting number of times active right now. Probably should !!


																if N > Nsplit: 	# two cell types.

																	axd = plt.subplot2grid( (verts,3), (2,0) )
																	axd.scatter( Hetero[srtCAbySize], cellsInStats[srtCAbySize,0], c=np.arange(M), s=50, alpha=0.3, cmap='jet')
																	axd.set_xlabel( str('Cell Type Heterogeneity #nonzero='+str( np.sum(Hetero>0) )+'/'+str(M)) )
																	axd.set_ylabel('CA size')
																	axd.set_xlim(0,1)
																	axd.set_aspect('auto')
																	axd.grid()
																	#
																	ax8 = plt.subplot2grid( (verts,3), (2,2) )
																	ax8.hist(Hetero)
																	ax8.grid()
																	ax8.set_xlabel( str('Hist of Hetero. #nonZero = '+str( np.sum(Hetero>0) )+'/'+str(M) ) )
																	#
																	# (3). Scatter Robustness vs. Heterogeneity
																	axe = plt.subplot2grid( (verts,3), (2,1) )
																	#axe.errorbar( cellOutType2Stats[:,1], cellInType2Stats[:,1], yerr=cellInType2Stats[:,2], xerr=cellOutType2Stats[:,2], alpha=0.2, linestyle='' )	
																	axe.scatter( Robust[srtCAbySize], Hetero[srtCAbySize],  c=np.arange(M), s=50, alpha=0.5, cmap='jet')
																	axe.errorbar( Robust.mean(), Hetero.mean(),  yerr=Hetero.std(), xerr=Robust.std() )
																	axe.set_xlim(0,1)
																	axe.set_ylim(0,1)
																	axe.set_ylabel( str('Heterogeneity') ) 
																	axe.set_xlabel( str('Robustness') )
																	axe.set_aspect('equal')
																	#axe.set_title( str('Cells In CA $P_{ia} \mu & \sigma$') )
																	#axe.plot([0,1], [0,1],'k--')
																	axe.grid()


																#
																plt.suptitle( str( fname.replace('_',' ') ) )
																#
																# SAVE PLOT.													# Info for saving the plot.
																CA_stats_figs_save_dir	= str( dirScratch + 'figs/PGM_analysis/PSTH_and_Rasters/' + init_dir + 'CA_stats/')
																#
																if not os.path.exists( str(CA_stats_figs_save_dir) ):
																	os.makedirs( str(CA_stats_figs_save_dir) )

																plt.savefig( str(CA_stats_figs_save_dir + fname + '.' + figSaveFileType ) ) 
																#plt.show()
																plt.close() 
















															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															#
															# (2). How similar are CA's in their temporal profiles. 
															# 			(across model learned for matching ones. Like cosSim for time.)
															# 			(and across CAs in single model to get at coactivation matrix.)
															if flg_match_CAs_temporal:
																#
																print('Finding temporal similarity of CA activations.')
																print('Q: Does spatial cosSim correlate with temporal cosSim, in same model?, and across models?')
																#
																temporalCSofCAs_dir 	= str( EM_learning_Dir + init_dir + 'CA_temporalCosSim/')
																temporalCSofCAs_fname 	= str( cellSubTypes + '_' + stim + '_' + str(msBins) + 'msBins_' + \
																	 str(bsPSTH) + 'msPSTHbins_'+ sample_longSWs_1st + 'Smp1st_' + priorType + '_rand' + str(rand) + Btag + '.npz')
																#
																if not os.path.exists( str(temporalCSofCAs_dir) ):
																		os.makedirs( str(temporalCSofCAs_dir) )
																#
																try:
																	# Load in npz file with temporal CA PSTH info from binned rasters. Then compare them across different models learned
																	# and maybe even across different SW samplings and priors too...
																	#
																	tPSTH = np.load( str(temporalCSofCAs_dir+temporalCSofCAs_fname) )
																	#
																	psthZ_accum = tPSTH['psthZ_accum']
																	binsPSTH = tPSTH['binsPSTH']
																	TcosSimMat = tPSTH['TcosSimMat'] 
																	cmk_resort = tPSTH['cmk_resort']

																except:
																	# if PSTH and temporal cosSim npz file isnt there to load, make it and save it.
																	numMS = 5000
																	binsPSTH = np.arange(0, numMS+1, bsPSTH)
																	psthZ_accum = np.zeros( (M, len(binsPSTH)-1 ) )
																	#
																	for k in range(M):																
																		#print(': CA#',k)
																		xx = list()
																		for i in range(numTrials): # combine all trials into one list (list comprehension doesnt work here for some reason.)
																			xx.extend( raster_Z_inferred_allSWs[i][k] ) 
																		#
																		ActTimes = [xx[i] for i in range(len(xx))] # if xx[i]<=numMS] # get rid of any spike after numMS
																		psthZ,_ = np.histogram( ActTimes, bins=binsPSTH )
																		psthZ_accum[k] = psthZ.T
																		#print( (psthZ[psthZ>0]/psthZ.sum()).sum() )
																	

																	TcosSimMat = cosine_similarity(psthZ_accum,psthZ_accum) # temporal cosine similarity.
																	
																	# Do Cuthill-Mckee Algorithm to make CosSim matrix block diagonal to group CAs together that are temporally coactive.
																	a = np.copy(TcosSimMat)
																	a[a<0.5]=0
																	TcosSimMatSp = spsp.csc_matrix( a )
																	cmk_resort = reverse_cuthill_mckee(TcosSimMatSp)
																	#
																	# Save an NPZ file: psthZ_accum, binsPSTH, TcosSimMat, cmk_resort
																	np.savez( str(temporalCSofCAs_dir+temporalCSofCAs_fname), psthZ_accum=psthZ_accum, binsPSTH=binsPSTH, TcosSimMat=TcosSimMat, cmk_resort=cmk_resort)
																#

																psthZ_sum = psthZ_accum.sum(axis=1)
																psthZ_normed = psthZ_accum/psthZ_sum[:,None]


																XcosSimMat = cosine_similarity(1-Pia.T,1-Pia.T)	# spatial cosine similairty - within one model.


																if True:
																	# Plot spatial and temporal cosSim matrices, normed PSTH and Z activation rate.
																	f = plt.figure( figsize=(20,10) ) # size units in inches
																	plt.rc('font', weight='bold', size=16)
																	plt.rc('text',usetex=False)
																	#	
																	ax1 = plt.subplot2grid( (2,3), (1,0), colspan=2)
																	ax1.imshow( psthZ_normed[cmk_resort] ) #, alpha=0.5, linewidth=2 )
																	ax1.set_aspect('auto')
																	xt = np.arange( 0, len(binsPSTH), 10 )
																	ax1.set_xticks( xt )
																	ax1.set_xticklabels( binsPSTH[xt]) #, fontsize=10 )
																	yt = np.arange( M )
																	ax1.set_yticks( yt )
																	ax1.set_yticklabels(cmk_resort, fontsize=8  )
																	#ax1.grid()
																	ax1.set_xlabel('time (ms)')
																	ax1.set_ylabel('CA')
																	ax1.set_title( str('normed PSTH (max='+str(psthZ_normed.max().round(2))+')' ) )
																	# ax1.set_xlim(0,len(binsPSTH)-1)
																	# ax1.set_ylim(0,psthZ_normed.max())
																	#ax1.set_title( str('Normed CA activations (binned at ' + str(bsPSTH) + 'ms)' ) )
																	#
																	ax4 = plt.subplot2grid( (2,3), (1,2))
																	ax4.imshow( 1-Pia[:,cmk_resort], vmin=0, vmax=1  )
																	xt = np.arange( M )
																	ax4.set_xticks( xt )
																	ax4.set_yticks([])
																	ax4.set_xticklabels(cmk_resort, fontsize=8, rotation=90 )
																	ax4.set_ylabel('$P_{ia}$')
																	ax4.set_xlabel('CA')
																	#
																	ax0 = plt.subplot2grid( (2,3), (0,1), colspan=1 )
																	ax0.imshow(TcosSimMat[np.ix_(cmk_resort,cmk_resort)], vmin=0, vmax=1)
																	xt = np.arange( M )
																	ax0.set_xticks( xt )
																	ax0.set_yticks( xt )
																	ax0.set_yticklabels([])
																	ax0.set_ylabel('CA')
																	ax0.set_xticklabels(cmk_resort, fontsize=8, rotation=90  )
																	ax0.set_title( str('Temporal Cos Sim') )
																	#
																	ax3 = plt.subplot2grid( (2,3), (0,2), colspan=1 )
																	ax3.imshow(XcosSimMat[np.ix_(cmk_resort,cmk_resort)], vmin=0, vmax=1)
																	xt = np.arange( M )
																	ax3.set_xticks( xt )
																	ax3.set_yticks( xt )
																	ax3.set_yticklabels([])
																	ax3.set_ylabel('CA')
																	ax3.set_xticklabels(cmk_resort, fontsize=8, rotation=90 )
																	ax3.set_title( str('Spatial Cos Sim') )
																	#
																	ax2 = plt.subplot2grid( (2,3), (0,0), colspan=1 )
																	ax2.scatter( cmk_resort, psthZ_sum[cmk_resort] )
																	xt = np.arange( M )
																	ax2.set_xticks( xt )
																	ax2.set_xticklabels( cmk_resort, fontsize=8, rotation=90 )
																	ax2.set_xlim([0,M])
																	ax2.grid()
																	ax2.set_title('CA # activs')

																	#fname = str( cellSubTypes + '_' + stim + '_' + str(msBins) + 'msBins_' + sample_longSWs_1st + 'Smp1st_' + priorType + '_rand' + str(rand) + Btag )
																	
																	plt.suptitle( str('Within 1 Model: ' + fname.replace('_',' ') + ' (binned for PSTH at ' + str(bsPSTH) + 'ms)' ) )

																	# SAVE PLOT.													# Info for saving the plot.
																	CA_temp_cosSim_figs_save_dir	= str( dirScratch + 'figs/PGM_analysis/PSTH_and_Rasters/' + init_dir + 'CA_temporal_cosSim/')
																	#
																	if not os.path.exists( str(CA_temp_cosSim_figs_save_dir) ):
																		os.makedirs( str(CA_temp_cosSim_figs_save_dir) )

																	plt.savefig( str(CA_temp_cosSim_figs_save_dir + fname + '.' + figSaveFileType ) ) 
																	#plt.show()
																	plt.close() 




																	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
																	#
																	# What is the spatial layout of CAs with most temporal similarity?
																	# 	NOTE: WE COULD ALSO DO THIS FOR MOST SYNERGYSTIC CELLS.
																	# 	SYNERGY = 1 - sqrt( XcosSimMat*TcosSimMat )
																	#
																	maxNumCAs_tempMatch = 3
																	bestTempMatchCAs = list()
																	bestTempMatchCAs_cosSimT = list()
																	bestTempMatchCAs_cosSimX = list()
																	#
																	for c in range(M):
																		yyy = np.where( TcosSimMat[c] > TcosSimMat[c].mean()+TcosSimMat[c].std() )[0] 	# find all CAs that match better than mean+std.
																		zzz = np.argsort(TcosSimMat[c][yyy])[::-1] 										# sort those best matching ones.
																		# use zzz = zzz[1:] to take self out. 

																		if zzz.size > maxNumCAs_tempMatch: 	 # take only top couple matching ones.
																			zzz = zzz[:maxNumCAs_tempMatch]

																		bestTempMatchCAs.append( yyy[zzz] ) 
																		bestTempMatchCAs_cosSimT.append( TcosSimMat[c][yyy[zzz]] )
																		bestTempMatchCAs_cosSimX.append( XcosSimMat[c][yyy[zzz]] )

																		
																		# Interesting plot.
																		if False:
																			plt.rc('font', weight='bold', size=12)
																			f = plt.figure(figsize=(20,10))
																			ax1 = plt.subplot2grid( (1,5),(0,0), colspan=4 )
																			ax1.plot(psthZ_accum[bestMatchCAs].T)
																			ax1.legend(bestMatchCAs)
																			ax1.grid()
																			#
																			ax2 = plt.subplot2grid( (1,5),(0,4) ) 
																			ax2.imshow(1 - Pia[:,bestMatchCAs], vmin=0, vmax=1)
																			ax2.set_aspect('auto')
																			ax2.set_xticks( np.arange(bestMatchCAs.size) )
																			ax2.set_xticklabels(bestMatchCAs)
																			ax2.set_xlabel('CA')


																			plt.suptitle( str(c), fontsize=20 )
																			plt.show()




															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															#
															# (4).  Save Rasters (lists of size  #Trials x #CA x "spikeTimes")
															#
															# 
															if flg_save_rasters4MAT:
																if not os.path.exists( rastersMATdir ): # Make directories for output data if not already there.
																	os.makedirs( rastersMATdir )
																#
																raster_fname = str( rastersMATdir + cellSubTypes + '_' + stim + '_' + str(msBins) + 'msBins.mat' )
																spio.savemat( raster_fname, mdict={'raster_Z_inferred_allSWs' : raster_Z_inferred_allSWs, 'raster_allSWs' : raster_allSWs} )
																	# , 'raster_Y_inferred_allSWs' : raster_Y_inferred_allSWs












															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #		
															#
															# (x). Are CA's Redundant (or Synergystic)?
															#		Define Reduncant = sqrt(cs_T * cs_X) and Synergy = 1 - Redundant.
															# 		That is, when temporally coactive, are they also spatially very similar?
															#		Actually, dunno if this is useful. A confound with high synergy being CAs
															# 		that have low temporal overlap. Maybe just show cs_X also.
															#
															if False:
																Redundant = np.sqrt(TcosSimMat*XcosSimMat)
																np.fill_diagonal(Redundant,0)
																Synergy = 1 - Redundant
																
																if False:
																	plt.imshow(Redundant)	
																	plt.colorbar()
																	plt.show()















															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# (X). Plot Spatial RF and temporal PSTH layout of CAs that match up best in time - defined by PSTH coactivity.
															#
															if flg_plot_tempCoactiveCAs_RFs_and_PSTH:

																psthZ_sum = psthZ_accum.sum(axis=1)
																psthZ_normed = psthZ_accum.T/psthZ_sum[None,:]

																# Build my own fucking colormap cause all these ones suck.
																colsAll = list()
																colsAll.append( [1,0,0,1] )
																colsAll.append( [0,1,0,1] )
																colsAll.append( [0,0,1,1] )
																colsAll.append( [0,1,1,1] )
																colsAll.append( [1,0,1,1] )
																colsAll.append( [0,0,0,1] )
																#colsAll.append( [1,1,0,1] ) # yellow sucks


																for MM in range(M):
																	#
																	# Plot RFs of Z in bottom right.
																	if N > Nsplit:
																		plt.rc('font', weight='bold', size=18)
																		f = plt.figure(figsize=(20,10))
																		#
																		axRF1 = plt.subplot2grid( (2,3), (0,0) )
																		axRF2 = plt.subplot2grid( (2,3), (0,1) )
																		axRC = plt.subplot2grid(  (4,3), (0,2) )
																		axRH = plt.subplot2grid(  (4,3), (1,2) ) 
																	else:
																		plt.rc('font', weight='bold', size=14)
																		f = plt.figure(figsize=(20,10))
																		#
																		axRF1 = plt.subplot2grid( (2,2), (0,0) )
																		axRC = plt.subplot2grid(  (2,2), (0,1) )
																		axRF2 = None
																	#
																	axPS = plt.subplot2grid( (2,1), (1,0) )
																	#
																	axRC.scatter(Robust, CrispX, s=150, facecolors='none', edgecolors='black' )
																	if N > Nsplit:
																		axRH.scatter(Robust, Hetero, s=150, facecolors='none', edgecolors='black' )
																	#
																	for i,A in enumerate( bestTempMatchCAs[MM] ):
																		if i==0:
																			axPS.plot( binsPSTH[:-1], psthZ_normed[:,A], color=colsAll[i], linewidth=3, label=str( str(A) + '  :  ' + str( int(psthZ_sum[A]) ) ), alpha=0.5 )
																		else:
																			Synergy = 1 - np.sqrt( bestTempMatchCAs_cosSimX[MM][i]*bestTempMatchCAs_cosSimT[MM][i] )
																			axPS.plot( binsPSTH[:-1], psthZ_normed[:,A], color=colsAll[i], linewidth=3, label=str( str(A) + '  :  ' + str( int(psthZ_sum[A]) ) + \
																				'  :  ' + str(bestTempMatchCAs_cosSimT[MM][i].round(2)) + '  :  ' + str(Synergy.round(2)) ), alpha=0.5 )
																		#
																		axRC.scatter(Robust[A], CrispX[A], s=100, color=colsAll[i], alpha=0.5)
																		#
																		if N>Nsplit:
																			axRH.scatter(Robust[A], Hetero[A], s=100, color=colsAll[i], alpha=0.5)
																		#
																		Bs = cellsIn_collect[A]
																		#
																		axRF1,axRF2 = pf.ellipse_RFz_multi_2ax(STRF_gauss[cellSubTypeIDs], Pia, A, axRF1, axRF2, Bs, colsAll[i], 0.5, Nsplit)
																		#

																	axRC.errorbar( Robust.mean(), CrispX.mean(), xerr=Robust.std(), yerr=CrispX.std() )
																	axRC.set_xlim(-.1,1.1)
																	axRC.set_ylim(-.1,1.1)
																	axRC.set_xlabel('R')
																	axRC.set_ylabel('Cx')
																	axRC.set_aspect('equal')
																	axRC.grid(True)
																	#
																	if N>Nsplit:
																		axRH.errorbar( Robust.mean(), Hetero.mean(), xerr=Robust.std(), yerr=Hetero.std() )
																		axRH.set_xlim(-.1,1.1)
																		axRH.set_ylim(-.1,1.1)
																		axRH.set_xlabel('R')
																		axRH.set_ylabel('H')
																		axRH.set_aspect('equal')
																		axRH.grid(True)
																	#
																	axPS.legend(loc='best', title=str('z : #Act : $cs_T$ : Syn')) #, bbox_to_anchor=(1, 0.5))
																	axPS.set_xlim(0,binsPSTH.max())
																	#axPS.set_ylim(0,1.1)
																	axPS.set_xlabel('Time (ms)')
																	axPS.set_ylabel('PSTH (normed)') # could also norm to make total sum = 1. Harder to see.
																	axPS.set_aspect('auto')
																	axPS.grid(True)
																	#
																	axRF1.set_xticklabels([])
																	axRF1.set_yticklabels([])
																	axRF1.set_aspect('equal')
																	axRF1.grid(True)
																	axRF1.set_xlabel( str('RFs (ct1)'), fontsize=16 )
																	if N > Nsplit:
																		axRF2.set_xticklabels([])
																		axRF2.set_yticklabels([])
																		axRF2.set_xlabel( str('RFs (ct2)'), fontsize=16 )
																		axRF2.set_aspect('equal')
																		axRF2.grid(True)

																	plt.suptitle( str( 'CAs coactive with z' + str(MM) + ' ' + fname.replace('_',' ') ) )
																	#
																	# SAVE PLOT.                                                    # Info for saving the plot.
																	timeCoactiveCAs_figs_save_dir = str( output_figs_dir+'../CA_RF_PSTHs_coactiveCAs/')
																	#
																	if not os.path.exists( str(timeCoactiveCAs_figs_save_dir) ):
																		os.makedirs( str(timeCoactiveCAs_figs_save_dir) )

																	plt.savefig( str(timeCoactiveCAs_figs_save_dir + fname + '_z' + str(MM) + '.' + figSaveFileType ) ) 
																	#plt.show()
																	plt.close() 

























															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															#
															# (5). Scatter one metric vs. another. Then show CAs that are high in that combo
															# 		of metrics. Tile RFs and plot PSTHs.
															#
															if flg_plot_crispRobust_RFs_and_PSTH:
																print('Plotting CAs arranged by different performance in metrics.')

																numCAs = 3		# choose 9 if colormap is Set1, 7 if its my made 'rgbcmk' colormap.
																numIters = np.floor(M/numCAs).astype(int)  	# np.floor(M/9).astype(int) to do as many as possible.

																# Scatter Robustness vs. Crispness.
																metricNames = ['R','Cx','dPy','bPy']
																metricData 	= [Robust, CrispX, cosdif_ZwGLMpy[:,0], cosdif_binGLM]
																print(metricNames)
																#
																pf.plot_crispRobust_RFs_and_PSTH( numCAs, numIters, metricData, metricNames, psthZ_accum, binsPSTH, 
																	Pia, STRF_gauss[cellSubTypeIDs], Nsplit, cellsIn_collect, output_figs_dir, fname, figSaveFileType )


																# Scatter difference from null model p(y) vs. how that changes with binning.
																metricNames = ['dPy','bPy','R','Cx']
																metricData 	= [cosdif_ZwGLMpy[:,0], cosdif_binGLM, Robust, CrispX]
																print(metricNames)
																#
																pf.plot_crispRobust_RFs_and_PSTH( numCAs, numIters, metricData, metricNames, psthZ_accum, binsPSTH, 
																	Pia, STRF_gauss[cellSubTypeIDs], Nsplit, cellsIn_collect, output_figs_dir, fname, figSaveFileType )



																# Scatter difference from null model p(y) vs. how that changes with binning.
																metricNames = ['R','bPy','Cx','dPy']
																metricData 	= [Robust, cosdif_binGLM, CrispX, cosdif_ZwGLMpy[:,0]]
																print(metricNames)
																#
																pf.plot_crispRobust_RFs_and_PSTH( numCAs, numIters, metricData, metricNames, psthZ_accum, binsPSTH, 
																	Pia, STRF_gauss[cellSubTypeIDs], Nsplit, cellsIn_collect, output_figs_dir, fname, figSaveFileType )



																# Scatter difference from null model p(y) vs. crispness. Is activity of crisp CAs well characterized by GLM model?
																metricNames = ['Cx','bPy','R','dPy']
																metricData 	= [CrispX, cosdif_binGLM, Robust, cosdif_ZwGLMpy[:,0]]
																print(metricNames)
																#
																pf.plot_crispRobust_RFs_and_PSTH( numCAs, numIters, metricData, metricNames, psthZ_accum, binsPSTH, 
																	Pia, STRF_gauss[cellSubTypeIDs], Nsplit, cellsIn_collect, output_figs_dir, fname, figSaveFileType )




																if N > Nsplit:
																	# Scatter Robustness vs. Heterogeneity 
																	metricNames = ['R','H','dPy','bPy']
																	metricData 	= [Robust, Hetero, cosdif_ZwGLMpy[:,0], cosdif_binGLM]
																	print(metricNames)
																	#
																	pf.plot_crispRobust_RFs_and_PSTH( numCAs, numIters, metricData, metricNames, psthZ_accum, binsPSTH, 
																		Pia, STRF_gauss[cellSubTypeIDs], Nsplit, cellsIn_collect, output_figs_dir, fname, figSaveFileType )
																	#
																	#
																	# Scatter Crispness vs. Heterogeneity (what the hell, why not?...)
																	metricNames = ['H','Cx','R','bPy']
																	metricData 	= [Hetero, CrispX, Robust, cosdif_binGLM]
																	print(metricNames)
																	#
																	pf.plot_crispRobust_RFs_and_PSTH( numCAs, numIters, metricData, metricNames, psthZ_accum, binsPSTH, 
																		Pia, STRF_gauss[cellSubTypeIDs], Nsplit, cellsIn_collect, output_figs_dir, fname, figSaveFileType )





															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# (5). Make some raster and PSTH plots of inferred z's, observed y's and their relationships.
															#
															# [[A]]. Raster Z inferred each Y observed participating in it below.

															if flg_raster_Z_andItsYs:
																pf.raster_YorZ_andItsMembers('z', 1-Pia, TH, minMemb, maxMemb, raster_allSWs, raster_Z_inferred_allSWs, \
																	STRF_gauss[cellSubTypeIDs], mean_CS_accModels[i_mdl], cosdif_ZwGLMpy[:,0], cosdif_binGLM, Zmatch_gather, Pia_gather, \
																	PSTHz_gather, binsPSTH, colsDark, output_figs_dir, model_file, Nsplit, Robust, CrispX, figSaveFileType) # 
																	# CA_coactivity_allSWs, Cell_coactivity_allSWs, 














															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
															# (6). Show stimulus at a couple time points before a cell assembly becomes most active
															# 		According to its PSTH.
															#
															if flg_show_stim_at_high_PSTH: # and False:
					
																numMS 				= 5000
																sampling_period 	= (numMS/timeBins) # time in ms of one movie frame.
																tBins = np.arange(0, numMS+1, sampling_period)


																

																for k in CAS_to_show_stim_at_high_PSTH: # Loop through each CA.
																	print(': CA#',k)
																	xx = list()
																	for i in range(numTrials): # combine all trials into one list (list comprehension doesnt work here for some reason.)
																		xx.extend( raster_Z_inferred_allSWs[i][k] ) 
																	#
																	ActTimes = [xx[i] for i in range(len(xx)) if xx[i]<=numMS] # get rid of any spike after numMS
																	psthZ,_ = np.histogram( ActTimes, bins=tBins )
																	ind = np.array([np.argmax(psthZ)]) #np.where( psthZ > psthZ.mean()+psthZ.std() )[0]
																	#
																	#plt.rc('font', weight='bold', size=16)

																# Plot RFs of Z in bottom right.
																	#if N > Nsplit:
																	plt.rc('font', weight='bold', size=18)
																	f = plt.figure(figsize=(20,10))
																	#
																	axSTM2 = plt.subplot2grid( (3,3), (0,0), rowspan=2 )
																	axRF1 = plt.subplot2grid( (3,3), (0,1), rowspan=2 )
																	axSTM1 = plt.subplot2grid( (3,3), (0,2), rowspan=2 )
																	#axRF2 = None # axRF2 = plt.subplot2grid( (2,3), (0,1) )
																	axPS = plt.subplot2grid(  (3,1), (2,0) )
																	# else:
																	# 	#
																	# 	axSTM1 = plt.subplot2grid( (10,2), (0,1), rowspan=9 )
																	# 	axSTM2 = plt.subplot2grid( (10,2), (0,1), rowspan=9 )
																	# 	axRF1 = plt.subplot2grid( (10,1), (0,0), rowspan=9 )
																	# 	axRF2 = None
																	# 	axPS = plt.subplot2grid(  (10,1), (9,0) )


																	# try:
																	if True:
																		for i in range(len(ind)):

																			indPast = ind[i]-20

																			axSTM1.imshow( Mov_stim[ :,:,ind[0]], cmap='bone')
																			axSTM1.set_title(str('Stim at Present (t='+str( (tBins[ind[0]]/1000).round(2))+'s)'), fontsize=10, fontweight='bold' )
																			axSTM1.set_xticks([])
																			axSTM1.set_yticks([])
																			axSTM1.set_aspect('equal')

																			axSTM2.imshow( Mov_stim[ :,:,indPast], cmap='bone')
																			axSTM2.set_title(str('Stim 333ms in Past (t='+str( (tBins[indPast]/1000).round(2))+'s)'), fontsize=10, fontweight='bold' )
																			axSTM2.set_xticks([])
																			axSTM2.set_yticks([])
																			axSTM2.set_aspect('equal')

																		#
																		# # Plot Cell Assembly Receptive Fields.
																		Bs = cellsIn_collect[k]
																		axRF1 = pf.ellipse_RFz_multiCelltype_1ax(STRF_gauss[cellSubTypeIDs], Pia, k, axRF1, Bs, 1, Nsplit)
																		axRF1.set_title(str( 'RFs'  ), fontsize=10, fontweight='bold' )
																		axRF1.set_xticks([])
																		axRF1.set_yticks([])
																		axRF1.set_aspect('equal')
																		

																		# Plot PSTH
																		#i+=1
																		axPS.plot( tBins[1:], psthZ )
																		axPS.scatter( tBins[ind], psthZ[ind], s=30, c='r', marker='.' )
																		axPS.plot( [tBins[indPast], tBins[indPast] ], [0, psthZ.max()], 'k--' )
																		axPS.set_xticks([0,1000,2000,3000,4000,5000])
																		axPS.set_xlim(0,5000)
																		axPS.set_xticklabels([0,1,2,3,4,5])	
																		axPS.set_ylabel('PSTH', fontsize=10, fontweight='bold')
																		axPS.set_xlabel('time (s)', fontsize=10, fontweight='bold')
																		axPS.set_yticks([0, psthZ.max()])
																		#axPS.set_yticklabels([0,1,2,3,4,5])
																		axPS.tick_params(axis='both', which='major', labelsize=8)
																		axPS.grid()


																		plt.suptitle( str( cellSubTypes+' CA#'+str(k) ), fontsize=18, fontweight='bold')
																	# except:
																	# 	print('meh')
																	#
																	# save png's to flip through quickly for analysis because pdfs files are too big and slow to load and manipulate.
																	HeteroCA_andStim_figs_save_dir = str( output_figs_dir+'../HeteroCA_andStim/')
																	#
																	if not os.path.exists( str(HeteroCA_andStim_figs_save_dir) ):
																		os.makedirs( str(HeteroCA_andStim_figs_save_dir) )

																	plt.savefig( str(HeteroCA_andStim_figs_save_dir+fname+'_z'+str(k)+'.png' ) ) 
																	
																	#plt.show()


																	plt.close() 

																	# # THESE DONT WORK BECAUSE STIM IS TOO SHORT...
																	#
																	# # Plot STA (kinda) - unweighted, just requires passing threshold of mean + 1std.
																	# i+=1
																	# axPS.imshow(Mov_stim[:,:,ind].mean(axis=2), cmap='bone')
																	# axPS.set_title(str('RTA'), fontsize=10, fontweight='bold' )
																	# axPS.set_xticks([])
																	# axPS.set_yticks([])
																	#
																	# # Plot more accurate STA (weighted by number of activations in time bin) - any difference?																		# Plot STA (kinda)	
																	# i+=1
																	# #
																	# cata0 = np.zeros_like( Mov_stim[:,:,0] ) 
																	# #print(tBins)
																	# for t in range(len(psthZ)):
																	# 	cata0 += Mov_stim[:,:,t]*psthZ[t] 
																	# axPS.imshow(cata0, cmap='bone')
																	# axPS.set_title(str('RTA2'), fontsize=10, fontweight='bold' )
																	# axPS.set_xticks([])
																	# axPS.set_yticks([])
																	# #










														# except:
														# 	print('Problem. Moving on.')





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #










# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Look at distributions of p(y) under GLM model for spikewords observed when each Za is active 
#
flg_plot_pOfy_underNullOLDONE = False
if flg_plot_pOfy_underNullOLDONE:
	print('plot_pOfy_underNullOLDONE')

	pZ = list() # marginal p(y) under GLM of observed SW when Z active.
	# pC = list() # conditional p(y|z) under PGM of observed SW when Z active. 
	pJ = list() # joint p(y,z) under PGM of observed SW when Z active. 
	#
	for zz in range(M):
		pZ.append( list() )
		# pC.append( list() )
		pJ.append( list() )
		#
		for tt in range(numTrials):
			pZ[zz].extend( pSW_forZ_GLM[tt][zz] )
			# pC[zz].extend( pSW_forZ_cdl[tt][zz] )
			pJ[zz].extend( pSW_forZ_jnt[tt][zz] )
		#
		# here, turn zeros into machine precision to avoid -inf when taking log.
		for zers in np.where(pZ[zz]==0)[0]:
			pZ[zz][zers] = np.finfo(float).eps	

	nBins = 50
	pZ_histX = np.zeros( (M,nBins+1) )
	pZ_histY = np.zeros( (M,nBins) )
	for zz in range(M): 
		try:
			# get rid of nans (which happen a lot when spikes happen outside allowed GLM windown)
			# and -infs which happen rather rarely when you take log of p(y)=0
			pZ_histY[zz],pZ_histX[zz] = np.histogram( np.log(pZ[zz])[ ~np.isnan(pZ[zz]) ], bins=nBins )
		except:
			print('Error w histogram for CA',zz)
			print( len(pZ[zz]), np.isfinite( np.log(pZ[zz]) ).sum(), np.isinf( np.log(pZ[zz]) ).sum(), np.isnan( np.log(pZ[zz]) ).sum() )
			# np.log(pZ[zz])[ ~np.isnan(pZ[zz]) ]
	#
	# Compute average p(y) under GLM for all spike-words regardless of z activity.
	allSum = 0
	allCnt = 0
	for tt in range(numTrials):
		inds = list(np.where(np.isfinite(np.log(pSW_nullGLM[tt])))[0])
		summ = [np.log(pSW_nullGLM[tt][xx]) for xx in inds]
		allSum += np.sum( summ )
		allCnt += len(inds)
	allMn = allSum/allCnt
	#
	allStd = 0
	for tt in range(numTrials):
		inds = list(np.where(np.isfinite(np.log(pSW_nullGLM[tt])))[0])
		summ = [ (np.log(pSW_nullGLM[tt][xx])-allMn)**2 for xx in inds]
		allStd += np.sum( summ )
	allStd = np.sqrt(allStd/allCnt)


	# Stats (mean, std, #timesInferred) for spike-words when each za is active
	mnn = np.zeros(M)	# for GLM p(y)
	std = np.zeros(M)
	cnt = np.zeros(M)
	# mnnC = np.zeros(M) 	# for conditional (using joint only)
	# stdC = np.zeros(M)
	mnnJ = np.zeros(M) 	# for joint.
	stdJ = np.zeros(M)
	#
	for zz in range(M): 
		mnn[zz] = np.nanmean( pZ[zz] )
		std[zz] = np.nanstd( pZ[zz] )
		cnt[zz] = len(pZ[zz])
		#
		# mnnC[zz] = np.nanmean( pC[zz] ) #  (using joint only)
		# stdC[zz] = np.nanstd( pC[zz] )
		#
		mnnJ[zz] = np.nanmean( pJ[zz] )
		stdJ[zz] = np.nanstd( pJ[zz] )
	#
	# Two options to sort CAs in the plots.
	srtPyNull = np.argsort( np.log(mnn) ) 							# (1). sort by p(y)
	#srtMnRob = np.argsort( mean_CS_accModels[i_mdl]+ mean_CST_accModels )[::-1]		# (2). sort by <cosSim>

	# Info for saving the plot.
	pofy_figs_save_dir	= str( dirScratch + 'figs/PGM_analysis/PSTH_and_Rasters/' + init_dir + 'plot_pofYnull_forCAs/')
	#fname = str( cellSubTypes + '_' + stim + '_' + whichSim + '_' + str(msBins) + 'msBins_' + sample_longSWs_1st + 'Smp1st_' + priorType + '_rand' + str(rand) + Btag )
	#
	if not os.path.exists( str(pofy_figs_save_dir) ):
		os.makedirs( str(pofy_figs_save_dir) )


	# (1). Plot histograms of p(y) under GLM during activation of each cell assembly za.
	f = plt.figure( figsize=(20,10) ) # size units in inches
	plt.rc('font', weight='bold', size=10)
	plt.rc('text',usetex=False)
	#
	#
	# (1). Plot full histograms of p(y) under null for each CA in model.
	ax0 = plt.subplot2grid(	(2,2), (0,0) ) 
	for zz in range(M): 
		ax0.plot(pZ_histX[zz][:-1],pZ_histY[zz], alpha=0.5, color=colors[zz])
	for zz in range(M): 
		ind = pZ_histY[zz].argmax()
		ax0.text( pZ_histX[zz][ind], pZ_histY[zz][ind], str(zz) ) 
	#
	ax0.plot( [allMn, allMn], [0, pZ_histY.max()],  'k--')
	ax0.plot( [allMn-allStd, allMn-allStd], [0, pZ_histY.max()],  'k:')
	ax0.text( allMn, pZ_histY.max(), str(r'$\mu,\sigma$ allSWs='+str(allMn.round(1))+','+str(allStd.round(1)) ), color='black', ha='right', va='top' )
	ax0.set_xlim([-50,0])
	#	
	ax0.set_title( str('log p(y) under '+whichSim) , fontsize=18, fontweight='bold' )
	ax0.set_ylabel( '# times z inferred' , fontsize=18, fontweight='bold' )
	ax0.grid()

	# (2). Scatter plot mean p(y) of SW when z active vs # times z inferred
	ax1 = plt.subplot2grid(	(2,4), (1,0) ) 
	ax1.scatter(cnt[srtPyNull], np.log(mnn[srtPyNull]), c=np.arange(M), s=100, alpha=0.5, cmap='jet')	
	for ii in range(M):
		ax1.text( cnt[ii], np.log(mnn[ii]), str(ii) )
	ax1.plot( [cnt.min(),cnt.max()], [allMn, allMn], 'k--')
	ax1.plot( [cnt.min(),cnt.max()], [allMn-allStd, allMn-allStd], 'k:')
	ax1.text( cnt.min(), allMn-allStd, str(r'$\mu,\sigma$ allSWs='+str(allMn.round(1))+','+str(allStd.round(1)) ), color='black', ha='left', va='top' )
	ax1.set_xlabel('# times inferred', fontsize=18, fontweight='bold')
	ax1.set_ylabel( str(r' $\mu$ log p(y) ' + whichSim), fontsize=18, fontweight='bold' )
	#ax1.set_title( 'CA generated SW prob. under null' )
	ax1.grid()



	# (2b). Scatter plot mean p(y) of SW when z active vs # times z inferred
	axc = plt.subplot2grid(	(2,4), (1,1) ) 
	axc.scatter(mean_CS_accModels[i_mdl][srtPyNull], mean_CST_accModels[srtPyNull], c=np.arange(M), s=100, alpha=0.5, cmap='jet')	
	for ii in range(M):
		axc.text( mean_CS_accModels[i_mdl][ii], mean_CST_accModels[ii], str(ii) )
	axc.plot([0,1],[0,1],'k--')
	axc.set_xlabel('<cosSim>_X', fontsize=18, fontweight='bold')
	axc.set_ylabel('<cosSim>_T', fontsize=18, fontweight='bold')
	axc.set_title('CA robustness accross models', fontsize=18, fontweight='bold')
	axc.set_xlim([0,1])
	axc.set_ylim([0,1])
	#ax1.set_title( 'CA generated SW prob. under null' )
	axc.grid()



	# xlab = list() 		# X-tick label including CA id, mean p(y|z) and mean p(y,z) for each CA.
	# for m in range(M):
	# 	xlab.append( str( str(srtPyNull[m]) + ', ' + str(mnnC[srtPyNull[m]].round(1))  + ', ' + str(mnnJ[srtPyNull[m]].round(1))  ) ) 


	# (3). Show Pia matrix reordering CAs by most improbable SWs under null model
	axb = plt.subplot2grid(	(4,2), (3,1)) #, rowspan=3) 
	try:
		axb.imshow(psthZ_allMods[i_mdl])
	except:
		aaa=0
	xt = np.arange( 0, len(binsPSTH), 10 )
	axb.set_xticks( xt )
	axb.set_xticklabels( (binsPSTH[xt]/1000).round(1) ) #, fontsize=10 )
	#
	axb.set_yticks(np.arange(M))
	axb.set_yticklabels( srtPyNull, fontsize=8)		
	for ytick, color in zip(axb.get_yticklabels(), colors): # colorcode CAs on yticks by srtPyNull for visual clarity.
		ytick.set_color(color)
	#	
	axb.set_ylabel('CA $z_a$', fontsize=18, fontweight='bold')
	axb.set_xlabel( 'time (sec)' , fontsize=18, fontweight='bold')
	axb.set_aspect('auto')
	#axb.grid()


	# (3). Show Pia matrix reordering CAs by most improbable SWs under null model
	ax2 = plt.subplot2grid(	(4,2), (2,1)) #, rowspan=3) 
	ax2.imshow(1-Pia[:,srtPyNull],vmin=0,vmax=1)
	ax2.plot( [0,M], [Nsplit, Nsplit], 'w--' )
	ax2.set_xlim(0,M-1)
	ax2.set_ylim(0,N-1)
	ax2.set_xticks(np.arange(M))
	ax2.set_xticklabels( srtPyNull, rotation=90, fontsize=8)
	for xtick, color in zip(ax2.get_xticklabels(), colors): # colorcode CAs on xticks by srtPyNull for visual clarity.
		xtick.set_color(color)
	#ax2.set_xlabel('Cell Assembly $z_a$', fontsize=18, fontweight='bold')
	ax2.set_ylabel( 'Cell $y_i$' , fontsize=18, fontweight='bold')
	ax2.set_aspect('auto')
	#ax2.grid()

	# (4). Plot mean cos sim for each CA with each matching CA in all other models.
	ax3 = plt.subplot2grid(	(4,2), (1,1)) #, rowspan=3) 
	ax3.scatter( np.arange(M), mean_CS_accModels[i_mdl][srtPyNull], label='$<cs>_X$' )
	try:
		ax3.scatter( np.arange(M), mean_CST_accModels[srtPyNull], label='$<cs>_T$' )
	except:
		aaa=0
	ax3.set_xticks(np.arange(M))
	ax3.set_xticklabels( srtPyNull, rotation=90, fontsize=8)
	for xtick, color in zip(ax3.get_xticklabels(), colors): # colorcode CAs on xticks by srtPyNull for visual clarity.
		xtick.set_color(color)
	ax3.set_xlim(0, M)
	#ax3.set_xlabel('$z_a$', fontsize=18, fontweight='bold')
	ax3.set_ylabel( '$<cosSim>$' , fontsize=18, fontweight='bold')
	#ax3.grid()
	ax3.legend()
	ax3.set_aspect('auto')

	# (5). Plot p(y)_{null}, Joint and conditional for PGM for each CA
	ax4 = plt.subplot2grid(	(4,2), (0,1)) 
	#ax4.plot(mnnC[srtPyNull],'k-', label='p(y|z) PGM')
	ax4.scatter( np.arange(M), mnnJ[srtPyNull], color='blue', label='p(y,z) PGM')
	ax4.scatter( np.arange(M), np.log(mnn[srtPyNull]), c=np.arange(M), s=50, alpha=0.5, cmap='jet')
	ax4.plot(np.log(mnn[srtPyNull]),'g-', label=str('p(y) '+whichSim))
	ax4.plot( [0, M], [allMn, allMn], 'k--')
	ax4.plot( [0, M], [allMn-allStd, allMn-allStd], 'k:')
	ax4.text( 0, allMn, str(r'$\mu,\sigma$ allSWs='+str(allMn.round(1))+','+str(allStd.round(1)) ), color='black', ha='left', va='bottom' )
	ax4.set_xticks(np.arange(M))
	ax4.set_xticklabels( srtPyNull, rotation=90, fontsize=8)
	for xtick, color in zip(ax4.get_xticklabels(), colors): # colorcode CAs on xticks by srtPyNull for visual clarity.
		xtick.set_color(color)
	ax4.set_xlim(0, M)
	ax4.set_ylabel( 'log p' , fontsize=18, fontweight='bold')
	#ax4.set_xlabel( '$z_a$' , fontsize=18, fontweight='bold')
	ax4.legend()
	#ax4.grid()

	#
	plt.suptitle( fname.replace('_',' '), fontsize=20, fontweight='bold' ) 
	#plt.tight_layout()
	plt.savefig( str(pofy_figs_save_dir + fname + '.' + figSaveFileType ) ) 
	plt.close() 

	# plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# (6). Compute "spike"-triggered-average using cell assemblies and stim movie to do reverse correlation.
#		Using Sonja Gruen's ELEPHANT toolbox.
#
if flg_compute_CATA:

	# plt.imshow(preprocessing.cov_matrix(Mov_stim[0])) # spatial correlations in frame 0
	# plt.colorbar()
	# xxx

	numMS 				= 5000
	sampling_period 	=(numMS/timeBins) # time in ms of one move frame.
	
	

	# for x in range(300,301): #range(xPix):
	# 	for y in range(300,303): #range(yPix):
			#
	print('neo.AnalogSignal')
	t0 = time.time()
	mov_sig = neo.AnalogSignal( Mov_stim_unrav, units=V, sampling_period=sampling_period*ms ) # Need to transpose?
	t1 = time.time()
	print('time:',t1-t0)
	#
	# CATA = np.zeros()


	for k in range(1): #M): # Loop through each CA.
		print(': CA#',k)#, :yPix',y,':xPix',x)
		print('Building ras_1CA')
		t0 = time.time()
		#
		xx = list()
		for i in range(numTrials): # combine all trials into one list (list comprehension doesnt work here for some reason.)
			xx.extend( raster_Z_inferred_allSWs[i][k] ) 
		#
		ras_1CA = [xx[i] for i in range(len(xx)) if xx[i]<=numMS] # get rid of any spike after numMS
		t1 = time.time()
		print('time:',t1-t0)
		#
		# #
		#
		print('Neo SpikeTrain')
		t0 = time.time()
		st = neo.SpikeTrain( ras_1CA*ms, t_stop=numMS*ms)
		t1 = time.time()
		print('time:',t1-t0)

		#binned_st = BinnedSpikeTrain(st, binsize=szHistBins*ms)

		# raster_Z_inferred_allSWs: is list of size: trial X cell X # activations (units in ms)
		# Mov_stim: 				 600 pix X 795 pix X 300 time pts. Time per frame is 16 & 2/3 ms.
		
		t0 = time.time()
		print('Elephant spike_triggered_average for one freaking CA!!')
		CATA = sta.spike_triggered_average( mov_sig, st, (-10*ms,0*ms) )
		t1=time.time()
		print('Time spent = ', t1-t0)
		#
		t0 = time.time()
		print('Elephant spike_field_coherence for one freaking CA!!')
		SFfq, SFC = sta.spike_field_coherence( mov_sig, st )
		#CATA = sig.coherence( mov_sig, st[0] )
		t1=time.time()
		print('Time spent = ', t1-t0)

		
		print('Spike - Triggered Avg' ,CATA)
		print('SpikeField Coherence' ,SFC)
		print('SFC Freqs' ,SFfq)





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
# (1). Look for periodicity by looking at CA ISIs in raster plots. Coefficient of Variation of CA activation ISIs.
if flg_compute_CA_ISIs:
	print('Compute ISIs of cell assembly activations. Are any periodic?')
	# THIS ISNT SUCH A VALID ANALYSIS BECAUSE WE ARE BINNING SPIKE TRAINS AT 5MS TO MAKE SPIKE WORDS.
	# THEN LEARNING CAS FROM THOSE SPIKE WORDS, WHICH ARE VERY REDUNDANT IN SHORT TIME. SO HERE,
	# MANY ISIS ARE 1MS OR 2MS.  AND THEN SOMETIMES ISIS ARE 1000MS BECAUSE OF THE LONG QUIET PERIODS.
	# I THINK THIS INFLATES THE FANO FACTOR. BECAUSE OF THE ISI VARIANCE.
	#
	# raster_Z_inferred_allSWs # list ( numTrials x M ) --> contains list of activation times.
	#

	f,ax = plt.subplots(1,2)

	CV = np.zeros(M)
	for m in range(M):
		ISI_accum = list()
		for tr in range(numTrials):
			ISI_accum.extend( np.diff(raster_Z_inferred_allSWs[tr][m]) )
		#
		ISI_accum = np.array(ISI_accum)
		CV[m] = ISI_accum.var()/ISI_accum.mean() # coefficient of variation.
		#ISIbins = np.unique( np.round( np.logspace( np.log(ISI_accum.min()), np.log(ISI_accum.max()), 100) ) )
		ISIbins = np.arange( ISI_accum.max() )
		ISI = np.histogram(ISI_accum, bins=ISIbins) #, normed=True)
		ax[0].plot(ISI[1][:-1], ISI[0], label=str( 'z'+str(m) ), alpha=0.5 )
	#
	ax[0].set_xlim([2,30]) # ms
	#ax[0].autoscale(enable=True, axis='both', tight=True)
	ax[0].grid()
	ax[0].set_title('ISI distributions')
	ax[0].set_xlabel('ms')
	#
	#xx,yy = np.histogram(CV)
	ax[1].grid()
	ax[1].scatter(np.arange(M),CV)
	ax[1].plot([0,M],[1,1],'k--' )
	ax[1].plot([0,M],[0,0],'k--' )
	#ax[1].text(.9*yy.max(),.9*xx.max(), str('M='+str(M)), ha='right', va='top' )
	ax[1].set_title('fano factor')
	#ax[1].set_xticks( np.unique(np.round(yy[:-1]) ) )
	ax[1].set_xlabel('$\sigma^2 / \mu$')

	#plt.show()

	#fname = str( cellSubTypes + '_' + stim + '_' + str(msBins) + 'msBins_' + sample_longSWs_1st + 'Smp1st_' + priorType + '_rand' + str(rand) + Btag )
	
	plt.suptitle( str( fname.replace('_',' ') ) )

	# SAVE PLOT.													# Info for saving the plot.
	CV_of_CAs_figs_save_dir	= str( dirScratch + 'figs/PGM_analysis/PSTH_and_Rasters/' + init_dir + 'CV_of_CAs/')
	#
	if not os.path.exists( str(CV_of_CAs_figs_save_dir) ):
		os.makedirs( str(CV_of_CAs_figs_save_dir) )

	plt.savefig( str(CV_of_CAs_figs_save_dir + fname + '.' + figSaveFileType ) ) 
	#plt.show()
	plt.close() 


