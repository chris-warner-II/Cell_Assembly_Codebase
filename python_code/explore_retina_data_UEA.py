

## This code (running as a script to start) will take in a matlab file which contains ...

##

# toolbox from Sonja Gruen's group to do Unitary Events Analysis and other parallel spike train analysis
import elephant.spike_train_generation as stgen 
from elephant.spike_train_correlation import cross_correlation_histogram, corrcoef, covariance
from quantities import Hz, s, ms
from elephant.kernels import GaussianKernel
from elephant.statistics import isi, cv, instantaneous_rate, mean_firing_rate
import neo

from elephant.conversion import BinnedSpikeTrain
import elephant.unitary_event_analysis as uea

import numpy as np
import scipy as sp
from scipy import io as io
from scipy import sparse as spsp

import sklearn.cluster as skc
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import time 						# to time operations for code analysis
import os.path 						# to check if a file or directory exists already
import sys

import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.plot_functions as pf
import utils.elephant_usage as el
import utils.retina_computation as rc

import pyflann as nnc 				# Fast Library for Approximate Nearest Neighbors (FLANN) toolbox (Muja & Lowe)
import pyclust as pc 				# Toolbox that contains k-medoids (for clustering binary data) algorithm
#import pycluster as pc2 			# Another toolbox that contains k-medoids
#import pyclustering as pc3 

import sys



## (0). Flags for what to run.
#
# Spike Rate Analysis
flg_instantaneousRate_Eleph 			= False 		# UPDATED
flg_meanRate_Eleph 						= False 		# UPDATED
flg_scatter_meanRate_trialAvg_Eleph 	= False 		# UPDATED
#
# Generating Spike Trains with Stochastic Models
flg_generateSpikeTrains_Eleph 			= False 		# UPDATED
flg_HPP_gen								= False 		# UPDATED
flg_HGP_gen								= False 		# UPDATED
flg_CPP_gen								= False 		# TO DO
flg_SIP_gen								= False 		# TO DO
flg_raster_spikeTrains					= False 		# TO DO
#
# Inter-Spike Interval Analysis
flg_computeCVs 							= False 		# UPDATED
flg_hist_CVs_all 						= False 		# UPDATED
flg_hist_CVs_split 						= False 			# TO DO
flg_compute_plot_ISIs 					= False 		# UPDATED
#
# Flags for the EM (inference & learning) algorithm.
verbose_EM 								= False
flg_ignore_Zeq0_inference  				= True 
flg_extract_spikeWords 					= True 		# WORKING HERE !!
flg_save_SW 							= True
flg_do_EM 								= True
flg_save_learned_model_EM 				= True
#
# Pairwise Spike Train Analysis: Cross Correlation Histograms, Pearson Correlation Coefficient & Covariance
flg_compute_XCorr_histograms 			= False 			# TO DO
flg_compute_XCorr_histograms_HPP		= False 			# TO DO # GET RID OF THIS AND ENCORPORATE IT INTO FLAG ABOVE.
flg_plot_XCorr_hists 					= False 			# TO DO
flg_compute_corrcoeff_and_covariance 	= False 			# TO DO
flg_plot_corrcoeff_trial_avg 			= False 			# TO DO
flg_plot_corrcoeff_vs_dist_pairwise		= False 			# TO DO
flg_plot_covariance_trial_avg 			= False 			# TO DO
#
# Find Unitary Events & do analysis of unique Spike Words
flg_UnitaryEventsAnalysis	= False 			# TO DO
flg_UEA_sw_len_histograms 	= False 			# TO DO
#
flg_swSim_stim				= [] 				# TO DO  	# ['wNoise','NatMov'] # a list including which stim you want to look at.
flg_similarity_func			= ['Kulczynski'] 	# TO DO 		# ['bitAND', 'Jaccard', 'Kulczynski' ] # a list including the similarity functions to consider.
flg_TH_similarity			= [2]	 			# TO DO 			# [2,1,0]  # number & value of thresholds on the bitwise AND matrix. 
flg_swSim_sparse			= False  			# TO DO
#
# Different clusterings for unique (or maybe all?) spike-words.
#	Many are from sklearn clustering toolbox.
#	Can quantify their performance by sklearn clustering metrics toolbox.
flg_cluster_sw_kmeans 			= False 			# TO DO 	# [Implemented.] : Non-binary cluster centers can be interpreted as probability of cell being active.
flg_cluster_sw_DBSCAN			= False 			# TO DO 	# [Implemented.] : I think, but slow and a memory hog
flg_cluster_sw_agglomerative	= False 			# TO DO 	# [Implemented.] : I think, but slow and a memory hog
#
flg_cluster_sw_KMedoids			= False 			# TO DO 	# like k-means but uses exemplar from data (for binary) - But we dont want to limit ourselves to exemplars as cluster centers I dont think.
flg_cluster_sw_LSH				= False 			# TO DO
flg_cluster_sw_FLANN			= False 			# TO DO 
#
# #											# None of these below implemented yet.
#
flg_cluster_swSim_CuthillMcKee	= False 			# TO DO
flg_plot_swSim_CuthillMcKee		= False 			# TO DO
flg_cluster_swSim_DBSCAN		= False 			# TO DO		# DBSCAN can be run on spike words directly or on Similarity matrix I think
flg_cluster_swSim_agglomerative	= False 			# TO DO
#
# #											# None of these below implemented yet.
#
flg_cluster_sw_hopfield			= False 			# TO DO 	# Look at Hillars examples. Need to use python 2.7
flg_cluster_sw_sparseCode		= False 			# TO DO
flg_cluster_sw_sparseCodePos	= False 			# TO DO
flg_cluster_sw_ProbGenModel		= False 			# TO DO
# # OTHERS?
#sklearn.mixture.BayesianGaussianMixture([…])	Variational Bayesian estimation of a Gaussian mixture.
#sklearn.mixture.GaussianMixture([n_components, …])	Gaussian Mixture.

flg_cluster_metrics = False 			# TO DO


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # Set up params, directories and load in spikes # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# Name of data files saved from my preprocessing in MATLAB of raw datafiles from Greg Field's Lab.
cell_type = 'OffBriskTransient'
GField_spikeData_File = str(cell_type + '_spikeTrains_CellXTrial.mat')
GField_STRFdata_File  = str('STRF_fits_' + cell_type + '_55cells.mat')

GField_GLMsimData_File = str('glm_cat_sim_fullpop_v2.mat')


## (1). Set up directories and create dirs if necessary.
dirHome, dirScratch = dm.set_dir_tree()
#

if flg_swSim_sparse:
	dirData_swSim = str( dirScratch + 'data/python_data/SpikeWordSimilarity/' )
	if not os.path.exists(dirData_swSim):
		os.makedirs(dirData_swSim)	
#
if flg_extract_spikeWords:
	SW_extracted_Dir = str( dirScratch + 'data/python_data/PGM_analysis/Greg_retina_data/SpikeWordsExtracted/')
	if not os.path.exists(SW_extracted_Dir):
		os.makedirs(SW_extracted_Dir)
#
if flg_do_EM:
	EM_learning_Dir = str( dirScratch + 'data/python_data/PGM_analysis/Greg_retina_data/Models_learned_EM/')
	if not os.path.exists(EM_learning_Dir):
		os.makedirs(EM_learning_Dir)
#
if flg_instantaneousRate_Eleph or flg_meanRate_Eleph or flg_scatter_meanRate_trialAvg_Eleph:
	SpikeRateDir = str( dirScratch + 'figs/spike_rate_plots/')
	if not os.path.exists(SpikeRateDir):
		os.makedirs(SpikeRateDir)	
#
if flg_computeCVs or flg_compute_plot_ISIs: 
	ISIdir = str( dirScratch + 'figs/ISI_analysis/')
	if not os.path.exists(ISIdir):
		os.makedirs(ISIdir)	

#
if flg_plot_XCorr_hists or flg_plot_corrcoeff_trial_avg or flg_plot_corrcoeff_vs_dist_pairwise	or flg_plot_covariance_trial_avg: 
	XCorrDir = str( dirScratch + 'figs/XCorr_plots/')
	if not os.path.exists(XCorrDir):
		os.makedirs(XCorrDir)







# # # # # # #
# Load in GLM simulation results of RGCs responding to natural movie stimuli
#
GLM = io.loadmat( str(dirScratch + 'data/GField_data/' + GField_GLMsimData_File) ) # forPython.mat') )
#
# OTHER VARS IN THERE.
#'cat_indpt_spike_preds'
#'cat_psth_indpt_perf'
#'cat_sim_triggers'
#
instFR = GLM['cat_indpt_inst_fr'][0] # Instantaneous firing rate from GLM simulation in ms bins of all 137 offBT, onBT and offBS cells.
#
instFR.shape # = (137,)
instFR[0].shape # = (200,6000) (trials,ms_time_bins)
# Questions: 1. Are these ms time bins or something smaller?
#			 2. values inside are in spikes per second? (some values > 1000). How to convert to p(y=1)?












## (2). Load in ndarray of spiketimes for 2 types of stimulus. They are of size: (#cells x #trials x #spikes)
spikes = io.loadmat( str(dirScratch + 'data/matlab_data/' + GField_spikeData_File) ) # forPython.mat') )
spikesInCat = spikes['spikesInCat']
spikesInWn = spikes['spikesInWn']
del spikes
#
numCells  = spikesInCat.shape[0]
numTrials = spikesInCat.shape[1] #  # note: the minus 1 at the end is because that last trial is weird (has spikes out to 15sec)
# 
for T in range(numTrials):
	for c in range(numCells):
		spikesInCat[c][T] = spikesInCat[c][T][0] # getting rid of another dimension or some ndarray nested in an ndarray.
		spikesInWn[c][T] = spikesInWn[c][T][0]
#
N = numCells # number of cells (length of binary Y-vector)
M_mod = np.round(model_CA_overcompleteness*N) # # number of Cell Assemblies (length of binary Z-vector)




## (3). Load in STRF parameters (Gaussian fits and temporal profiles. Compute distance matrix between pairwise RFs)
cellRFs = io.loadmat( str(dirScratch + 'data/matlab_data/' + GField_STRFdata_file) )
STRFtime = cellRFs['STRF_TimeParams']
STRFgauss = cellRFs['STRF_GaussParams']  # [Amplitude, x_mean, x_width, y_mean, y_width, orientation_angle]
del cellRFs
#
# Compute pairwise distance between Receptive Fields in microns on retina
pix2ret = 15*4 # microns. (15 monitor pixels per movie pixel that we have & 4 microns on retina per monitor pixel). Info from G. Field.
distRFs = pix2ret*sp.spatial.distance_matrix( np.array([STRFgauss[:,1], STRFgauss[:,3]]).T, np.array([STRFgauss[:,1], STRFgauss[:,3]]).T )

















## (0). Some parameters I may use for particular analyses below
numMS = 5500  # Each data collection / stimulus presentation is ~5.5 seconds.
szHistBins = 1    # for Histograms
#
cellsToLookAtInstRate = np.array([0,15,23,50]) # random choice of cells to plot instantaneous spike rate for.
instRateSampPd = 5 # (ms) in Elephant doc (pg 13): "Time stamp resolution of the spike times. The same resolution will be assumed for the kernel."
#	


whichStim = [0,1] # list of which stimuli to process. Numbers references into stims = ['NatMov', 'Wnoise', 'HPP', 'HGP', 'CPP', 'SIP']
#
# Parameter initializations for EM algorithm to learn model parameters
model_CA_overcompleteness 	= 2 # how many times more cell assemblies we have than cells (1 means complete - N=M, 2 means 2x overcomplete)
params_init 			= 'NoisyConst' 	# Options: {'True', 'RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise) }
C_noise 				= np.array([4/M_mod, 1, 1 ])		#[q, pi, pia] Mean of parameters for 'NoisyConst' initializations. Scalar between 0 and 1. 1 means almost silent. 0 means almost always firing.
sig_init 				= np.array([0.01, 0.05, 0.05 ])		# STD on parameter initializations from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
#
learning_rate 			= 1e0
lRateScale_Pi 			= 1		# Multiplicative scaling to Pi learning rate. If set to zero, Pi taken out of model essentially.
#
num_EM_Samples 			= 100		# number of steps to run full EM algorithm - alternating Inference and Learning.

pjt_tol = 10

SW_bins = [1] # [0, 1, 2 ] # ms. Build up spikewords from groups of cells in the same trial that fire within SW_bins of eachother.





stims = ['NatMov', 'Wnoise', 'HPP', 'HGP', 'CPP', 'SIP']
colorsForPlot = ['blue','green','red','cyan','magenta','yellow','black']




















# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # Look at Mean & Instantaneous Spike Rates # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #





# (4). Compute Instantaneous Spike Rate Profile using built-in Elephant function
if flg_instantaneousRate_Eleph:	
	InstRate_file_path = str(SpikeRateDir + cell_type + '_instantRates_Eleph_trialAvg_' + str(instRateSampPd) + 'msBins_wNoise_v_NatMov_cells' + str(cellsToLookAtInstRate) + '.png')
	#
	if os.path.isfile(InstRate_file_path):
		print( str('Figure ' + InstRate_file_path + ' already exists. Not reconstructing') )
	else:
		el.plot_instSpkRate_fewCells(spikesInCat, spikesInWn, instRateSampPd, cellsToLookAtInstRate, numMS, InstRate_file_path)



# (5). Compute Mean Spike Rate using built-in Elephant function
if flg_meanRate_Eleph:
	fRateCat, fRateWn = el.compute_mean_spike_rate(spikesInCat, spikesInWn, numMS)
	
	# Plot average spike rates across all trials for each cell & each stim.
	if flg_scatter_meanRate_trialAvg_Eleph:
		MeanRate_file_path = str(SpikeRateDir + cell_type + '_meanRatesScatter_Eleph_trialAvg_wNoise_v_NatMov_all' + str(numCells) + 'cells.png')
		#
		if os.path.isfile(MeanRate_file_path):
			print( str('Figure ' + MeanRate_file_path + ' already exists. Not reconstructing') )
		else:
			el.errorbar_meanSpkRate_trialAvg(fRateCat, fRateWn, MeanRate_file_path)


# (6). This function from the Elephant toolbox generates spiketrains from different null models.
if flg_generateSpikeTrains_Eleph: # These mostly work, but not using them right now. Come back to it.
	fRateAll = np.vstack( ( fRateCat, fRateWn ) )
	
	# (a). homogeneous_poisson_process
	if flg_HPP_gen:
		t0 = time.time()
		spiketrain_HPP = [ stgen.homogeneous_poisson_process(rate=1000*fRateAll.mean()*Hz, t_start=0*ms, t_stop=5500*ms ) for i in range(numCells*numTrials) ]
		t1 = time.time()
		print(t1-t0)
		#

	# (b). compound_poisson_process (CPP's amplitude distribution. A[j] represents the probability of a synchronous event of size j among the generated spike trains. The sum over all entries of A must be equal to one.)
	if flg_CPP_gen:
		t0 = time.time()
		A = np.array([0.35,0.25,0.2,0.1,0.1])
		spiketrain_CPP = [ stgen.compound_poisson_process(rate=1000*np.mean(fRateAll)*Hz, A=A, t_start=0*ms, t_stop=5500*ms, shift=None ) for i in range(numCells*numTrials) ]
		t1 = time.time()
		print(t1-t0)
		#

	# (c). homogeneous_gamma_process (a = gamma shape parameter, b = rate parameter)
	if flg_HGP_gen:
		t0 = time.time() 					
		HGP_shape=2		
		spiketrain_HGP = [ stgen.homogeneous_gamma_process(a=HGP_shape , b=1000*np.mean(fRateAll)*Hz, t_start=0*ms, t_stop=5500*ms ) for i in range(numCells*numTrials) ]
		t1 = time.time()
		print(t1-t0)
		#

	# (d). single_interaction_process
	if flg_SIP_gen:
		t0 = time.time()
		spiketrain_SIP = [ stgen.single_interaction_process(rate=1000*fRateAll.mean()*Hz, rate_c=0.01*1000*fRateAll.mean()*Hz, t_start=0*ms, t_stop=5500*ms ) for i in range(numCells*numTrials) ]
		t1 = time.time()
		print(t1-t0)
		#


	




# (7). Plot spike trains (just for sanity checking, dont really want to make figures.)
if flg_raster_spikeTrains:

	if whichStim==0:
		spiketrain = spikesInCat
	elif whichStim==1:
		spiketrain = spikesInWn
	else:
		#spiketrain = DO SOMETHING WITH HPP HERE.
		print('DO SOMETHING WITH HPP HERE')

	for j in range(numTrials):
		for i in range(numCells):
			t=spiketrain[i][j]
			plt.plot(t, i * np.ones_like(t), 'k.', markersize=2)

		plt.axis('tight')
		plt.xlim(0, 5500)
		plt.xlabel('Time (ms)', fontsize=16)
		plt.ylabel('Spike Train Index', fontsize=16)
		plt.title( str( 'Trial # ' + str(j) ), fontsize=18)
		plt.gca().tick_params(axis='both', which='major', labelsize=14)
		plt.show()







## (4). Compute and plot coefficient of variation (CV) from interspike intervals (ISIs) distributions
# 		Recall CV = std/mn. Poisson firing CV = 1.  CV > 1 means less regular than Poisson. Do it for:
#		(a) all cells & all trials; 
#		(b) all cells in each trial; All trials for each cell (all on same figure too.)
if flg_computeCVs:



	# CVs_Wn = [cv(isi(spiketrain)) for spiketrain in list(spikesInWn)]
	# CVs_Wn = np.asarray(CVs_Wn)

	t0 = time.time()
	CVs_Wn=np.zeros([numCells,numTrials])
	CVs_Cat=np.zeros([numCells,numTrials])
	for j in range(numTrials):
		for i in range(numCells):
			CVs_Wn[i,j] = cv(isi(spikesInWn[i][j]),axis=1)
			CVs_Cat[i,j] = cv(isi(spikesInCat[i][j]),axis=1)
	t1 = time.time()
	print(t1-t0)		
	#
	all_CVs = np.concatenate( (CVs_Cat.flatten(), CVs_Wn.flatten()) )	
	if flg_HPP_gen:
		CVs_HPP = [cv(isi(spiketrain)) for spiketrain in spiketrain_HPP]
		CVs_HPP = np.asarray(CVs_HPP)
		all_CVs = np.concatenate( (all_CVs, CVs_HPP) )
	if flg_HGP_gen:
		CVs_HGP = [cv(isi(spiketrain)) for spiketrain in spiketrain_HGP]
		CVs_HGP = np.asarray(CVs_HGP)		
		all_CVs = np.concatenate( (all_CVs, CVs_HGP) )
	if flg_CPP_gen:
		CVs_CPP = [cv(isi(spiketrain)) for spiketrain in spiketrain_CPP]
		CVs_CPP = np.asarray(CVs_CPP)		
		all_CVs = np.concatenate( (all_CVs, CVs_CPP) )
	if flg_SIP_gen:	
		CVs_SIP = [cv(isi(spiketrain)) for spiketrain in spiketrain_SIP]
		CVs_SIP = np.asarray(CVs_SIP)		
		all_CVs = np.concatenate( (all_CVs, CVs_SIP) )
	#		
	maxCV = all_CVs.max()
	minCV = all_CVs.min()
	del all_CVs
	#
	# (a). show CV histogram for all cells in all trials
	if flg_hist_CVs_all:
		CV_hists_file_path = str(ISIdir + cell_type + '_CV_hists_allTogether_wNoise_v_natMov_v_nullModels.png')
		if os.path.isfile(CV_hists_file_path):
			print( str('Figure ' + CV_hists_file_path + ' already exists. Not reconstructing') )
		else:
			numBins = 100
			bins  = np.linspace( minCV, maxCV, numBins ) 
			f, ax = plt.subplots(1, 1)
			a,_ = np.histogram(CVs_Cat,bins)
			ax.plot(bins[:-1],a,label='Nat Mov')
			#
			a,_ = np.histogram(CVs_Wn,bins)
			ax.plot(bins[:-1],a,label='Wht Noiz')
			#
			if flg_HPP_gen:
				a,_ = np.histogram(CVs_HPP,bins)
				ax.plot(bins[:-1],a,label='HPP')
			if flg_HGP_gen:
				a,_ = np.histogram(CVs_HGP,bins)
				ax.plot(bins[:-1],a,label=str('HGP w/ \gamma='+str(HGP_shape)) )
			if flg_CPP_gen:
				a,_ = np.histogram(CVs_CPP,bins)
				ax.plot(bins[:-1],a,label='CPP')
			if flg_SIP_gen:	
				a,_ = np.histogram(CVs_SIP,bins)
				ax.plot(bins[:-1],a,label='SIP')	


			ax.set_xlabel('Coefficient of Variation', fontsize=16)
			ax.set_ylabel('count', fontsize=16)
			ax.legend()
			ax.grid()
			ax.set_title( str(cell_type + 'CV across all cells & trials'), fontsize=18 )
			#plt.gca().tick_params(axis='both', which='major', labelsize=14)
			f.set_size_inches(14,8)
			f.savefig( CV_hists_file_path )


			#			pf.plot_histograms(f, ax, all_CVs, bins)
			#			plt.legend(stims)
	#
	# (b). show CV histograms on 4 different subplots contrasting White Noise vs. Natural Movie
	#							               Histograms binned across Cells vs. Across Trials
	### SKIPPING OVER THIS FOR NOW.
	if flg_hist_CVs_split:
		CV_split_hists_file_path = str(ISIdir + cell_type + '_CV_hists_split_vTrials_and_vCells_wNoise_v_natMov')
		if os.path.isfile(CV_split_hists_file_path):
			print( str('Figure ' + CV_split_hists_file_path + ' already exists. Not reconstructing') )
		else:
			numBinsCells=50
			bins_cells  = np.linspace( np.min(all_CVs), np.max(all_CVs), numBinsCells ) 
			numBinsTrials=15
			bins_trials = np.linspace( np.min(all_CVs), np.max(all_CVs), numBinsTrials ) 
			#
			f, ax = plt.subplots(2, 2) 
			pf.plot_histograms(f, ax[0][0], CVs_Wn, bins_cells)
			pf.plot_histograms(f, ax[0][1], CVs_Wn.T, bins_trials)
			pf.plot_histograms(f, ax[1][0], CVs_Cat, bins_cells)
			pf.plot_histograms(f, ax[1][1], CVs_Cat.T, bins_trials)
			#
			ax[0][0].set_ylabel('Stim = White Noise', fontsize=18 )
			ax[1][0].set_ylabel('Stim = Natural Movie', fontsize=18 )
			#
			ax[0][0].set_title( str('Hist over all ' + str(numTrials) + ' trials for each of ' + str(numCells) + ' cells'), fontsize=18 )
			ax[0][1].set_title( str('Hist over all ' + str(numCells) + ' cells for each of ' + str(numTrials) + ' trials'), fontsize=18 )
			#
			ax[1][1].set_xlabel('CV', fontsize=16 )
			ax[1][1].set_ylabel('counts', fontsize=16 )
			#
			f.suptitle('Coefficient of Variation Histograms',fontsize=20)
			f.set_size_inches(14,8)
			f.savefig( CV_split_hists_file_path )
	



## (5). Plot ISI distributions
if flg_compute_plot_ISIs:
	ISIs_file_path = str(ISIdir + cell_type + '_ISI_hists_allTogether_wNoise_v_natMov')
	if os.path.isfile(ISIs_file_path):
		print( str('Figure ' + ISIs_file_path + ' already exists. Not reconstructing') )
	else:
		ISIs_Wn = np.array([])
		ISIs_Cat = np.array([])
		for j in range(numTrials):
			for i in range(numCells):
				ISIs_Wn = np.append(ISIs_Wn, isi(spikesInWn[i][j])[0] )
				ISIs_Cat = np.append(ISIs_Cat, isi(spikesInCat[i][j])[0] )
		#
		all_ISIs = np.concatenate( (ISIs_Cat.flatten(), ISIs_Wn.flatten()) )		
		if flg_HPP_gen:
			ISIs_HPP = [isi(spiketrain) for spiketrain in spiketrain_HPP]
			ISIs_HPP = np.concatenate(ISIs_HPP).ravel()
			ISIs_HPP = np.asarray(ISIs_HPP)
			all_ISIs = np.concatenate( (all_ISIs, ISIs_HPP) )
		if flg_HGP_gen:
			ISIs_HGP = [isi(spiketrain) for spiketrain in spiketrain_HGP]		
			ISIs_HGP = np.concatenate(ISIs_HGP).ravel()
			ISIs_HGP = np.asarray(ISIs_HGP)
			all_ISIs = np.concatenate( (all_ISIs, ISIs_HGP) )
		if flg_CPP_gen:
			ISIs_CPP = [isi(spiketrain) for spiketrain in spiketrain_CPP]
			ISIs_CPP = np.concatenate(ISIs_CPP).ravel()
			ISIs_CPP = np.asarray(ISIs_CPP)		
			all_ISIs = np.concatenate( (all_ISIs, ISIs_CPP) )
		if flg_SIP_gen:	
			ISIs_SIP = [isi(spiketrain) for spiketrain in spiketrain_SIP]	
			ISIs_SIP = np.concatenate(ISIs_SIP).ravel()
			ISIs_SIP = np.asarray(ISIs_SIP)	
			all_ISIs = np.concatenate( (all_ISIs, ISIs_SIP) )	
			#		
		maxISI = all_ISIs.max()
		minISI = all_ISIs.min()
		del all_ISIs	
		#
		numBins = 100		
		bins = np.linspace(0,maxISI,numBins)
		f, ax = plt.subplots(1, 1)
		a,_ = np.histogram(ISIs_Cat,bins)
		ax.plot(bins[:-1],a,label='NatMov')
		#
		a,_ = np.histogram(ISIs_Wn,bins)
		ax.plot(bins[:-1],a,label='Wnoise')
		#
		if flg_HPP_gen:
			a,_ = np.histogram(ISIs_HPP,bins)
			ax.plot(bins[:-1],a,label='HPP')
		if flg_HGP_gen:
			a,_ = np.histogram(ISIs_HGP,bins)
			ax.plot(bins[:-1],a,label=str('HGP w/ \gamma='+str(HGP_shape)) )
		if flg_CPP_gen:
			a,_ = np.histogram(ISIs_CPP,bins)
			ax.plot(bins[:-1],a,label='CPP')
		if flg_SIP_gen:	
			a,_ = np.histogram(ISIs_SIP,bins)
			ax.plot(bins[:-1],a,label='SIP')

		ax.set_xlabel('Inter-Spike Interval (ms)', fontsize=16)
		ax.set_ylabel('count', fontsize=16)
		ax.legend()
		ax.grid()
		ax.set_title( str(cell_type + ' ISI across all cells & trials'), fontsize=18 )
		#plt.gca().tick_params(axis='both', which='major', labelsize=14)
		f.set_size_inches(14,8)
		f.savefig( ISIs_file_path )



## (6). - Calculate and save cross correlation histograms for all spike train pairs averaged over all trials.. 
if flg_compute_XCorr_histograms:
	whichStim = 0 # 0 = natMov, 1 = wNoise, 2 = HPP
	XCorr_histogram_file_path = str(dirDataOut + cell_type + '_CrossCorrelationHistograms_' + str(szHistBins) + 'msBins_meanOverTrials_' + stims[whichStim] + '.npz')

	if os.path.isfile(XCorr_histogram_file_path):
		data = np.load(XCorr_histogram_file_path)
		CCH_means = data['arr_0']
		binIDs = data['arr_1']
		stim = data['arr_2']
		del data

	else:
		numBins = np.int(numMS*2/szHistBins - 1)
		CCH_means = np.ndarray( (numCells, numCells, numBins) )

		if whichStim==0:
			spikesInData = spikesInCat
		elif whichStim==1:
			spikesInData = spikesInWn
		else:
			#spikesInData = DO SOMETHING WITH HPP HERE.
			print('DO SOMETHING WITH HPP HERE')


		for i in range(numCells):
			for j in range(numCells):

				print( (i, j) )
				CCH = np.ndarray( (numTrials,numBins) )

				for k in range(numTrials):

					st1 = neo.SpikeTrain( spikesInData[i][k]*ms, t_stop=numMS*ms)
					st2 = neo.SpikeTrain( spikesInData[j][k]*ms, t_stop=numMS*ms)
					binned_st1 = BinnedSpikeTrain(st1, binsize=szHistBins*ms)	
					binned_st2 = BinnedSpikeTrain(st2, binsize=szHistBins*ms)	

					cch,binIDs = cross_correlation_histogram(binned_st1, binned_st2, window='full', border_correction=False, 
															binary=False, kernel=None, method='speed', cross_corr_coef=False)

					CCH[k] = cch.as_array().reshape(numBins,)

				CCH_means[i][j] = np.mean(CCH,axis=0)	# number of spikes that line up in 1 trial (on average) if I shift one spiketrain relative to the other by x msec.
				#plt.plot( binIDs, np.mean(CCH,axis=0) )
				#plt.show()

		np.savez( XCorr_histogram_file_path, CCH_means, binIDs, stim )		





## (6b). - Calculate and save cross correlation histograms for spikeTrain pairs generated from homogeneous_poisson_process
# 			TO DO ::: STILL WORKING HERE ::: DO LATER ...
#			INCLUDE THIS IN 6A ABOVE !!!
if flg_compute_XCorr_histograms_HPP:
	
	numBins = np.int(numMS*2/szHistBins - 1)
	CCH_means = np.ndarray( (numCells, numCells, numBins) )

	spikesInData = spikesInCat # spikesInCat or spikesInWn
	stim = 'natMov' # 'natMov' or 'wNoise'

	for i in range(numCells):
		for j in range(numCells):

			print( (i, j) )
			CCH = np.ndarray( (numTrials,numBins) )

			for k in range(numTrials):

				binned_st1= BinnedSpikeTrain(st1, binsize=szHistBins*ms)

				cch,binIDs = cross_correlation_histogram(binned_st1, binned_st2, window='full', border_correction=False, 
														binary=False, kernel=None, method='speed', cross_corr_coef=False)

				CCH[k] = cch.as_array().reshape(numBins,)

			CCH_means[i][j] = np.mean(CCH,axis=0)	# number of spikes that line up in 1 trial (on average) if I shift one spiketrain relative to the other by x msec.
			#plt.plot( binIDs, np.mean(CCH,axis=0) )
			#plt.show()

	np.savez( str(dirDataOut + cell_type + '_CrossCorrelationHistograms_meanOverTrials_' + stim), CCH_means, binIDs, stim )		





## (8). Explore and plot CCH for cell pairs averaged across
#	    NOTE: CCH is the number of spikes that line up in 1 trial (on average) if I shift one spiketrain relative to the other.
#		TO DO: PUT THIS INTO 6A ALSO, LOAD IN BOTH CAT & WN & MAYBE HPP ALSO FOR THE PLOT.
#		ALSO: THINK ABOUT HOW TO PLOT THIS STUFF BETTER TO COMPARE THEM.
if flg_plot_XCorr_hists:
	stim = 'NatMov' # 'NatMov' or 'wNoise'
	dataCat = np.load( str(dirDataOut + cell_type + '_CrossCorrelationHistograms_' + str(szHistBins) + 'msBins_meanOverTrials_' + stim + '.npz') )
	CCH_means_Cat = dataCat['arr_0']
	binsCCH_Cat = dataCat['arr_1']
	stim_Cat = dataCat['arr_2']
	del dataCat

	stim = 'wNoise' # 'natMov' or 'wNoise'
	dataWn = np.load( str(dirDataOut + cell_type + '_CrossCorrelationHistograms_' + str(szHistBins) + 'msBins_meanOverTrials_' + stim + '.npz') )
	CCH_means_Wn = dataWn['arr_0']
	binsCCH_Wn = dataWn['arr_1']
	stim_Wn = dataWn['arr_2']
	del dataWn

	#
	cell_ref = 0
	cell_beg = 1
	cell_end = 2 #numCells
	ms_xaxis = numMS-1
	#
	#h, ax = plt.subplots(1,2)
	#ax[0]

	plt.plot( binsCCH_Cat, CCH_means_Cat[cell_ref][range(cell_beg,cell_end)].T, color='red' )
	plt.plot( binsCCH_Wn, CCH_means_Wn[cell_ref][range(cell_beg,cell_end)].T, color='blue'  )
	plt.xlim(0, ms_xaxis)
	plt.xlabel('relative time of spikes (msec)')
	plt.ylabel('# of coincident spikes (mean across trials)')
	plt.title( str('CCH between cell #' + str(cell_ref) + ' and others with ' + stim + ' and ' + str(szHistBins) + 'msBins' ) )
	plt.legend( stims ) # range(cell_beg,cell_end)
	plt.rc('text', usetex=False)
	plt.grid()
	plt.show()

	# STILL WORKING HERE...











## (7). Compute Pearson Correlation Coefficient (corrcoef) and Covariance between spike train pairs using built-in Elephant functions.
#		This will compute a (#Cells)x(#Cells) matrix for each trial. We will then average across trials to get a single (#Cells)x(#Cells) matrix
#		of Covariance and one for Pearson CC for each stimulus type
#		NOTE: This can & should be done in 6 - when we compute cross_correlation_histograms.
if flg_compute_corrcoeff_and_covariance:
	PearsonCC_Covar_file_path = str(dirDataOut + cell_type + '_PearsonCC_and_Covariance_' + str(szHistBins) + 'msBins_wNoise_and_natMov.npz')

	if os.path.isfile(PearsonCC_Covar_file_path):
		data = np.load(PearsonCC_Covar_file_path)
		cc_mat_Cat = data['arr_0']
		covar_mat_Cat = data['arr_1'] 
		cc_mat_Wn = data['arr_2'] 
		covar_mat_Wn = data['arr_3'] 
		numTrials = data['arr_4'] 
		numCells = data['arr_5']
		del data

	else:
		cc_mat_Cat = np.zeros([numTrials,numCells,numCells])
		covar_mat_Cat = np.zeros([numTrials,numCells,numCells])
		cc_mat_Wn = np.zeros([numTrials,numCells,numCells])
		covar_mat_Wn = np.zeros([numTrials,numCells,numCells])

		# Try computing these not averaging acrods trials but treating spike train as one long thing.
		spike_trains = []
		for k in range(numTrials):
			print(k)
			xx = spikesInCat.T
			spike_trains.append( list(k*numMS + xx[k][0][0].T) )
			#
			flat_list = [item for sublist in l for item in sublist]
			flattened_list = [y for x in list_of_lists for y in x]

		for k in range(numTrials):
			print(k)
			xx = spikesInCat.T
			stxx = [neo.SpikeTrain( xx[k][i]*ms, t_stop=numMS*ms) for i in range(numCells)]
			cc_mat_Cat[k][:][:] = corrcoef(BinnedSpikeTrain(stxx, binsize=szHistBins*ms))
			covar_mat_Cat[k][:][:] = covariance(BinnedSpikeTrain(stxx, binsize=szHistBins*ms)) 
			#
			xx = spikesInWn.T
			stxx = [neo.SpikeTrain( xx[k][i]*ms, t_stop=numMS*ms) for i in range(numCells)]
			cc_mat_Wn[k][:][:] = corrcoef(BinnedSpikeTrain(stxx, binsize=szHistBins*ms))
			covar_mat_Wn[k][:][:] = covariance(BinnedSpikeTrain(stxx, binsize=szHistBins*ms)) 

		np.savez( PearsonCC_Covar_file_path, cc_mat_Cat, covar_mat_Cat, cc_mat_Wn, covar_mat_Wn, numTrials, numCells )	
	


	if flg_plot_corrcoeff_trial_avg:
		CA_Xcorr_Avg_file_path = str(XCorrDir + cell_type + '_PearsonCC_trialAvg_' + str(szHistBins) + 'msBins_wNoise_v_natMov.png')
		if os.path.isfile(CA_Xcorr_Avg_file_path):
			print( str('Figure ' + CA_Xcorr_Avg_file_path + ' already exists. Not reconstructing') )
		else:
			# compute mean & std across trials of Pearson Correlation Coefficient.
			zeroOutDiag = 1 - np.diag( np.ones(numCells) )
			cc_mat_Cat_trialMean	= np.multiply( np.mean(cc_mat_Cat,axis=0), zeroOutDiag )
			cc_mat_Cat_trialStd 	= np.std(cc_mat_Cat,axis=0)
			cc_mat_Wn_trialMean 	= np.multiply( np.mean(cc_mat_Wn,axis=0), zeroOutDiag )
			cc_mat_Wn_trialStd 		= np.std(cc_mat_Wn,axis=0)
			cc_allStim_trialMean	= np.concatenate( ( cc_mat_Cat_trialMean.flatten(), cc_mat_Wn_trialMean.flatten() ) )	
			cc_allStim_trialStd		= np.concatenate( ( cc_mat_Cat_trialStd.flatten(), cc_mat_Wn_trialStd.flatten() ) )
			f,ax = plt.subplots(2,2)
			plt.rc('text', usetex=True)
			#
			hcc=ax[0][0].imshow( cc_mat_Cat_trialMean, vmin=np.min(cc_allStim_trialMean), vmax=np.max(cc_allStim_trialMean) )
			hco=ax[1][0].imshow( cc_mat_Cat_trialStd,  vmin=np.min(cc_allStim_trialStd),  vmax=np.max(cc_allStim_trialStd)  )
			hcc=ax[0][1].imshow( cc_mat_Wn_trialMean,  vmin=np.min(cc_allStim_trialMean), vmax=np.max(cc_allStim_trialMean) )
			hco=ax[1][1].imshow( cc_mat_Wn_trialStd,   vmin=np.min(cc_allStim_trialStd),  vmax=np.max(cc_allStim_trialStd)  )
			#
			ax[0][0].set_title( str('Stim = Natural Movie'), fontsize=18 )
			ax[0][1].set_title( str('Stim = White Noise'), fontsize=18 )
			ax[0][0].set_ylabel('$\mu$ c.c.', fontsize=16 )
			ax[1][0].set_ylabel('$\sigma$ c.c.', fontsize=16 )
			ax[1][1].set_xlabel('cell \#', fontsize=14 )
			ax[1][1].set_ylabel('cell \#', fontsize=14 )
			#
			f.subplots_adjust(right=0.8)
			cbar_ax_cc = f.add_axes([0.85, 0.6, 0.05, 0.3])
			cbar_ax_cc.set_title('$\mu$',fontsize=14)
			f.colorbar(hcc, cax=cbar_ax_cc,)
			#
			cbar_ax_co = f.add_axes([0.85, 0.1, 0.05, 0.3])
			cbar_ax_co.set_title('$\sigma$',fontsize=14)
			f.colorbar(hco, cax=cbar_ax_co)
			#
			f.suptitle( str('Pearson Correlation Coefficient - (Averaged across ' + str(numTrials) + ' trials)  (t = ' + str(numMS) + ' ms)  (binSize = ' + str(szHistBins) + ' ms)'), fontsize=22 )
			#
			f.set_size_inches(14,8)
			f.savefig( CA_Xcorr_Avg_file_path )




	if flg_plot_covariance_trial_avg:
		CA_Covar_Avg_file_path = str( XCorrDir + cell_type + '_Covariance_trialAvg_' + str(szHistBins) + 'msBins_wNoise_v_natMov')
		if os.path.isfile(CA_Covar_Avg_file_path):
			print( str('Figure ' + CA_Covar_Avg_file_path + ' already exists. Not reconstructing') )
		else:
			# compute mean & std across trials of Covariance.
			zeroOutDiag = 1 - np.diag( np.ones(numCells) )
			covar_mat_Cat_trialMean = np.multiply( np.mean(covar_mat_Cat,axis=0), zeroOutDiag )
			covar_mat_Cat_trialStd 	= np.std(covar_mat_Cat,axis=0)
			covar_mat_Wn_trialMean 	= np.multiply( np.mean(covar_mat_Cat,axis=0), zeroOutDiag )
			covar_mat_Wn_trialStd 	= np.std(covar_mat_Cat,axis=0)
			covar_allStim_trialMean	= np.concatenate( ( covar_mat_Cat_trialMean.flatten(), covar_mat_Wn_trialMean.flatten() ) )
			covar_allStim_trialStd	= np.concatenate( ( covar_mat_Cat_trialStd.flatten(), covar_mat_Wn_trialStd.flatten() ) )

			f,ax = plt.subplots(2,2)
			plt.rc('text', usetex=True)
			#
			hcc=ax[0][0].imshow( covar_mat_Cat_trialMean, vmin=np.min(covar_allStim_trialMean), vmax=np.max(covar_allStim_trialMean) )
			hco=ax[1][0].imshow( covar_mat_Cat_trialStd,  vmin=np.min(covar_allStim_trialStd),  vmax=np.max(covar_allStim_trialStd) )
			hcc=ax[0][1].imshow( covar_mat_Wn_trialMean,  vmin=np.min(covar_allStim_trialMean), vmax=np.max(covar_allStim_trialMean) )
			hco=ax[1][1].imshow( covar_mat_Wn_trialStd,	  vmin=np.min(covar_allStim_trialStd),  vmax=np.max(covar_allStim_trialStd) )
			#
			ax[0][0].set_title( str('Stim = Natural Movie'), fontsize=18 )
			ax[0][1].set_title( str('Stim = White Noise'), fontsize=18 )
			ax[0][0].set_ylabel('$\mu$ Covariance', fontsize=16 )
			ax[1][0].set_ylabel('$\sigma$ Covariance', fontsize=16 )
			ax[1][1].set_xlabel('cell \#', fontsize=14 )
			ax[1][1].set_ylabel('cell \#', fontsize=14 )
			#
			f.subplots_adjust(right=0.8)
			cbar_ax_cc = f.add_axes([0.85, 0.6, 0.05, 0.3])
			cbar_ax_cc.set_title('$\mu$',fontsize=14)
			f.colorbar(hcc, cax=cbar_ax_cc,)
			#
			cbar_ax_co = f.add_axes([0.85, 0.1, 0.05, 0.3])
			cbar_ax_co.set_title('$\sigma$',fontsize=14)
			f.colorbar(hco, cax=cbar_ax_co)
			#
			f.suptitle( str('Pairwise Covariance - (Averaged across ' + str(numTrials) + ' trials)  (t = ' + str(numMS) + ' ms)  (binSize = ' + str(szHistBins) + ' ms)'), fontsize=22 )
			#
			f.set_size_inches(14,8)
			f.savefig( CA_Covar_Avg_file_path )







	if flg_plot_corrcoeff_vs_dist_pairwise:
		#
		zeroOutDiag 			= 1 - np.diag( np.ones(numCells) )
		cc_mat_Cat_trialMean	= np.multiply( np.mean(cc_mat_Cat,axis=0), zeroOutDiag )
		cc_mat_Cat_trialStd 	= np.std(cc_mat_Cat,axis=0)
		cc_mat_Wn_trialMean 	= np.multiply( np.mean(cc_mat_Wn,axis=0), zeroOutDiag )
		cc_mat_Wn_trialStd 		= np.std(cc_mat_Wn,axis=0)
		#
		inds = np.argsort(distRFs.ravel(),axis=None)
		plt.scatter( distRFs.ravel()[inds], cc_mat_Cat_trialMean.ravel()[inds],s=10,c='r',label='Natural Movie')
		plt.scatter( distRFs.ravel()[inds], cc_mat_Wn_trialMean.ravel()[inds],s=10,c='b',label='White Noise')
		plt.grid()
		#
		plt.xlabel('Distance between RFs on retina ($\mu$ m)')
		plt.ylabel('Pearson Cross Correlation (p)')
		plt.title( str('Spiking Correlation (' + str(szHistBins) + 'ms bins) vs. Distance in Rat ' + cell_type + ' (a la Pitkow & Meister 2012)') )
		#
		plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # Unitary Events Analysis # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if flg_UnitaryEventsAnalysis:

	UEA_file_path = str(dirDataOut + cell_type + '_UEA_SpikeWordHashing_' + str(szHistBins) + 'msBins_wNoise_and_natMov.npz')

	if os.path.isfile(UEA_file_path):
		print('Loading ',UEA_file_path)
		data = np.load(UEA_file_path)
		swWn_hash 			= data['arr_0']
		swWn_time 			= data['arr_1']
		swWn_trial			= data['arr_2']
		swCat_hash 			= data['arr_3']
		swCat_time 			= data['arr_4']
		swCat_trial 		= data['arr_5']
		#
		swuWn_hash 			= data['arr_6']
		indx_swuWn_hash 	= data['arr_7']
		swuCat_hash 		= data['arr_8']
		indx_swuCat_hash	= data['arr_9']
		swWn_bool 			= data['arr_10']
		swuWn_bool 			= data['arr_11']
		swCat_bool 			= data['arr_12'] 
		swuCat_bool 		= data['arr_13']		

		 # Make sure these 4 variables I define match those recorded in data file.
		if not( (szHistBins 	== np.int(data['arr_14'][True][0])) & 
				(numTrials 	== np.int(data['arr_15'][True][0])) & 
				(numMS 		== np.int(data['arr_16'][True][0])) & 
				(numCells 	== np.int(data['arr_17'][True][0])) ):
			print('Warning: UEA params do not match expected values. Breaking')
			sys.exit()

		del data
		

	else:

		# Construct boolean array of spikes of all cells across all time points. Looping over all trials too.
		stWn_hash = np.zeros( (numTrials, numMS) )
		stCat_hash = np.zeros( (numTrials, numMS) )
		for k in range(numTrials):
			print('Trial',str(k))
			stWn = [ neo.SpikeTrain( spikesInWn[i][k]*ms, t_stop=numMS*ms) for i in range(numCells)]
			binned_stWn = BinnedSpikeTrain(stWn, binsize=szHistBins*ms).to_bool_array()
			stWn_hash[k] = uea.hash_from_pattern(binned_stWn, N=numCells) 
			#
			stCat = [ neo.SpikeTrain( spikesInCat[i][k]*ms, t_stop=numMS*ms) for i in range(numCells)]
			binned_stCat = BinnedSpikeTrain(stCat, binsize=szHistBins*ms).to_bool_array()
			stCat_hash[k] = uea.hash_from_pattern(binned_stCat, N=numCells) 
			#
		w = stWn_hash!=0
		swWn_hash = stWn_hash[w]	
		swWn_trial,swWn_time = np.where(w) # location of each spike word in (trial, time) space
		#
		c = stCat_hash!=0
		swCat_hash = stCat_hash[c]	
		swCat_trial,swCat_time = np.where(c) # location of each spike word in (trial, time) space
		
	
		# find unique spike words sorted and their indexes in spk_words (this is fast.)
		print('find unique spike words')
		swuWn_hash, indx_swuWn_hash = np.unique(swWn_hash,return_index=True)
		swuCat_hash, indx_swuCat_hash = np.unique(swCat_hash,return_index=True)



		# convert hashed spike words back to binary patterns! (this can take some time... ~1min)
		print('converting hash to binary =  ~1min')
		t0 = time.time()
		#
		swWn_bool 		 	= uea.inverse_hash_from_pattern(swWn_hash, N=numCells).astype(bool)
		swuWn_bool 			= uea.inverse_hash_from_pattern(swuWn_hash, N=numCells).astype(bool)
		#
		swCat_bool 			= uea.inverse_hash_from_pattern(swCat_hash, N=numCells).astype(bool)
		swuCat_bool 		= uea.inverse_hash_from_pattern(swuCat_hash, N=numCells).astype(bool)
		#
		t1 = time.time()
		print('Time = ',t1-t0)
		#
		print('Saving ',UEA_file_path)
		np.savez( UEA_file_path, swWn_hash, swWn_time, swWn_trial, swCat_hash, swCat_time, swCat_trial,
				swuWn_hash, indx_swuWn_hash, swuCat_hash, indx_swuCat_hash,
				swWn_bool, swuWn_bool, swCat_bool, swuCat_bool,
				szHistBins, numTrials, numMS, numCells )


	# number of cells involved in each spike word.
	lenSW_Wn  = swWn_bool.sum(axis=0).astype(np.uint8)
	lenSW_Cat = swCat_bool.sum(axis=0).astype(np.uint8)			
	#
	# number of cells involved in each unique spike word 
	lenSWU_Wn  = swuWn_bool.sum(axis=0).astype(np.uint8)
	lenSWU_Cat = swuCat_bool.sum(axis=0).astype(np.uint8)	
	#
	# number of spike words (total or unique)
	numSW_Cat 	= lenSW_Cat.size
	numSWU_Cat 	= lenSWU_Cat.size
	numSW_Wn 	= lenSW_Wn.size
	numSWU_Wn 	= lenSWU_Wn.size


	# Print the number of spike words (filtered in different ways) for both stimuli
	print('Number of Spike Words: For (White Noise, Natural Movie)')
	print( 'Including Single Cells:  ', numSW_Wn, numSW_Cat )
	print( 'Only unique Spike Words: ', numSWU_Wn, numSWU_Cat )





	if flg_UEA_sw_len_histograms:

		big = np.max(np.hstack([lenSW_Wn,lenSW_Cat])) # largest number of cells that participate in a spike word
		binsH = np.linspace(0,big,big)
		xhWn, y, z 		= plt.hist( lenSW_Wn,   binsH)
		xhCat, y, z 	= plt.hist( lenSW_Cat,  binsH)
		xhWnU, y, z 	= plt.hist( lenSWU_Wn,  binsH)
		xhCatU, y, z 	= plt.hist( lenSWU_Cat, binsH)
		plt.close()
		#
		f,ax = plt.subplots(3,1)
		plt.rc('text', usetex=True)
		ax[0].plot(binsH[1:], np.log10(xhWn), color='blue', linestyle='--', linewidth=2, marker='o')
		ax[0].plot(binsH[1:], np.log10(xhCat), color='red', linestyle='--', linewidth=2, marker='o')
		ax[0].plot(binsH[1:], np.log10(xhWnU), color='blue', linestyle='-', linewidth=2, marker='^')
		ax[0].plot(binsH[1:], np.log10(xhCatU), color='red', linestyle='-', linewidth=2, marker='^')
		ax[0].axis([0, 1.05*big, 0, 1.05*np.max(np.log10(xhWn-xhWnU))])
		ax[0].set_ylabel('log_{10}(\#Occ)', fontsize=16, fontweight='bold')
		ax[0].legend([str( 'wNoise All sw (\# = ' + str( xhWn.sum().astype(int) ) + ')'),
					  str( 'natMov All sw (\# = ' + str( xhCat.sum().astype(int) ) + ')'),
					  str( 'wNoise Unique (\# = ' + str( xhWnU.sum().astype(int) ) + ' )'),
					  str( 'natMov Unique (\# = ' + str( xhCatU.sum().astype(int) ) + ' )')])
		ax[0].grid()
		#
		ax[1].plot(binsH[1:], (xhWn-xhWnU), color='blue', linestyle='-', linewidth=2, marker='o')
		ax[1].plot(binsH[1:], (xhCat-xhCatU), color='red', linestyle='-', linewidth=2, marker='o')
		ax[1].axis([0, 1.05*big, 0, 1.05*np.max(xhWn-xhWnU)])
		ax[1].set_ylabel('$\Delta$ \#Occ Non-Unique', fontsize=16, fontweight='bold')
		ax[1].legend(['wNoise (All - Unique)','natMov'])
		ax[1].grid()
		#
			#
		ax[2].plot(binsH[1:], (xhCatU-xhWnU), color='green', linestyle='-', linewidth=2, marker='o')
		ax[2].plot(binsH[1:], np.ones_like(binsH[1:]), color='black', linestyle='--', linewidth=2)
		ax[2].axis([0, 1.05*big, 1.05*np.min(xhCatU-xhWnU), 1.05*np.max(xhCatU-xhWnU)])
		ax[2].set_xlabel('Number of neurons that participate in each Spike Word', fontsize=16, fontweight='bold')
		ax[2].set_ylabel('$\Delta$ \#Occ Unique', fontsize=16, fontweight='bold')
		ax[2].legend(['NatMov - wNoise'])
		ax[2].grid()
		#
		f.set_size_inches(14,8)
		f.suptitle( str('Histogram of Spike Word Length with ' + str(szHistBins) + 'ms time resolution'), fontsize=20, fontweight='bold' )
		f.savefig( str(dirRoot + 'cell_assembly_analysis/' + cell_type + '_UEA_SpikeWordLengthHistogram_' + str(szHistBins) + 'msBins_wNoise_v_natMov') )
		# MAYBE LOOK AT CDF INSTEAD OF HISTOGRAM?

		del xhCat, xhCatU, xhWn, xhWnU






	# Try some clustering directly in binary spikeword space
	if flg_cluster_sw_kmeans: 
		file_path = str(dirDataOut + 'Clustering/K-means/' + cell_type + '_Kmeans' + str(numClust) + '_' + str(szHistBins) + 'msBins_wNoise.npz')

		if os.path.isfile(file_path):
			print('Loading ',file_path)
			data = np.load(file_path)
			kmCenters = data['arr_0']
			kmLabels = data['arr_1']
			kmParams = data['arr_2']
			del data

		else:
			print('Compute Kmeans directly on binary spike words')	
			t = time.time()
			km = skc.KMeans(n_clusters=numClust, n_init=10, max_iter=300, tol=0.0001, verbose=True).fit(swuWn_bool.T)
			print( 'Time = ', time.time()-t )
			#
			kmLabels = km.labels_
			kmCenters = km.cluster_centers_
			kmParams = km
			np.savez(file_path, kmCenters, kmLabels, kmParams)
		#





		# predict clusters in a hold out set of data.
		# km.predict('hold-out-data')

		kmeans_analyze = False
		if kmeans_analyze:
			# Compute and sort Cell Assemblies by their size (# of 1's)
			CA_size = np.round(kmCenters).sum(axis=1).astype(np.uint8)
			indSZ = np.argsort(CA_size)


			# Quantify how close to binary (corners of hypercube) cluster centers are.
			devBin = np.sqrt( (np.round(kmCenters) - kmCenters )**2 ).T
			#
			devBin.mean(axis=0)
			devBin.std(axis=0)
			#
			plt.imshow(devBin)
			plt.set_cmap('bone')
			plt.title( str('Cell Assemblies - deviation from binary [0,1] (k-means w/ k = ' + str(numClust) + ')') )
			plt.xlabel('Cell Assembly # (sorted by size)')
			plt.ylabel('Cell #')
			plt.colorbar()
			plt.show()
			

			# Show all Cell Assemblies sorted by their size (number of 1's)
			plt.imshow( np.round(kmCenters)[indSZ].T )
			plt.set_cmap('bone')
			plt.title( str('Cell Assemblies - cluster centers of unique spike words (k-means w/ k = ' + str(numClust) + ')') )
			plt.xlabel('Cell Assembly # (sorted by size)')
			plt.ylabel('Cell #')
			plt.show()

			# Compute distribution/histogram of Cell Assemblies' sizes (number of 1's).
			CAsz_bins = np.arange( 1, CA_size.max()+1, 1 )
			CAsz_hist, _ = np.histogram( CA_size, CAsz_bins )



			# Find number of unique spike words included in each cluster
			CA_numSW = []
			for i in range(numClust):
				CA_numSW.append( (kmLabels==i).sum() )
			CA_numSW = np.array( CA_numSW )	
			CAnumSW_hist, CAnumSW_bins = np.histogram( CA_numSW, 50 )

			# Plot histogram of Cluster sizes (# spike words in cluster)
			plt.plot( CAnumSW_bins[:-1], CAnumSW_hist )
			plt.title( str('Cluster sizes Distribution - number spike words in each cluster (k-means)') )
			plt.xlabel( str('# Spike Words in Cluster ( Total #sw = ' + str(numSWU_Wn) + ' )') )
			plt.ylabel( str('# Clusters of size ( Total #clusters = ' + str(numClust) + ' )') )
			plt.show()



			# for i in range(km.n_clusters):
			# 	ind = (kmLabels==i)
			# 	swInC = swuWn_bool.T[ind].T
			# 	CA = np.round(kmCenters[i]).reshape(CA.size,1).astype(np.bool)
			# 	ClusterContents = np.concatenate((swInC,CA),axis=1)
			# 	spikesMissedInC = swInC.astype(np.int8) - CA.astype(np.int8)

			# 	plt.imshow(spikesMissedInC)
			# 	plt.set_cmap('bone')
			# 	plt.colorbar()
			# 	plt.show()






	if flg_cluster_sw_KMedoids: # use pyclust toolbox
		print('Kmedoids')	
		file_path = str(dirDataOut + 'Clustering/K-medoids/' + cell_type + '_Kmedoids' + str(numClust) + '_' + str(szHistBins) + 'msBins_wNoise.npz')
		if os.path.isfile(file_path):
			print('Loading ',file_path)
			data = np.load(file_path)
			kmCenters = data['arr_0']
			kmLabels = data['arr_1']
			kmParams = data['arr_2']
			del data

		else:
			print('Compute Kmedoids directly on binary spike words')	
			t = time.time()
			km = pc.KMedoids(n_clusters= numClust, distance='euclidean', n_trials=10, max_iter=100, tol=0.001).fit(swuWn_bool.T[0:5000])
															# these 3 params chosen to match skc.kmeans (10,300,0.0001)
			print( 'Time = ', time.time()-t )
			#
			kmLabels = km.labels_
			kmCenters = km.cluster_centers_
			kmParams = km
			np.savez(file_path, kmCenters, kmLabels, kmParams)
		#








		# pc.BisectKMeans  
		# pc.GMM           
		# pc.KMeans        
		# pc.KMedoids      
		# pc.KernelKMeans  	


		
		if flg_cluster_metrics:

			# Unsupervised Clustering Metrics  (DO NOT NEED GROUND TRUTH) 
			#
			# (1). skm.cluster.calinski_harabaz_score 
			# (2). skm.cluster.silhouette_score   
			#			skm.cluster.silhouette_samples        

			t = time.time()
			CHS = skm.cluster.calinski_harabaz_score(swuWn_bool.T, kmLabels)
			print('CHS = ',CHS)
			# silh_score = skm.cluster.silhouette_samples(swuWn_bool.T, kmLabels, metric='euclidean')
			# silh_avg = skm.cluster.silhouette_score(swuWn_bool.T, kmLabels, metric='euclidean')
			print('Time = ', time.time() - t)


		



	# Do hierarchical agglomerative clustering on spike words.
	if flg_cluster_sw_agglomerative:

		print('Compute Agglomerative Clustering directly on binary spike words')	
		t = time.time()
		ac = skc.AgglomerativeClustering(n_clusters=numClust, affinity='euclidean', linkage='ward').fit(swuWn_bool.T)
		print('Time = ',time.time()-t)


	# Do hierarchical agglomerative clustering on spike words.
	if flg_cluster_sw_DBSCAN:

		print('Compute DBSCAN Clustering directly on binary spike words')	
		t = time.time()
		db = DBSCAN(eps=5, min_samples=50).fit(swuWn_bool.T)
		print('Time = ',time.time()-t)	
		

 				
		# TO DO:
		#			(2). LOOK INTO TIME (WITHIN TRIAL) DISTRIBUTION/REGULARITY WHEN SPIKE WORDS OCCUR
		#			(3). LOOK INTO SPATIAL DISTRIBUTION OF Receptive Fields OF SPIKE WORDS
		#			(4). LOOK INTO STIMULUS AT THE TIME OF THESE SPIKE WORDS 




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # COMPUTE CLUSTERINGS ON PAIRWISE SPIKE WORD SIMILARITY # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #





	# Spike word similarity for unique spike words responding to stim
	for _,stim in enumerate(flg_swSim_stim): # Loop thru different stimulus

		if stim is 'wNoise':
			numSWU 		= numSWU_Wn
			swU_bool 	= swuWn_bool
			lenSWU 		= lenSWU_Wn
			del numSWU_Wn, swuWn_bool, lenSWU_Wn
			# swWn_hash, swWn_time, swWn_trial, swuWn_hash, indx_swuWn_hash, swWn_bool, numSW_Wn,lenSW_Wn, 

		elif stim is 'NatMov':
			numSWU 		= numSWU_Cat
			swU_bool 	= swuCat_bool
			lenSWU 		= lenSWU_Cat
			del numSWU_Cat, swuCat_bool, lenSWU_Cat

		else:
			print('I do not understand the stimulus. ',stim)
			sys.exit()
		#
		# # Other variables not used currently, but avaiable.
		# sw_hash = swWn_hash 
		# sw_time = swWn_time 
		# sw_trial = swWn_trial
		# swU_hash = swuWn_hash
		# indx_swU_hash =indx_swuWn_hash 
		# sw_bool = swWn_bool
		# swU_bool = swuWn_bool
		# lenSW = lenSW_Wn
		# numSW = numSW_Wn	

		for _,TH in enumerate(flg_TH_similarity): # loop thru thresholds on the bitwise AND similarity matrix

			for _,simFunc in enumerate(flg_similarity_func): # loop thru similarity functions
				print(stim,' , ',simFunc,' , TH=',TH)

				# Load in NPZ file containing similarity matrix with a particular threshold, similarity measure
				if flg_swSim_sparse:
					fname = str(dirData_swSim + simFunc + '/' +  cell_type + '_swSim_' + str(szHistBins) + 'msBins_' + stim + '_sparse_gt' + str(TH) + '.npz')
				else:
					fname = str(dirData_swSim + simFunc + '/' + cell_type + '_swSim_' + str(szHistBins) + 'msBins_' + stim + '.npz')
				#	
				swSim = dm.load_similarity_mat( flg_swSim_sparse, fname )

				if (swSim is None):
					if simFunc is 'Jaccard': # load in bitwise AND similarity and compute/save Jaccard similarity.
						print('Load in bitwise AND similarity and compute/save Jaccard similarity.') 
						fname = str(dirData_swSim + 'bitAND/' +  cell_type + '_swSim_' + str(szHistBins) + 'msBins_' + stim + '_sparse_gt' + str(TH) + '.npz')
						swSim = dm.load_similarity_mat( flg_swSim_sparse, fname )
						fname = str(dirData_swSim + simFunc + '/' +  cell_type + '_swSim_' + str(szHistBins) + 'msBins_' + stim + '_sparse_gt' + str(TH) + '.npz')
						swSim = dm.Jaccard_sim(swSim, lenSWU, fname)					

					elif simFunc is 'Kulczynski':
						print('Load in bitwise AND similarity and compute/save Kulczynski similarity.') 
						fname = str(dirData_swSim + 'bitAND/' +  cell_type + '_swSim_' + str(szHistBins) + 'msBins_' + stim + '_sparse_gt' + str(TH) + '.npz')
						swSim = dm.load_similarity_mat( flg_swSim_sparse, fname )
						fname = str(dirData_swSim + simFunc + '/' +  cell_type + '_swSim_' + str(szHistBins) + 'msBins_' + stim + '_sparse_gt' + str(TH) + '.npz')
						swSim = dm.Kulczynski_sim(swSim, lenSWU, fname)

					elif simFunc is 'bitAND':
						print('Maybe Construct the bitwise AND, but it should already exist.')
						sys.exit()

					else:
						print('Uh oh, I dont recognize that similarity function.')
						sys.exit()


				# Do 'clustering' with Cuthill-Mckee reordering algorithm
				if flg_cluster_swSim_CuthillMcKee:
					sym=True
					#fname = str(dirData_swSim + 'bitAND/' + cell_type + '_swSim_' + str(szHistBins) + 'msBins_NatMov_sparse_gt' + str(TH) +'.npz')
					swSim = dm.threshold_similarity_mat(swSim, TH, numSWU, fname)
					perm = dm.cuthill_mckee(swSim,sym) # computing actual cuthill-mckee reordering is pretty quick.

					if flg_plot_swSim_CuthillMcKee: # Plot Cuthill-Mckee reordering of Spike Word Similarity Matrix.
						print('This is not completed yet.')
						spy_cuthill_mckee(swSim, fname)












				if flg_cluster_swSim_agglomerative: # use scikit learn toolbox and/or maybe FLANN toolbox.
					
					print('Agglomerative Clustering')	
					t = time.time()
					AC = skc.AgglomerativeClustering(n_clusters=numClust, affinity='manhattan',compute_full_tree=False, linkage='ward')
					AC.fit(1-swSim) 
					#
					# 1 - sparse matrix is not supported :/
					# maybe need spsp.csr_matrix(1-swSim) because this may become a full matrix.
					# should also check its sparsity level
					print('Time = '.time.time()-t)



				if flg_cluster_swSim_DBSCAN:	
					print('DBSCAN Clustering on Similarity Matrix.')	
					t = time.time()
					DB = skc.DBSCAN(min_samples)
					DB.fit(swSim)
					print('Time = ', time.time()-t)


						




	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
					# # # STUFF WE ONLY HAVE TO DO ONCE.
					# Compute and save a dense bitwise AND similarity matrix.
					if False:
						fname = str(dirData_swSim + 'bitAND/' +  cell_type + '_swSim_' + str(szHistBins) + 'msBins_' + stim + '.npz')
						swSim = dm.construct_AND_similarity_mat( numSWU, swU_bool, fname )


					# Create and save a CSR sparse matrix of bitwise AND similarity from dense matrix.
					if False:
						fname = str(dirData_swSim + 'bitAND/' +  cell_type + '_swSim_' + str(szHistBins) + 'msBins_' + stim + '_sparse_gt0.npz')
						dm.convert_dense_to_sparse(swSim, fname)


					# Threshold similarity matrix - index into sparse Similarity Matrix to grab 
					# values > some threshold values and save new sparse matrix.
					if False:
						for TH in range(1,3)[::-1]: # go thru these values backwards.
							fname = str(dirData_swSim + 'bitAND/' +  cell_type + '_swSim_' + str(szHistBins) + 'msBins_' + stim + '_sparse_gt' + str(TH) +'.npz')
							dm.threshold_similarity_mat(swSim, TH, numSWU, fname)	






	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 














		# Analysis and plot: Look at other spike-words with which a given spike-word has a particular degree of overlap.
		# FLESH THIS OUT LATER. JUST AN EXAMPLE OF AN ANALYSIS I CAN DO ON SPIKE WORD OVERLAP.
		if False:
			xx = np.where( swWn_AND[3] == swWn_AND[3][3] )

			f,ax = plt.subplots(2,1)
			ax[0].imshow( swuWn_bool.T[xx].T, interpolation='none', cmap=plt.get_cmap('gray') )
			ax[0].set_title('Other spike words with which spike word \#3 shares all cells.')
			ax[0].set_ylabel('cell\#')
			#ax[0].set_xlabel('spike word')
			ax[0].set_aspect('auto')
			#
			ax[1].plot( swuWn_bool.T[xx].sum(axis=1) )
			#ax[1].set_title('Number of cells in each spike word')
			ax[1].set_ylabel('\# cells involved in word')
			ax[1].set_xlabel('spike word')
			ax[1].set_aspect('auto')
			ax[1].axis('tight')
			plt.show()








		# look at all spike words in a given cluster.
		if False:
			plt.imshow(xx)
			plt.show()
			#
			#
			# compute SVD on all spike words in a given cluster
			print('Computing SVD. Input size =  ', np.shape(xx) )
			t0 = time.time()
			U, s, V = np.linalg.svd(xx)
			t1 = time.time()
			print('Time = ',t1-t0,' seconds')






		# # Plot these using PCA, SVD or Eigenvectors in 2 or 3D coloring them by.
		# # NOT POSSIBLE: This takes WAAY too much time and blows up memory.	
		# if False:
		# 	print('Computing SVD. Input size =  ', np.shape(swuWn_bool) )
		# 	t0 = time.time()
		# 	U, s, V = np.linalg.svd(swuWn_bool)
		# 	t1 = time.time()
		# 	print('Time = ',t1-t0,' seconds')
		


# Look in SciKitLearn for more clustering ideas: 
#	http://scikit-learn.org/stable/modules/clustering.html
#
#	get_bin_seeds            
#	MiniBatchKMeans 		                         
#	SpectralBiclustering 	                          
#	FeatureAgglomeration				
#	SpectralCoclustering 
#	bicluster 	            
#	estimate_bandwidth 		 
#
# 	Unsupervised Learning:
#		KMeans (no good for binary)
#		DBSCAN (HDBSCAN?) - density based (also no good for binary I think)
#		AffinityPropagation
#		Mean Shift	
#		Spectral Clustering
#		Birch
#	- - - - - - - - - - - - - - - - - - - 
#		hierarchical. (a whole toolbox)
#		AgglomerativeClustering 
#		ward_tree
#		linkage_tree 

 


if flg_cluster_sw_FLANN:
	#nnc.

	import numpy as np
	import pyflann as nnc





	dataset = np.array(
	    [[1., 1, 1, 2, 3],
	     [10, 10, 10, 3, 2],
	     [100, 100, 2, 30, 1] 
	     ])#.astype(np.float64) # note: can be [numpy.float32, numpy.float64, numpy.uint8, numpy.int32]
	testset = np.array(
	    [[1., 1, 1, 1, 1],
	     [90, 90, 10, 10, 1]
	     ])#.astype(np.float64)
	flann = nnc.FLANN()
	result, dists = flann.nn(dataset, testset, 1, algorithm="kmeans", branching=32, iterations=7, checks=16)
	print( result )
	print( dists )


	#nnc.set_distance_type(distance_type, order=0) #Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.


	# dataset = np.random.rand(10000, 128)
	# testset = np.random.rand(1000, 128)
	# flann = nnc.FLANN()
	# result, dists = flann.nn(dataset, testset, 5, algorithm="kmeans", branching=32, iterations=7, checks=16)
	# print( result )
	# print( dists )






# A BERNOULLI MIXTURE MODEL FOR CLUSTERING BINARY DATA
#
# BINARY LOCALITY-SENSITIVE HASHING (BINARY LSH) - IN FLANN TOOLBOX.
#
# HIERARCHICAL CLUSTERING WITH HAMMING DISTANCE - IN FLANN TOOLBOX (& SCIKIT LEARN TOO?)
#
# GNAT ALGORITHM. BY SERGEY BRIN?


# Look in SciKitLearn for clustering metrics: 
# SCORERS                                fbeta_score                            pairwise_distances_argmin
# accuracy_score                         get_scorer                             pairwise_distances_argmin_min
# adjusted_mutual_info_score             hamming_loss                           pairwise_fast
# adjusted_rand_score                    hinge_loss                             pairwise_kernels
# auc                                    homogeneity_completeness_v_measure     precision_recall_curve
# average_precision_score                homogeneity_score                      precision_recall_fscore_support
# base                                   jaccard_similarity_score               precision_score
# brier_score_loss                       label_ranking_average_precision_score  r2_score
# classification                         label_ranking_loss                     ranking
# classification_report                  log_loss                               recall_score
# cluster                                make_scorer                            regression
# cohen_kappa_score                      matthews_corrcoef                      roc_auc_score
# completeness_score                     mean_absolute_error                    roc_curve
# confusion_matrix                       mean_squared_error                     scorer
# consensus_score                        median_absolute_error                  silhouette_samples
# coverage_error                         mutual_info_score                      silhouette_score
# euclidean_distances                    normalized_mutual_info_score           v_measure_score
# explained_variance_score               pairwise                               zero_one_loss
# f1_score                               pairwise_distances 






	"""
	plt.scatter( swWn_time[indx_swuWn_hash], lenSWU_Wn, s=5,color='blue', alpha=0.2 )
	plt.scatter( swCat_time[indx_swuCat_hash], lenSWU_Cat, s=5, color='red', alpha=0.2 )
	plt.title( 'Number of neurons that participate in each Spike Word' )
	plt.xlabel( 'time in msec' )
	plt.ylabel( ' # neurons ' )
	#plt.legend('wNoise','natMov')
	plt.grid()
	plt.show()
	"""



	

	"""
	plt.scatter( sw_trial[indx_unique_spk_words_hash], len_unique_spk_words_bool )
	plt.title( 'Number of neurons that participate in each Spike Word' )
	plt.xlabel( 'trial' )
	plt.ylabel( ' # neurons ' )
	plt.grid()
	plt.show()
	"""

























	# 			THE OTHER THINGS WE CAN COMPUTE WITH UNITARY EVENTS ANALYSIS.
	#
	# (1). gen_pval_anal: 						computes the expected coincidences and p-value for given empirical coincidences 
	#											(for hash constructed from spike train)
	#
	# (2). jointJ_window_analysis & jointJ:		surprise measure for better visualization of significant events
	#											logarithmic transformation of joint-p-value into surprise measure
	#
	# (3). n_emp_mat_sum_trial & n_emp_mat:		Empirical number of observed patterns in spike trains and summed across trials
	#
	# (4). n_exp_mat_sum_trial & n_exp_mat:		Expected number of observed patterns in spike trains and summed across trials
	#
	# (5). CONSTRUCT SPIKE TRAINS BASED ON STATISTICAL PROCESSES AND COMPARE RESULTS
	#
	# (6). SURROGATE ANALYSIS TO BREAK HIGHER ORDER CORRELATIONS IN OBSERVED SPIKE TRAINS
	#
	# (7). CUBIC.
	#
	# (8). ASSET.

