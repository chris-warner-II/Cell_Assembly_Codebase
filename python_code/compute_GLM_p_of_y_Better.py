import numpy as np
import scipy as sp
from scipy import io as io
import matplotlib.pyplot as plt
import time 						# to time operations for code analysis
import os.path 						# to check if a file or directory exists already
import sys

import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.plot_functions as pf
import utils.elephant_usage as el
import utils.retina_computation as rc


whichSim = 'GLM' # 'GLM' or 'LNP'


# Name of data files saved from my preprocessing in MATLAB of raw datafiles from Greg Field's Lab.
GField_spikeData_File = str('allCells_spikeTrains_CellXTrial.mat')
#
if whichSim == 'GLM':
	GField_GLMsimData_File = str('GLM_cat_sim_fullpop.mat') 	# Independent GLM with spike history.
elif whichSim == 'LNP':
	GField_GLMsimData_File = str('LNP_cat_sim_fullpop.mat')	# LNP model without spike history.
else:
	print('I dont understand whichSim',whichSim)


#Set up directories and create dirs if necessary.
dirHome, dirScratch = dm.set_dir_tree()


GLMpY_saveDir = str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/GLM_p_of_y/')
#
if not os.path.exists( GLMpY_saveDir ):
	os.makedirs( GLMpY_saveDir )	



# LearnedModel_Dir 	= 

# CA_Raster_Dir 		=	


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# (1). Get cell types and cell type ids. Load in ndarray of spiketimes for 2 types
# 						of stimulus. They are of size: (#cells x #trials x #spikes)
#
if False: 					# DONT THINK I NEED TO DO THIS RIGHT NOW.
	spikes = io.loadmat( str(dirScratch + 'data/matlab_data/' + GField_spikeData_File) )
	#print(spikes.keys())
	allCellTypes = spikes['allCellTypes']
	cellTypeIDs = spikes['cellTypeIDs']

	del spikes








# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# (2). Load in GLM simulation results of RGCs responding to natural movie stimuli
#
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



# # FOR GLM VERSION:
# instFR.shape # = (137,)
# instFR[0].shape # = (200,6000) (trials,time_bins)
# # Questions: 1. Time bins are 1/20th of stim time bins ( (1/60)/20 ). Or 5 sec / 6000 bins =  0.833333333333333ms
# #			 2. The units of the finely sampled firing rate are spikes/s. Convert to p(y=1) by p(y=1) = instFR / 
# #
# # FOR LNP VERSION:




tBins_FR = 5 / nBins # bin size for finely binned spike rate vectors (units = sec/bin). 5 seconds for full stim.



# Set time bounds to only look at spike words that occur between these times.
tiFin = 5000 # ms.
tiBeg = int( np.round(400*tBins_FR*1000) ) 	 # ms. # Kiersten suggests this because of GLM history dependence. int( np.round(400*tBins_FR*1000) )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# (3). Extract spike words - or load in file with already extracted spike-words in it.
#
SW_extracted_Dir = str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')
if not os.path.exists(SW_extracted_Dir):
	os.makedirs(SW_extracted_Dir)

cell_types = ['[offBriskTransient]'] # ['[offBriskTransient,onBriskTransient]', '[offBriskTransient,offBriskSustained]', '[offBriskTransient]'] #			
stims = ['NatMov'] # ,'Wnoise'] #
SW_bins = [2] #[2,1]				# ms. Build up spikewords from groups of cells in the same trial that fire within SW_bins of eachother.







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
# Params to load in Learned model and CA raster files.
#
which_train_rand = [0] #0, 1, 2] # Vector of which rand values to use. Choose a "good" model. 
train_2nd_modelS = [True] #,True]# False if there is no B file, True to run only B file, [True,False] to run both.

flg_EgalitarianPriorS = [ True ] # False] # 
sample_longSWs_1stS = ['Dont']#,'Prob'] # Options are: {'Dont', 'Prob', 'Hard'}










# Grab firing rates and cell ids for cells within the cell types.
#
for i,ct in enumerate(cell_types): # Loop through cell type combinations
	#
	# Convert cell id in spike-word to index in instFR array knowing that the order
	# of cells is: 137 cells = {55 offBT, 43 offBS, 39 onBT}.
	if ct == '[offBriskTransient]':
		N = 55
		FR = instFR[:N] # offBT
		cellSet = set(range(N))
	elif ct == '[offBriskTransient,offBriskSustained]':	
		N = 55+43
		FR = instFR[:N] # offBT and offBS
		cellSet = set(range(N))
	elif ct == '[offBriskTransient,onBriskTransient]':
		N = 55+39
		if whichSim == 'GLM':
			xx = instFR[:55] # offBT
			yy = instFR[-39:] # onBT
		elif whichSim == 'LNP':
			xx = instFR[:55].T # offBT
			yy = instFR[-39:].T # onBT
		FR = np.hstack( (xx,yy) )
		del xx, yy
		cellSet = set(range(N))
	else:
		print('Dont understand cell type combination ',ct)
	#
	# #
	#
	for stim in stims: # Loop through stimulus
		#
		for SW_bin in SW_bins: # Loop through binning of spike trains
			msBins = 2*SW_bin+1
			#
			SWs_ext = np.load( str(SW_extracted_Dir + ct + '_' + stim +'_' + str(msBins) + 'msBins.npz') )
			SWs 	= SWs_ext['SWs'] 
			SWtimes = SWs_ext['SWtimes']
			#
			nTrials = len(SWs)
			pSW_nullGLM = list()



			print(ct, stim, msBins,'msBins')


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
			#
			# Convert firing rate into probability of fewer than 1 spike within 0.83ms interval, i.e. p(yi=0)
			xx = np.zeros( (N,nBins) )
			for i in range(N):
				xx[i] = FR[i].mean(axis=0)
			pyiEq0 = np.exp( -xx*tBins_FR ) # is N x tBins_FR
			del xx
			








			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			#
			# Get CA activation times so we only look at these p(y|za=1)
			#
			# Load in CA Model and Raster files.
			#











			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			#
			# OK, here is what makes this one better than the original. 
			# Want to compute p(y) under null model for a particular observed spike-word
			# for all times in the stimulus. This is to see if there is a systematic offset
			# between when a synchronous firing pattern is most likely under the GLM and
			# when we observe it using our model.
			#
			pyiEq0_Big =np.zeros( (pyiEq0.shape[0], pyiEq0.shape[1], bFwd+bBck) )
			#
			shifts = np.arange( -bBck,bFwd )
			for h,s in enumerate(shifts):
				#print(h,s)
				pyiEq0_Big[:,:,h] = np.roll(pyiEq0,s,axis=1) # Adds a 3rd dimension to array and puts a time shifted version of pyiEq0.

			



			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			#
			#
			# Loop through all trials and find when CA is active. Then get p(y) at all times for that spikeword.
			#
			for tr in range(1): #nTrials): # Loop through trials

				print(tr)
				pSW_nullGLM.append( list() )				
				#
				for t,sw in enumerate(SWs[tr]): # Loop through spike words in a trial
					ti = SWtimes[tr][t]			# time of a spike word.

					if (ti<(tiFin-bFwd)) and (ti>(tiBeg-bBck)): # 5 seconds. GLM doesnt have results past this.
					#
						print(tr, ti, list(sw) )

						tBin_indx = int( np.round( ti/(tBins_FR*1000) ) ) 	# find bin index in GLM data (6000 bins) 
																			# converting from 5000 bins. Different sampling.
						#
						inn = list( sw )									# cells in the spike-word.
						out = list( cellSet.difference(sw) )				# cells not in the spike-word.
						#

						# cxxx

						pOff = pyiEq0_Big[out].prod(axis=2)		# prob of each cell that is off to be off
						pOn = 1 - pyiEq0_Big[inn].prod(axis=2)	# prob of all cell that is on to be on
						pY = np.vstack([pOn,pOff]).prod(axis=0)	# prob of whole spike word.
						

						f,ax = plt.subplots( 1,1 )
						ax.plot(pY)
						ax.scatter(ti,pY[ti],50,c='red',marker='o')
						ax.grid()

						plt.show()




					# 	pSW_nullGLM[tr].append( pY ) 		# make same shape as SWs and SWtimes.	
					# 	#
					# else:
					# 	pSW_nullGLM[tr].append( np.nan ) 	# time beyond 5sec is past where GLM has data.


			

			# SAVE np.array(pSW_nullGLM) to a file.	
			fname_GLMpY = str(GLMpY_saveDir + ct + '_' + stim +'_' + whichSim + '_' + str(msBins) + 'msBins_stAt' + str(tiBeg) + 'ms.npz')
			np.savez( fname_GLMpY, tBins_FR=tBins_FR, tiBeg=tiBeg, tiFin=tiFin, pSW_nullGLM=pSW_nullGLM )	




			




			






			

















