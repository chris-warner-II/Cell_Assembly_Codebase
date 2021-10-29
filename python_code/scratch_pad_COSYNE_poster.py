import numpy as np
import scipy as sp


import matplotlib.pyplot as plt
from matplotlib import ticker
import time 						# to time operations for code analysis
import os.path 						# to check if a file or directory exists already
import sys

import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.plot_functions as pf
import utils.retina_computation as rc




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (1). File name setup and stuff.
#
cell_type = ['offBriskTransient']
			# One of these 5 things below.
			# [ ['offBriskTransient'], ['offBriskSustained'], ['onBriskTransient'], \
			# ['offBriskTransient','offBriskSustained'], \
			# ['offBriskTransient','onBriskTransient'] ] 

N = 55		# Ns = [55, 39, 43, 98, 94]			


stim = 'Wnoise' # 'Wnoise' or 'NatMov'
dirHome, dirScratch = dm.set_dir_tree()
SW_extracted_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')


plt_save_dir = str( dirHome + 'documentation_2018/Cosyne_2019/Figures/' )
if not os.path.exists(plt_save_dir):
    os.makedirs(plt_save_dir)




SW_bin=2
msBins = 1+2*SW_bin

minTms = 0
maxTms = 6000

CST = str(cell_type).replace('\'','').replace(' ','') # convert cell_type list into a string of expected format.
	





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (2).  Get extracted spikewords from spike data to look at statistics of spiking activity.
#
# Extract spike words from raw data or load in npz file if it has already been done and saved.
		
if not os.path.exists( SW_extracted_Dir ): # Make directories for output data if  not already there.
	os.makedirs( SW_extracted_Dir )
#
print('Extracting spikewords')
t0 = time.time()
fname_SWs = str( SW_extracted_Dir + CST + '_' + stim + '_' + str(msBins) + 'msBins.npz' )
spikesIn = list()
SWs, SWtimes = rc.extract_spikeWords(spikesIn, msBins, fname_SWs)
numTrials = len(SWs)
t1 = time.time()
print('Done Extracting spikewords: time = ',t1-t0)






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (3).  Convert SWs into 3D tensor that is CellsCAs x Trials x TimesActive
#
print('Converting SWs,Zinf,Yinf into 3D tensor that is CellsCAs x Trials x TimesActive.')
t0 = time.time()
#
raster_allSWs = rc.compute_raster_list(SWtimes, SWs, N, minTms, maxTms ) 
#
t1 = time.time()
print('Done Converting SWs,Zinf,Yinf into 3D tensor that is CellsCAs x Trials x TimesActive: time = ',t1-t0)





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (4).  Compute spike rates (a). for all spikes (regardless of |y|) from raster vectors
# 						  and (b). for all |y|=1 spike words that each cell participates in.
#
if True:
	# (a). Compute spike rate, individually (ignoring spike words and SW length.)
	spkRate = np.zeros(N)
	for y in range(N):
		spkRate[y] = np.array([ len(raster_allSWs[T][y]) for T in range(numTrials) ]).sum()
	spkRate = spkRate/(numTrials*(maxTms-minTms))*1000 # spike rate in units of spikes/second.	
	#
	#
	# (b). Calculate rate at which each cell spikes on its own.
	SWsR, SWtimesR = rc.extract_spikeWords(spikesIn, msBins, fname_SWs)
	xxx = list()
	for j in range(len(SWsR)):
		xxx.append( [SWsR[j][i].pop() for i in range(len(SWsR[j])) if len(SWsR[j][i])==1] )
	#
	flat_list = [item for sublist in xxx for item in sublist]
	#
	yEq1 = [ (np.array(flat_list)==i).sum() for i in range(N) ]

	yEq1_SWrate = np.array(yEq1)/len(flat_list)

	del SWsR, SWtimesR, xxx, flat_list	
	#
	#
	# (c). Scatter plot spike rate vs. |y|=1 SW rate for each cell.
	#		[Kind of a linear relationship.]
		plt.scatter(spkRate, yEq1_SWrate)
		plt.show()















# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (5).  Compute and plot (a). PSTH across Trials (number of trials that each cell spiked in each timebin).
# 					   and (b). PSTH across Cells  (number of cells active in each timebin for each trial).			
#
# (a). PSTH across Trials (number of trials that each cell spiked in each timebin).
PSTH_acrossTrials = np.zeros( (N,maxTms-minTms) )
for cel in range(N):
	print(cel)
	for trl in range(numTrials):
		PSTH_acrossTrials[cel][ raster_allSWs[trl][cel] ]+=1



plt.figure( figsize=(12,6) ) # size units in inches
plt.rc('font', weight='bold', size=16)
plt.imshow(PSTH_acrossTrials/numTrials, aspect='auto', cmap='jet')		
plt.title( str('PSTH across trials: Cell type = ' + str(cell_type) + ' stim = ' + stim) )
plt.xlabel('time (ms)')
plt.ylabel('cell id')
plt.xlim(0,5000)
#
cbar = plt.colorbar(ticks=[0, np.max(PSTH_acrossTrials/numTrials)])
cbar.set_label( str('% of trials (out of ' + str(numTrials) + ')' ), rotation=270)
# 
plt.savefig( str(plt_save_dir + 'PSTH_acrossTrials_' + str(cell_type) + '_' + stim + '_SWbin' + str(SW_bin) + '.png' ) ) 
#
# #
#
# (b). PSTH across Cells  (number of cells active in each timebin for each trial).	
PSTH_acrossCells = np.zeros( (numTrials,maxTms-minTms) )
for trl in range(numTrials):
	print(trl)
	for cel in range(N):
		PSTH_acrossCells[trl][ raster_allSWs[trl][cel] ]+=1

plt.figure( figsize=(12,6) ) # size units in inches
plt.rc('font', weight='bold', size=16)
plt.imshow(PSTH_acrossCells/N, aspect='auto', cmap='jet')		
plt.title( str('PSTH across trials: Cell type = ' + str(cell_type) + ' stim = ' + stim ) )
plt.xlabel('time (ms)')
plt.ylabel('trial #')
plt.xlim(0,5000)
#
cbar = plt.colorbar(ticks=[0, np.max(PSTH_acrossCells/N)])
cbar.set_label( str('% of cells (out of ' + str(N) + ')' ), rotation=270)
# 
plt.savefig( str(plt_save_dir + 'PSTH_acrossCells_' + str(cell_type) + '_' + stim + '_SWbin' + str(SW_bin) + '.png' ) ) 








