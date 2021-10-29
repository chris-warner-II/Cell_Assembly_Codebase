# toolbox from Sonja Gruen's group to do Unitary Events Analysis and other parallel spike train analysis
import elephant.spike_train_generation as stgen 
from elephant.spike_train_correlation import cross_correlation_histogram, corrcoef, covariance
from quantities import Hz, s, ms
from elephant.kernels import GaussianKernel
from elephant.statistics import isi, cv, instantaneous_rate, mean_firing_rate
import neo
from elephant.conversion import BinnedSpikeTrain
import elephant.unitary_event_analysis as uea


# Other things
import matplotlib.pyplot as plt
import numpy as np



def plot_instSpkRate_fewCells(spikesInCat, spikesInWn, kernBins, cells, numMS, f_path):		
	# Compute and plot the instantaneous spike rate for a couple of chosen cells.
	#
	# Inputs:
	# 		spikesInCat - 2D ndarray of spiketimes under natural movie stimulus. It is numCells x numTrials.
	# 		spikesInWn  - ndarray of spiketimes under white noise stimulus 
	#		kernBins 	- Bin size in ms when spiketrains are convolved with Gaussian Kernel to compute spike rate.
	# 		cells 		- subset of cells for which to plot their instantaneous rates.
	# 		numMS 		- trial duration in ms. 
	# 		f_path 		- path to save output figure.

	numCells  = spikesInCat.shape[0]
	numTrials = spikesInCat.shape[1] - 1 # not taking last trial because it extends for much longer.

	GK = GaussianKernel(sigma=kernBins*ms)
	tBins = int(numMS/kernBins)
	f,ax = plt.subplots(2,1)
	plt.rc('text', usetex=True)
	#
	for c in cells:
		instRateCat = np.zeros( (numTrials, tBins ) )
		instRateWn = np.zeros(  (numTrials, tBins ) )
		#
		for k in range(numTrials):
			stCat = neo.SpikeTrain( spikesInCat[c][k]*ms, t_stop=numMS*ms)
			instRateCat[k] = instantaneous_rate(stCat, kernBins*ms, kernel=GK, cutoff=5.0, t_start=None, t_stop=None, trim=False).as_array().ravel()
			#
			stWn = neo.SpikeTrain( spikesInWn[c][k]*ms, t_stop=numMS*ms)
			instRateWn[k] = instantaneous_rate(stWn, kernBins*ms, kernel=GK, cutoff=5.0, t_start=None, t_stop=None, trim=False).as_array().ravel()
		#
		ax[0].plot(instRateCat.mean(axis=0), label=str('cell '+str(c)) )
		ax[0].fill_between(range(tBins), instRateCat.mean(axis=0)+instRateCat.std(axis=0),instRateCat.mean(axis=0)-instRateCat.std(axis=0), alpha=0.5)
		ax[0].legend()
		ax[1].plot(instRateWn.mean(axis=0), label=str('cell '+str(c)) )
		ax[1].fill_between(range(tBins), instRateWn.mean(axis=0)+instRateWn.std(axis=0), instRateWn.mean(axis=0)-instRateWn.std(axis=0), alpha=0.5)
	#
	ax[0].set_title( str('Natural Movie'), fontsize=18 )
	ax[0].set_ylabel('Rate (sec$^{-1}$)', fontsize=16 )
	ax[0].set_xticklabels( kernBins*ax[0].get_xticks().astype(int) )
	#
	ax[1].set_title( str('White Noise '), fontsize=18 )
	ax[1].set_xlabel('Time (msec)', fontsize=16 )
	ax[1].set_ylabel('Rate (sec$^{-1}$)', fontsize=16 )
	ax[1].set_xticklabels( kernBins*ax[1].get_xticks().astype(int) )
	#
	plt.suptitle( str( 'Instantaneous Spike Rates for ' + str(np.size(cells)) + ' cells ( Gaussian Kernel w/ ' + str(kernBins) + 'ms Bins )'), fontsize=20  )
	f.set_size_inches(14,8)
	f.savefig( f_path )




def compute_mean_spike_rate(spikesInCat, spikesInWn, numMS):

	numCells  = spikesInCat.shape[0]
	numTrials = spikesInCat.shape[1] - 1 # not taking last trial because it extends for much longer.

	fRateCat = np.zeros( (numCells,numTrials) )
	fRateWn = np.zeros( (numCells,numTrials) )
	for i in range(numCells):
		for k in range(numTrials):
			stCat = neo.SpikeTrain( spikesInCat[i][k]*ms, t_stop=numMS*ms)
			fRateCat[i][k] = mean_firing_rate(stCat, t_start=None, t_stop=None, axis=None)
			#
			stWn = neo.SpikeTrain( spikesInWn[i][k]*ms, t_stop=numMS*ms)
			fRateWn[i][k] = mean_firing_rate(stWn, t_start=None, t_stop=None, axis=None)

	return fRateCat, fRateWn	






def errorbar_meanSpkRate_trialAvg(fRateCat, fRateWn, MeanRate_file_path):

	numCells  = fRateCat.shape[0]
	numTrials = fRateCat.shape[1]
	fRateAll = np.vstack( ( fRateCat, fRateWn ) )

	f,ax = plt.subplots()
	plt.errorbar( range(numCells), np.mean(fRateCat,axis=1), np.std(fRateCat,axis=1), color='red', fmt='o' )
	plt.errorbar( range(numCells), np.mean(fRateWn,axis=1), np.std(fRateWn,axis=1), color='blue', fmt='o' )
	#
	plt.errorbar( (-3,numCells+1), ( fRateCat.mean(), fRateCat.mean() ), ( fRateCat.std(), fRateCat.std() ), color='red')
	plt.errorbar( (-2,numCells+2), ( fRateAll.mean(), fRateAll.mean() ), ( fRateAll.std(), fRateAll.std() ), color='black')
	plt.errorbar( (-1,numCells+3), ( fRateWn.mean(),  fRateWn.mean()  ), ( fRateWn.std(),  fRateWn.std()  ), color='blue')
	#
	plt.text(0, fRateCat.mean(), str( '$\mu$ rate NatMov = ' + str(fRateCat.mean().astype(int) )), color='red',   verticalalignment='bottom', fontsize=16, fontweight='bold')
	plt.text(0, fRateWn.mean(),  str( '$\mu$ rate wNoise = ' + str(fRateWn.mean().astype(int)  )), color='blue',  verticalalignment='bottom', fontsize=16, fontweight='bold')
	plt.text(0, fRateAll.mean(), str( '$\mu$ rate All = '    + str(fRateAll.mean().astype(int) )), color='black', verticalalignment='bottom', fontsize=16, fontweight='bold') 
	#
	plt.xlim(-4,numCells+4)
	plt.rc('text', usetex=True)
	plt.ylabel('Spike Rate (sec$^{-1}$)', fontsize=18)
	plt.xlabel('Cell \#', fontsize=18)
	plt.title('Trial Averaged Mean Spike Rates for Different Stimuli', fontsize=20)
	plt.legend(['NatMov','wNoise', '$\mu$ NatMov', '$\mu$ wNoise', '$\mu$ All'], loc='best')
	plt.grid()
	ax.tick_params(axis='both', which='major', labelsize=16)
	f.set_size_inches(14,8)
	f.savefig( MeanRate_file_path )