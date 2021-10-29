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



#Set up directories and create dirs if necessary.
dirHome, dirScratch = dm.set_dir_tree()
GLMpYdir 			= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/GLM_p_of_y/')
SW_extracted_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')
spkRasterDir  		= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Rasters_trial_vs_time_zInferred/')



# Params to loop through
cell_types = ['[offBriskTransient,onBriskTransient]', '[offBriskTransient,offBriskSustained]', '[offBriskTransient]'] #			
stims = ['NatMov'] # ,'Wnoise'] #
SW_bins = [2] # 2,				# ms. Build up spikewords from groups of cells in the same trial that fire within SW_bins of eachother.

Ns = [94,98,55]


howSampleS 	= ['Prob','Dont'] 	# 'Prob' or 'Dont'
whichPriorS 	= ['EgalQ','BinomQ'] 	# 'BinomQ' or 'EgalQ'

randsToTry = ['0','0B','1','1B','2','2B']


# Grab firing rates and cell ids for cells within the cell types.
#
for ii,ct in enumerate(cell_types): # Loop through cell type combinations
	N = Ns[ii]
	M = N
	#
	for stim in stims: # Loop through stimuli
		#
		for SW_bin in SW_bins: # Loop through binning of spike trains
			msBins = 2*SW_bin+1
			#
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			#
			# Load in spike-words extracted from real retinal data.
			SWs_ext = np.load( str(SW_extracted_Dir + ct + '_' + stim +'_' + str(msBins) + 'msBins.npz') )
			SWs 	= SWs_ext['SWs'] 
			SWtimes = SWs_ext['SWtimes']
			nTrials = len(SWs)


			numSWs = np.sum([len(x) for x in SWs])

			#
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			#
			# Load in p(Y) computed for each observed spike-word real retinal data under GLM and LNP models.
			GLM = np.load( str(GLMpYdir + ct + '_' + stim +'_GLM_' + str(msBins) + 'msBins.npz') ) 
			LNP = np.load( str(GLMpYdir + ct + '_' + stim +'_LNP_' + str(msBins) + 'msBins.npz') ) 
			#
			# print('GLM keys = ',GLM.keys())
			# print('LNP keys = ',LNP.keys())
			#
			pY_GLM = GLM['pSW_nullGLM']
			pY_LNP = LNP['pSW_nullGLM']
			#
			if (GLM['tiBeg'] == LNP['tiBeg']) and (GLM['tiFin'] == LNP['tiFin'])  and (GLM['tBins_FR'] == LNP['tBins_FR']):
				tiBeg 		= GLM['tiBeg']
				tiFin 		= GLM['tiFin']
				tBins_FR	= GLM['tBins_FR']
			else:
				print('Error: expecting tiBeg, tiFin and tBins_FR to be same in GLM and LNP.')


			del GLM, LNP, SWs_ext # free up some space.


			secBins = msBins/1000 	# convert ms to sec for spike-word binning in spike train.
			b = secBins/tBins_FR	# number of fine-time GLM bins in one of our larger SW bins.
			#
			bBck = np.int( np.floor( (b-1)/2 ) ) 
			bFwd = np.int( np.ceil( (b-1)/2 ) )
			


			for howSample in howSampleS:
				#
				for whichPrior in whichPriorS:
					#
					for rand in randsToTry:

						#if True:
						try:	
							print( 'Running for:  ',ct, 'sample', howSample, 'prior', whichPrior, 'rand', rand )

							#
							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
							#
							# Load in a raster(?) file from our model which contains p(Y|Z) that we can compare. Fair comparison?
							init_dir = str( 'Init_NoisyConst_5hot_mPi1.0_mPia1.0_sI[0.01 0.05 0.05]_LR0pt5_LRsc[1.0, 0.1, 0.1]/')
							model_dir = str( ct + '_N' + str(N) + '_M' + str(M) + '_zeq0_ylims0_1000_yMinSW1_' + howSample + 'Smp1st_' + whichPrior + '/' )
							rasFname = str( 'rasterZ_allSWs_allTrials_' + stim + '_' + str(numSWs) + 'SWs_0pt5trn_' + str(msBins) + 'msBins_allSamps_rand' + rand + '.npz' )
							#

							PGM = np.load( str(spkRasterDir + init_dir + model_dir + rasFname) ) 
							#
							# print(PGM.keys())
							#
							zInf_PGM = PGM['Z_inferred_allSWs'] 	# which Z's are inferred for each spike-word
							pCnd_PGM = PGM['cond_inferred_allSWs']	# conditional probability p(Y|Z) for each SW
							pJnt_PGM = PGM['pj_inferred_allSWs']	# joint probability p(Y,Z) for each SW

							del PGM # free up some space.



							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
							#
							# Compute average p(y) of null models (GLM & LNP) across all spikewords.
							#
							GLMsum = 0
							GLMcnt = 0
							GLMinfCnt = 0
							LNPsum = 0
							LNPcnt = 0
							LNPinfCnt = 0
							PGMsum = 0
							PGMcnt = 0
							for tr in range(nTrials):
								finites = np.isfinite( np.log(pY_GLM[tr]) )
								GLMinfCnt += np.isinf( np.log(pY_GLM[tr]) ).sum()
								GLMsum += np.nansum( np.log(pY_GLM[tr])*finites )
								GLMcnt += finites.sum()
								#
								finites = np.isfinite( np.log(pY_LNP[tr]) )
								LNPinfCnt += np.isinf( np.log(pY_LNP[tr]) ).sum()
								LNPsum += np.nansum( np.log(pY_LNP[tr])*finites )
								LNPcnt += finites.sum()
								#
								PGMsum += np.nansum(pJnt_PGM[tr])
								PGMcnt += np.isfinite(pJnt_PGM[tr]).sum()
								#
							GLMmean = GLMsum/GLMcnt
							LNPmean = LNPsum/LNPcnt
							PGMmean = PGMsum/PGMcnt
							#
							# Compute standard deviation also.
							GLMstdSum = 0
							LNPstdSum = 0
							PGMstdSum = 0
							for tr in range(nTrials):
								finites = np.isfinite( np.log(pY_GLM[tr]) )
								GLMstdSum += np.nansum( ( np.log(pY_GLM[tr])*finites - GLMmean)**2 )
								finites = np.isfinite( np.log(pY_LNP[tr]) )
								LNPstdSum += np.nansum( ( np.log(pY_LNP[tr])*finites - LNPmean)**2 )
								PGMstdSum += np.nansum( ( pJnt_PGM[tr]  - PGMmean)**2 )
							#
							GLMstd = GLMstdSum/GLMcnt
							LNPstd = LNPstdSum/LNPcnt
							PGMstd = PGMstdSum/PGMcnt




							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
							#
							# Make a list of lists for PGM p(y,z), GLM p(y), LNP p(y), SWtimes.
							# 	(1). Outer list ranges  over M cell assemblies.
							# 	(2). Inner list ranges over each time CA is active.
							#
							pPGMz 	= list() 	# PGM p(y,z)
							pGLMz 	= list()	# GLM p(y)
							pLNPz 	= list()	# LNP p(y)
							tiSWz 	= list()	# SW times
							trSWz 	= list()	# SW trial
							SWz 	= list()	# SWs
							#
							for m in range(M+1):				# (1). Outer list ranges over M cell assemblies.
								pPGMz.append( list() )
								pGLMz.append( list() )
								pLNPz.append( list() )
								tiSWz.append( list() )
								trSWz.append( list() )
								SWz.append( list() )
								#
							#----
							#
							# (2). Now fill in the lists when a certain cell assembly is active.			
							#
							for tr in range(nTrials):
								#print('trial#',tr)
								#
								for i in range(len(zInf_PGM[tr])): # Loop through all indices 
									#
									if not zInf_PGM[tr][i]: 			# If there are no active cell assemblies
										if np.isfinite( np.log(pY_GLM[tr][i]) ) and np.isfinite( np.log(pY_LNP[tr][i]) ): # nan if they are outside the time limits of GLM and inf if p(y) is exactly zero.
											#
											pPGMz[M].append( pJnt_PGM[tr][i] )
											pGLMz[M].append( np.log(pY_GLM[tr][i]) )
											pLNPz[M].append( np.log(pY_LNP[tr][i]) )
											tiSWz[M].append( SWtimes[tr][i] )
											trSWz[M].append( tr )
											SWz[M].append( SWs[tr][i] )
										#
									else:								# If there are active cell assemblies
										for ca in zInf_PGM[tr][i]:
											#
											if np.isfinite( np.log(pY_GLM[tr][i]) ) and np.isfinite( np.log(pY_LNP[tr][i]) ): # nan if they are outside the time limits of GLM and inf if p(y) is exactly zero.
												#
												#print(i, ca, pJnt_PGM[tr][i].round(2), np.log(pY_GLM[tr][i]).round(2), np.log(pY_LNP[tr][i]).round(2) )
												#
												pPGMz[ca].append( pJnt_PGM[tr][i] )
												pGLMz[ca].append( np.log(pY_GLM[tr][i]) )
												pLNPz[ca].append( np.log(pY_LNP[tr][i]) )
												tiSWz[ca].append( SWtimes[tr][i] )
												trSWz[ca].append( tr )
												SWz[ca].append( SWs[tr][i] )



							mnPGM = [ np.mean(xx) for xx in pPGMz ]
							#sdPGM = [ np.std(xx) for xx in pPGMz ]
							mnGLM = [ np.mean(xx) for xx in pGLMz ]
							mnLNP = [ np.mean(xx) for xx in pLNPz ]




							# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
							#
							# Scatter Plot of PGM p(y,z) vs. Null model p(y)
							#
							if True:
								#
								f = plt.figure( figsize=(10,10) ) # size units in inches   
								plt.rc('font', weight='bold', size=12)
								#
								minn = -90 #np.nanmin([mnPGM,mnGLM,mnLNP])#, PGMmean, GLMmean, LNPmean])
								maxx = 0 #np.nanmax([mnPGM,mnGLM,mnLNP])#, PGMmean, GLMmean, LNPmean])
								#
								plt.scatter(mnGLM, mnPGM, s=100, color='blue', alpha=0.5, label='GLM w/ z')
								plt.scatter(mnLNP, mnPGM, s=100, color='green',  alpha=0.5, label='LNP w/ z')
								#
								# label z's when p(y) under null is low.
								for z in range(M):
									plt.text(mnGLM[z], mnPGM[z], str(z), color='black', va='center', ha='center', fontsize=6, fontweight='bold')
									plt.text(mnLNP[z], mnPGM[z], str(z), color='black', va='center', ha='center', fontsize=6, fontweight='bold')
								#
								plt.scatter(mnGLM[M], mnPGM[M], s=110, marker='o', facecolor='none', edgecolor='magenta', label='GLM no z')
								plt.scatter(mnLNP[M], mnPGM[M], s=110, marker='x', edgecolor='magenta', label='LNP no z')
								#
								plt.scatter( GLMmean, PGMmean, s=150, marker='o', facecolor='none', edgecolor='red', label='GLM all y')
								plt.scatter( LNPmean, PGMmean, s=150, marker='x', edgecolor='red', label='LNP all y')
								#
								plt.text( minn+10, minn, 'z#', color='black', va='center', ha='center', fontsize=12, fontweight='bold')
								plt.plot( [minn,maxx], [minn,maxx], 'k--' )
								plt.axis([minn-2, maxx+2, minn-2, maxx+2])
								#plt.axis('equal')
								plt.ylabel('log p(y,z) of PGM')
								plt.xlabel('log p(y) of Null')
								plt.title( str( ct + ' N,M'+str(N) + ' ' + stim + ' ' + str(msBins)+'msBins ' + howSample + ' ' + whichPrior ) )
								plt.legend(title='null model')
								plt.grid()
								#plt.show()

								
								#
								# For comparison, what is average p(y) under null?
								# Or full distribution. Tails of distrib. Statistical significance.
								#
								
								# # Save plot. Visualize Good CAs.
								# if not os.path.exists( str('path') ):
								# 	os.makedirs( str('path') )
								#

								fname_save = str(ct+'_N'+str(N)+'_M'+str(M)+'_'+stim+'_'+str(msBins)+'msBins_'+howSample+'_'+whichPrior+'_rand'+rand)
								figSaveFileType = 'png'

								plt.savefig( str('./' + fname_save + '.' + figSaveFileType ) )
								plt.close() 






							# Plot p(y) values for each model (PGM, GLM, LNP) for all time points in trial (NOT SO USEFUL)
							if False:
								f, ax = plt.subplots(2,1)
								#
								ax[0].plot( np.log(pY_LNP[tr]), label=r'LNP $p(\vec{y})$', color='yellow', alpha=0.5 )
								ax[0].plot( np.log(pY_GLM[tr]), label=r'GLM $p(\vec{y})$', color='blue', alpha=0.5 )
								ax[0].plot( pJnt_PGM[tr], label=r'PGM $p(\vec{y},\vec{z})$', color='red', alpha=0.5 )
								ax[0].legend()
								#
								ax[1].plot( pJnt_PGM[tr] - np.log(pY_LNP[tr]), label='PGM - LNP', color='yellow', alpha=0.5 )
								ax[1].plot( pJnt_PGM[tr] - np.log(pY_GLM[tr]), label=r'PGM - GLM', color='blue', alpha=0.5 )
								#ax[1].plot( np.array([0, pJnt_PGM[tr] ]), np.array([0, 0]) , 'k--', alpha=0.5 )
								ax[1].legend()
								#
								plt.show()


							# SAVE np.array(pSW_nullGLM) to a file.	
							if False:
								fname_GLMpY = str(GLMpY_saveDir + ct + '_' + stim +'_' + whichSim + '_' + str(msBins) + 'msBins.npz')
								np.savez( fname_GLMpY, tBins_FR=tBins_FR, tiBeg=tiBeg, tiFin=tiFin, pSW_nullGLM=pSW_nullGLM )	



						except:
							print('There is a problem with it. Moving on.')

