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

#
# This function makes plots of distributions for spike-word statistics in order to compare observed spike-word 
# statistics for synthetic data generated from models with various parameter values to those spike-word 
# statistics in real retinal data. This is discussed in the section "Fitting Model parameters to spike-word 
# statistics" of the paper and displayed in Fig. 2, "Fitting synthetic model to spike-word moments". This 
# allows us to choose the best model parameters to synthesize data based on QQ-plots comparing spike-word 
# stats between real and synthesized data.
#


# Number of Synthetic Model Construction Parameters grid searched for each param in compare_SWdists_realNsynthData
numK = 12
numC = 5
num_muPia = 14
num_sigPia = 2
num_muPi = 2 
num_sigPi = 2

wts = np.array([2., 1., 3.]) 	# weights on different components of distance measure (on each distribution) 
								# [yc, y1, y2] - yc = cardinality, y1 = single cell activity, y2 = pairwise coactivity.



sample_longSWs_1st = 'Dont' # 'Dont' or 'Prob'
flg_EgalitarianPrior = True
flg_include_Zeq0_infer = True

figSaveFileType = 'png' # 'png' or 'pdf'	

verbose = True
plot_CDFs_n_PDFs = True

nBestFits = 1
samps = 50000


if flg_include_Zeq0_infer:
	z0_tag='_zeq0'
else:
	z0_tag='_zneq0'

#
if flg_EgalitarianPrior:	
	priorType = 'EgalQ' 
else:
	priorType = 'BinomQ'







# # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # 
#
# Parameters for Real Data (Spike Words Extracted).
#
cellSubTypes = [ '[offBriskTransient]','[offBriskTransient,offBriskSustained]', '[offBriskTransient,onBriskTransient]']
			# ,'[offBriskSustained]','[onBriskTransient]', 
Ns = [55, 98, 94] # 43, 39,
Ms = Ns
numSWs_MovS = [571276, 809134, 717525] # 559505, 353174,
numSWs_WnzS = [740267, 852944, 861660] # 422947, 440264,
#
SW_binS =  [2, 1, 0] #[0, 1, 2] 	# ms. Build up spikewords from groups of cells in the same trial that fire within SW_bins of eachother.

stimS = ['Mov','Wnz']

yLo		= 0		# for both real and synth data.
yHi 	= 1000 	# for both real and synth data.

yMinSW 	= 1 # [1, 3] # [1,2,3]

bernoulli_Pi = 1.0









# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (0). Set up directories and create dirs if necessary.
dirHome, dirScratch = dm.set_dir_tree()
synthData_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/Models_learned_EM/')
realData_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')
realRaster_dir  = str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Rasters_trial_vs_time_zInferred/')
BestParams_dir = str( dirScratch + 'data/python_data/PGM_analysis/compareSWs_realNsynth/')

# Infer_postLrn_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/inference_postLrn/')
# EM_learnStats_Dir	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/inferStats_from_EM_learning/')
plt_save_dir = str( dirScratch + 'figs/PGM_analysis/compareSWs_realNsynth/')
#
if not os.path.exists( str(plt_save_dir) ):
	os.makedirs( str(plt_save_dir) )

gridNum_str = str( str(numK)+'Ks_'+str(numC)+'Cs_'+str(num_muPia) +'m'+str(num_sigPia)+\
									'sPiaS_'+str(num_muPi)+'m'+str(num_sigPi)+'sPiS')

plt_grid_dir = str( plt_save_dir + '/searchGrid' + gridNum_str + '_wtsC12'+str(wts)+'/' )
#
if not os.path.exists( str(plt_grid_dir) ):
	os.makedirs( str(plt_grid_dir) )






# Loop through different real data files and for each of them loop through all the synth parameter sweeps.
for indReal in range(1,len(cellSubTypes)):
	CST = cellSubTypes[indReal]
	N = Ns[indReal]
	M = Ms[indReal]
	M_mod	= M
	numSWs_Mov = numSWs_MovS[indReal]
	numSWs_Wnz = numSWs_WnzS[indReal]
	#
	for SW_bin in SW_binS:

		for stim in stimS:

			realData_str = str( CST + '_' + stim + '_' + str(1+2*SW_bin) + 'msBins')
			print('Real Data:  ',realData_str, ' searching grid ',gridNum_str)

			# # # # # # # # # # # # # # # # # # # # # # # # # 
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
			# # # # # # # # # # # # # # # # # # # # # # # # # 
			#
			# Load in Spike Words Extracted from real retinal data.
			print('Load in Spike Words Extracted from real retinal data.')
			t0 = time.time()
			#
			try:
				# Try to load in stats directly that were computed on ALL spike words. From stats files in SpikeWords Extracted directory
				fname_SWstats_Wnz = str( realData_Dir + str(CST) + '_Wnoise_' + str(1+2*SW_bin) + 'msBins_yMin' + str(yMinSW) + '_SWstats.npz' )
				data_Wnz = np.load(fname_SWstats_Wnz)
				nYr_Wnz = data_Wnz['nY'] 
				Ycell_histR_Wnz = data_Wnz['Ycell_hist'] 
				Cell_coactivityR_Wnz = data_Wnz['Cell_coactivity']
				del data_Wnz
			except:

				# Try to load in stats directly that were computed on ALL spike words. From rasterZ_allSWs files.
				fname_raster_Wnz = str( realRaster_dir + 'Init_NoisyConst_5hot_mPi1.0_mPia1.0_sI[0.01 0.05 0.05]_LR0pt5_LRsc[1.0, 0.1, 0.1]/' + \
					str(CST) + '_N' + str(N) +'_M' + str(M) + z0_tag + '_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) + '_' + \
					sample_longSWs_1st + 'Smp1st_' + priorType + '/' + \
					'rasterZ_allSWs_allTrials_Wnoise_' + str(numSWs_Wnz) + 'SWs_0pt5trn_' + str(1+2*SW_bin) + 'msBins_allSamps_rand0.npz')
			
				data_RasWnz = np.load(fname_raster_Wnz)
				nYr_Wnz = data_RasWnz['nY_allSWs'] 
				Ycell_histR_Wnz = data_RasWnz['Ycell_hist_allSWs'] 
				Cell_coactivityR_Wnz = data_RasWnz['Cell_coactivity_allSWs']
				del data_RasWnz
			#
			# # #
			#
			try:
				fname_SWstats_Mov = str( realData_Dir + str(CST) + '_NatMov_' + str(1+2*SW_bin) + 'msBins_yMin' + str(yMinSW) + '_SWstats.npz' )
				data_Mov = np.load(fname_SWstats_Mov)
				nYr_Mov = data_Mov['nY'] 
				Ycell_histR_Mov = data_Mov['Ycell_hist'] 
				Cell_coactivityR_Mov = data_Mov['Cell_coactivity']
				del data_Mov
			except:
				fname_raster_Mov = str( realRaster_dir + 'Init_NoisyConst_5hot_mPi1.0_mPia1.0_sI[0.01 0.05 0.05]_LR0pt5_LRsc[1.0, 0.1, 0.1]/' + \
					str(CST) + '_N' + str(N) +'_M' + str(M) + z0_tag + '_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + str(yMinSW) + '_' + \
					sample_longSWs_1st + 'Smp1st_' + priorType + '/' + \
					'rasterZ_allSWs_allTrials_NatMov_' + str(numSWs_Mov) + 'SWs_0pt5trn_' + str(1+2*SW_bin) + 'msBins_allSamps_rand0.npz')
				
				data_RasMov = np.load(fname_raster_Mov)
				nYr_Mov = data_RasMov['nY_allSWs'] 
				Ycell_histR_Mov = data_RasMov['Ycell_hist_allSWs'] 
				Cell_coactivityR_Mov = data_RasMov['Cell_coactivity_allSWs']
				del data_RasMov
			#
			t1 = time.time()
			print('Time: ',t1-t0)







			# # # # # # # # # # # # # # # # # # # # # # # # # 
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
			# # # # # # # # # # # # # # # # # # # # # # # # # 
			#
			# Load in QQ dist from Unity metric matrix.
			#
			print('Load in QQ dist from Unity metric matrix.')
			t0 = time.time()
			#
			fname_str = str( CST+'_'+str(1+2*SW_bin)+'msBins_'+gridNum_str+'_'+str(samps)+'SWs' )
			#
			data = np.load( str(BestParams_dir + fname_str + '.npz') )
			#
			QQ_yc_sm = data['QQ_yc_sm']
			QQ_yc_sn = data['QQ_yc_sn'] 
			QQ_y1_sm = data['QQ_y1_sm']
			QQ_y1_sn = data['QQ_y1_sn']
			QQ_y2_sm = data['QQ_y2_sm']
			QQ_y2_sn = data['QQ_y2_sn']
			#
			K_iter = data['K_iter']
			Kmin_iter = data['Kmin_iter']
			Kmax_iter = data['Kmax_iter']
			#
			C_iter = data['C_iter']
			Cmin_iter = data['Cmin_iter'] 
			Cmax_iter = data['Cmax_iter']
			#
			mu_Pia_iter = data['mu_Pia_iter'] 
			sig_Pia_iter = data['sig_Pia_iter'] 
			mu_Pi_iter = data['mu_Pi_iter']
			sig_Pi_iter = data['sig_Pi_iter']
			#
			t1 = time.time()
			print('Time: ',t1-t0)




			# # # # # # # # # # # # # # # # # # # # # # # # # 
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
			# # # # # # # # # # # # # # # # # # # # # # # # # 
			#
			# Sort through matrices of QQ measurements and grab the top couple and
			# Loop through the different parameters for synthetic model data generation.
			#
			#
			# # 
			if stim=='Mov':
				measure = wts[0]*QQ_yc_sm + wts[1]*QQ_y1_sm + wts[2]*QQ_y2_sm # yc = cardinality, y1 = single cell activity, y2 = pairwise coactivity.
			elif stim=='Wnz':
				measure = wts[0]*QQ_yc_sn + wts[1]*QQ_y1_sn + wts[2]*QQ_y2_sn # yc = cardinality, y1 = single cell activity, y2 = pairwise coactivity.
			else:
				print('Dont understand stim?', stim)
			#
			srtd = np.sort(measure, axis=None)
			for ii in range(nBestFits): 
				indx = np.where(measure == srtd[ii])
				#
				K 	 = K_iter[ indx[0][0] ]
				Kmin = Kmin_iter[ indx[0][0] ]
				Kmax = Kmax_iter[ indx[0][0] ]
				C 	 = C_iter[ indx[1][0] ]
				Cmin = Cmin_iter[ indx[1][0] ]
				Cmax = Cmax_iter[ indx[1][0] ]
				
				mu_Pia = mu_Pia_iter[ indx[2][0] ]
				sig_Pia = sig_Pia_iter[ indx[3][0] ]
				mu_Pi = mu_Pi_iter[ indx[4][0] ]
				sig_Pi = sig_Pi_iter[ indx[5][0] ]

				#
				paramsFit_str = str( 'K'+str(K)+'_'+str(Kmin)+'_'+str(Kmax)+'_C'+str(C)+'_'+str(Cmin)+'_'+ \
					str(Cmax)+'_Pia'+str(mu_Pia)+'_'+str(sig_Pia)+'_Pi'+str(mu_Pi)+'_'+str(sig_Pi) )
				print('Top ',str(ii),' synth params:  ',paramsFit_str)

				# # # # # # # # # # # # # # # # # # # # # # # # # 
				# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
				# # # # # # # # # # # # # # # # # # # # # # # # # 
				#
				# Construct synthetic model and generate spike words data.
				print('Construct synthetic model.')
				#
				t0 = time.time()
				q = None #ri = None #ria = None #ria_mod = None
				while q==None: # while statement to resynthesize model if any cells participate in 0 assemblies.
					q, ri, ria, ria_mod = rc.synthesize_model(N,M,M_mod, C,K, Cmin,Cmax, bernoulli_Pi,mu_Pi,sig_Pi, mu_Pia,sig_Pia, verbose=False)
				#
				t1 = time.time()
				print('Time: ',t1-t0)
				#
				print('Generate spike words data synth model')
				t0 = time.time()
				Y_synth, Z_synth = rc.generate_data(samps, q, ri, ria, Cmin, Cmax, Kmin, Kmax)
				t1 = time.time()
				print('Time: ',t1-t0)

				print('compute_dataGen_Histograms on synth data.')
				t0 = time.time()
				Ycell_histS, Zassem_histS, nYs, nZs, CA_coactivityS, Cell_coactivityS = \
					rc.compute_dataGen_Histograms( Y_synth, Z_synth, M, N )
				t1 = time.time()
				print('Time: ',t1-t0)


				# # # # # # # # # # # # # # # # # # # # # # # # # 
				# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
				# # # # # # # # # # # # # # # # # # # # # # # # # 
				#
				# Histogram distributions and compute KL-divergence similarity measure. 
				#
				print('compute KL divergence between Synth, Movie, Noise, |y|, single activity, and pairwise coactivity.')
				t0 = time.time()
				#
				yc_sm, yc_sn, yc_nm, y1_sm, y1_sn, y1_nm, y2_sm, y2_sn, y2_nm, \
				mx1, nx1, sx1, mx2, nx2, sx2, mx3, nx3, sx3, \
				m1, n1, s1, m2, n2, s2, m3, n3, s3 = rc.compute_QQ_diff_metric( \
					nYr_Mov, nYr_Wnz, nYs, Ycell_histR_Mov[:-1], Ycell_histR_Wnz[:-1], Ycell_histS,  \
					Cell_coactivityR_Mov, Cell_coactivityR_Wnz, Cell_coactivityS)
				#
				t1 = time.time()
				print('Time: ',t1-t0)


				# MAYBE TODO. PLOT AND FIT DISTRIBUTIONS OF ROWSUMS IN PAIRWISE CORRELATIONS. IF SO, WHERE TO STOP ??
				if plot_CDFs_n_PDFs:
					params_str = str( realData_str + '_top' + str(ii) + '_' + paramsFit_str )
					pf.plot_QQ_bestFit_dists(yc_sm, yc_sn, yc_nm, y1_sm, y1_sn, y1_nm, y2_sm, y2_sn, y2_nm, mx1, nx1, sx1, \
						mx2, nx2, sx2, mx3, nx3, sx3, m1, n1, s1, m2, n2, s2, m3, n3, s3, plt_grid_dir, params_str, figSaveFileType)
	








