import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import stats as st
from textwrap import wrap

import pandas as pd
import argparse
import scipy as sp
from scipy import io as io
import matplotlib.pyplot as plt
import time 						# to time operations for code analysis
import os.path 						# to check if a file or directory exists already
import os
import sys

import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.plot_functions as pf
import utils.retina_computation as rc



def compare_SWdists_realNsynthData(args):

	print('Running compare_SWdists_realNsynthData to grid search over synth model params and match some y-vector distributions observed in real data.')



	print(args)

	# Extract variables from args input into function from command line
	argsRec = args
	globals().update(vars(args))


	# 		my_list = [int(item) for item in args.list.split(',')]
	# 		or
	#		my_list = [float(item) for item in args.list.split(',')]



	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # (0). Set up directories and create dirs if necessary.
	dirHome, dirScratch = dm.set_dir_tree()
	synthData_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/Models_learned_EM/')
	realData_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')
	realRaster_dir  	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Rasters_trial_vs_time_zInferred/')



	# Infer_postLrn_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/inference_postLrn/')
	# EM_learnStats_Dir	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/inferStats_from_EM_learning/')
	plt_save_dir = str( dirScratch + 'figs/PGM_analysis/compareSWs_realNsynth/')
	dat_save_dir = str( dirScratch + 'data/python_data/PGM_analysis/compareSWs_realNsynth/')
	#
	if not os.path.exists( str(plt_save_dir) ):
		os.makedirs( str(plt_save_dir) )
	#
	if not os.path.exists( str(dat_save_dir) ):
		os.makedirs( str(dat_save_dir) )

	#flg_include_Zeq0_infer = True
	if flg_include_Zeq0_infer:
		z0_tag='_zeq0'
	else:
		z0_tag='_zneq0'



	print('Real Data:  ', CST, str(1+2*SW_bin), 'msBins')

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
		fname_raster_Wnz = str( realRaster_dir + 'Init_DiagonalPia_2hot_mPi1.0_mPia1.0_sI[0.01 0.05 0.05]_LR0pt5_LRpi1pt0/' + \
			str(CST) + '_N' + str(N) +'_M' + str(M) + z0_tag + '_ylims' + str(yLo) + '_' + str(yHiR) + '_yMinSW' + str(yMinSW) + '/' + \
			'rasterZ_allSWs_Wnoise_' + str(numSWs_Wnz) + 'SWs_0pt5trn_' + str(1+2*SW_bin) + 'msBins_rand0.npz')
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
		fname_raster_Mov = str( realRaster_dir + 'Init_DiagonalPia_2hot_mPi1.0_mPia1.0_sI[0.01 0.05 0.05]_LR0pt1_LRpi1pt0/' + \
			str(CST) + '_N' + str(N) +'_M' + str(M) + z0_tag + '_ylims' + str(yLo) + '_' + str(yHiR) + '_yMinSW' + str(yMinSW) + '/' + \
			'rasterZ_allSWs_NatMov_' + str(numSWs_Mov) + 'SWs_0pt5trn_' + str(1+2*SW_bin) + 'msBins_rand0.npz')
		data_RasMov = np.load(fname_raster_Mov)
		nYr_Mov = data_RasMov['nY_allSWs'] 
		Ycell_histR_Mov = data_RasMov['Ycell_hist_allSWs'] 
		Cell_coactivityR_Mov = data_RasMov['Cell_coactivity_allSWs']
		del data_RasMov
	#
	t1 = time.time()
	print('Time: ',t1-t0)



	# Turn comma delimited strings input into lists. These are the parameters to loop over.
	K_iter 			= [int(item) for item in args.K_iter.split(',')]
	Kmin_iter 		= [int(item) for item in args.Kmin_iter.split(',')]
	Kmax_iter 		= [int(item) for item in args.Kmax_iter.split(',')]
	#
	C_iter 			= [int(item) for item in args.C_iter.split(',')]
	Cmin_iter 		= [int(item) for item in args.Cmin_iter.split(',')]
	Cmax_iter 		= [int(item) for item in args.Cmax_iter.split(',')]
	#
	mu_Pia_iter 	= [float(item) for item in args.mu_Pia_iter.split(',')]
	sig_Pia_iter 	= [float(item) for item in args.sig_Pia_iter.split(',')]
	mu_Pi_iter 		= [float(item) for item in args.mu_Pi_iter.split(',')]
	sig_Pi_iter 	= [float(item) for item in args.sig_Pi_iter.split(',')]



	# Multidimensional matrices to hold results of distance from unity line on QQ-plot.
	QQ_yc_sm = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for |y| distributions for synth and movie
	QQ_yc_sn = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for |y| distributions for synth and noise
	#
	QQ_y1_sm = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for single y activity distributions for synth and movie
	QQ_y1_sn = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for single y activity distributions for synth and noise
	#
	QQ_y2_sm = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for pairwise y coactivity distributions for synth and movie
	QQ_y2_sn = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for pairwise y coactivity distributions for synth and noise

	print('Number of tests to run is: ',QQ_yc_sm.size)




	# # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # 
	#
	# Loop through the different parameters for synthetic model data generation.
	#
	#
	for Kind in range(len(K_iter)):	# Number of	cell assemblies active, given a Bernoulli distribution ("hotness" of Z-vector)
		K = K_iter[Kind]
		Kmin = Kmin_iter[Kind]
		Kmax = Kmax_iter[Kind]
		#
		for Cind in range(len(C_iter)):	# Number of cells participating in cell assemblies. ("hotness" of each row in Pia)
			C = C_iter[Cind]
			Cmin = Cmin_iter[Cind]
			Cmax = Cmax_iter[Cind]
			#
			for a,mu_Pia in enumerate(mu_Pia_iter): 	# Binary values in Pia matrix made continuous by sampling from Gaussian centered at 1-mu_Pia or 0+mu_Pia.
				for b,sig_Pia in enumerate(sig_Pia_iter):	# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
					#
					for c,mu_Pi in enumerate(mu_Pi_iter): 	# Binary values in Pi vector made continuous by sampling from Gaussian centered at 1-mu_Pi or 0+mu_Pi.
						for d,sig_Pi in enumerate(sig_Pi_iter): 	# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.


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
							Y_synth, Z_synth = rc.generate_data(synthSamps, q, ri, ria, Cmin, Cmax, Kmin, Kmax)
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


							yc_sm, yc_sn, yc_nm, y1_sm, y1_sn, y1_nm, y2_sm, y2_sn, y2_nm, \
							mx1, nx1, sx1, mx2, nx2, sx2, mx3, nx3, sx3, \
							m1, n1, s1, m2, n2, s2, m3, n3, s3 = rc.compute_QQ_diff_metric( \
								nYr_Mov, nYr_Wnz, nYs, Ycell_histR_Mov[:-1], Ycell_histR_Wnz[:-1], Ycell_histS,  \
								Cell_coactivityR_Mov, Cell_coactivityR_Wnz, Cell_coactivityS)
							#
							QQ_yc_sm[Kind,Cind,a,b,c,d] = yc_sm
							QQ_yc_sn[Kind,Cind,a,b,c,d] = yc_sn
							#
							QQ_y1_sm[Kind,Cind,a,b,c,d] = y1_sm
							QQ_y1_sn[Kind,Cind,a,b,c,d] = y1_sn
							#
							QQ_y2_sm[Kind,Cind,a,b,c,d] = y2_sm
							QQ_y2_sn[Kind,Cind,a,b,c,d] = y2_sn
							#
							t1 = time.time()
							print('Time: ',t1-t0)

						



	# # # # # # # # # # # #
	#
	# Save a file for each real data combo.
	#
	fname_str = str( CST+'_'+str(1+2*SW_bin)+'msBins_'+str(len(K_iter))+'Ks_'+str(len(C_iter))+'Cs_'+str(len(mu_Pia_iter)) \
		+'m'+str(len(sig_Pia_iter))+'sPiaS_'+str(len(mu_Pi_iter))+'m'+str(len(sig_Pi_iter))+'sPiS_'+str(synthSamps)+'SWs' )
	#
	np.savez( str(dat_save_dir+fname_str), 
		QQ_yc_sm=QQ_yc_sm, QQ_yc_sn=QQ_yc_sn, 
		QQ_y1_sm=QQ_y1_sm, QQ_y1_sn=QQ_y1_sn, 
		QQ_y2_sm=QQ_y2_sm, QQ_y2_sn=QQ_y2_sn, 
		K_iter=K_iter, Kmin_iter=Kmin_iter, Kmax_iter=Kmax_iter, 
		C_iter=C_iter, Cmin_iter=Cmin_iter, Cmax_iter=Cmax_iter,
		mu_Pia_iter=mu_Pia_iter, sig_Pia_iter=sig_Pia_iter, 
		mu_Pi_iter=mu_Pi_iter, sig_Pi_iter=sig_Pi_iter)
								
			





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == "__main__":


	#
	# # Set up Parser
	#
	print('Trying to run as a command line function call.')
	parser = argparse.ArgumentParser(description="run pgmCA_realData from command line")

	# CST
	parser.add_argument('-CST', '--CST', dest='CST', type=str, default='[offBriskTransient]',
		help='Cell Sub-types to look at.')
	# N
	parser.add_argument('-N', '--N', dest='N', type=int, default=55,
		help='Number of cells in collected data.')
	# M
	parser.add_argument('-M', '--M', dest='M', type=int, default=55,
		help='Number of cell assemblies in model. Often same as N, but doesnt have to be.')
	# M_mod
	parser.add_argument('-M_mod', '--M_mod', dest='M_mod', type=int, default=55,
		help='Number of cell assemblies in model. Same as M basically.')
	# numSWs_Mov
	parser.add_argument('-numSWs_Mov', '--numSWs_Mov', dest='numSWs_Mov', type=int, default=571276,
		help='Number of spike words in response to movie stim for a certain cell subtype.')		
	# numSWs_Wnz
	parser.add_argument('-numSWs_Wnz', '--numSWs_Wnz', dest='numSWs_Wnz', type=int, default=740267,
		help='Number of spike words in response to white noise stim for a certain cell subtype.')	
	# SW_bin
	parser.add_argument('-SW_bin', '--SW_bin', dest='SW_bin', type=int, default=0,
		help='Binning spike words into ms bins 1ms+/-SW_bin on either side.')
	# synthSamps
	parser.add_argument('-synthSamps', '--synthSamps', dest='synthSamps', type=int, default=10000,
		help='Number of samples to compute distributions from in synthesized data.')
	#
	# #
	#
	# # # # # #
	# 
	# Below are delimited lists. They work in the following way:
	# 		parser = ArgumentParser()
	# 		parser.add_argument('-l', '--list', help='delimited list input', type=str)
	# 		args = parser.parse_args()
	# 		my_list = [int(item) for item in args.list.split(',')]
	# 		or
	#		my_list = [float(item) for item in args.list.split(',')]
	#
	# 		Then, call it like...
	# 	python test.py -l "265340,268738,270774,270817" [other arguments]
	# 	or,
	# 	python test.py -l 265340,268738,270774,270817 [other arguments]
	#
	#
	# K_iter
	parser.add_argument('-K', '--K_iter', dest='K_iter', type=str, default='1,2,3',
		help='Comma delimited string to be converted into a list of K values to be grid searched over.')
	# Kmin_iter
	parser.add_argument('-Kmin', '--Kmin_iter', dest='Kmin_iter', type=str, default='1,2,3',
		help='Kmin values to be grid searched over. Same length as K_iter.')
	# Kmax_iter 
	parser.add_argument('-Kmax', '--Kmax_iter', dest='Kmax_iter', type=str, default='1,2,3',
		help='Kmax values to be grid searched over. Same length as K_iter.')
	# C_iter
	parser.add_argument('-C', '--C_iter', dest='C_iter', type=str, default='1,2,3',
		help='')
	# Cmin_iter
	parser.add_argument('-Cmin', '--Cmin_iter', dest='Cmin_iter', type=str, default='1,2,3',
		help='')
	# Cmax_iter
	parser.add_argument('-Cmax', '--Cmax_iter', dest='Cmax_iter', type=str, default='1,2,3',
		help='')
	# mu_Pia_iter
	parser.add_argument('-mu_Pia', '--mu_Pia_iter', dest='mu_Pia_iter', type=str, default='1,2,3',
		help='')
	# sig_Pia_iter
	parser.add_argument('-sig_Pia', '--sig_Pia_iter', dest='sig_Pia_iter', type=str, default='1,2,3',
		help='')
	# mu_Pi_iter
	parser.add_argument('-mu_Pi', '--mu_Pi_iter', dest='mu_Pi_iter', type=str, default='1,2,3',
		help='')
	# sig_Pi_iter
	parser.add_argument('-sig_Pi', '--sig_Pi_iter', dest='sig_Pi_iter', type=str, default='1,2,3',
		help='')
	#
	# #
	#

	# bernoulli_Pi
	parser.add_argument('-bernoulli_Pi', '--bernoulli_Pi', dest='bernoulli_Pi', type=float, default=1.,
		help='Probability of drawing a 0 (very noisy cell) in Pi vector.')
	# yLo
	parser.add_argument('-yLo', '--yLo', dest='yLo', type=int, default=0,
		help='If |y| < yLo, ignore the spike word for learning. Minimum number of active cells in a time bin to be considered a spike word and used to train model.')
	# yHi
	parser.add_argument('-yHi', '--yHi', dest='yHi', type=int, default=1000,
		help='If |y| > yHi, z=0 not allowed for inference. Assume there must be at least one CA on.')
	# yHiR
	parser.add_argument('-yHiR', '--yHiR', dest='yHiR', type=int, default=1000,
		help='If |y| > yHi, z=0 not allowed for inference. Assume there must be at least one CA on.')
	# yMinSW
	parser.add_argument('-yMinSW', '--yMinSW', dest='yMinSW', type=int, default=1,
		help='If |y| < yMinSW, throw it away and dont use it for training model.')


	# flg_include_Zeq0_infer
	parser.add_argument('-z0', '--flg_include_Zeq0_infer', dest='flg_include_Zeq0_infer', action='store_true', default=False,
		help=', Flag to include the z=0''s vector in inference if True. ex( True or False))')
	# verbose_EM
	parser.add_argument('-v', '--verbose_EM', dest='verbose_EM', action='store_true', default=False,
		help=', Flag to dislay additional output messages (Sanity Checking). ex( True or False))')	




	# # K
	# parser.add_argument('-K', '--K', dest='K', type=int, default=2,
	# 	help='Number of active cell assemblies in a spike word |z-vec|. Actual sample pulled from a bernoulli distribution w/ p(z_a=1) = K/M.')		
	# # Kmin
	# parser.add_argument('-Kmin', '--Kmin', dest='Kmin', type=int, default=0,
	# 	help='Minimum number of active cell assemblies in a sampled spike word |z-vec|.')
	# # Kmax
	# parser.add_argument('-Kmax', '--Kmax', dest='Kmax', type=int, default=2,
	# 	help='Maximum number of active cell assemblies in a sampled spike word |z-vec|.')
	# # C
	# parser.add_argument('-C', '--C', dest='C', type=int, default=2,
	# 	help='Number of active cells in a given cell assembly. Actual sample pulled from a bernoulli distribution w/ p(z_a=1) = C/N.')		
	# # Cmin
	# parser.add_argument('-Cmin', '--Cmin', dest='Cmin', type=int, default=2,
	# 	help='Minimum number of active cell in a given cell assembly. Resample if not satisfied.')
	# # Cmax
	# parser.add_argument('-Cmax', '--Cmax', dest='Cmax', type=int, default=6,
	# 	help='Maximum number of active cell in a given cell assembly. Resample if not satisfied.')

	# # mu_Pia
	# parser.add_argument('-mu_Pia', '--mu_Pia', dest='mu_Pia', type=float, default=0.,
	# 	help='"Mean" (distance from binary 0 or 1 values) of Pia matrix parameters in GT synthesized model used to construct data')	
	# # sig_Pia
	# parser.add_argument('-sig_Pia', '--sig_Pia', dest='sig_Pia', type=float, default=0.1,
	# 	help='Spread or STD of Pia matrix parameters in GT synthesized model used to construct data')
	# # bernoulli_Pi
	# parser.add_argument('-bernoulli_Pi', '--bernoulli_Pi', dest='bernoulli_Pi', type=float, default=1.,
	# 	help='Probability of drawing a 0 (very noisy cell) in Pi vector.')
	# # mu_Pi
	# parser.add_argument('-mu_Pi', '--mu_Pi', dest='mu_Pi', type=float, default=0.,
	# 	help='"Mean" (distance from binary 0 or 1 values) of Pi vector parameters in GT synthesized model used to construct data')		
	# # sig_Pi
	# parser.add_argument('-sig_Pi', '--sig_Pi', dest='sig_Pi', type=float, default=0.1,
	# 	help='Spread or STD of Pi vector parameters in GT synthesized model used to construct data')	
				



	#
	# # Get args from Parser.
	#
	args = parser.parse_args()
	compare_SWdists_realNsynthData(args)  


