import argparse
import numpy as np 
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import os
import time

import utils.data_manipulation as dm
import utils.plot_functions as pf
import utils.retina_computation as rc
from textwrap import wrap

import difflib





def raster_zs_inferred_xValSWs_given_model(args): # y=DEFAULTS["y"], ):  #


	# Extract variables from args input into function from command line
	argsRaster = args

	print(args)
	globals().update(vars(args))

	# Turn comma delimited strings input into lists.
	lRateScale 	= [float(item) for item in args.lRateScale.split(',')]

	sig_init 		= np.array([ sigQ_init, sigPi_init, sigPia_init ])
	params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )

	msBins = 1+2*SW_bin

	if flg_include_Zeq0_infer:
		z0_tag='_zeq0'
	else:
		z0_tag='_zneq0'

	if train_2nd_model:
		ToBorNot2B = 'B'
	else:
		ToBorNot2B = ''	

	if not np.isnan( maxSamps ):
		maxSampTag = str( '_'+str( int(maxSamps ) )+'Samps' )
	else:
		maxSampTag = '_allSamps'

	if not np.isnan( maxRasTrials ):
		maxTrialTag = str( str( int(maxRasTrials ) )+'Trials' )
	else:
		maxTrialTag = 'allTrials'	


	if flg_EgalitarianPrior:	
		priorType = 'EgalQ' 
	else:
		priorType = 'BinomQ'		


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # (0). Set up directories and create dirs if necessary.
	dirHome, dirScratch = dm.set_dir_tree()
	EM_learning_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Models_learned_EM/')
	SW_extracted_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')
	data_save_dir  		= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Rasters_trial_vs_time_zInferred/')


	init_dir = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') + \
			'_LRsc' + str(lRateScale) +'/' )
	#
	# #
	# Find directory (model_dir) with unknown N and M that matches cell_type and yMin
	CST = str(cell_type).replace('\'','').replace(' ','') # convert cell_type list into a string of expected format.
	#
	subDirs = os.listdir( str(EM_learning_Dir + init_dir) ) 

	model_dir = [s for s in subDirs if CST in s and str( z0_tag + '_ylims' + str(yLo) + '_' + str(yHi) + '_yMinSW' + \
					str(yMinSW) + '_' + sample_longSWs_1st + 'Smp1st_' + priorType ) in s ]
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
	# #
	# Find npz file (model_file) inside model_dir with unknown numSWs but matching stim, msBins, pctXvalTrain, EMsamps and rand.
	filesInDir = os.listdir( str(EM_learning_Dir + init_dir + model_dir) )

	if whichGLM=='real':
		GLMtag = '';
	else:
		GLMtag = str('_GLM'+whichGLM)


	print(filesInDir)
	print(str('LearnedModel' + GLMtag + '_' + stim))
	print(str( 'SWs_' + str(pct_xVal_train).replace('.','pt') + 'trn_' + str(msBins) + 'msBins' + maxSampTag + '_rand' + str(rand) + ToBorNot2B + '.npz' ) )
	#
	model_files = [s for s in filesInDir if str('LearnedModel' + GLMtag + '_' + stim) in s and \
			str( 'SWs_' + str(pct_xVal_train).replace('.','pt') + 'trn_' + str(msBins) + 'msBins' + maxSampTag + '_rand' + str(rand) + ToBorNot2B + '.npz' ) in s ]
	#
	if len(model_files) != 1:
		print('I am expecting one matching file. I have ',len(model_files))
		print(model_files)
	#
	model_file = model_files[0]
	#
	# Find numSWs.
	a = model_file.find(stim)
	b = model_file.find('SWs_')
	print(a, b, model_file[a+1+len(stim):b])
	numSWs = int( model_file[a+1+len(stim):b] )
	print('numSWs = ',numSWs)


	



	# Set up directory for output raster npz file. Also, check if it already exists, and if so, say fuck it.
	if not os.path.exists( str(data_save_dir + init_dir + model_dir) ): # Make a directory for output plots if it is not already there.
		os.makedirs( str(data_save_dir + init_dir + model_dir) )
	#
	#inferZ_data_save_file = str(data_save_dir + init_dir + model_dir + model_file.replace('LearnedModel_','inferZ_allSWs_'))
	
	rasterZ_data_save_file = str(data_save_dir + init_dir + model_dir + model_file.replace('LearnedModel_',str('rasterZ_xValSWs_'+maxTrialTag+'_') ))
	print( rasterZ_data_save_file )







	#flg_checkNPZvars = True # CHECK THAT THE LIST BELOW IS COORECT AND CONTAINS EVERTHING WE ARE SAVING
	# INTO NPZ FILES BEFORE RUNNING THIS FLAG=TRUE. WILL DELETE FILES THAT DO NOT CONTAIN THESE VARIABLES.
	#
	realKeys = [ 'Z_inferred_allSWs', 'pyiEq1_gvnZ_allSWs', 'pj_inferred_allSWs', 'cond_inferred_allSWs', \
		'Ycell_hist_allSWs', 'YcellInf_hist_allSWs', 'Zassem_hist_allSWs', 'nY_allSWs',  'nYinf_allSWs', \
		'nZ_allSWs', 'CA_coactivity_allSWs', 'Cell_coactivity_allSWs', 'CellInf_coactivity_allSWs', \
		'argsRecModelLearn', 'argsRaster' ]




	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# 
	# Check that npz file contains the all the right, most current keys.
	# If it doesnt, remove the file.
	fExists = False
	#
	if os.path.isfile(rasterZ_data_save_file):
		fExists=True
		#
		if flg_checkNPZvars: 
			data = np.load(rasterZ_data_save_file)
			if not all (k in data.keys() for k in realKeys):
				print('Old File. Removing it: ',rasterZ_data_save_file)
				os.remove(rasterZ_data_save_file)
				fExists=False

	else:
		fExists = False



	#
	if fExists:
		print('File(s) already exist for this rand', str(rand) ,' and pct_xVal_train', pct_xVal_train ,'. Not replacing em.')
		return
	else:
		print('File(s) do not exist for this rand', str(rand) ,' and pct_xVal_train', pct_xVal_train ,'.')
	



		# Load in model from saved npz file.
		print(str( 'Loading ' + EM_learning_Dir + init_dir + model_dir + model_file ) )
		try:
			data = np.load( str(EM_learning_Dir + init_dir + model_dir + model_file) )
		except:
			#delete npz file
			# os.remove( str(EM_learning_Dir + init_dir + model_dir + model_file)  )
			print('Learned Model NPZ file corrupt. Uncomment to Remove it!')
		#
		#print(data.keys())

		qp = data['qp']
		rip = data['rip']
		riap = data['riap']


		if train_2nd_model:
			SWs = data['SWs_train']	# if running on the rand0B model 
		else:
			SWs = data['SWs_test'] 	# if running on the rand0 model

		meanT 	= np.mean((minTms,maxTms))
		SWtimes = meanT*np.ones_like(SWs)


		argsRecModelLearn = data['argsRec']

		print('any nans?: ',np.any(np.isnan(qp)) or np.any(np.isnan(rip)) or np.any(np.isnan(riap)) )



		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		# # (2).  Get extracted spikewords from spike data to look at statistics of spiking activity.
		#
		# Extract spike words from raw data or load in npz file if it has already been done and saved.
		if not os.path.exists( SW_extracted_Dir ): # Make directories for output data if  not already there.
			os.makedirs( SW_extracted_Dir )


		CAs = np.nan # Because there are no GT CAs
		numTrials=1 # HARD CODED BECAUSE OF HOW SWs_TEST AND SWs_TRAIN ARE SAVED.
		SWs = [ list(SWs) ]
		SWtimes = [ list(SWtimes) ]
		# NOTE: Need to send in SWs and SWtimes as lists of lists so that indexing inside works.
		# because infer_allSWs function was intended to process SWs as ndarray of lists of sets
		# and SWtimes as ndarray of lists of int64s.


		# # Infer Y and Z for every spike word.
		print('Inferring Z,Y,pjoint for all Spike Words with model post-EM-learning')
		t0 = time.time()
		Z_inferred_allSWs, pyiEq1_gvnZ_allSWs, pj_inferred_allSWs, cond_inferred_allSWs = rc.inferZ_allSWs( SWs, \
			SWtimes , CAs, numTrials, minTms, maxTms, qp, rip, riap, flg_include_Zeq0_infer, yLo, yHi, verbose=False )
		t1 = time.time()										
		print('Done Inferring Z,Y,pjoint for all Spike Words with model post-EM-learning: time = ',t1-t0)


		# print(type(Z_inferred_allSWs))
		# print( len(Z_inferred_allSWs) )
		# print( len(Z_inferred_allSWs[0]) )
		# print( len(Z_inferred_allSWs[0][0]) )
		# print( Z_inferred_allSWs )


		# # SWs is a list of lists (of sets of cells in an assembly). Unwrap it into a single list of sets.
		# print('Computing Statistics and histograms on all spike words and Z & Y inferred using model post-EM-learning')
		# t0 = time.time()
		# allSWs = list()
		# allSWtimes = list()
		# Zinf_allSWs_list = list()
		# pYgvnZ_allSWs_list = list()
		# #
		# for i in range(numTrials):
		# 	allSWs.extend(SWs[i])
		# 	allSWtimes.extend(SWtimes[i])
		# 	Zinf_allSWs_list.extend(Z_inferred_allSWs[i])
		# 	pYgvnZ_allSWs_list.extend(pyiEq1_gvnZ_allSWs[i])
		# # numSWs	= len(allSWs)
		# #
		# # #
		# #
		# Ycell_hist_allSWs, Zassem_hist_allSWs, nY_allSWs, nZ_allSWs, CA_coactivity_allSWs, Cell_coactivity_allSWs = \
		# 					rc.compute_dataGen_Histograms(allSWs, Zinf_allSWs_list, M, N)
		# #	
		# Ycell_hist_allSWs, YcellInf_hist_allSWs, nY_allSWs, nYinf_allSWs, CellInf_coactivity_allSWs, Cell_coactivity_allSWs = \
		# 					rc.compute_dataGen_Histograms(allSWs, pYgvnZ_allSWs_list, M, N)
		# #
		# t1 = time.time()
		# print('Computing Statistics and histograms on all spike words and Z & Y inferred using model post-EM-learning: time = ',t1-t0)



		Ycell_hist_allSWs 			= 0 
		Zassem_hist_allSWs 			= 0 
		nY_allSWs 					= 0 
		nZ_allSWs 					= 0
		CA_coactivity_allSWs 		= 0
		Cell_coactivity_allSWs 		= 0
		Ycell_hist_allSWs 			= 0 
		YcellInf_hist_allSWs 		= 0
		nY_allSWs 					= 0
		nYinf_allSWs 				= 0
		CellInf_coactivity_allSWs 	= 0
		Cell_coactivity_allSWs 		= 0


		
		# Save the inferred Zs and Ys on all spike words to an npz file
		# Change filename back to : infer_Z_data_save_file
		np.savez( rasterZ_data_save_file, Z_inferred_allSWs=Z_inferred_allSWs, pyiEq1_gvnZ_allSWs=pyiEq1_gvnZ_allSWs, 
			pj_inferred_allSWs=pj_inferred_allSWs, cond_inferred_allSWs=cond_inferred_allSWs, 
			Ycell_hist_allSWs=Ycell_hist_allSWs, YcellInf_hist_allSWs=YcellInf_hist_allSWs, \
			Zassem_hist_allSWs=Zassem_hist_allSWs, nY_allSWs=nY_allSWs,  nYinf_allSWs= nYinf_allSWs, nZ_allSWs=nZ_allSWs, \
			CA_coactivity_allSWs=CA_coactivity_allSWs, Cell_coactivity_allSWs=Cell_coactivity_allSWs, \
			CellInf_coactivity_allSWs=CellInf_coactivity_allSWs, argsRecModelLearn=argsRecModelLearn, argsRaster=argsRaster )
			










		# Here construct a raster of trial and time when each Z is inferred, Y is inferred, Y is active in observed SW.
		# NOTE: THIS IS VERY FAST AND IS CAUSING JOBS WITH MORE CELLS TO RUN OUT OF MEMORY. SO DONT DO IT.
		if False:
			print('Converting SWs,Zinf,Yinf into 3D tensor that is CellsCAs x Trials x TimesActive.')
			t0 = time.time()
			#
			raster_Z_inferred_allSWs = rc.compute_raster_list(SWtimes, Z_inferred_allSWs, M, minTms, maxTms )
			raster_Y_inferred_allSWs = rc.compute_raster_list(SWtimes, Y_inferred_allSWs, N, minTms, maxTms )
			raster_allSWs 			 = rc.compute_raster_list(SWtimes, SWs, N, minTms, maxTms ) 
			#
			t1 = time.time()
			print('Done Converting SWs,Zinf,Yinf into 3D tensor that is CellsCAs x Trials x TimesActive: time = ',t1-t0)

			# print(raster_allSWs) # Trials x Cells/CAs x spikeTimes
			# print(type(raster_allSWs))
			# print( len(raster_allSWs) )
			# print( len(raster_allSWs[0]) )
			# print( len(raster_allSWs[0][0]) )
			#
			# #
			# Save the rasters of inferred Zs, Ys and all spike words to an npz file
			np.savez( rasterZ_data_save_file, raster_Z_inferred_allSWs=raster_Z_inferred_allSWs, raster_Y_inferred_allSWs=raster_Y_inferred_allSWs, \
											raster_allSWs=raster_allSWs, argsRecModelLearn=argsRecModelLearn, argsRaster=argsRaster)


			# NOTE: ALL THESE IN INFER DATA FILE.
			# Z_inferred_allSWs=Z_inferred_allSWs, Y_inferred_allSWs=Y_inferred_allSWs, pj_inferred_allSWs=pj_inferred_allSWs, \
			# Ycell_hist_allSWs=Ycell_hist_allSWs, YcellInf_hist_allSWs=YcellInf_hist_allSWs, Zassem_hist_allSWs=Zassem_hist_allSWs, nY_allSWs=nY_allSWs, \
			# nYinf_allSWs= nYinf_allSWs, nZ_allSWs=nZ_allSWs, CA_coactivity_allSWs=CA_coactivity_allSWs, Cell_coactivity_allSWs=Cell_coactivity_allSWs, \
			# CellInf_coactivity_allSWs=CellInf_coactivity_allSWs,  )
			# 







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == "__main__":

	#
	# # Set up Parser
	#
	print('Trying to run as a command line function call.')
	parser = argparse.ArgumentParser(description="run PSTH_zs_inferred_allSWs_given_model from command line")





	#
	# # Info about which real data to process.
	#
	# cell_type
	parser.add_argument('-ct', '--cell_type', dest='cell_type', type=str, default=['offBriskTransient','offBriskSustained'], #default=DEFAULTS["cell_type"],
		help='What cell types to process from GField retina data.')
	# whichCells
	parser.add_argument('-cll', '--whichCells', dest='whichCells', type=str, default='', 
		help='Which subtypes of cells GLM simulations was run on. Either "offBT" or "offBT_offBS" or "offBT_onBT" or "" for real data.')
	# whichPop
	parser.add_argument('-pop', '--whichPop', dest='whichPop', type=str, default='',
		help='Which popluation of cells GLM simulation was run on "fullpop" for 137 cells or "subpop" of 8 or 9 cells or "" for real data')
	# whichGLM
	parser.add_argument('-glm', '--whichGLM', dest='whichGLM', type=str, default='real',
		help='Which type of GLM simulation was run "ind" or "cpl" or "real" for real data')
	# stim
	parser.add_argument('-st', '--stim', dest='stim', type=str, default='NatMov',
		help='Which stimuli presentations to process. ex( ''NatMov'' or ''Wnoise'' )')
	# SW_bin
	parser.add_argument('-sw', '--SW_bin', dest='SW_bin', type=int, default=0,
		help='Bin width in ms within which to consider spikes part of a spike word, ex([0,1,2]) --> 1(+/-)bin --> [1,3,5]ms ')
	# yLo
	parser.add_argument('-yLo', '--yLo', dest='yLo', type=int, default=1,
		help='If |y| < yLo, ignore the spike word for learning. Minimum number of active cells in a time bin to be considered a spike word and used to train model.')
	# yHi
	parser.add_argument('-yHi', '--yHi', dest='yHi', type=int, default=9999,
		help='If |y| > yHi, z=0 not allowed for inference. Assume there must be at least one CA on.')
	# yMinSW
	parser.add_argument('-ySW', '--yMinSW', dest='yMinSW', type=int, default=1,
		help='If |y| < yLo, ignore the spike word for learning. Minimum number of active cells in a time bin to be considered a spike word and used to train model.')
	# pct_xVal_train
	parser.add_argument('-xvp', '--pct_xVal_train', dest='pct_xVal_train', type=float, default=0.5,
		help='Percent of spike words to put into the training data set. Put the rest into test data set to perform cross-validation.')
	# num_test_samps_4xVal
	parser.add_argument('-nXs', '--num_test_samps_4xVal', dest='num_test_samps_4xVal', type=int, default=1,
		help='Number of test data samples to compute pjoint on and average over to implement cross validation.')
	# rand
	parser.add_argument('-rnd', '--rand', dest='rand', type=int, default=0,
		help='Due to order of spike words sampled, model learned can be different. Model can be learned with a different random sampling of spike words.')
	#
	# # Model Initialization parameters 
	#
	# params_init
	parser.add_argument('-pi', '--params_init', dest='params_init', type=str, default='NoisyConst',
		help='How to initialize the model for EM algorithm. ex(NoisyConst True NoisyTrue RandomUniform NoisyConst) ')
	# sigQ_init
	parser.add_argument('-si_q', '--sigQ_init', dest='sigQ_init', type=float, default=0.01,
		help='Variability / sigma in model params initializations. Distributed around centers defined by [C_noise_ri, C_noise_ria, overcomp, Z_hot') 
	# sigPi_init
	parser.add_argument('-si_ri', '--sigPi_init', dest='sigPi_init', type=float, default=0.05,
		help='Variability / sigma in model params initializations. Distributed around centers defined by [C_noise_ri, C_noise_ria, overcomp, Z_hot') 
	# sigPia_init
	parser.add_argument('-si_ria', '--sigPia_init', dest='sigPia_init', type=float, default=0.05,
		help='Variability / sigma in model params initializations. Distributed around centers defined by [C_noise_ri, C_noise_ria, overcomp, Z_hot') 
	
	# overcomp
	parser.add_argument('-oc', '--overcomp', dest='overcomp', type=int, default=1,
		help='How over overcomplete model is vs cells. M = (mo)*N')
	# Z_hot
	parser.add_argument('-z1', '--Z_hot', dest='Z_hot', type=int, default=4,
		help='How many 1''s in the sparse z-vector')
	# C_noise_ri
	parser.add_argument('-mi_ri', '--C_noise_ri', dest='C_noise_ri', type=float, default=1.,
		help='Center or mean of distributions on Pi in initializations. ex(=1 means all r_i initialized near silent)')
	# C_noise_ria
	parser.add_argument('-mi_ria', '--C_noise_ria', dest='C_noise_ria', type=float, default=1.,
		help='Center or mean of distributions on Pia in initializations. ex(=1 means all r_ia initialized near silent)')

	#
	# # Learning rate stuff and some options on Inference.
	#
	# learning_rate,
	parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=1.,
		help='Learning rate in model learning at each iteration. ex(1 works well)')  
	# lRateScale
	parser.add_argument('-LRS', '--lRateScale', dest='lRateScale', type=str, default='1.0,0.1,0.1',
		help='Comma delimited string to be converted into a list of learning rate scales for Pia, Pi and Q.')
	# train_2nd_model
	parser.add_argument('-2nd', '--train_2nd_model', dest='train_2nd_model', action='store_true', default=False,
		help='Flag to train a 2nd model with 50% test data so we can compare CAs found by each. ex( True or False))')	
	# flg_include_Zeq0_infer
	parser.add_argument('-z0', '--flg_include_Zeq0_infer', dest='flg_include_Zeq0_infer', action='store_true', default=False,
		help=', Flag to include the z=0''s vector in inference if True. ex( True or False))')
	# verbose_EM
	parser.add_argument('-v', '--verbose_EM', dest='verbose_EM', action='store_true', default=False,
		help=', Flag to display additional output messages (Sanity Checking). ex( True or False))')	
	# sample_longSWs_1st
	parser.add_argument('-smp1st', '--sample_longSWs_1st', dest='sample_longSWs_1st', type=str, default='Dont',
		help="Sample Spikewords for learning randomly, with probability proportional to their length, or sorted by there SW length. Options are: {'Dont', 'Prob', 'Hard'} ")
	# flg_EgalitarianPrior
	parser.add_argument('-Egal', '--flg_EgalitarianPrior', dest='flg_EgalitarianPrior', action='store_true', default=False,
		help='Flag to use Egalitarian activity based prior on z-vec by Q-vec. Otherwise use Binomial Q-scalar prior.. ex( True or False))')	
	# flg_checkNPZvars
	parser.add_argument('-chkNPZ', '--flg_checkNPZvars', dest='flg_checkNPZvars', action='store_true', default=False,
		help='Flag to check keys / variables in the NPZ file and get rid of the file if they dont match expected values.')
	# maxSamps
	parser.add_argument('-ms', '--maxSamps', dest='maxSamps', type=float, default=np.nan,
		help='Number of Spikewords to use as samples. A scalar number or NaN to use all available data.') 	
	# maxRasTrials
	parser.add_argument('-mrt', '--maxRasTrials', dest='maxRasTrials', type=float, default=np.nan,
		help='Number of Trials to infer post learning. A scalar number or NaN to use all available data.') 	


	#
	# # Max and min times to consider activity in spiketrains to compute PSTHs
	#
	# maxTms
	parser.add_argument('-Thi', '--maxTms', dest='maxTms', type=int, default=6000,
		help='Maximum time (in ms) to consider activity in spiketrains to compute PSTHs - Ignore the Spikes in last trial are recorded waaay longer. ')
	# minTms
	parser.add_argument('-Tlo', '--minTms', dest='minTms', type=int, default=0,
		help='Minimum time (in ms) to consider activity in spiketrains to compute PSTHs - Can ignore stim onset response with this. ')

	#
	# # Get args from Parser.
	#
	args = parser.parse_args()
	raster_zs_inferred_xValSWs_given_model(args) #args.x, args.y)  




