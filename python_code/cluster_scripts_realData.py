import numpy as np
import os
import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.sbatch_scripts as ss



# In Greg Field's data :::
#        Cell Types : Num Cells
# ------------------------------
# offBriskTransient : 55 cells
# offBriskSustained : 43 cells
#  onBriskTransient : 39 cells
#      offExpanding : 13 cells
#      offTransient :  4 cells
#  onBriskSustained :  6 cells
#       onTransient :  7 cells
#       dsOnoffDown :  7 cells
#      dsOnoffRight :  3 cells
#       dsOnoffLeft :  3 cells
#         dsOnoffUp :  2 cells


#
dirHomeLoc, dirScratch = dm.set_dir_tree()

# Parameters we can loop over.
stims = ['NatMov']#,'Wnoise']
cell_types = ['allCells'] 
cellSubTypeCombinations = [ ['offBriskTransient'], ['offBriskTransient','offBriskSustained'], ['offBriskTransient','onBriskTransient'] ] # a list of lists. Each internal list is a combination of cell sub types to consider as a group to find Cell Assemblies within them.
#cellSubTypeCombinations = [ ['offBriskTransient','onBriskTransient'] ]
num_test_samps_4xValS 	= [1] #[1, 10, 100] 
model_CA_overcompleteness = [1] 		# [1,2] 	# how many times more cell assemblies we have than cells (1 means complete - N=M, 2 means 2x overcomplete)
SW_bins = [2]#, 2,1,0] 					# ms. Build up spikewords from groups of cells in the same trial that fire within SW_bins of eachother.
learning_rates = [0.5] #[1.0, 0.5, 0.1]
lRateScaleS = [[1., 0.1, 0.1]]# , [1., 0.1, 1.] ]  # Multiplicative scaling to Pia,Pi,Q learning rates. Can set to zero and take Pi taken out of model essentially.



yLo_Vals 	= [0] #[1] 		# If |y|<=yLo, then we force the z=0 inference solution and change Pi. This defines cells assemblies to be more than 1 cell.
yHi_Vals	= [1000] 		# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution and change Pia.
yMinSWs 	= [1]			# DOING BELOW THING WITH YYY inside pgm functions. --> (set to 0, so does nothing) 
								# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<yLo.
#

train_2nd_model = True
if train_2nd_model:
	pct_xVal_train = 0.5
else:
	pct_xVal_train = 0.9

num_EM_rands = 3

maxSampsS = [np.nan] #[50000,100000] # np.nan if you want to use all the SWs for samples or a scalar to use only a certain value

maxRasTrials = np.nan # Single scalar value. np.nan to use all trials when inferring all spikewords post learning. Otherwise use a small number here.


ds_fctr_snapshots 	= 1000 	# Take a sample of model parameters every *ds_fctr* time steps to compute and plot MSE after we determine permutation of learned CA's to True CA's
xVal_snapshot 		= 1000
xVal_batchSize 		= 100


mem = 7500 #False #	# amount of memory (in MB) to allocate per CPU for each job submitted to cluster. 
# pgmR needs more memory, (3.5G?) when running for 500k EM samples.
# rasZ needs more memory (7.5) when running for all spike words also for N=98
			# set to False to not specify amount of memory to use.



what_to_run = 'rasZ' 	# (1). 	'pgmR' - Learn model on real data 
						# (2). 	'rasZ' - Infer z's for all spike words using learned model
						# (2). 	'rasX' - Infer z's for test-set spikewords for cross validation for model comparison.
						# (1&2) 'PnRr' - Do #1 and then #2 after, in serial
						# (3). 	'statI'- compute statistics on inference step - post learning.



		
# if what_to_run=='pgmG':
# 	whichCells = ['OffBT', 'OffBT_OffST']
# 	whichGLM = ['indep','cpled']
# else:
# 	whichCells = ['xxx'] # If NOT running pgmG, have one entry in these and it wont be used.
# 	whichGLM = ['xxx'] 	 # If there are multiple entries, it will run real data stuff multiple times.



flg_checkNPZvars = True # True or False. Check variables saved in NPZ file and delete file and regenerate it if it does not contain the expected variables (hardcoded in each funciton.)

# NEEDED FOR THE rasZ and statI
#N 	 	= 55 # number of cells in real data
maxTms 	= 6000 # ms 
minTms 	= 0 # ms 


sample_longSWs_1stS = ['Prob','Dont']    # Options are: {'Dont', 'Prob', 'Hard'}

flg_EgalitarianPriorS = [True]#,False]

# Flags for running on clusters (nersc or cortex)
whichClust = 'cortex' # 'cortex' or 'nersc'.
n_cores=1
time='10-00:00:00' # hhhh:mm:ss



							 	# number of times to randomize samples and rerun EM on same data generated from single synthetic model.
params_init 	= 'NoisyConst' #'DiagonalPia'						# Options: {'True', RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise), 'DiagonalPia' (w/ sig_init & C_noise) }
#sig_init 		= np.array([ 0.01, 0.05, 0.05 ])	# STD on parameter initializations from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigQ_init 		= 0.01	# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigPi_init 		= 0.05	# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigPia_init 	= 0.05	# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.

# Parameter initializations for EM algorithm to learn model parameters
Z_hot 				= 5 	# initialization for Q value (how many 1's expected in binary z-vector)
C_noise_ri 			= 1.
C_noise_ria 		= 1.




# Flags for the EM (inference & learning) algorithm.
flg_include_Zeq0_inferS  = [True] # [True, False]
verbose_EM 				 = False
flg_compute_StatsPostLrn = True
flg_compute_StatsDuringEM = True



# Set up directory structure and make dirs if they dont exist already.
if whichClust=='nersc':
	homeDir = '/global/homes/w/warner/'
if whichClust=='cortex':
	homeDir = '/global/home/users/cwarner/'
cluster_path = str( homeDir + 'Projects/G_Field_Retinal_Data/cluster_scripts/realData/')
#	
base_path = str(dirHomeLoc + 'cluster_scripts/realData/')
if not os.path.exists(base_path):
	os.makedirs(base_path)







# Loop through all parameters and write sbatch scripts to submit jobs to the cluster.		
wrapper = open( str(base_path + what_to_run + '.wrap') , 'w') # open wrapper file to call all the sbatch scripts written in the loop.

i = 0
for rand in range(num_EM_rands):
	#
	for num_test_samps_4xVal in num_test_samps_4xValS:
		#
		for yLo in yLo_Vals:
			#
			for yHi in yHi_Vals:
				#
				for yMinSW in yMinSWs:
					#
					for cell_type in cell_types:
						#
						for cellSubTypes in cellSubTypeCombinations:
							#
							for k,stim in enumerate(stims):
								#
								for learning_rate in learning_rates:
									#
									for lRateScale in lRateScaleS:
										#
										for overcomp in model_CA_overcompleteness:
											#
											for SW_bin in SW_bins:
												#
												for flg_include_Zeq0_infer in flg_include_Zeq0_inferS:
													#
													for flg_EgalitarianPrior in flg_EgalitarianPriorS:
														#
														for sample_longSWs_1st in sample_longSWs_1stS:
															# 
															for maxSamps in maxSampsS:
																#
																i+=1
																fname = str( what_to_run + str(i) )
																script_path = str(base_path    + fname + '.s')
																output_path = str(cluster_path + 'out_files/' + fname + '.o')
																error_path  = str(cluster_path + 'out_files/' + fname + '.e')	

																# write each sbatch call in wrapper file that I can just call once. NOTE: directory here is dir on NERSC cluster.
																wrapper.write( str( 'sbatch ' + cluster_path + fname + '.s \n') ) 
																#wrapper.write( str( 'sleep 1 \n') ) 	





																if what_to_run is 'pgmR':
																	ss.write_sbatch_script_pgmCA_realData(script_path=script_path, output_path=output_path, error_path=error_path, 
																		whichClust=whichClust, n_cores=n_cores, mem=mem, time=time, job_name=fname, cell_type=cell_type, 
																		cellSubTypes=cellSubTypes, stim=stim, num_test_samps_4xVal=num_test_samps_4xVal, rand=rand, SW_bin=SW_bin, 
																		yLo=yLo, yHi=yHi, yMinSW=yMinSW, params_init=params_init, sigQ_init=sigQ_init, sigPi_init=sigPi_init, 
																		sigPia_init=sigPia_init, overcomp=overcomp, Z_hot=Z_hot, C_noise_ri=C_noise_ri, C_noise_ria=C_noise_ria, 
																		learning_rate=learning_rate, lRateScale=lRateScale, ds_fctr_snapshots=ds_fctr_snapshots, 
																		pct_xVal_train=pct_xVal_train, xVal_snapshot=xVal_snapshot, xVal_batchSize=xVal_batchSize, 
																		flg_EgalitarianPrior=flg_EgalitarianPrior, train_2nd_model=train_2nd_model, flg_include_Zeq0_infer=flg_include_Zeq0_infer, 
																		sample_longSWs_1st=sample_longSWs_1st, maxSamps=maxSamps, flg_checkNPZvars=flg_checkNPZvars, verbose_EM=verbose_EM)






																if what_to_run is 'rasZ':
																	#
																	# Infer Z's for all Spike Words (in train data set if train_2nd_model=False; in test dataset if train_2nd_model=True)
																	ss.write_sbatch_script_rasterZ_realData(script_path=script_path, output_path=output_path, error_path=error_path, 
																		whichClust=whichClust, n_cores=n_cores, mem=mem, time=time, job_name=fname, cell_type=cellSubTypes, stim=stim, 
																		num_test_samps_4xVal=num_test_samps_4xVal, rand=rand, SW_bin=SW_bin, yLo=yLo, yHi=yHi, yMinSW=yMinSW, params_init=params_init, 
																		sigQ_init=sigQ_init, sigPi_init=sigPi_init, sigPia_init=sigPia_init, overcomp=overcomp, Z_hot=Z_hot, 
																		C_noise_ri=C_noise_ri, C_noise_ria=C_noise_ria, learning_rate=learning_rate, lRateScale=lRateScale, 
																		maxTms=maxTms, minTms=minTms, pct_xVal_train=pct_xVal_train, flg_EgalitarianPrior=flg_EgalitarianPrior, train_2nd_model=train_2nd_model, 
																		flg_include_Zeq0_infer=flg_include_Zeq0_infer, sample_longSWs_1st=sample_longSWs_1st, maxSamps=maxSamps, maxRasTrials=maxRasTrials, 
																		flg_checkNPZvars=flg_checkNPZvars, verbose_EM=verbose_EM)
																	# 
																	# Infer Z's for all Spike Words in train data set if train_2nd_model=True)
																	if train_2nd_model: 
																		i+=1
																		fname = str( what_to_run + str(i) )
																		script_path = str(base_path    + fname + '.s')
																		output_path = str(cluster_path + 'out_files/' + fname + '.o')
																		error_path  = str(cluster_path + 'out_files/' + fname + '.e')	
																		wrapper.write( str( 'sbatch ' + cluster_path + fname + '.s \n') ) 
																		#
																		ss.write_sbatch_script_rasterZ_realData(script_path=script_path, output_path=output_path, error_path=error_path, 
																			whichClust=whichClust, n_cores=n_cores, mem=mem, time=time, job_name=fname, cell_type=cellSubTypes, stim=stim, 
																			num_test_samps_4xVal=num_test_samps_4xVal, rand=rand, SW_bin=SW_bin, yLo=yLo, yHi=yHi, yMinSW=yMinSW, params_init=params_init, 
																			sigQ_init=sigQ_init, sigPi_init=sigPi_init, sigPia_init=sigPia_init, overcomp=overcomp, Z_hot=Z_hot, 
																			C_noise_ri=C_noise_ri, C_noise_ria=C_noise_ria, learning_rate=learning_rate, lRateScale=lRateScale, 
																			maxTms=maxTms, minTms=minTms, pct_xVal_train=pct_xVal_train, flg_EgalitarianPrior=flg_EgalitarianPrior, train_2nd_model=False, 
																			flg_include_Zeq0_infer=flg_include_Zeq0_infer, sample_longSWs_1st=sample_longSWs_1st, maxSamps=maxSamps, maxRasTrials=maxRasTrials, 
																			flg_checkNPZvars=flg_checkNPZvars, verbose_EM=verbose_EM)







																if what_to_run is 'rasX':
																	#
																	# Infer Z's for all Spike Words (in train data set if train_2nd_model=False; in test dataset if train_2nd_model=True)
																	ss.write_sbatch_script_rasterZ_xVal_realData(script_path=script_path, output_path=output_path, error_path=error_path, 
																		whichClust=whichClust, n_cores=n_cores, mem=mem, time=time, job_name=fname, cell_type=cellSubTypes, stim=stim, 
																		num_test_samps_4xVal=num_test_samps_4xVal, rand=rand, SW_bin=SW_bin, yLo=yLo, yHi=yHi, yMinSW=yMinSW, params_init=params_init, 
																		sigQ_init=sigQ_init, sigPi_init=sigPi_init, sigPia_init=sigPia_init, overcomp=overcomp, Z_hot=Z_hot, 
																		C_noise_ri=C_noise_ri, C_noise_ria=C_noise_ria, learning_rate=learning_rate, lRateScale=lRateScale, 
																		maxTms=maxTms, minTms=minTms, pct_xVal_train=pct_xVal_train, flg_EgalitarianPrior=flg_EgalitarianPrior, train_2nd_model=train_2nd_model, 
																		flg_include_Zeq0_infer=flg_include_Zeq0_infer, sample_longSWs_1st=sample_longSWs_1st, maxSamps=maxSamps, maxRasTrials=maxRasTrials, 
																		flg_checkNPZvars=flg_checkNPZvars, verbose_EM=verbose_EM)
																	# 
																	# Infer Z's for all Spike Words in train data set if train_2nd_model=True)
																	if train_2nd_model: 
																		i+=1
																		fname = str( what_to_run + str(i) )
																		script_path = str(base_path    + fname + '.s')
																		output_path = str(cluster_path + 'out_files/' + fname + '.o')
																		error_path  = str(cluster_path + 'out_files/' + fname + '.e')	
																		wrapper.write( str( 'sbatch ' + cluster_path + fname + '.s \n') ) 
																		#
																		ss.write_sbatch_script_rasterZ_xVal_realData(script_path=script_path, output_path=output_path, error_path=error_path, 
																			whichClust=whichClust, n_cores=n_cores, mem=mem, time=time, job_name=fname, cell_type=cellSubTypes, stim=stim, 
																			num_test_samps_4xVal=num_test_samps_4xVal, rand=rand, SW_bin=SW_bin, yLo=yLo, yHi=yHi, yMinSW=yMinSW, params_init=params_init, 
																			sigQ_init=sigQ_init, sigPi_init=sigPi_init, sigPia_init=sigPia_init, overcomp=overcomp, Z_hot=Z_hot, 
																			C_noise_ri=C_noise_ri, C_noise_ria=C_noise_ria, learning_rate=learning_rate, lRateScale=lRateScale, 
																			maxTms=maxTms, minTms=minTms, pct_xVal_train=pct_xVal_train, flg_EgalitarianPrior=flg_EgalitarianPrior, train_2nd_model=False, 
																			flg_include_Zeq0_infer=flg_include_Zeq0_infer, sample_longSWs_1st=sample_longSWs_1st, maxSamps=maxSamps, maxRasTrials=maxRasTrials, 
																			flg_checkNPZvars=flg_checkNPZvars, verbose_EM=verbose_EM)







																if what_to_run is 'statI':
																	ss.write_sbatch_script_StatsInfPL_realData( script_path=script_path, output_path=output_path, error_path=error_path, 
																		whichClust=whichClust, n_cores=n_cores, mem=mem, time=time, job_name=fname, cellSubTypes=cellSubTypes, stim=stim, 
																		num_test_samps_4xVal=num_test_samps_4xVal, rand=rand, SW_bin=SW_bin, yLo=yLo, yHi=yHi,  yMinSW=yMinSW, params_init=params_init, 
																		sigQ_init=sigQ_init, sigPi_init=sigPi_init, sigPia_init=sigPia_init, Z_hot=Z_hot, C_noise_ri=C_noise_ri, 
																		C_noise_ria=C_noise_ria, learning_rate=learning_rate, lRateScale=lRateScale,
																		maxTms=maxTms, minTms=minTms, train_2nd_model=train_2nd_model, flg_include_Zeq0_infer=flg_include_Zeq0_infer,
																		flg_compute_StatsPostLrn=flg_compute_StatsPostLrn, flg_compute_StatsDuringEM=flg_compute_StatsDuringEM, verbose_EM=verbose_EM)
																		# TO ADD: sample_longSWs_1st=sample_longSWs_1st, maxSamps, flg_checkNPZvars
																	#
																	if train_2nd_model: 
																		i+=1
																		fname = str( what_to_run + str(i) )
																		script_path = str(base_path    + fname + '.s')
																		output_path = str(cluster_path + 'out_files/' + fname + '.o')
																		error_path  = str(cluster_path + 'out_files/' + fname + '.e')	
																		wrapper.write( str( 'sbatch ' + cluster_path + fname + '.s \n') ) 
																		#
																		ss.write_sbatch_script_StatsInfPL_realData( script_path=script_path, output_path=output_path, error_path=error_path, 
																			whichClust=whichClust, n_cores=n_cores, mem=mem, time=time, job_name=fname, cellSubTypes=cellSubTypes, stim=stim, 
																			num_test_samps_4xVal=num_test_samps_4xVal, rand=rand, SW_bin=SW_bin, yLo=yLo, yHi=yHi,  yMinSW=yMinSW, params_init=params_init, 
																			sigQ_init=sigQ_init, sigPi_init=sigPi_init, sigPia_init=sigPia_init, Z_hot=Z_hot, C_noise_ri=C_noise_ri, 
																			C_noise_ria=C_noise_ria, learning_rate=learning_rate, lRateScale=lRateScale,
																			maxTms=maxTms, minTms=minTms, train_2nd_model=False, flg_include_Zeq0_infer=flg_include_Zeq0_infer,
																			flg_compute_StatsPostLrn=flg_compute_StatsPostLrn, flg_compute_StatsDuringEM=flg_compute_StatsDuringEM, verbose_EM=verbose_EM)
																			# TO ADD: sample_longSWs_1st=sample_longSWs_1st, maxSamps, flg_checkNPZvars





wrapper.close()	# close wrapper file	

print( str('made ' + str(i) + ' sbatch job scripts'  ) )			

