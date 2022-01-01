import numpy as np
import os
import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.sbatch_scripts as ss



#
dirHomeLoc, dirScratch = dm.set_dir_tree()

# Parameters we can loop over.
num_SWs_tot 	= [1000] #[100000] #[10000, 100000]
num_Cells 		= [55]		# Looping over N values
num_CAs_true 	= [55]		# Looping over M values used to build the model
model_CA_overcompleteness = [1]		# how many times more cell assemblies the model assumes than are in true model (1 means complete - M_mod=M, 2 means 2x overcomplete)
#
ds_fctr_snapshots 	= 1000 	# Take a sample of model parameters every *ds_fctr* time steps to compute and plot MSE after we determine permutation of learned CA's to True CA's
xVal_snapshot 		= 1000
xVal_batchSize 		= 100

xTimesThruTrain = 1

num_EM_rands	= 1 	# number of times to randomize samples and rerun EM on same data generated from single synthetic model.
#

train_2nd_model = True

pct_xVal_train 	= 0.50 	# percentage of spikewords (total data) to use as training data. Set the rest aside for test data for cross validation.

resample_available_spikewords = True # ALWAYS TRUE FROM NOW ON !!!
pct_xVal_train_prev = 0.5	


sample_longSWs_1stS = ['Prob']#,'Dont'] # Options are: {'Dont', 'Prob', 'Hard'}
flg_EgalitarianPriorS = [False,True] # False means binomial prior. True means Egalitarian thing.


what_to_run = 'pgmS' 	# 'pgmS' or 'infPL' or 'PnIS' .... or one day, 'visS'?
 						# (1). 	 'pgmS' - Learn model on synth data 
						# (2). 	 'infPL' - Infer z's for all spike words using learned model
						# (1&2). 'PnIS' -  Do #1 and then #2 after, in serial
						# (X).   'visS' - one day. Not implemented.





#mk_plots 	= False 	# a quick flag to make all the plots. Only used in pgmS right now.

flg_checkNPZvars = True # True or False. Check variables saved in NPZ file and delete file and regenerate it if it does not contain the expected variables (hardcoded in each funciton.)



# Flags for running on clusters (nersc or cortex)
whichClust 	= 'cortex' 		# 'cortex' or 'nersc'.
n_cores 	= 1 			# number of CPU's to run on.
time 		= '15-00:00:00' 	# dd-hhhh:mm:ss

mem =  False # amount of memory (in MB) to allocate per CPU for each job submitted to cluster. pgmR needs more memory when running for 500k EM samples.
		# 7500 	# set to False to not specify amount of memory to use.

# Synthetic Model Construction Parameters
Ks 				= [1,2] 	# Number of	cell assemblies active, given a Bernoulli distribution ("hotness" of Z-vector)
Kmins 			= [0,0] 	# Max & Min number of cell assemblies active 
Kmaxs 			= [4,4] 	# 
#
Cs 				= [6,2] 	# number of cells participating in cell assemblies. ("hotness" of each row in Pia)
Cmins 			= [2,2] 	# Max & Min number of cell active to call it a cell assembly
Cmaxs 			= [6,6] 	# 
#
mu_PiaS			= [0.30, 0.55]	# Binary values in Pia matrix made continuous by sampling from Gaussian centered at 1-mu_Pia or 0+mu_Pia.
sig_PiaS		= [0.10, 0.05]	# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
#
mu_PiS			= [0.04, 0.04]	# Binary values in Pi vector made continuous by sampling from Gaussian centered at 1-mu_Pi or 0+mu_Pi.
sig_PiS			= [0.02, 0.02] 	# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
bernoulli_Pi	= 1.0   		# Each value in binary Pi vector randomly sampled from bernoulli with p(Pi=1) = bernoulli_Pi.	

#
yLo_Vals 		= [0] 		# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<=yLo.
yHi_Vals 		= [1000] 	# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution.
yMinSWs 		= [1] #,3] 	# Only grab spike words that are at least this length for training.


num_test_samps_4xValS = [1] 	# Number of test data points to to use to calculate pjoint_test (take their average) for each single train data point.
								# Increasing this should smooth out the pjoint curve and make cross validation easier.

# Parameter initializations for EM algorithm to learn model parameters
params_initS 	= ['NoisyConst'] #['DiagonalPia', 'NoisyConst'] 	# Options: {'True', 'RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise) }
sigQ_init 		= 0.01			# STD on Q initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
sigPi_init 		= 0.05			# STD on Pi initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
sigPia_init 	= 0.05			# STD on Pia initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
Z_hot 			= 5 			# Mean of initialization for Q value (how many 1's expected in binary z-vector)
C_noise_riS 	= [1.0]			# Mean of initialization of Pi values (1 means mostly silent) with variability defined by sigPia_init
C_noise_riaS 	= [1.0]			# Mean of initialization of Pia values (1 means mostly silent) with variability defined by sigPia_init



# Learning rates for EM algorithm
learning_rates 	= [0.5]#, 0.1] 	 	# Learning rates to loop through
lRateScaleS = [ [1., 0.1, 0.1]]#, [1., 0.1, 1.] ]  # Multiplicative scaling to Pia,Pi,Q learning rates. Can set to zero and take Pi taken out of model essentially.





# Flags for the EM (inference & learning) algorithm.
flg_include_Zeq0_infer  = True
flg_recordRandImprove 	= False
verbose_EM 				= False



# Set up directory structure and make dirs if they dont exist already.
if whichClust=='nersc':
	homeDir = '/global/homes/w/warner/'
if whichClust=='cortex':
	homeDir = '/global/home/users/cwarner/'
cluster_path = str( homeDir + 'Projects/G_Field_Retinal_Data/cluster_scripts/synthData/')
#	
base_path = str(dirHomeLoc + 'cluster_scripts/synthData/')
if not os.path.exists(base_path):
	os.makedirs(base_path)



# Loop through all parameters and write sbatch scripts to submit jobs to the cluster.
wrapper = open( str(base_path + what_to_run + '.wrap') , 'w') # open wrapper file to call all the sbatch scripts written in the loop.

i=0
for rand in range(num_EM_rands):
	#
	for num_SWs in num_SWs_tot:
		#
		for abc in range(len(num_Cells)):
			N = num_Cells[abc]
			M = num_CAs_true[abc]
			#
			for Cn_ind in range(len(C_noise_riaS)):
				C_noise_ria = C_noise_riaS[Cn_ind]
				C_noise_ri = C_noise_riS[Cn_ind]
				#
				for xyz in range( len(Ks) ):
					#
					K 	 = Ks[xyz]
					Kmin = Kmins[xyz]
					Kmax = Kmaxs[xyz]
					#
					C 	 = Cs[xyz]
					Cmin = Cmins[xyz]
					Cmax = Cmaxs[xyz]
					#
					mu_Pia = mu_PiaS[xyz]
					mu_Pi = mu_PiS[xyz]
					sig_Pia = sig_PiaS[xyz]
					sig_Pi = sig_PiS[xyz]
					#
					for yMinSW in yMinSWs:
							#
							for yLo in yLo_Vals:
								#
								for yHi in yHi_Vals:
									#
									for num_test_samps_4xVal in num_test_samps_4xValS:
										#
										for params_init in params_initS:
											#
											for learning_rate in learning_rates:
												#
												for lRateScale in lRateScaleS:
													#
													for overcomp in model_CA_overcompleteness:
														#
														for flg_EgalitarianPrior in flg_EgalitarianPriorS:
															#
															for sample_longSWs_1st in sample_longSWs_1stS:
																i+=1
																fname = str( what_to_run + str(i) )
																script_path = str(base_path    + fname + '.s')
																output_path = str(cluster_path + 'out_files/' + fname + '.o')
																error_path  = str(cluster_path + 'out_files/' + fname + '.e')

																if what_to_run is 'pgmS':
																	#
																	# write each sbatch call in wrapper file that I can just call once. NOTE: directory here is dir on NERSC cluster.
																	wrapper.write( str( 'sbatch ' + cluster_path + fname + '.s \n') )	
																	#
																	ss.write_sbatch_script_pgmCA_synthData(script_path=script_path, output_path=output_path, error_path=error_path, whichClust=whichClust,
																		n_cores=n_cores, mem=mem, time=time, job_name=fname, N=N, M=M, K=K, Kmin=Kmin, Kmax=Kmax, C=C, Cmin=Cmin, Cmax=Cmax,	yLo=yLo, yHi=yHi, yMinSW=yMinSW,
																		mu_Pia=mu_Pia, sig_Pia=sig_Pia, bernoulli_Pi=bernoulli_Pi, mu_Pi=mu_Pi, sig_Pi=sig_Pi, params_init=params_init, sigQ_init=sigQ_init, 
																		sigPi_init=sigPi_init, sigPia_init=sigPia_init, overcomp=overcomp, Z_hot=Z_hot, C_noise_ri=C_noise_ri, C_noise_ria=C_noise_ria, num_SWs=num_SWs, 
																		pct_xVal_train=pct_xVal_train, pct_xVal_train_prev=pct_xVal_train_prev, xVal_snapshot=xVal_snapshot, xVal_batchSize=xVal_batchSize, xTimesThruTrain=xTimesThruTrain, 
																		num_test_samps_4xVal=num_test_samps_4xVal, rand=rand, learning_rate=learning_rate, lRateScale=lRateScale, ds_fctr_snapshots=ds_fctr_snapshots,
																		flg_EgalitarianPrior=flg_EgalitarianPrior, flg_include_Zeq0_infer=flg_include_Zeq0_infer, flg_recordRandImprove=flg_recordRandImprove, 
																		train_2nd_model=train_2nd_model, resample_available_spikewords=resample_available_spikewords, sample_longSWs_1st=sample_longSWs_1st, 
																		flg_checkNPZvars=flg_checkNPZvars, verbose_EM=verbose_EM)




																if what_to_run is 'infPL':
																	#
																	# write each sbatch call in wrapper file that I can just call once. NOTE: directory here is dir on NERSC cluster.
																	wrapper.write( str( 'sbatch ' + cluster_path + fname + '.s \n') )	
																	#
																	# Infer Z's for all Spike Words (in train data set if train_2nd_model=False; in test dataset if train_2nd_model=True)
																	ss.write_sbatch_script_infer_postLrn_synthData(script_path=script_path, output_path=output_path, error_path=error_path, whichClust=whichClust,
																		n_cores=n_cores, mem=mem, time=time, job_name=fname, N=N, M=M, K=K, Kmin=Kmin, Kmax=Kmax, C=C, Cmin=Cmin, Cmax=Cmax, yLo=yLo, yHi=yHi, yMinSW=yMinSW,
																		mu_Pia=mu_Pia, sig_Pia=sig_Pia, bernoulli_Pi=bernoulli_Pi, mu_Pi=mu_Pi, sig_Pi=sig_Pi, params_init=params_init, 
																		sigQ_init=sigQ_init, sigPi_init=sigPi_init, sigPia_init=sigPia_init, overcomp=overcomp, Z_hot=Z_hot, C_noise_ri=C_noise_ri, 
																		C_noise_ria=C_noise_ria, num_SWs=num_SWs, pct_xVal_train=pct_xVal_train, pct_xVal_train_prev=pct_xVal_train_prev, xTimesThruTrain=xTimesThruTrain,
																		num_test_samps_4xVal=num_test_samps_4xVal, rand=rand, learning_rate=learning_rate, lRateScale=lRateScale, ds_fctr_snapshots=ds_fctr_snapshots, 
																		flg_EgalitarianPrior=flg_EgalitarianPrior, flg_include_Zeq0_infer=flg_include_Zeq0_infer, train_2nd_model=train_2nd_model, resample_available_spikewords=resample_available_spikewords, 
																		sample_longSWs_1st=sample_longSWs_1st, flg_checkNPZvars=flg_checkNPZvars, verbose_EM=verbose_EM)
																	# 
																	# Infer Z's for all Spike Words in train data set if train_2nd_model=True)
																	if train_2nd_model: 
																		#
																		i+=1
																		fname = str( what_to_run + str(i) )
																		script_path = str(base_path    + fname + '.s')
																		output_path = str(cluster_path + 'out_files/' + fname + '.o')
																		error_path  = str(cluster_path + 'out_files/' + fname + '.e')	
																		#
																		# write each sbatch call in wrapper file that I can just call once. NOTE: directory here is dir on NERSC cluster.
																		wrapper.write( str( 'sbatch ' + cluster_path + fname + '.s \n') )	
																		#
																		ss.write_sbatch_script_infer_postLrn_synthData(script_path=script_path, output_path=output_path, error_path=error_path, whichClust=whichClust,
																			n_cores=n_cores, mem=mem, time=time, job_name=fname, N=N, M=M, K=K, Kmin=Kmin, Kmax=Kmax, C=C, Cmin=Cmin, Cmax=Cmax, yLo=yLo, yHi=yHi, yMinSW=yMinSW,
																			mu_Pia=mu_Pia, sig_Pia=sig_Pia, bernoulli_Pi=bernoulli_Pi, mu_Pi=mu_Pi, sig_Pi=sig_Pi, params_init=params_init, 
																			sigQ_init=sigQ_init, sigPi_init=sigPi_init, sigPia_init=sigPia_init, overcomp=overcomp, Z_hot=Z_hot, C_noise_ri=C_noise_ri, 
																			C_noise_ria=C_noise_ria, num_SWs=num_SWs, pct_xVal_train=pct_xVal_train, pct_xVal_train_prev=pct_xVal_train_prev, xTimesThruTrain=xTimesThruTrain,
																			num_test_samps_4xVal=num_test_samps_4xVal, rand=rand, learning_rate=learning_rate, lRateScale=lRateScale, ds_fctr_snapshots=ds_fctr_snapshots, 
																			flg_EgalitarianPrior=flg_EgalitarianPrior, flg_include_Zeq0_infer=flg_include_Zeq0_infer, train_2nd_model=False,
																			resample_available_spikewords=resample_available_spikewords, sample_longSWs_1st=sample_longSWs_1st, flg_checkNPZvars=flg_checkNPZvars, verbose_EM=verbose_EM)




																if what_to_run is 'PnIS':
																	#
																	# write each sbatch call in wrapper file that I can just call once. NOTE: directory here is dir on NERSC cluster.
																	wrapper.write( str( 'sbatch ' + cluster_path + fname + '.s \n') )	
																	#
																	ss.write_sbatch_script_pgmCA_and_infPL_synthData(script_path=script_path, output_path=output_path, error_path=error_path, whichClust=whichClust,
																		n_cores=n_cores, mem=mem, time=time, job_name=fname, N=N, M=M, K=K, Kmin=Kmin, Kmax=Kmax, C=C, Cmin=Cmin, Cmax=Cmax, yLo=yLo, yHi=yHi, yMinSW=yMinSW,
																		mu_Pia=mu_Pia, sig_Pia=sig_Pia, bernoulli_Pi=bernoulli_Pi, mu_Pi=mu_Pi, sig_Pi=sig_Pi, params_init=params_init, sigQ_init=sigQ_init, 
																		sigPi_init=sigPi_init, sigPia_init=sigPia_init, overcomp=overcomp, Z_hot=Z_hot, C_noise_ri=C_noise_ri, C_noise_ria=C_noise_ria, num_SWs=num_SWs, 
																		pct_xVal_train=pct_xVal_train, pct_xVal_train_prev=pct_xVal_train_prev, xVal_snapshot=xVal_snapshot, xVal_batchSize=xVal_batchSize, 
																		xTimesThruTrain=xTimesThruTrain, num_test_samps_4xVal=num_test_samps_4xVal, rand=rand, learning_rate=learning_rate, lRateScale=lRateScale,
																		ds_fctr_snapshots=ds_fctr_snapshots,  flg_EgalitarianPrior=flg_EgalitarianPrior, flg_include_Zeq0_infer=flg_include_Zeq0_infer, 
																		flg_recordRandImprove=flg_recordRandImprove, train_2nd_model=train_2nd_model, resample_available_spikewords=resample_available_spikewords, 
																		sample_longSWs_1st=sample_longSWs_1st, flg_checkNPZvars=flg_checkNPZvars, verbose_EM=verbose_EM)





																if what_to_run is 'visS':
																	print('I have not implemented this yet.')




		
wrapper.close()	# close wrapper file				

print( str('made ' + str(i) + ' sbatch job scripts'  ) )		