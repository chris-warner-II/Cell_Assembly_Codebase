# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def write_sbatch_script_pgmCA_synthData(script_path, output_path, error_path, whichClust, n_cores, mem, time, job_name, 
		N, M, K, Kmin, Kmax, C, Cmin, Cmax,	yLo, yHi, yMinSW, mu_Pia, sig_Pia, bernoulli_Pi, mu_Pi, sig_Pi, 
		params_init, sigQ_init, sigPi_init, sigPia_init, overcomp, Z_hot, C_noise_ri, C_noise_ria, num_SWs, 
		pct_xVal_train, pct_xVal_train_prev, xVal_snapshot, xVal_batchSize, xTimesThruTrain, num_test_samps_4xVal, rand, learning_rate, lRateScale, ds_fctr_snapshots, 
		flg_EgalitarianPrior, flg_include_Zeq0_infer, flg_recordRandImprove, train_2nd_model, resample_available_spikewords, sample_longSWs_1st, flg_checkNPZvars, verbose_EM):
	


	# open file
	file = open(script_path, 'w')
	# sbatch header
	file.write('#!/bin/bash\n')
	
	if whichClust=='nersc':
		homeDir = '/global/homes/w/warner/'
		# shared queue
		file.write('#SBATCH -q shared\n')
		# number of cores - optional argument
		file.write('#SBATCH -n ' + str(n_cores) + '\n')
		# email for failure
		file.write('#SBATCH --mail-user=cwarner@berkeley.edu\n')
		file.write('#SBATCH --mail-type=FAIL\n\n')
	
	if whichClust=='cortex':
		homeDir = '/global/home/users/cwarner/'
		# specify the partition as 'cortex' if on cortex.
		file.write('#SBATCH --partition=' + whichClust + '\n')	
		# Constrain Nodes:
		# file.write('#SBATCH --constraint=cortex_nogpu \n')
		# # Processors:
		# file.write('#SBATCH --ntasks=4 \n')
		# Memory:
		if mem:
			file.write( str('#SBATCH --mem-per-cpu='+str(mem)+'M \n') )

	# amount of time - optional argument 
	file.write('#SBATCH --time=' + time + '\n')
	# job name - optional argument
	file.write('#SBATCH --job-name=' + job_name + '\n')
	# path for job output
	file.write('#SBATCH -o ' + output_path + '\n')
	# path for job errors
	file.write('#SBATCH -e ' + error_path + '\n')

	# import python and anaconda
	file.write('module load python \n')
	# python command
	# which selections to run

	file.write(
		'python3 ~/Projects/G_Field_Retinal_Data/'	+ \
		'Chris_working_code_2019/python_code/pgmCA_synthData.py' + \
		' --N=' 					+ str(N) + \
		' --M=' 					+ str(M) + \
		' --K=' 					+ str(K) + \
		' --Kmin=' 					+ str(Kmin) + \
		' --Kmax=' 					+ str(Kmax) + \
		' --C=' 					+ str(C) + \
		' --Cmin=' 					+ str(Cmin) + \
		' --Cmax=' 					+ str(Cmax) + \
		' --yLo=' 					+ str(yLo) + \
		' --yHi=' 					+ str(yHi) + \
		' --yMinSW=' 				+ str(yMinSW) + \
		' --mu_Pia=' 				+ str(mu_Pia) + \
		' --sig_Pia=' 				+ str(sig_Pia) + \
		' --bernoulli_Pi='			+ str(bernoulli_Pi) + \
		' --mu_Pi=' 				+ str(mu_Pi) + \
		' --sig_Pi=' 				+ str(sig_Pi) + \
		' --xTimesThruTrain=' 		+ str(xTimesThruTrain) + \
		' --xVal_snapshot=' 		+ str(xVal_snapshot) + \
		' --xVal_batchSize=' 		+ str(xVal_batchSize) + \
		' --num_test_samps_4xVal=' 	+ str(num_test_samps_4xVal) + \
		' --sample_longSWs_1st='	+ str(sample_longSWs_1st) + \
		' --rand=' 					+ str(rand) + \
		' --num_SWs=' 				+ str(num_SWs) + \
		' --pct_xVal_train=' 		+ str(pct_xVal_train) + \
		' --pct_xVal_train_prev='	+ str(pct_xVal_train_prev) + \
		' --learning_rate=' 		+ str(learning_rate) + \
		' --lRateScale=' 			+ str(lRateScale).replace(' ','').replace('[','').replace(']','') + \
		' --params_init=' 			+ params_init + \
		' --sigQ_init=' 			+ str(sigQ_init) + \
		' --sigPi_init=' 			+ str(sigPi_init) + \
		' --sigPia_init=' 			+ str(sigPia_init) + \
		' --overcomp=' 				+ str(overcomp) + \
		' --Z_hot='					+ str(Z_hot) + \
		' --C_noise_ri=' 			+ str(C_noise_ri) + \
		' --C_noise_ria=' 			+ str(C_noise_ria) + \
		' --ds_fctr_snapshots=' 	+ str(ds_fctr_snapshots) )


	# Include boolean flags.	
	if flg_EgalitarianPrior:
		file.write(' --flg_EgalitarianPrior')

	if flg_include_Zeq0_infer:
		file.write(' --flg_include_Zeq0_infer')

	if flg_recordRandImprove:
		file.write(' --flg_recordRandImprove')	

	if train_2nd_model:	
		file.write(' --train_2nd_model')

	if resample_available_spikewords:	
		file.write(' --resample_available_spikewords')		

	if flg_checkNPZvars:	
		file.write(' --flg_checkNPZvars')	

	if verbose_EM:
		file.write(' --verbose_EM')	

	file.write('\n')	

	# close file
	file.close()




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def write_sbatch_script_infer_postLrn_synthData(script_path, output_path, error_path, whichClust, n_cores, mem, time, job_name, 
		N, M, K, Kmin, Kmax, C, Cmin, Cmax,	yLo, yHi, yMinSW, mu_Pia, sig_Pia, bernoulli_Pi, mu_Pi, sig_Pi, 
		params_init, sigQ_init, sigPi_init, sigPia_init, overcomp, Z_hot, C_noise_ri, C_noise_ria, num_SWs, pct_xVal_train, 
		pct_xVal_train_prev, num_test_samps_4xVal, xTimesThruTrain, rand, learning_rate, lRateScale, ds_fctr_snapshots, 
		flg_EgalitarianPrior, flg_include_Zeq0_infer, train_2nd_model, resample_available_spikewords, sample_longSWs_1st, flg_checkNPZvars, verbose_EM):


	

	# open file
	file = open(script_path, 'w')
	# sbatch header
	file.write('#!/bin/bash\n')
	
	if whichClust=='nersc':
		homeDir = '/global/homes/w/warner/'
		# shared queue
		file.write('#SBATCH -q shared\n')
		# number of cores - optional argument
		file.write('#SBATCH -n ' + str(n_cores) + '\n')
		# email for failure
		file.write('#SBATCH --mail-user=cwarner@berkeley.edu\n')
		file.write('#SBATCH --mail-type=FAIL\n\n')
	
	if whichClust=='cortex':
		homeDir = '/global/home/users/cwarner/'
		# specify the partition as 'cortex' if on cortex.
		file.write('#SBATCH --partition=' + whichClust + '\n')	
		# Constrain Nodes:
		# file.write('#SBATCH --constraint=cortex_nogpu \n')
		# # Processors:
		# file.write('#SBATCH --ntasks=4 \n')
		# Memory:
		if mem:
			file.write( str('#SBATCH --mem-per-cpu='+str(mem)+'M \n') )

	# amount of time - optional argument 
	file.write('#SBATCH --time=' + time + '\n')
	# job name - optional argument
	file.write('#SBATCH --job-name=' + job_name + '\n')
	# path for job output
	file.write('#SBATCH -o ' + output_path + '\n')
	# path for job errors
	file.write('#SBATCH -e ' + error_path + '\n')

	# import python and anaconda
	file.write('module load python \n')
	# python command
	# which selections to run

	file.write(
		'python3 ~/Projects/G_Field_Retinal_Data/'	+ \
		'Chris_working_code_2019/python_code/infer_postLrn_synthData.py' + \
		' --N=' 					+ str(N) + \
		' --M=' 					+ str(M) + \
		' --K=' 					+ str(K) + \
		' --Kmin=' 					+ str(Kmin) + \
		' --Kmax=' 					+ str(Kmax) + \
		' --C=' 					+ str(C) + \
		' --Cmin=' 					+ str(Cmin) + \
		' --Cmax=' 					+ str(Cmax) + \
		' --yLo=' 					+ str(yLo) + \
		' --yHi=' 					+ str(yHi) + \
		' --yMinSW=' 				+ str(yMinSW) + \
		' --mu_Pia=' 				+ str(mu_Pia) + \
		' --sig_Pia=' 				+ str(sig_Pia) + \
		' --bernoulli_Pi='			+ str(bernoulli_Pi) + \
		' --mu_Pi=' 				+ str(mu_Pi) + \
		' --sig_Pi=' 				+ str(sig_Pi) + \
		' --num_test_samps_4xVal=' 	+ str(num_test_samps_4xVal) + \
		' --sample_longSWs_1st='	+ str(sample_longSWs_1st) + \
		' --rand=' 					+ str(rand) + \
		' --num_SWs=' 				+ str(num_SWs) + \
		' --pct_xVal_train=' 		+ str(pct_xVal_train) + \
		' --pct_xVal_train_prev='	+ str(pct_xVal_train_prev) + \
		' --xTimesThruTrain='		+ str(xTimesThruTrain) + \
		' --learning_rate=' 		+ str(learning_rate) + \
		' --lRateScale=' 			+ str(lRateScale).replace(' ','').replace('[','').replace(']','') + \
		' --params_init=' 			+ params_init + \
		' --sigQ_init=' 			+ str(sigQ_init) + \
		' --sigPi_init=' 			+ str(sigPi_init) + \
		' --sigPia_init=' 			+ str(sigPia_init) + \
		' --overcomp=' 				+ str(overcomp) + \
		' --Z_hot='					+ str(Z_hot) + \
		' --C_noise_ri=' 			+ str(C_noise_ri) + \
		' --C_noise_ria=' 			+ str(C_noise_ria) + \
		' --ds_fctr_snapshots=' 	+ str(ds_fctr_snapshots) )




	# Include boolean flags.	
	if flg_EgalitarianPrior:
		file.write(' --flg_EgalitarianPrior')

	if flg_include_Zeq0_infer:
		file.write(' --flg_include_Zeq0_infer')

	if train_2nd_model:	
		file.write(' --train_2nd_model')	

	if resample_available_spikewords:	
		file.write(' --resample_available_spikewords')	

	if flg_checkNPZvars:	
		file.write(' --flg_checkNPZvars')		

	if verbose_EM:
		file.write(' --verbose_EM')	

	file.write('\n')	

	# close file
	file.close()



























# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def write_sbatch_script_pgmCA_and_infPL_synthData(script_path, output_path, error_path, whichClust, n_cores, mem, time, job_name, 
	N, M, K, Kmin, Kmax, C, Cmin, Cmax,	yLo, yHi, yMinSW, mu_Pia, sig_Pia, bernoulli_Pi, mu_Pi, sig_Pi, 
	params_init, sigQ_init, sigPi_init, sigPia_init, overcomp, Z_hot, C_noise_ri, C_noise_ria, num_SWs, 
	pct_xVal_train, pct_xVal_train_prev, xVal_snapshot, xVal_batchSize, xTimesThruTrain, num_test_samps_4xVal, rand, learning_rate, lRateScale, ds_fctr_snapshots, 
	flg_EgalitarianPrior, flg_include_Zeq0_infer, flg_recordRandImprove, train_2nd_model, resample_available_spikewords, sample_longSWs_1st, flg_checkNPZvars, verbose_EM):


	# open file
	file = open(script_path, 'w')
	# sbatch header
	file.write('#!/bin/bash\n')
	
	if whichClust=='nersc':
		homeDir = '/global/homes/w/warner/'
		# shared queue
		file.write('#SBATCH -q shared\n')
		# number of cores - optional argument
		file.write('#SBATCH -n ' + str(n_cores) + '\n')
		# email for failure
		file.write('#SBATCH --mail-user=cwarner@berkeley.edu\n')
		file.write('#SBATCH --mail-type=FAIL\n\n')
	
	if whichClust=='cortex':
		homeDir = '/global/home/users/cwarner/'
		# specify the partition as 'cortex' if on cortex.
		file.write('#SBATCH --partition=' + whichClust + '\n')	
		# Constrain Nodes:
		# file.write('#SBATCH --constraint=cortex_nogpu \n')
		# # Processors:
		# file.write('#SBATCH --ntasks=4 \n')
		# Memory:
		if mem:
			file.write( str('#SBATCH --mem-per-cpu='+str(mem)+'M \n') )

	# amount of time - optional argument 
	file.write('#SBATCH --time=' + time + '\n')
	# job name - optional argument
	file.write('#SBATCH --job-name=' + job_name + '\n')
	# path for job output
	file.write('#SBATCH -o ' + output_path + '\n')
	# path for job errors
	file.write('#SBATCH -e ' + error_path + '\n')

	# import python and anaconda
	file.write('module load python \n \n')
	# python command
	# which selections to run

	file.write(
		'python3 ~/Projects/G_Field_Retinal_Data/'	+ \
		'Chris_working_code_2019/python_code/pgmCA_synthData.py' + \
		' --N=' 					+ str(N) + \
		' --M=' 					+ str(M) + \
		' --K=' 					+ str(K) + \
		' --Kmin=' 					+ str(Kmin) + \
		' --Kmax=' 					+ str(Kmax) + \
		' --C=' 					+ str(C) + \
		' --Cmin=' 					+ str(Cmin) + \
		' --Cmax=' 					+ str(Cmax) + \
		' --yLo=' 					+ str(yLo) + \
		' --yHi=' 					+ str(yHi) + \
		' --yMinSW=' 				+ str(yMinSW) + \
		' --mu_Pia=' 				+ str(mu_Pia) + \
		' --sig_Pia=' 				+ str(sig_Pia) + \
		' --bernoulli_Pi='			+ str(bernoulli_Pi) + \
		' --mu_Pi=' 				+ str(mu_Pi) + \
		' --sig_Pi=' 				+ str(sig_Pi) + \
		' --xTimesThruTrain=' 		+ str(xTimesThruTrain) + \
		' --xVal_snapshot=' 		+ str(xVal_snapshot) + \
		' --xVal_batchSize=' 		+ str(xVal_batchSize) + \
		' --num_test_samps_4xVal=' 	+ str(num_test_samps_4xVal) + \
		' --sample_longSWs_1st='	+ str(sample_longSWs_1st) + \
		' --rand=' 					+ str(rand) + \
		' --num_SWs=' 				+ str(num_SWs) + \
		' --pct_xVal_train=' 		+ str(pct_xVal_train) + \
		' --pct_xVal_train_prev='	+ str(pct_xVal_train_prev) + \
		' --learning_rate=' 		+ str(learning_rate) + \
		' --lRateScale=' 			+ str(lRateScale).replace(' ','').replace('[','').replace(']','') + \
		' --params_init=' 			+ params_init + \
		' --sigQ_init=' 			+ str(sigQ_init) + \
		' --sigPi_init=' 			+ str(sigPi_init) + \
		' --sigPia_init=' 			+ str(sigPia_init) + \
		' --overcomp=' 				+ str(overcomp) + \
		' --Z_hot='					+ str(Z_hot) + \
		' --C_noise_ri=' 			+ str(C_noise_ri) + \
		' --C_noise_ria=' 			+ str(C_noise_ria) + \
		' --ds_fctr_snapshots=' 	+ str(ds_fctr_snapshots) )


	# Include boolean flags.	
	if flg_EgalitarianPrior:
		file.write(' --flg_EgalitarianPrior')

	if flg_include_Zeq0_infer:
		file.write(' --flg_include_Zeq0_infer')

	if flg_recordRandImprove:
		file.write(' --flg_recordRandImprove')	

	if train_2nd_model:	
		file.write(' --train_2nd_model')

	if resample_available_spikewords:	
		file.write(' --resample_available_spikewords')	

	if flg_checkNPZvars:	
		file.write(' --flg_checkNPZvars')			

	if verbose_EM:
		file.write(' --verbose_EM')	

	file.write('\n \n')		


	# # # # # # # # # # # # # # # # # # # # # # # # # # # #


	file.write(
		'python3 ~/Projects/G_Field_Retinal_Data/'	+ \
		'Chris_working_code_2019/python_code/infer_postLrn_synthData.py' + \
		' --N=' 					+ str(N) + \
		' --M=' 					+ str(M) + \
		' --K=' 					+ str(K) + \
		' --Kmin=' 					+ str(Kmin) + \
		' --Kmax=' 					+ str(Kmax) + \
		' --C=' 					+ str(C) + \
		' --Cmin=' 					+ str(Cmin) + \
		' --Cmax=' 					+ str(Cmax) + \
		' --yLo=' 					+ str(yLo) + \
		' --yHi=' 					+ str(yHi) + \
		' --yMinSW=' 				+ str(yMinSW) + \
		' --mu_Pia=' 				+ str(mu_Pia) + \
		' --sig_Pia=' 				+ str(sig_Pia) + \
		' --bernoulli_Pi='			+ str(bernoulli_Pi) + \
		' --mu_Pi=' 				+ str(mu_Pi) + \
		' --sig_Pi=' 				+ str(sig_Pi) + \
		' --num_test_samps_4xVal=' 	+ str(num_test_samps_4xVal) + \
		' --sample_longSWs_1st='	+ str(sample_longSWs_1st) + \
		' --rand=' 					+ str(rand) + \
		' --num_SWs=' 				+ str(num_SWs) + \
		' --pct_xVal_train=' 		+ str(pct_xVal_train) + \
		' --pct_xVal_train_prev='	+ str(pct_xVal_train_prev) + \
		' --xTimesThruTrain='		+ str(xTimesThruTrain) + \
		' --learning_rate=' 		+ str(learning_rate) + \
		' --lRateScale=' 			+ str(lRateScale).replace(' ','').replace('[','').replace(']','') + \
		' --params_init=' 			+ params_init + \
		' --sigQ_init=' 			+ str(sigQ_init) + \
		' --sigPi_init=' 			+ str(sigPi_init) + \
		' --sigPia_init=' 			+ str(sigPia_init) + \
		' --overcomp=' 				+ str(overcomp) + \
		' --Z_hot='					+ str(Z_hot) + \
		' --C_noise_ri=' 			+ str(C_noise_ri) + \
		' --C_noise_ria=' 			+ str(C_noise_ria) + \
		' --ds_fctr_snapshots=' 	+ str(ds_fctr_snapshots) )


	# Include boolean flags.	
	if flg_EgalitarianPrior:
		file.write(' --flg_EgalitarianPrior')

	if flg_include_Zeq0_infer:
		file.write(' --flg_include_Zeq0_infer')

	if train_2nd_model:	
		file.write(' --train_2nd_model')	

	if resample_available_spikewords:	
		file.write(' --resample_available_spikewords')	

	if flg_checkNPZvars:	
		file.write(' --flg_checkNPZvars')		

	if verbose_EM:
		file.write(' --verbose_EM')	

	file.write('\n \n')	





	# # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# if training 2nd model do inference on the one where 'train_2nd_model' would have been False.
	if train_2nd_model:

		file.write(
			'python3 ~/Projects/G_Field_Retinal_Data/'	+ \
			'Chris_working_code_2019/python_code/infer_postLrn_synthData.py' + \
			' --N=' 					+ str(N) + \
			' --M=' 					+ str(M) + \
			' --K=' 					+ str(K) + \
			' --Kmin=' 					+ str(Kmin) + \
			' --Kmax=' 					+ str(Kmax) + \
			' --C=' 					+ str(C) + \
			' --Cmin=' 					+ str(Cmin) + \
			' --Cmax=' 					+ str(Cmax) + \
			' --yLo=' 					+ str(yLo) + \
			' --yHi=' 					+ str(yHi) + \
			' --yMinSW=' 				+ str(yMinSW) + \
			' --mu_Pia=' 				+ str(mu_Pia) + \
			' --sig_Pia=' 				+ str(sig_Pia) + \
			' --bernoulli_Pi='			+ str(bernoulli_Pi) + \
			' --mu_Pi=' 				+ str(mu_Pi) + \
			' --sig_Pi=' 				+ str(sig_Pi) + \
			' --num_test_samps_4xVal=' 	+ str(num_test_samps_4xVal) + \
			' --sample_longSWs_1st='	+ str(sample_longSWs_1st) + \
			' --rand=' 					+ str(rand) + \
			' --num_SWs=' 				+ str(num_SWs) + \
			' --pct_xVal_train=' 		+ str(pct_xVal_train) + \
			' --pct_xVal_train_prev='	+ str(pct_xVal_train_prev) + \
			' --xTimesThruTrain='		+ str(xTimesThruTrain) + \
			' --learning_rate=' 		+ str(learning_rate) + \
			' --lRateScale=' 			+ str(lRateScale).replace(' ','').replace('[','').replace(']','') + \
			' --params_init=' 			+ params_init + \
			' --sigQ_init=' 			+ str(sigQ_init) + \
			' --sigPi_init=' 			+ str(sigPi_init) + \
			' --sigPia_init=' 			+ str(sigPia_init) + \
			' --overcomp=' 				+ str(overcomp) + \
			' --Z_hot='					+ str(Z_hot) + \
			' --C_noise_ri=' 			+ str(C_noise_ri) + \
			' --C_noise_ria=' 			+ str(C_noise_ria) + \
			' --ds_fctr_snapshots=' 	+ str(ds_fctr_snapshots) )


		# Include boolean flags.	

		if flg_EgalitarianPrior:
			file.write(' --flg_EgalitarianPrior')	

		if flg_include_Zeq0_infer:
			file.write(' --flg_include_Zeq0_infer')

		if resample_available_spikewords:	
			file.write(' --resample_available_spikewords')	

		if flg_checkNPZvars:	
			file.write(' --flg_checkNPZvars')	

		if verbose_EM:
			file.write(' --verbose_EM')	

	file.write('\n')	

	# close file
	file.close()
























# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def write_sbatch_script_pgmCA_realData(script_path, output_path, error_path, whichClust, n_cores, mem, time, job_name, 
		cell_type, cellSubTypes, stim, num_test_samps_4xVal, rand, SW_bin, yLo, yHi, yMinSW, params_init, sigQ_init, 
		sigPi_init, sigPia_init, overcomp, Z_hot, C_noise_ri, C_noise_ria, learning_rate, lRateScale, ds_fctr_snapshots, 
		pct_xVal_train, xVal_snapshot, xVal_batchSize, flg_EgalitarianPrior, train_2nd_model, flg_include_Zeq0_infer, 
		sample_longSWs_1st, maxSamps, flg_checkNPZvars, verbose_EM):

	# open file
	file = open(script_path, 'w')
	# sbatch header
	file.write('#!/bin/bash\n')
	
	if whichClust=='nersc':
		homeDir = '/global/homes/w/warner/'
		# shared queue
		file.write('#SBATCH -q shared\n')
		# number of cores - optional argument
		file.write('#SBATCH -n ' + str(n_cores) + '\n')
		# email for failure
		file.write('#SBATCH --mail-user=cwarner@berkeley.edu\n')
		file.write('#SBATCH --mail-type=FAIL\n\n')
	
	if whichClust=='cortex':
		homeDir = '/global/home/users/cwarner/'
		# specify the partition as 'cortex' if on cortex.
		file.write('#SBATCH --partition=' + whichClust + '\n')	
		# Constrain Nodes:
		# file.write('#SBATCH --constraint=cortex_nogpu \n')
		# # Processors:
		# file.write('#SBATCH --ntasks=4 \n')
		# Memory:
		if mem:
			file.write( str('#SBATCH --mem-per-cpu='+str(mem)+'M \n') )

	# amount of time - optional argument 
	file.write('#SBATCH --time=' + time + '\n')
	# job name - optional argument
	file.write('#SBATCH --job-name=' + job_name + '\n')
	# path for job output
	file.write('#SBATCH -o ' + output_path + '\n')
	# path for job errors
	file.write('#SBATCH -e ' + error_path + '\n')

	# import python and anaconda
	file.write('module load python \n')
	# python command
	# which selections to run


	#print( str('\'' + str(cellSubTypes) + '\'') )
	# print(cellSubTypes)
	# print(type(cellSubTypes))
	CST_str = str(cellSubTypes).replace(' ','') # have to do this because of weirdness in calling function and passing in this var.
	# print(CST_str)
	# print(type(CST_str))

	file.write(
		'python3 ' + homeDir + 'Projects/G_Field_Retinal_Data/'	+ \
		'Chris_working_code_2019/python_code/pgmCA_realData.py' + \
		' --cell_type=' 			+ cell_type + \
		' --cellSubTypes=' 			+ CST_str + \
		' --stim=' 					+ stim + \
		' --num_test_samps_4xVal=' 	+ str(num_test_samps_4xVal) + \
		' --sample_longSWs_1st='	+ str(sample_longSWs_1st) + \
		' --rand=' 					+ str(rand) + \
		' --SW_bin=' 				+ str(SW_bin) + \
		' --yLo=' 					+ str(yLo) + \
		' --yHi=' 					+ str(yHi) + \
		' --yMinSW=' 				+ str(yMinSW) + \
		' --params_init=' 			+ params_init + \
		' --sigQ_init=' 			+ str(sigQ_init) + \
		' --sigPi_init=' 			+ str(sigPi_init) + \
		' --sigPia_init=' 			+ str(sigPia_init) + \
		' --overcomp=' 				+ str(overcomp) + \
		' --Z_hot=' 				+ str(Z_hot) + \
		' --C_noise_ri=' 			+ str(C_noise_ri) + \
		' --C_noise_ria=' 			+ str(C_noise_ria) + \
		' --learning_rate=' 		+ str(learning_rate) + \
		' --lRateScale=' 			+ str(lRateScale).replace(' ','').replace('[','').replace(']','') + \
		' --maxSamps=' 				+ str(maxSamps) + \
		' --ds_fctr_snapshots=' 	+ str(ds_fctr_snapshots) + \
		' --pct_xVal_train=' 		+ str(pct_xVal_train) + \
		' --xVal_snapshot=' 		+ str(xVal_snapshot) + \
		' --xVal_batchSize=' 		+ str(xVal_batchSize) )
		
	# Include boolean flags.	
	if flg_include_Zeq0_infer:
		file.write(' --flg_include_Zeq0_infer')

	if train_2nd_model:	
		file.write(' --train_2nd_model')			
		
	if verbose_EM:
		file.write(' --verbose_EM')	

	if flg_EgalitarianPrior:
		file.write(' --flg_EgalitarianPrior')	

	if flg_checkNPZvars:	
		file.write(' --flg_checkNPZvars')	


	file.write('\n')	

	# close file
	file.close()












# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def write_sbatch_script_rasterZ_realData(script_path, output_path, error_path, whichClust, n_cores, mem, time, job_name, 
		cell_type, stim, num_test_samps_4xVal, rand, SW_bin, yLo, yHi, yMinSW, params_init, sigQ_init, sigPi_init, sigPia_init, 
		overcomp, Z_hot, C_noise_ri, C_noise_ria, learning_rate, lRateScale, maxTms, minTms, pct_xVal_train, flg_EgalitarianPrior, 
		train_2nd_model, flg_include_Zeq0_infer, sample_longSWs_1st, maxSamps, maxRasTrials, flg_checkNPZvars, verbose_EM):

	#print('2nd model: ',train_2nd_model)

	# open file
	file = open(script_path, 'w')
	# sbatch header
	file.write('#!/bin/bash\n')
	
	if whichClust=='nersc':
		homeDir = '/global/homes/w/warner/'
		# shared queue
		file.write('#SBATCH -q shared\n')
		# number of cores - optional argument
		file.write('#SBATCH -n ' + str(n_cores) + '\n')
		# email for failure
		file.write('#SBATCH --mail-user=cwarner@berkeley.edu\n')
		file.write('#SBATCH --mail-type=FAIL\n\n')
	
	if whichClust=='cortex':
		homeDir = '/global/home/users/cwarner/'
		# specify the partition as 'cortex' if on cortex.
		file.write('#SBATCH --partition=' + whichClust + '\n')	
		# Constrain Nodes:
		# file.write('#SBATCH --constraint=cortex_nogpu \n')
		# # Processors:
		# file.write('#SBATCH --ntasks=4 \n')
		# Memory:
		if mem:
			file.write( str('#SBATCH --mem-per-cpu='+str(mem)+'M \n') )

	# amount of time - optional argument 
	file.write('#SBATCH --time=' + time + '\n')
	# job name - optional argument
	file.write('#SBATCH --job-name=' + job_name + '\n')
	# path for job output
	file.write('#SBATCH -o ' + output_path + '\n')
	# path for job errors
	file.write('#SBATCH -e ' + error_path + '\n')

	# import python and anaconda
	file.write('module load python \n')
	# python command
	# which selections to run

	CST_str = str(cell_type).replace(' ','') # have to do this because of weirdness in calling function and passing in this var.
	

	file.write(
		'python3 ' + homeDir + 'Projects/G_Field_Retinal_Data/'	+ \
		'Chris_working_code_2019/python_code/raster_zs_inferred_allSWs_given_model.py' + \
		' --cell_type=' 			+ CST_str + \
		' --whichCells=' 			+ '' + \
		' --whichGLM=' 				+ 'real' + \
		' --whichPop=' 				+ '' + \
		' --stim=' 					+ stim + \
		' --num_test_samps_4xVal=' 	+ str(num_test_samps_4xVal) + \
		' --sample_longSWs_1st='	+ str(sample_longSWs_1st) + \
		' --rand=' 					+ str(rand) + \
		' --SW_bin=' 				+ str(SW_bin) + \
		' --yLo=' 					+ str(yLo) + \
		' --yHi=' 					+ str(yHi) + \
		' --yMinSW=' 				+ str(yMinSW) + \
		' --params_init=' 			+ params_init + \
		' --sigQ_init=' 			+ str(sigQ_init) + \
		' --sigPi_init=' 			+ str(sigPi_init) + \
		' --sigPia_init=' 			+ str(sigPia_init) + \
		' --overcomp=' 				+ str(overcomp) + \
		' --Z_hot=' 				+ str(Z_hot) + \
		' --C_noise_ri=' 			+ str(C_noise_ri) + \
		' --C_noise_ria=' 			+ str(C_noise_ria) + \
		' --learning_rate=' 		+ str(learning_rate) + \
		' --lRateScale=' 			+ str(lRateScale).replace(' ','').replace('[','').replace(']','') + \
		' --maxTms=' 				+ str(maxTms) + \
		' --minTms=' 				+ str(minTms) + \
		' --pct_xVal_train=' 		+ str(pct_xVal_train) + \
		' --maxSamps=' 				+ str(maxSamps) + \
		' --maxRasTrials=' 			+ str(maxRasTrials) )



	# Include boolean flags.	
	if flg_EgalitarianPrior:
		file.write(' --flg_EgalitarianPrior')
	
	if flg_include_Zeq0_infer:
		file.write(' --flg_include_Zeq0_infer')

	if train_2nd_model:	
		print('writing in file train 2nd: ',job_name)
		file.write(' --train_2nd_model')	

	if flg_checkNPZvars:	
		file.write(' --flg_checkNPZvars')

	if verbose_EM:
		file.write(' --verbose_EM')		

	file.write('\n')	

	# close file
	file.close()







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def write_sbatch_script_rasterZ_xVal_realData(script_path, output_path, error_path, whichClust, n_cores, mem, time, job_name, 
		cell_type, stim, num_test_samps_4xVal, rand, SW_bin, yLo, yHi, yMinSW, params_init, sigQ_init, sigPi_init, sigPia_init, 
		overcomp, Z_hot, C_noise_ri, C_noise_ria, learning_rate, lRateScale, maxTms, minTms, pct_xVal_train, flg_EgalitarianPrior, 
		train_2nd_model, flg_include_Zeq0_infer, sample_longSWs_1st, maxSamps, maxRasTrials, flg_checkNPZvars, verbose_EM):

	#print('2nd model: ',train_2nd_model)

	# open file
	file = open(script_path, 'w')
	# sbatch header
	file.write('#!/bin/bash\n')
	
	if whichClust=='nersc':
		homeDir = '/global/homes/w/warner/'
		# shared queue
		file.write('#SBATCH -q shared\n')
		# number of cores - optional argument
		file.write('#SBATCH -n ' + str(n_cores) + '\n')
		# email for failure
		file.write('#SBATCH --mail-user=cwarner@berkeley.edu\n')
		file.write('#SBATCH --mail-type=FAIL\n\n')
	
	if whichClust=='cortex':
		homeDir = '/global/home/users/cwarner/'
		# specify the partition as 'cortex' if on cortex.
		file.write('#SBATCH --partition=' + whichClust + '\n')	
		# Constrain Nodes:
		# file.write('#SBATCH --constraint=cortex_nogpu \n')
		# # Processors:
		# file.write('#SBATCH --ntasks=4 \n')
		# Memory:
		if mem:
			file.write( str('#SBATCH --mem-per-cpu='+str(mem)+'M \n') )

	# amount of time - optional argument 
	file.write('#SBATCH --time=' + time + '\n')
	# job name - optional argument
	file.write('#SBATCH --job-name=' + job_name + '\n')
	# path for job output
	file.write('#SBATCH -o ' + output_path + '\n')
	# path for job errors
	file.write('#SBATCH -e ' + error_path + '\n')

	# import python and anaconda
	file.write('module load python \n')
	# python command
	# which selections to run

	CST_str = str(cell_type).replace(' ','') # have to do this because of weirdness in calling function and passing in this var.
	

	file.write(
		'python3 ' + homeDir + 'Projects/G_Field_Retinal_Data/'	+ \
		'Chris_working_code_2019/python_code/raster_zs_inferred_xValSWs_given_model.py' + \
		' --cell_type=' 			+ CST_str + \
		' --whichCells=' 			+ '' + \
		' --whichGLM=' 				+ 'real' + \
		' --whichPop=' 				+ '' + \
		' --stim=' 					+ stim + \
		' --num_test_samps_4xVal=' 	+ str(num_test_samps_4xVal) + \
		' --sample_longSWs_1st='	+ str(sample_longSWs_1st) + \
		' --rand=' 					+ str(rand) + \
		' --SW_bin=' 				+ str(SW_bin) + \
		' --yLo=' 					+ str(yLo) + \
		' --yHi=' 					+ str(yHi) + \
		' --yMinSW=' 				+ str(yMinSW) + \
		' --params_init=' 			+ params_init + \
		' --sigQ_init=' 			+ str(sigQ_init) + \
		' --sigPi_init=' 			+ str(sigPi_init) + \
		' --sigPia_init=' 			+ str(sigPia_init) + \
		' --overcomp=' 				+ str(overcomp) + \
		' --Z_hot=' 				+ str(Z_hot) + \
		' --C_noise_ri=' 			+ str(C_noise_ri) + \
		' --C_noise_ria=' 			+ str(C_noise_ria) + \
		' --learning_rate=' 		+ str(learning_rate) + \
		' --lRateScale=' 			+ str(lRateScale).replace(' ','').replace('[','').replace(']','') + \
		' --maxTms=' 				+ str(maxTms) + \
		' --minTms=' 				+ str(minTms) + \
		' --pct_xVal_train=' 		+ str(pct_xVal_train) + \
		' --maxSamps=' 				+ str(maxSamps) + \
		' --maxRasTrials=' 			+ str(maxRasTrials) )



	# Include boolean flags.	
	if flg_EgalitarianPrior:
		file.write(' --flg_EgalitarianPrior')
	
	if flg_include_Zeq0_infer:
		file.write(' --flg_include_Zeq0_infer')

	if train_2nd_model:	
		print('writing in file train 2nd: ',job_name)
		file.write(' --train_2nd_model')	

	if flg_checkNPZvars:	
		file.write(' --flg_checkNPZvars')

	if verbose_EM:
		file.write(' --verbose_EM')		

	file.write('\n')	

	# close file
	file.close()




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def write_sbatch_script_StatsInfPL_realData( script_path, output_path, error_path, whichClust, n_cores, mem,
	time, job_name, cellSubTypes, stim, num_test_samps_4xVal, rand, SW_bin, yLo, yHi, yMinSW, params_init, sigQ_init, 
	sigPi_init, sigPia_init, Z_hot, C_noise_ri, C_noise_ria, learning_rate, lRateScale, maxTms, 
	minTms, train_2nd_model, flg_include_Zeq0_infer, flg_compute_StatsPostLrn, flg_compute_StatsDuringEM, verbose_EM ):
	# TO ADD: sample_longSWs_1st, and ' --sample_longSWs_1st='	+ str(sample_longSWs_1st) + \


	# open file
	file = open(script_path, 'w')
	# sbatch header
	file.write('#!/bin/bash\n')
	
	if whichClust=='nersc':
		homeDir = '/global/homes/w/warner/'
		# shared queue
		file.write('#SBATCH -q shared\n')
		# number of cores - optional argument
		file.write('#SBATCH -n ' + str(n_cores) + '\n')
		# email for failure
		file.write('#SBATCH --mail-user=cwarner@berkeley.edu\n')
		file.write('#SBATCH --mail-type=FAIL\n\n')
	
	if whichClust=='cortex':
		homeDir = '/global/home/users/cwarner/'
		# specify the partition as 'cortex' if on cortex.
		file.write('#SBATCH --partition=' + whichClust + '\n')	
		# Constrain Nodes:
		# file.write('#SBATCH --constraint=cortex_nogpu \n')
		# # Processors:
		# file.write('#SBATCH --ntasks=4 \n')
		# Memory:
		if mem:
			file.write( str('#SBATCH --mem-per-cpu='+str(mem)+'M \n') )

	# amount of time - optional argument 
	file.write('#SBATCH --time=' + time + '\n')
	# job name - optional argument
	file.write('#SBATCH --job-name=' + job_name + '\n')
	# path for job output
	file.write('#SBATCH -o ' + output_path + '\n')
	# path for job errors
	file.write('#SBATCH -e ' + error_path + '\n')

	# import python and anaconda
	file.write('module load python \n')
	# python command
	# which selections to run


	#print( str('\'' + str(cellSubTypes) + '\'') )
	print(cellSubTypes)
	print(type(cellSubTypes))
	CST_str = str(cellSubTypes).replace(' ','') # have to do this because of weirdness in calling function and passing in this var.
	print(CST_str)
	print(type(CST_str))

	file.write(
		'python3 ' + homeDir + 'Projects/G_Field_Retinal_Data/'	+ \
		'Chris_working_code_2019/python_code/StatsInfPL_realData.py' + \
		' --cellSubTypes=' 			+ CST_str + \
		' --stim=' 					+ stim + \
		' --num_test_samps_4xVal=' 	+ str(num_test_samps_4xVal) + \
		' --rand=' 					+ str(rand) + \
		' --SW_bin=' 				+ str(SW_bin) + \
		' --yLo=' 					+ str(yLo) + \
		' --yHi=' 					+ str(yHi) + \
		' --yMinSW=' 				+ str(yMinSW) + \
		' --params_init=' 			+ params_init + \
		' --sigQ_init=' 			+ str(sigQ_init) + \
		' --sigPi_init=' 			+ str(sigPi_init) + \
		' --sigPia_init=' 			+ str(sigPia_init) + \
		' --Z_hot=' 				+ str(Z_hot) + \
		' --C_noise_ri=' 			+ str(C_noise_ri) + \
		' --C_noise_ria=' 			+ str(C_noise_ria) + \
		' --learning_rate=' 		+ str(learning_rate) + \
		' --lRateScale=' 			+ str(lRateScale).replace(' ','').replace('[','').replace(']','') + \
		' --maxTms=' 				+ str(maxTms) + \
		' --minTms=' 				+ str(minTms))

		




	# Include boolean flags.	
	if flg_include_Zeq0_infer:
		file.write(' --flg_include_Zeq0_infer')

	if train_2nd_model:	
		file.write(' --train_2nd_model')	

	if flg_compute_StatsPostLrn:
		file.write(' --flg_compute_StatsPostLrn')

	if flg_compute_StatsDuringEM:	
		file.write(' --flg_compute_StatsDuringEM')			

	if verbose_EM:
		file.write(' --verbose_EM')			
		




	file.write('\n')	

	# close file
	file.close()	






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def write_sbatch_script_compare_SWdists_realNsynth(script_path, output_path, error_path, whichClust, n_cores, mem, time, job_name, 
	CST, N, M, M_mod, numSWs_Mov, numSWs_Wnz, SW_bin, synthSamps, K_iter, Kmin_iter, Kmax_iter, C_iter, Cmin_iter, Cmax_iter, 
	mu_Pia_iter, sig_Pia_iter, mu_Pi_iter, sig_Pi_iter, bernoulli_Pi, yHiR, yHi, yLo, yMinSW, flg_include_Zeq0_infer, verbose): 	


	# open file
	file = open(script_path, 'w')
	# sbatch header
	file.write('#!/bin/bash\n')
	
	if whichClust=='nersc':
		homeDir = '/global/homes/w/warner/'
		# shared queue
		file.write('#SBATCH -q shared\n')
		# number of cores - optional argument
		file.write('#SBATCH -n ' + str(n_cores) + '\n')
		# email for failure
		file.write('#SBATCH --mail-user=cwarner@berkeley.edu\n')
		file.write('#SBATCH --mail-type=FAIL\n\n')
	
	if whichClust=='cortex':
		homeDir = '/global/home/users/cwarner/'
		# specify the partition as 'cortex' if on cortex.
		file.write('#SBATCH --partition=' + whichClust + '\n')	
		# Constrain Nodes:
		# file.write('#SBATCH --constraint=cortex_nogpu \n')
		# # Processors:
		# file.write('#SBATCH --ntasks=4 \n')
		# Memory:
		if mem:
			file.write( str('#SBATCH --mem-per-cpu='+str(mem)+'M \n') )

	# amount of time - optional argument 
	file.write('#SBATCH --time=' + time + '\n')
	# job name - optional argument
	file.write('#SBATCH --job-name=' + job_name + '\n')
	# path for job output
	file.write('#SBATCH -o ' + output_path + '\n')
	# path for job errors
	file.write('#SBATCH -e ' + error_path + '\n')

	# import python and anaconda
	file.write('module load python \n')
	# python command
	# which selections to run


	# #print( str('\'' + str(cellSubTypes) + '\'') )
	# print(cellSubTypes)
	# print(type(cellSubTypes))
	# CST_str = str(cellSubTypes).replace(' ','') # have to do this because of weirdness in calling function and passing in this var.
	# print(CST_str)
	# print(type(CST_str))

	file.write(
		'python3 ' + homeDir + 'Projects/G_Field_Retinal_Data/'	+ \
		'Chris_working_code_2019/python_code/compare_SWdists_realNsynthData.py' + \

		' --CST='			+ CST + \
		' --N='				+ str(N) + \
		' --M='				+ str(M) + \
		' --M_mod='			+ str(M_mod) + \
		' --numSWs_Mov='	+ str(numSWs_Mov) + \
		' --numSWs_Wnz='	+ str(numSWs_Wnz) + \
		' --SW_bin='		+ str(SW_bin) + \
		' --synthSamps='	+ str(synthSamps) + \
		' --K_iter='		+ str(K_iter).replace(' ','').replace('[','').replace(']','') + \
		' --Kmin_iter='		+ str(Kmin_iter).replace(' ','').replace('[','').replace(']','') + \
		' --Kmax_iter='		+ str(Kmax_iter).replace(' ','').replace('[','').replace(']','') + \
		' --C_iter='		+ str(C_iter).replace(' ','').replace('[','').replace(']','') + \
		' --Cmin_iter='		+ str(Cmin_iter).replace(' ','').replace('[','').replace(']','') + \
		' --Cmax_iter='		+ str(Cmax_iter).replace(' ','').replace('[','').replace(']','') + \
		' --mu_Pia_iter='	+ str(mu_Pia_iter).replace(' ','').replace('[','').replace(']','') + \
		' --sig_Pia_iter='	+ str(sig_Pia_iter).replace(' ','').replace('[','').replace(']','') + \
		' --mu_Pi_iter='	+ str(mu_Pi_iter).replace(' ','').replace('[','').replace(']','') + \
		' --sig_Pi_iter='	+ str(sig_Pi_iter).replace(' ','').replace('[','').replace(']','') + \
		' --bernoulli_Pi='	+ str(bernoulli_Pi) + \
		' --yHiR='			+ str(yHiR) + \
		' --yHi='			+ str(yHi) + \
		' --yLo='			+ str(yLo) + \
		' --yMinSW='		+ str(yMinSW) )

	
	# Include boolean flags.	
	if flg_include_Zeq0_infer:
		file.write(' --flg_include_Zeq0_infer')

	if verbose:
		file.write(' --verbose')			

	file.write('\n')	

	# close file
	file.close()	




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def write_sbatch_script_pgmCA_GLMsimData(script_path, output_path, error_path, whichClust, n_cores, mem, time, job_name, 
		whichCell, whichGLM, whichPop, cell_type, cellSubTypes, stim, num_test_samps_4xVal, rand, SW_bin, yLo, yHi, yMinSW, params_init, sigQ_init, 
		sigPi_init, sigPia_init, overcomp, Z_hot, C_noise_ri, C_noise_ria, learning_rate, lRateScale, ds_fctr_snapshots, 
		pct_xVal_train, xVal_snapshot, xVal_batchSize, flg_EgalitarianPrior, train_2nd_model, flg_include_Zeq0_infer, 
		sample_longSWs_1st, maxSamps, flg_checkNPZvars, verbose_EM):

	# open file
	file = open(script_path, 'w')
	# sbatch header
	file.write('#!/bin/bash\n')
	
	if whichClust=='nersc':
		homeDir = '/global/homes/w/warner/'
		# shared queue
		file.write('#SBATCH -q shared\n')
		# number of cores - optional argument
		file.write('#SBATCH -n ' + str(n_cores) + '\n')
		# email for failure
		file.write('#SBATCH --mail-user=cwarner@berkeley.edu\n')
		file.write('#SBATCH --mail-type=FAIL\n\n')
	
	if whichClust=='cortex':
		homeDir = '/global/home/users/cwarner/'
		# specify the partition as 'cortex' if on cortex.
		file.write('#SBATCH --partition=' + whichClust + '\n')	
		# Constrain Nodes:
		# file.write('#SBATCH --constraint=cortex_nogpu \n')
		# # Processors:
		# file.write('#SBATCH --ntasks=4 \n')
		# Memory:
		if mem:
			file.write( str('#SBATCH --mem-per-cpu='+str(mem)+'M \n') )

	# amount of time - optional argument 
	file.write('#SBATCH --time=' + time + '\n')
	# job name - optional argument
	file.write('#SBATCH --job-name=' + job_name + '\n')
	# path for job output
	file.write('#SBATCH -o ' + output_path + '\n')
	# path for job errors
	file.write('#SBATCH -e ' + error_path + '\n')

	# import python and anaconda
	file.write('module load python \n')
	# python command
	# which selections to run


	#print( str('\'' + str(cellSubTypes) + '\'') )
	# print(cellSubTypes)
	# print(type(cellSubTypes))
	CST_str = str(cellSubTypes).replace(' ','') # have to do this because of weirdness in calling function and passing in this var.
	# print(CST_str)
	# print(type(CST_str))

	file.write(
		'python3 ' + homeDir + 'Projects/G_Field_Retinal_Data/'	+ \
		'Chris_working_code_2019/python_code/pgmCA_GLMsimData.py' + \
		' --whichCells=' 			+ whichCell + \
		' --whichGLM=' 				+ whichGLM + \
		' --whichPop=' 				+ whichPop + \
		' --cell_type=' 			+ cell_type + \
		' --cellSubTypes=' 			+ CST_str + \
		' --stim=' 					+ stim + \
		' --num_test_samps_4xVal=' 	+ str(num_test_samps_4xVal) + \
		' --sample_longSWs_1st='	+ str(sample_longSWs_1st) + \
		' --rand=' 					+ str(rand) + \
		' --SW_bin=' 				+ str(SW_bin) + \
		' --yLo=' 					+ str(yLo) + \
		' --yHi=' 					+ str(yHi) + \
		' --yMinSW=' 				+ str(yMinSW) + \
		' --params_init=' 			+ params_init + \
		' --sigQ_init=' 			+ str(sigQ_init) + \
		' --sigPi_init=' 			+ str(sigPi_init) + \
		' --sigPia_init=' 			+ str(sigPia_init) + \
		' --overcomp=' 				+ str(overcomp) + \
		' --Z_hot=' 				+ str(Z_hot) + \
		' --C_noise_ri=' 			+ str(C_noise_ri) + \
		' --C_noise_ria=' 			+ str(C_noise_ria) + \
		' --learning_rate=' 		+ str(learning_rate) + \
		' --lRateScale=' 			+ str(lRateScale).replace(' ','').replace('[','').replace(']','') + \
		' --maxSamps=' 				+ str(maxSamps) + \
		' --ds_fctr_snapshots=' 	+ str(ds_fctr_snapshots) + \
		' --pct_xVal_train=' 		+ str(pct_xVal_train) + \
		' --xVal_snapshot=' 		+ str(xVal_snapshot) + \
		' --xVal_batchSize=' 		+ str(xVal_batchSize) )
		
	# Include boolean flags.	
	if flg_include_Zeq0_infer:
		file.write(' --flg_include_Zeq0_infer')

	if train_2nd_model:	
		file.write(' --train_2nd_model')			
		
	if verbose_EM:
		file.write(' --verbose_EM')	

	if flg_EgalitarianPrior:
		file.write(' --flg_EgalitarianPrior')	

	if flg_checkNPZvars:	
		file.write(' --flg_checkNPZvars')	


	file.write('\n')	

	# close file
	file.close()







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def write_sbatch_script_rasterZ_GLMsimData(script_path, output_path, error_path, whichClust, n_cores, mem, time, job_name, 
		cell_type, whichCells, whichGLM, whichPop, stim, num_test_samps_4xVal, rand, SW_bin, yLo, yHi, yMinSW, params_init, sigQ_init, sigPi_init, sigPia_init, 
		overcomp, Z_hot, C_noise_ri, C_noise_ria, learning_rate, lRateScale, maxTms, minTms, pct_xVal_train, flg_EgalitarianPrior, 
		train_2nd_model, flg_include_Zeq0_infer, sample_longSWs_1st, maxSamps, maxRasTrials, flg_checkNPZvars, verbose_EM):

	#print('2nd model: ',train_2nd_model)

	# open file
	file = open(script_path, 'w')
	# sbatch header
	file.write('#!/bin/bash\n')
	
	if whichClust=='nersc':
		homeDir = '/global/homes/w/warner/'
		# shared queue
		file.write('#SBATCH -q shared\n')
		# number of cores - optional argument
		file.write('#SBATCH -n ' + str(n_cores) + '\n')
		# email for failure
		file.write('#SBATCH --mail-user=cwarner@berkeley.edu\n')
		file.write('#SBATCH --mail-type=FAIL\n\n')
	
	if whichClust=='cortex':
		homeDir = '/global/home/users/cwarner/'
		# specify the partition as 'cortex' if on cortex.
		file.write('#SBATCH --partition=' + whichClust + '\n')	
		# Constrain Nodes:
		# file.write('#SBATCH --constraint=cortex_nogpu \n')
		# # Processors:
		# file.write('#SBATCH --ntasks=4 \n')
		# Memory:
		if mem:
			file.write( str('#SBATCH --mem-per-cpu='+str(mem)+'M \n') )

	# amount of time - optional argument 
	file.write('#SBATCH --time=' + time + '\n')
	# job name - optional argument
	file.write('#SBATCH --job-name=' + job_name + '\n')
	# path for job output
	file.write('#SBATCH -o ' + output_path + '\n')
	# path for job errors
	file.write('#SBATCH -e ' + error_path + '\n')

	# import python and anaconda
	file.write('module load python \n')
	# python command
	# which selections to run

	CST_str = str(cell_type).replace(' ','') # have to do this because of weirdness in calling function and passing in this var.

	file.write(
		'python3 ' + homeDir + 'Projects/G_Field_Retinal_Data/'	+ \
		'Chris_working_code_2019/python_code/raster_zs_inferred_allSWs_given_model.py' + \
		' --whichCells=' 			+ whichCells + \
		' --whichGLM=' 				+ whichGLM + \
		' --whichPop=' 				+ whichPop + \
		' --cell_type=' 			+ CST_str + \
		' --stim=' 					+ stim + \
		' --num_test_samps_4xVal=' 	+ str(num_test_samps_4xVal) + \
		' --sample_longSWs_1st='	+ str(sample_longSWs_1st) + \
		' --rand=' 					+ str(rand) + \
		' --SW_bin=' 				+ str(SW_bin) + \
		' --yLo=' 					+ str(yLo) + \
		' --yHi=' 					+ str(yHi) + \
		' --yMinSW=' 				+ str(yMinSW) + \
		' --params_init=' 			+ params_init + \
		' --sigQ_init=' 			+ str(sigQ_init) + \
		' --sigPi_init=' 			+ str(sigPi_init) + \
		' --sigPia_init=' 			+ str(sigPia_init) + \
		' --overcomp=' 				+ str(overcomp) + \
		' --Z_hot=' 				+ str(Z_hot) + \
		' --C_noise_ri=' 			+ str(C_noise_ri) + \
		' --C_noise_ria=' 			+ str(C_noise_ria) + \
		' --learning_rate=' 		+ str(learning_rate) + \
		' --lRateScale=' 			+ str(lRateScale).replace(' ','').replace('[','').replace(']','') + \
		' --maxTms=' 				+ str(maxTms) + \
		' --minTms=' 				+ str(minTms) + \
		' --pct_xVal_train=' 		+ str(pct_xVal_train) + \
		' --maxSamps=' 				+ str(maxSamps) + \
		' --maxRasTrials=' 			+ str(maxRasTrials) )



	# Include boolean flags.	
	if flg_EgalitarianPrior:
		file.write(' --flg_EgalitarianPrior')
	
	if flg_include_Zeq0_infer:
		file.write(' --flg_include_Zeq0_infer')

	if train_2nd_model:	
		print('writing in file train 2nd: ',job_name)
		file.write(' --train_2nd_model')	

	if flg_checkNPZvars:	
		file.write(' --flg_checkNPZvars')

	if verbose_EM:
		file.write(' --verbose_EM')		

	file.write('\n')	

	# close file
	file.close()


