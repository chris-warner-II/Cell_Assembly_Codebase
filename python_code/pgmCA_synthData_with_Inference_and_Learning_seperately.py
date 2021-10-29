import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#import torch as pt


import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.retina_computation as rc 
import utils.plot_functions as pf

# sys.float_info # this gives info about maximum representable values and epsilon, etc..


result = None
while result is None: # errors out sometimes because of nans
	#try:
	if True:

		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		# # # # # # # # # # # # MODEL SYNTHESIS # # # # # # # # # # # #
		# # # # # # # # # # # # # # - & - # # # # # # # # # # # # # # #
		# # # # # # # # # # # # DATA GENERATION # # # # # # # # # # # #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		# (0). Parameters for Data Generation of Spike Words to train 
		#	   Probabalistic Generative Model for Cell Assemblies.

		N 						= 50		# number of cells			(cardinality of Y-vector)
		M 						= 40   		# number of cell assemblies (cardinality of Z-vector)
		M_mod 					= 2*M 		# Number of cell assemblies in model (can be larger, smaller or equal to true M.)
		#
		K 						= 2 		# Number of	cell assemblies active, given a Bernoulli distribution ("hotness" of Z-vector)
		Kmin 					= 0 		# Max & Min number of cell assemblies active 
		Kmax 					= 2 		# 
		#
		C 						= 2			# number of cells participating in cell assemblies. ("hotness" of each row in Pia)
		Cmin 					= 2 		# Max & Min number of cell active to call it a cell assembly
		Cmax 					= 6 		# 
		#
		num_SWs 				= 3000 		# number of spike words to draw (sampling z from Bernoulli and piping thru Pia to construct y, using Pi also)
		#
		mu_Pia					= 0.0
		sig_Pia 				= 0.3		# Standard deviation of variation from 0 & 1 in the Pia matrix.
		# 									# Note: 'bernoulli_Pia' (aka. q) is determined by K & M.							
		bernoulli_Pi 			= 1.0 		# Probability that a cell is silent (Pi=1) rather than chattery (Pi=0) outside of cell assemblies
		mu_Pi 					= 0.0		# How far a Pi value can be from 0 or 1.
		sig_Pi 					= 0.1		# Standard deviation in the Pi vector from the 1-mu_Pi or 0+mu_Pi points
		#
		# Parameter initializations for learning & EM algorithm
		params_init 			= 'NoisyConst' 	# Options: {'True', 'RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise) }
		C_noise 				= np.array([4/M_mod, 1, 1 ])		#[q, pi, pia] Mean of parameters for 'NoisyConst' initializations. Scalar between 0 and 1. 1 means almost silent. 0 means almost always firing.
		sig_init 				= np.array([0.01, 0.05, 0.05 ])		# STD on parameter initializations from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
		#
		learning_rate 			= 1e0
		lRateScale_Pi 			= 1		# Multiplicative scaling to Pi learning rate. If set to zero, Pi taken out of model essentially.
		#
		numInferenceSamples 	= 0		# number of times to perform inference of Y, Z-vector given fixed model params.
		numLearningSamples 		= 0		# number of times to perform learning of model params given fixed / known Y, Z-vectors.
		num_EM_Samples 			= 1000		# number of steps to run full EM algorithm - alternating Inference and Learning.
		#
		ds_fctr_snapshots 		= 100 	# Take a sample of model parameters every *ds_fctr* time steps to compute and plot MSE after we determine permutation of learned CA's to True CA's
		numPia_snapshots 		= np.round(num_EM_Samples/ds_fctr_snapshots).astype(int)
		samps2snapshot_4Pia 	= (np.hstack([ 1, np.arange( np.round(num_EM_Samples/numPia_snapshots).astype(int), num_EM_Samples, 
									np.round(num_EM_Samples/numPia_snapshots).astype(int)  ), num_EM_Samples]) -1).astype(int)
		numPia_snapshots 		+= 1
		#
		num_EM_randomizations	= 1 # number of times to randomize samples and rerun EM on same data generated from single synthetic model.


		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		# (1).  Flags for display or plotting
		#
		# Flags for Statistics for Model Synthesis and Data Generation
		dataGen_hist_flag 					= False
		#
		# Flags for Learning results (using ground truth z vectors). It works.
		flg_do_just_learning 				= False 	
		verbose_learning 					= False
		plt_learning_MSE_flg 				= False
		plt_learning_derivs_flg 			= False
		plt_learning_params_init_n_final 	= False
		learning_SWsamps_hist_flg 			= False
		#
		# Flags for Inference (using ground truth or noised up versions of model)
		flg_do_just_inference 				= False 	# The code within is old and functions have been modified. They are up to date in EM.
		verbose_inference 					= False
		pjt_tol 							= 10 # number of decimal points to round log joint to consider Cell Assemblies to be equivalent.
												 # ^^^ NOT BEING USED RIGHT NOW (except in 1st order), BUT MAYBE USEFUL TO PUT IN.
		flg_do_approx_inference 			= False # Always finds the z-vector = 0 solution to have largest p-joint (as it should mathematically)
		flg_do_greedy_inference 			= False
		flg_do_1shot_inference 				= False
		flg_do_1stOrdr_inference 			= True
		flg_do_2ndOrdr_inference 			= False
		flg_do_Hopfield_inference			= False

		flg_plt_infer_performance_stats 	= False
		flg_plot_compare_YinferVsTrueVsObs	= False
		#
		# Flags for Full Expectation Maximization Algorithm (alternating Inference and Learning)
		verbose_EM 							= False
		#
		flg_ignore_Zeq0_inference			= True
		#
		plt_EMlearning_MSE_flg 				= True		
		plt_EMlearning_derivs_flg 			= True
		plt_EMlearning_params_init_n_final 	= True
		EMlearning_SWsamps_hist_flg 		= True
		plt_dataGen_andSampling_hist_flg 	= True
		plt_EMinfer_performance_stats 		= True
		flg_temporal_EM_inference_analysis 	= True
		flg_save_learned_model_EM 			= True




		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		# (2).  Directory structure for output figures
		#
		ModelType = str( 'SyntheticData_N' + str(N) + '_M' + str(M) + '_Mmod' + str(M_mod)
							+ '_K' + str(K) + '_' + str(Kmin) + '_' + str(Kmax) 
							+ '_C' + str(C) + '_' + str(Cmin)+ '_' + str(Cmax) 
							+ '_mPia' + str(mu_Pia) + '_sPia' + str(sig_Pia) 
							+ '_bPi' + str(bernoulli_Pi) + '_mPi' + str(mu_Pi) + '_sPi' + str(sig_Pi) 
							+ '_SWs' + str(num_SWs) + '/' )
		#
		dirHome, dirScratch = dm.set_dir_tree()
		#
		learn_figs_dir = str( dirScratch + 'figs/PGM_analysis/Learning/' + 'Model_' + ModelType )
		if not os.path.exists(learn_figs_dir):
			os.makedirs(learn_figs_dir)
		#
		infer_figs_dir = str( dirScratch + 'figs/PGM_analysis/Inference/' + 'Model_' + ModelType )
		if not os.path.exists(infer_figs_dir):
			os.makedirs(infer_figs_dir)
		#
		dataGen_figs_dir = str( dirScratch + 'figs/PGM_analysis/DataGen/' + 'Model_' + ModelType )
		if not os.path.exists(dataGen_figs_dir):
			os.makedirs(dataGen_figs_dir)
		#
		EM_figs_dir = str( dirScratch + 'figs/PGM_analysis/EM_Algorithm/' + 'Model_' + ModelType )
		if not os.path.exists(EM_figs_dir):
			os.makedirs(EM_figs_dir)
		#
		EM_data_dir = str( dirScratch + 'data/python_data/PGM_analysis/Synthetic_Data/Models_learned_EM/' + 'Model_' + ModelType )
		if not os.path.exists(EM_data_dir):
			os.makedirs(EM_data_dir)



		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		# (1). Synthesize model from user input parameters.
		q = None #ri = None #ria = None #ria_mod = None
		while q==None: # while statement to resynthesize model if any cells participate in 0 assemblies.
			q, ri, ria, ria_mod = rc.synthesize_model(N,M,M_mod, C,K, Cmin,Cmax, bernoulli_Pi,mu_Pi,sig_Pi, mu_Pia,sig_Pia, verbose=False)

		Q 		= rc.sig(q)			# these 4
		Pi 		= rc.sig(ri)		# are
		Pia 	= rc.sig(ria) 		# for quicker trouble shooting .  
		Pia_mod = rc.sig(ria_mod)	# Should not be in final version of things.












		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		# (3). Generate Data Vectors Y & Z using Pia & Pi and some sampling assumptions.
		# 	(a). Sample Z vector. 
		# 	(b). Construct Y vector from Z and model (Pia, Pi).
		Y_list,Z_list = rc.generate_data(num_SWs, q, ri, ria, M, Cmin, Cmax, Kmin, Kmax)
		CA_ovl = rc.compute_CA_Overlap(ria)

		# Compute and plot Statistics on Generated Synthetic Data.
		# If there are biases (just random ones), the learning algorithm will learn them.
		Ycell_hist, Zassem_hist, nY, nZ, CA_coactivity, Cell_coactivity = rc.compute_dataGen_Histograms(Y_list, Z_list, M, N, num_SWs)
		




		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		# (4). Initialize learned model in the non-probability q, ri, ria space and Store the parameter init values so I can compare to them later.
		# 		For NoisyTrue case set sigma_std in probability space and pipe it thru the logistic function sig(sigma_std)
		#		For P = {0,1}, set r = {-b,+b} where b is something reasonably big (but not machine precision). 
		#		Nice Bayesian interpretation that we can never be absolutely certain of cell being active or silent.
		#

		# when dealing with real data we dont know the model parameters so we set them to these uniform, reasonable values.
		# When dealing with synthetic model and generated data, these will have been defined above.

		q_init, ri_init, ria_init, params_init_param = rc.init_parameters(q, ri, ria_mod, params_init, sig_init, C_noise, N, M_mod)
		#
		qp = q_init
		rip = ri_init
		riap = ria_init






		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#
		# (5). RUN FULL EXPECTATION MAXIMIZATION ALGORITHM.
		#		(A). Infer Z-vector from observed Y-vector and current model parameters.
		#		(B). Learn model parameters (via gradient descent) given Y-vector and Z-vector.
		#
		#

		num_rand=0 # Note: this may have to go outside the while result loop because there is a break inside this while loop. Not sure.

		dot_prod_Tru2Lrn_diffRand  = list() 
		translate_Tru2Lrn_diffRand = list() 
		dot_prod_Lrn2Tru_diffRand  = list() 
		translate_Lrn2Tru_diffRand = list() 

		while num_rand<num_EM_randomizations:

			print('Random Sampling of Spike Words #',num_rand)
			rand_EM_figs_dir = str(EM_figs_dir + 'rand' + str(num_rand) + '/')
			fname_EMlrn = str( EM_data_dir + 'Init_' + params_init + params_init_param + '_logLR' + str(int(np.log10(learning_rate))) + '_LRpi' + \
				  str(lRateScale_Pi).replace('.','pt') + '_' + str(num_SWs) + 'SWs_' + str(num_EM_Samples) + 'EMsamps_rand' + str(num_rand) + '.npz' )


			# if os.path.exists(rand_EM_figs_dir) and os.path.isfile(fname_EMlrn):
			# 	print('Already ran for random sampling #',num_rand,'. Delete figure dir and npz file if you want to run again')
			# 	print( rand_EM_figs_dir )
			# 	print( fname_EMlrn )
				
			# 	data = np.load(fname_EMlrn)
			# 	num_rand+=1
			# 	break


			qp, rip, riap, Z_inferred_EM, Y_inferred_EM, smp_EM, pjoint_EM, ria_snapshots_EM, q_deriv_EM, ri_deriv_EM,  \
				ria_deriv_EM,q_MSE_EM, ri_MSE_EM, ria_MSE_EM = rc.run_EM_algorithm(qp, rip, riap,  q, ri, ria, N, M_mod, \
				Z_list, Y_list, num_EM_Samples, num_SWs, samps2snapshot_4Pia, pjt_tol, learning_rate, lRateScale_Pi, \
				flg_ignore_Zeq0_inference, flg_save_learned_model_EM, fname_EMlrn, verbose_EM)












			if True:
				# Permute Piap so that it best matches Pia. But translate should have each entry only 1 time.
				# When we have ground truth for the Cell Assembly matrix (Pia) and we have learned through the EM algorithm
				# a Piap matrix, the learned matrix can be a permutation of the original in the simplest case where the Piap
				# matrix is complete (M_a = M_b). The learned Piap matrix can also be over-complete or under-complete (M_a </> M_b).
				# This function works in two ways. Depending on the order in which you feed in Pia and Piap, or which one you
				# call matrix A when passing them into the function.
				# It finds the column (axis=1) in matrix B that best matches each column (axis=1) in matrix A.
				# M_a = number of columns in matrix A and 
				# M_b = number of columns in matrix B.
				# Three cases are: M_a </=/> M_b
				#
				Piap=rc.sig(riap)
				#
				A=(1-Piap) # Learned Piap matrix. Cell Assemblies.
				Atag = 'Learned Piap'
				B=(1-Pia)  # Gnd Truth Pia matrix. Cell Assemblies.
				Btag = 'GndTruth Pia'
				translate_Lrn2TruShuff,dot_prod_Lrn2Tru,translate_Lrn2Tru, translate_Lrn2Lrn = rc.translate_CAs_LrnAndTru(A=A, Atag=Atag, B=B, Btag=Btag, plt_save_dir=rand_EM_figs_dir, verbose=False)
				#
				
				B=(1-Piap) # Learned Piap matrix. Cell Assemblies.
				Btag = 'Learned Piap'
				A=(1-Pia)  # Gnd Truth Pia matrix. Cell Assemblies.
				Atag = 'GndTruth Pia'
				translate_Tru2LrnShuff,dot_prod_Tru2Lrn,translate_Tru2Lrn, translate_Tru2Tru = rc.translate_CAs_LrnAndTru(A=A, Atag=Atag, B=B, Btag=Btag, plt_save_dir=rand_EM_figs_dir, verbose=False)
				#

				# Collect up found cell assemblies and their closeness to (dot product with) GT cell assemblies for different randomizations.
				dot_prod_Tru2Lrn_diffRand.append( dot_prod_Tru2Lrn )
				translate_Tru2Lrn_diffRand.append( translate_Tru2LrnShuff ) 
				dot_prod_Lrn2Tru_diffRand.append( dot_prod_Lrn2Tru ) 
				translate_Lrn2Tru_diffRand.append( translate_Lrn2TruShuff ) 





				# Compute Pia MSE using translations: Then plot below
				print('Pia MSE')
				ria_MSE_EM = np.zeros( (numPia_snapshots, 3) )	
				for i in range( numPia_snapshots ):
					#print(i) 
					Piap_snap = rc.sig(ria_snapshots_EM[i])
					dp_significant = (dot_prod_Lrn2Tru[0]>0.1)[None,:]
					x = Pia_mod[:,translate_Lrn2TruShuff[0]]*dp_significant
					y = Piap_snap[:,translate_Lrn2Lrn]*dp_significant
					xx = ( x - y )**2
					ria_MSE_EM[i,:] = np.array([xx.mean(), xx.std(), xx.max()])
					#print(ria_MSE_EM)

					if False:
						f,ax = plt.subplots(1,3)
						ax[0].imshow(x)
						ax[1].imshow(y)
						ax[2].imshow(xx)
						plt.show()





			## COMPUTE INFERENCE STATISTICS AND CONFUSION MATRICES WITH CORRECT / PERMUTED CELL ASSEMBLIES
			#
			# Compute statistics on inferred Cell / Cell Assembly activity comparing them to true y and z.  
			# NOTE: Will have to adapt this when true z-vector is not known.
			Kinf_EM 	= np.zeros(num_EM_Samples)
			KinfDiff_EM = np.zeros(num_EM_Samples)
			zCapture_EM = np.zeros(num_EM_Samples)
			zMissed_EM 	= np.zeros(num_EM_Samples) 
			zExtra_EM 	= np.zeros(num_EM_Samples)
			yCapture_EM = np.zeros(num_EM_Samples)
			yMissed_EM 	= np.zeros(num_EM_Samples)
			yExtra_EM 	= np.zeros(num_EM_Samples)
			#
			inferCA_Confusion_EM 	= np.zeros( (M_mod+1,M_mod+1) )
			zInferSampled_EM 		= np.zeros( M_mod+1 )
			zInferSampledRaw_EM 	= np.zeros( M_mod+1 )
			zInferSampledT_EM 		= np.zeros( M_mod+1 )
			inferCell_Confusion_EM	= np.zeros( (N+1,N+1) )
			yInferSampled_EM 		= np.zeros( N+1 )
			#
			for sample in range(num_EM_Samples):
				zTrue = Z_list[smp_EM[sample]]
				zHyp = set(translate_Lrn2Tru[ list(Z_inferred_EM[sample]) ])
				y = Y_list[smp_EM[sample]] 
				yHyp = Y_inferred_EM[sample]
				if verbose_EM:
					print('Sample:',sample,' zTrue = ',zTrue,' zHyp = ',zHyp)
					print('OR    :  zTrue = ',set(translate_Tru2Lrn[ list(Z_list[smp_EM[sample]]) ]),' zHyp = ',Z_inferred_EM[sample] ) 

				zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, yUnion, yIntersect, yDiff, ySubed, yAdded, yCapture, yMissed, yExtra, Kinf, KinfDiff = rc.compute_inference_statistics(zTrue, zHyp, M_mod, y, yHyp, N, verbose=verbose_EM)
				#
				Kinf_EM[sample] 	= Kinf
				KinfDiff_EM[sample]	= KinfDiff
				zCapture_EM[sample] = zCapture
				zMissed_EM[sample] 	= zMissed 
				zExtra_EM[sample] 	= zExtra
				yCapture_EM[sample] = yCapture
				yMissed_EM[sample] 	= yMissed
				yExtra_EM[sample] 	= yExtra
				#
				# Compute confusion matrix between inferred z's (zHyp) and true z's (z). (see below for details ...)
				grid = np.ix_(zAdded,zSubed)
				inferCA_Confusion_EM[grid]+=1							# add to off-diagonal (i,j) for mix-ups
				inferCA_Confusion_EM[( zIntersect,zIntersect )]+=1 		# add to diagonal (i,i) for correct inference
				zInferSampled_EM[list(zHyp)]+=1 						# keeping track of how many times each cell assembly was active in an inference sample.
				zInferSampledRaw_EM[list(Z_inferred_EM[sample])]+=1
				zInferSampledT_EM[list(zTrue)]+=1 
				#
				# Compute confusion matrix between inferred y's (yHyp) and observed y's (y). (see below for details ...)
				grid = np.ix_(yAdded,ySubed)
				inferCell_Confusion_EM[grid]+=1							# add to off-diagonal (i,j) for mix-ups
				inferCell_Confusion_EM[( yIntersect,yIntersect )]+=1 	# add to diagonal (i,i) for correct inference
				yInferSampled_EM[list(yHyp)]+=1 						# keeping track of how many times each cell assembly was active in an inference sample.
			


			result=1

			# # MAKE SOME DIAGNOSTIC PLOTS FROM THE FULL EM ALGORITHM.
			# Compute statistics on EM sampling to see if that has introduced any biases
			Z_smp_EM_list = [ Z_list[i] for i in list(smp_EM) ]
			Y_smp_EM_list = [ Y_list[i] for i in list(smp_EM) ]
			Ycell_hist_Samp, Zassem_hist_Samp, nY_Samp, nZ_Samp, CA_coactivity_Samp, Cell_coactivity_Samp = rc.compute_dataGen_Histograms(Y_smp_EM_list, Z_smp_EM_list, M_mod, N, num_EM_Samples)
			

			if plt_dataGen_andSampling_hist_flg:
				pf.hist_dataGen_andSampling_statistics(Ycell_hist, Zassem_hist, nY, nZ, CA_coactivity, Cell_coactivity, Ycell_hist_Samp, Zassem_hist_Samp, nY_Samp, nZ_Samp, CA_coactivity_Samp, Cell_coactivity_Samp, translate_Tru2LrnShuff[0],translate_Tru2Tru, translate_Lrn2TruShuff[0],translate_Lrn2Lrn, CA_ovl, num_SWs, num_EM_Samples, N, M, Kmax, rand_EM_figs_dir)
				#
				pf.hist_SWsampling4learning_stats(smp_EM, num_SWs, num_EM_Samples, ria, ri, N, M, C, Cmin, Cmax, rand_EM_figs_dir)

			if plt_EMlearning_MSE_flg:
				pf.plot_params_MSE_during_learning(q_MSE_EM, ri_MSE_EM, ria_MSE_EM, samps2snapshot_4Pia, num_EM_Samples, N, M, learning_rate,lRateScale_Pi, params_init, params_init_param, rand_EM_figs_dir)

			if plt_EMlearning_derivs_flg:
				pf.plot_params_derivs_during_learning(q_deriv_EM, ri_deriv_EM, ria_deriv_EM, num_EM_Samples, N, M, learning_rate,lRateScale_Pi, ds_fctr_snapshots, params_init, params_init_param, rand_EM_figs_dir)

			if plt_EMlearning_params_init_n_final:
				pf.plot_params_init_n_learned(q, ri, ria, qp, rip, riap, q_init, ri_init, ria_init, translate_Tru2LrnShuff[0],translate_Tru2Tru, translate_Lrn2TruShuff[0],translate_Lrn2Lrn, zInferSampledRaw_EM, num_EM_Samples, N, M, M_mod, learning_rate, lRateScale_Pi, params_init, params_init_param, rand_EM_figs_dir)

			if plt_EMinfer_performance_stats:
				pf.plot_CA_inference_performance(inferCA_Confusion_EM, inferCell_Confusion_EM, CA_ovl, CA_coactivity, zInferSampledRaw_EM, zInferSampledT_EM, Zassem_hist, yInferSampled_EM, Kinf_EM, KinfDiff_EM, N, M, M_mod, translate_Tru2LrnShuff[0],translate_Tru2Tru, translate_Lrn2TruShuff[0],translate_Lrn2Lrn, num_SWs, params_init, params_init_param, rand_EM_figs_dir, approx=False, inferType='1stOrdr')
				
			# # QUANTIFY IF INFERENCE IS IMPROVING WITH TIME (AND LEARNING) IN THE FULL EM ALGORITHM. IT SHOULD BE.
			if flg_temporal_EM_inference_analysis:
				pf.plot_CA_inference_temporal_performance(Z_inferred_EM, Z_list, smp_EM, M_mod, M, translate_Tru2Lrn,translate_Tru2Tru, translate_Lrn2Tru,translate_Lrn2Lrn, num_SWs, params_init, params_init_param, rand_EM_figs_dir, approx=False, inferType='1stOrdr')


			# # Plot #Cells correct vs. dropped vs. added in observed y vector vs in Noisy Interpretation of thresholded Pia for active za's
			# if flg_plot_compare_YinferVsTrueVsObs:
			# 	pf.plot_compare_YinferVsTrueVsObs(numCellsCorrectInYobs, numCellsAddedInYobs, numCellsDroppedInYobs, numCellsTotalInYobs, yCapture_Collect_Ordr1, yExtra_Collect_Ordr1, yMissed_Collect_Ordr1, yCapture_binVobs, yExtra_binVobs, yMissed_binVobs, numInferenceSamples, Q, bernoulli_Pi, mu_Pi, sig_Pi, mu_Pia, sig_Pia, params_init, params_init_param, infer_figs_dir, inferType='1stOrdr')





			# Print average value of parameter values < and > 0.5 to quantify how different from binary they are.
			print('Pia mean variation from binary: (',Pia[Pia<0.5].mean(),' , ',Pia[Pia>0.5].mean(),')')

			print('Pi mean variation from binary: (',Pi[Pi<0.5].mean(),' , ',Pi[Pi>0.5].mean(),')')

			print('Q: (',Q,')(Qp:(',rc.sig(qp),') --> Qp converges to Kinf_EM.sum()/num_EM_Samples/M ')

			result = 1
			num_rand+=1

	#except:
	#	pass



	# Look at different GT CA's learned with different random samplings of the same collection of spike words.
	dot_prod_Threshold = 0.5
	translate_Lrn2Tru_allRand = set()
	translate_Tru2Lrn_allRand = set()
	f = open(str(EM_figs_dir + 'Improvement_with_Random_Samplings.txt'), 'w')
	for r in range(num_EM_randomizations):
		f.write( str('After '+str(r)+' random samplings ('+str(num_EM_Samples)+' samples from '+str(num_SWs)+' spike words), at or above a dot product overlap of '+str(dot_prod_Threshold)+'\n') )
		inds = np.where(dot_prod_Lrn2Tru_diffRand[r]>dot_prod_Threshold)
		x = list(inds[0])
		y = list(inds[1])
		for i in range(len(x)):
			translate_Lrn2Tru_allRand.update( [ translate_Lrn2Tru_diffRand[r][x[i]][y[i]] ] )
		f.write( str('Captured '+str(len(translate_Lrn2Tru_allRand))+' True (Tru2Lrn) CAs: '+str(translate_Lrn2Tru_allRand)+'\n') ) 
		#
		inds = np.where(dot_prod_Tru2Lrn_diffRand[r]>dot_prod_Threshold)
		x = list(inds[0])
		y = list(inds[1])
		for i in range(len(x)):
			translate_Tru2Lrn_allRand.update( [ translate_Tru2Lrn_diffRand[r][x[i]][y[i]] ] )
		f.write( str('with '+str(len(translate_Tru2Lrn_allRand))+' Learned (Tru2Lrn) CAs: '+str(translate_Tru2Lrn_allRand)+'\n') )
		f.write('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n')
		f.write('\n')
	f.close()	

















	







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # OLD CODE FOR  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # SEPARATE LEARNING # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # AND # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # SEPARATE INFERENCE  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # (UNCOMMENT TO USE !!) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #







# 	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 	# (4). MAP INFERENCE: TAKE Pi, Pia, q FROM GROUND TRUTH & INFER Z-vector
# 	#	   IN GREEDY WAY. ITERATIVELY, CHOOSE SINGLE Z-element THAT MAXIMALLY   
# 	# 	   INCREASES THE MAP (POSTERIOR PROB)
# 	#
# 	#
# 	zInferSampledT 		= np.zeros(M+1)
# 	zInferSampledT[M] 	= numInferenceSamples		# Keeping track of the total number of inference samples.


# 	# To see if Spike words (y's) differ from truth (thresholded/binarized Pia column with active za's) for 'noisy' model.
# 	numCellsDroppedInYobs 		= np.zeros(numInferenceSamples).astype(int)
# 	numCellsAddedInYobs 		= np.zeros(numInferenceSamples).astype(int)
# 	numCellsCorrectInYobs 		= np.zeros(numInferenceSamples).astype(int)
# 	numCellsTotalInYobs 		= np.zeros(numInferenceSamples).astype(int)
# 	#
# 	Kinf_binVobs 				= np.zeros(numInferenceSamples).astype(int)
# 	yCapture_binVobs 			= np.zeros(numInferenceSamples).astype(int)
# 	yMissed_binVobs 			= np.zeros(numInferenceSamples).astype(int)
# 	yExtra_binVobs 				= np.zeros(numInferenceSamples).astype(int)
# 	inferCell_Confusion_binVobs = np.zeros( (N+1,N+1) )
# 	yInferSampled_binVobs 		= np.zeros(N+1)
# 	yInferSampled_binVobs[N]	= numInferenceSamples


# 	# # Preallocate memory to store statistics about inference results.
# 	if flg_do_greedy_inference:
# 		# Collecting performance results for Full Greedy inference strategy
# 		Kinf_Collect		= np.zeros(numInferenceSamples).astype(int)	#	(2). How many active cell assemblies in inferred z.
# 		KinfDiff_Collect	= np.zeros(numInferenceSamples).astype(int)	#	(2). How much larger is inferred K than true K.
# 		zCapture_Collect 	= np.zeros(numInferenceSamples).astype(int)	#	(3a). True Positives in inferred z.
# 		zMissed_Collect 	= np.zeros(numInferenceSamples).astype(int)	#	(3b). False Negatives in inferred z.
# 		zExtra_Collect 		= np.zeros(numInferenceSamples).astype(int)	#	(3c). False Positives in inferred z.
# 		yCapture_Collect 	= np.zeros(numInferenceSamples).astype(int)	#	(4a). True Positives in y from inferred z.
# 		yMissed_Collect 	= np.zeros(numInferenceSamples).astype(int)	#	(4b). False Negatives in y from inferred z.
# 		yExtra_Collect 		= np.zeros(numInferenceSamples).astype(int)	#	(4c). False Positives in y from inferred z.
# 		pJoint_Diff_Collect = np.zeros(numInferenceSamples) 			# 	(5). Difference between joint calculated from inferred z vs. true z.
# 		inferCA_Confusion 	= np.zeros( (M+1,M+1) )
# 		zInferSampledE 		= np.zeros(M+1)
# 		zInferSampledE[M] 	= numInferenceSamples	# Keeping track of the total number of inference samples.
# 		#
# 		if flg_do_approx_inference:
# 			# Collecting performance results for Approximate Greedy inference strategy
# 			Kinf_CollectA		= np.zeros(numInferenceSamples).astype(int)	#	(2). How many active cell assemblies in inferred z.
# 			KinfDiff_CollectA	= np.zeros(numInferenceSamples).astype(int)	#	(2). How much larger is inferred K than true K.
# 			zCapture_CollectA 	= np.zeros(numInferenceSamples).astype(int)	#	(3a). True Positives in inferred z.
# 			zMissed_CollectA 	= np.zeros(numInferenceSamples).astype(int)	#	(3b). False Negatives in inferred z.
# 			zExtra_CollectA 	= np.zeros(numInferenceSamples).astype(int)	#	(3c). False Positives in inferred z.
# 			yCapture_CollectA 	= np.zeros(numInferenceSamples).astype(int)	#	(4a). True Positives in y from inferred z.
# 			yMissed_CollectA 	= np.zeros(numInferenceSamples).astype(int)	#	(4b). False Negatives in y from inferred z.
# 			yExtra_CollectA		= np.zeros(numInferenceSamples).astype(int)	#	(4c). False Positives in y from inferred z.
# 			pJoint_Diff_CollectA= np.zeros(numInferenceSamples) 			# 	(5). Difference between joint calculated from inferred z vs. true z.
# 			inferCA_ConfusionA 	= np.zeros( (M+1,M+1) )
# 			zInferSampledA 		= np.zeros(M+1)
# 			zInferSampledA[M] 	= numInferenceSamples	# Keeping track of the total number of inference samples.
# 			#




# 	if flg_do_1shot_inference:
# 		# Collecting performance results for Full One-Shot inference strategy
# 		Kinf_Collect1s			= np.zeros(numInferenceSamples).astype(int)	#	(2). How many active cell assemblies in inferred z.
# 		KinfDiff_Collect1s		= np.zeros(numInferenceSamples).astype(int)	#	(2). How much larger is inferred K than true K.
# 		zCapture_Collect1s 		= np.zeros(numInferenceSamples).astype(int)	#	(3a). True Positives in inferred z.
# 		zMissed_Collect1s 		= np.zeros(numInferenceSamples).astype(int)	#	(3b). False Negatives in inferred z.
# 		zExtra_Collect1s 		= np.zeros(numInferenceSamples).astype(int)	#	(3c). False Positives in inferred z.
# 		yCapture_Collect1s 		= np.zeros(numInferenceSamples).astype(int)	#	(4a). True Positives in y from inferred z.
# 		yMissed_Collect1s 		= np.zeros(numInferenceSamples).astype(int)	#	(4b). False Negatives in y from inferred z.
# 		yExtra_Collect1s 		= np.zeros(numInferenceSamples).astype(int)	#	(4c). False Positives in y from inferred z.
# 		pJoint_Diff_Collect1s 	= np.zeros(numInferenceSamples) 			# 	(5). Difference between joint calculated from inferred z vs. true z.
# 		inferCA_Confusion1s 	= np.zeros( (M+1,M+1) )
# 		zInferSampledE1s 		= np.zeros(M+1)
# 		zInferSampledE1s[M] 	= numInferenceSamples # Keeping track of the total number of inference samples.
# 		#
# 		if flg_do_approx_inference:
# 			# Collecting performance results for Approximate One-Shot inference strategy
# 			Kinf_CollectA1s			= np.zeros(numInferenceSamples).astype(int)	#	(2). How many active cell assemblies in inferred z.
# 			KinfDiff_CollectA1s		= np.zeros(numInferenceSamples).astype(int)	#	(2). How much larger is inferred K than true K.
# 			zCapture_CollectA1s 	= np.zeros(numInferenceSamples).astype(int)	#	(3a). True Positives in inferred z.
# 			zMissed_CollectA1s 		= np.zeros(numInferenceSamples).astype(int)	#	(3b). False Negatives in inferred z.
# 			zExtra_CollectA1s 		= np.zeros(numInferenceSamples).astype(int)	#	(3c). False Positives in inferred z.
# 			yCapture_CollectA1s 	= np.zeros(numInferenceSamples).astype(int)	#	(4a). True Positives in y from inferred z.
# 			yMissed_CollectA1s 		= np.zeros(numInferenceSamples).astype(int)	#	(4b). False Negatives in y from inferred z.
# 			yExtra_CollectA1s		= np.zeros(numInferenceSamples).astype(int)	#	(4c). False Positives in y from inferred z.
# 			pJoint_Diff_CollectA1s	= np.zeros(numInferenceSamples) 			# 	(5). Difference between joint calculated from inferred z vs. true z.
# 			inferCA_ConfusionA1s	= np.zeros( (M+1,M+1) )
# 			zInferSampledA1s 		= np.zeros(M+1)
# 			zInferSampledA1s[M] 	= numInferenceSamples # Keeping track of the total number of inference samples.


# 	if flg_do_1stOrdr_inference:
# 		# Collecting performance results for 1st order inference approximation.
# 		Kinf_Collect_Ordr1			= np.zeros(numInferenceSamples).astype(int)	#	(2). How many active cell assemblies in inferred z.
# 		KinfDiff_Collect_Ordr1		= np.zeros(numInferenceSamples).astype(int)	#	(2). How much larger is inferred K than true K.
# 		zCapture_Collect_Ordr1 		= np.zeros(numInferenceSamples).astype(int)	#	(3a). True Positives in inferred z.
# 		zMissed_Collect_Ordr1 		= np.zeros(numInferenceSamples).astype(int)	#	(3b). False Negatives in inferred z.
# 		zExtra_Collect_Ordr1 		= np.zeros(numInferenceSamples).astype(int)	#	(3c). False Positives in inferred z.
# 		yCapture_Collect_Ordr1 		= np.zeros(numInferenceSamples).astype(int)	#	(4a). True Positives in y from inferred z.
# 		yMissed_Collect_Ordr1 		= np.zeros(numInferenceSamples).astype(int)	#	(4b). False Negatives in y from inferred z.
# 		yExtra_Collect_Ordr1		= np.zeros(numInferenceSamples).astype(int)	#	(4c). False Positives in y from inferred z.
# 		pJoint_Diff_Collect_Ordr1	= np.zeros(numInferenceSamples) 			# 	(5). Difference between joint calculated from inferred z vs. true z.
# 		inferCA_Confusion_Ordr1		= np.zeros( (M+1,M+1) )
# 		zInferSampled_Ordr1 		= np.zeros(M+1)
# 		zInferSampled_Ordr1[M] 		= numInferenceSamples # Keeping track of the total number of inference samples.
# 		inferCell_Confusion_Ordr1	= np.zeros( (N+1,N+1) )
# 		yInferSampled_Ordr1 		= np.zeros(N+1)
# 		yInferSampled_Ordr1[N] 		= numInferenceSamples # Keeping track of the total number of inference samples.

# 	if flg_do_2ndOrdr_inference:
# 		# Collecting performance results for 2nd order inference approximation.
# 		Kinf_Collect_Ordr2			= np.zeros(numInferenceSamples).astype(int)	#	(2). How many active cell assemblies in inferred z.
# 		KinfDiff_Collect_Ordr2		= np.zeros(numInferenceSamples).astype(int)	#	(2). How much larger is inferred K than true K.
# 		zCapture_Collect_Ordr2 		= np.zeros(numInferenceSamples).astype(int)	#	(3a). True Positives in inferred z.
# 		zMissed_Collect_Ordr2 		= np.zeros(numInferenceSamples).astype(int)	#	(3b). False Negatives in inferred z.
# 		zExtra_Collect_Ordr2 		= np.zeros(numInferenceSamples).astype(int)	#	(3c). False Positives in inferred z.
# 		yCapture_Collect_Ordr2 		= np.zeros(numInferenceSamples).astype(int)	#	(4a). True Positives in y from inferred z.
# 		yMissed_Collect_Ordr2 		= np.zeros(numInferenceSamples).astype(int)	#	(4b). False Negatives in y from inferred z.
# 		yExtra_Collect_Ordr2		= np.zeros(numInferenceSamples).astype(int)	#	(4c). False Positives in y from inferred z.
# 		pJoint_Diff_Collect_Ordr2	= np.zeros(numInferenceSamples) 			# 	(5). Difference between joint calculated from inferred z vs. true z.
# 		inferCA_Confusion_Ordr2		= np.zeros( (M+1,M+1) )
# 		zInferSampled_Ordr2 		= np.zeros(M+1)
# 		zInferSampled_Ordr2[M] 		= numInferenceSamples # Keeping track of the total number of inference samples.	





# 	#
# 	for sample in range(numInferenceSamples):

# 		if verbose_inference:
# 			print('')
# 			print('~ ~ ~ ~ ><((((> ~ ~ ~ ~ ><((((> ~ ~ ~ ~ ><((((> ~ ~ ~ ~ ><((((> ~ ~ ~ ~')
# 		print('Sample #',sample)
# 		#
# 		y = set()
# 		while len(y)==0:
# 			whichSW = np.int( np.random.uniform(num_SWs) )
# 			y = Y_list[whichSW] 	 # actual cells involved in spikeword.
# 		z = Z_list[whichSW]	 		 # actual cell assemblies involved in spikeword.
# 		zInferSampledT[list(z)]+=1 
# 		Y_true = rc.set2boolVec(y,N) # convert set of active cells to boolean vector
# 		Z_true = rc.set2boolVec(z,M) # convert sets of active cell assemblies to boolean vector
# 		#
# 		# Compute joint probability of True solution with Full and Approximate LMAP.
# 		pjt, Q_part, Pi_part, Pia_part, mix_part, mix_approx, x = rc.LMAP(q_init, ri_init, ria_init, Y_true, Z_true, verbose=False)
# 		pJoint_true = pjt.sum()	
# 		pJoint_trueApprox = (pjt - mix_part).sum()
# 		#
# 		zRec = list(z)
# 		yBin = set( np.where( (Pia[:,zRec]<0.5).sum(axis=1) )[0] )
# 		numCellsDroppedInYobs[sample] 	= len(np.setdiff1d(list(yBin),list(y)))
# 		numCellsAddedInYobs[sample] 	= len(np.setdiff1d(list(y),list(yBin)))
# 		numCellsCorrectInYobs[sample] 	= len(np.intersect1d(list(y),list(yBin)))
# 		numCellsTotalInYobs[sample] 	= len(y)

# 		if verbose_inference:
# 			print(' True z : ',z,' ::: Observed y : ',y,' ::: pJoint True :', pJoint_true) #,' ::: pJoint True Approx:', pJoint_trueApprox)
# 			print('"Noiseless" y (from Binarized active za''s) = ',yBin,' ::: # Correct Cells: ',numCellsCorrectInYobs[sample], '  ::  # Dropped Participating Cells: ',numCellsDroppedInYobs[sample], ' :: # Added Nonparticiating Cells: ',numCellsAddedInYobs[sample] )
# 			print('  ')
# 			print(' --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ')
# 			#
# 			print(' --- Explaining Away Active Cells By Adding Each Za --- ')
# 			cellsInZtrue = []
# 			setLens = []
			
# 			for i in zRec:
# 				yInZ = set( np.where(1-Pia[:,i]>0.5)[0] )
# 				cellsInZtrue.append( yInZ )
# 				setLens.append( len(yInZ) )
# 			#
# 			cellsAlready = set()
# 			cellsNowTot = set()
# 			lenPrev = len(cellsAlready)
# 			inds = np.argsort(setLens)[::-1]
# 			for i in inds:
# 				cellsNowTot.update(cellsInZtrue[i])
# 				lenNow = len(cellsNowTot)
# 				#print('cellsNowTot: ',cellsNowTot)
# 				#print('cellsAlready: ',cellsAlready)
# 				print('Added z',zRec[i],' includes cells ',cellsInZtrue[i],'  ::  Cells Already ',cellsAlready,'  ::  # New Cells ',lenNow-lenPrev)  #,' :: # Dropped? Cells',len(np.setdiff1d(list(cellsNowTot),list(y))), '  ::  # Unexplained Cells ', len(np.setdiff1d(list(y),list(cellsNowTot))) )
# 				cellsAlready.update(cellsInZtrue[i])
# 				lenPrev = len(cellsAlready)
# 			print('  ')
# 			print(' --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ')
			


# 		# Quantify 'noisyness' of spike word generation by comparing y (observed spike word) with yBin (derived from thresholded Pia and active za's)
# 		if True:
# 			zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, yUnion, yIntersect, yDiff, ySubed, yAdded, yCapture, yMissed, yExtra, Kinf, KinfDiff = rc.compute_inference_statistics(z, z, M, y, yBin, N, verbose=False)
# 			#
# 			Kinf_binVobs[sample] 		= Kinf
# 			yCapture_binVobs[sample] 	= yCapture
# 			yMissed_binVobs[sample] 	= yMissed
# 			yExtra_binVobs[sample] 		= yExtra
# 			#
# 			# Compute confusion matrix between inferred y's (yHyp) and observed y's (y). (see below for details ...)
# 			grid = np.ix_(yAdded,ySubed)
# 			inferCell_Confusion_binVobs[grid]+=1							# add to off-diagonal (i,j) for mix-ups
# 			inferCell_Confusion_binVobs[( yIntersect,yIntersect )]+=1 	# add to diagonal (i,i) for correct inference
# 			yInferSampled_binVobs[list(yBin)]+=1 						# keeping track of how many times each cell assembly was active in an inference sample.
# 			#		





# 		if flg_do_1shot_inference:
# 			# One-shot inference and combinatorial cleanup.  (ie. looking at ordered 1stOrdr pjoint for each za and then
# 			# computing pjoint for combinations of )
# 			if verbose_inference:
# 				print('Full one-shot Inference:')
# 			#
# 			# (1). One-shot Inference: Compute 1stOrdr joint probability of each 1-hot z-vectors and the z=0's vector.
# 			CA_sets, pj_vals = rc.MAP_inferZ_oneshot(y, N, M, q_init, ri_init, ria_init, approx=False, verbose=False)
# 			#
# 			# (2). Combinatorial Cleanup: Compute joint probability for z-vectors (2-hot, 3-hot) that are combinations of za's
# 			#      with individual pjoint >= that of the all 0's z-vector.
# 			print('CA_sets: ',CA_sets)
# 			print('pj_vals: ',pj_vals)

# 			zHyp_1s, yHyp_1s = rc.MAP_inferZ_Comb(y, N, M, q_init, ri_init, ria_init, CA_sets, pj_vals, approx=False, verbose=False)
# 			#
# 			Y_hyp1s = rc.set2boolVec(yHyp_1s,N)
# 			Z_hyp1s = rc.set2boolVec(zHyp_1s,M) 
# 			pjt, Q_part, Pi_part, Pia_part, mix_part, mix_approx, x = rc.LMAP(q_init, ri_init, ria_init, Y_true, Z_hyp1s) # , pJ_approx_err, _
# 			pJoint_infrd1s = (pjt).sum()
# 			#
# 			# Compute statistics on inferred Cell / Cell Assembly activity comparing them to true y and z.  
# 			# NOTE: Will have to adapt this when true z-vector is not known.
# 			zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, yUnion, yIntersect, yDiff, ySubed, yAdded, yCapture, yMissed, yExtra, Kinf, KinfDiff = rc.compute_inference_statistics(z, zHyp_1s, M, y, yHyp_1s, N, verbose=verbose_inference)
# 			#
# 			Kinf_Collect1s[sample] 		= Kinf
# 			KinfDiff_Collect1s[sample] 	= KinfDiff
# 			zCapture_Collect1s[sample] 	= zCapture
# 			zMissed_Collect1s[sample]	= zMissed 
# 			zExtra_Collect1s[sample] 	= zExtra
# 			yCapture_Collect1s[sample] 	= yCapture
# 			yMissed_Collect1s[sample] 	= yMissed
# 			yExtra_Collect1s[sample] 	= yExtra
# 			pJoint_Diff_Collect1s[sample] = pJoint_infrd1s - pJoint_true
# 			if verbose_inference:
# 				print('pjoint one-shot = ',pJoint_infrd1s,' & Difference = ',pJoint_Diff_Collect1s[sample])
# 			#
# 			# Compute confusion matrix between inferred z's (zHyp) and true z's (z). (see below for details ...)
# 			grid = np.ix_(zAdded,zSubed)
# 			inferCA_Confusion1s[grid]+=1						# add to off-diagonal (i,j) for mix-ups
# 			inferCA_Confusion1s[( zIntersect,zIntersect )]+=1 	# add to diagonal (i,i) for correct inference
# 			zInferSampledE1s[list(zHyp_1s)]+=1 					# keeping track of how many times each cell assembly was active in an inference sample.
# 			#



# 			# [B]. For Approximate One-Shot Inference
# 			if flg_do_approx_inference:
# 				CA_setsA1s, pj_valsA1s = rc.MAP_inferZ_oneshot(y, N, M, q_init, ri_init, ria_init, approx=True, verbose=False)
# 				zHyp_A1s, yHyp_A1s = rc.MAP_inferZ_Comb(y, N, M, q_init, ri_init, ria_init, CA_setsA1s, pj_valsA1s, approx=True, verbose=False)

# 				Y_hypA1s = rc.set2boolVec(yHyp_A1s,N)
# 				Z_hypA1s = rc.set2boolVec(zHyp_A1s,M) 
# 				pjt, Q_part, Pi_part, Pia_part, mix_part, mix_approx, x = rc.LMAP(q_init, ri_init, ria_init, Y_true, Z_hypA1s) # , pJ_approx_err, _
# 				pJoint_approx = (pjt - mix_part).sum()
# 				#
# 				# Compute statistics on inferred Cell / Cell Assembly activity comparing them to true y and z.  
# 				# NOTE: Will have to adapt this when true z-vector is not known.
# 				zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, yUnion, yIntersect, yDiff, ySubed, yAdded, yCapture, yMissed, yExtra, Kinf, KinfDiff = rc.compute_inference_statistics(z, zHyp_A1s, M, y, yHyp_A1s, N, verbose=verbose_inference)
# 				#
# 				Kinf_CollectA1s[sample] 		= Kinf
# 				KinfDiff_CollectA1s[sample] 	= KinfDiff
# 				zCapture_CollectA1s[sample] 	= zCapture
# 				zMissed_CollectA1s[sample]		= zMissed 
# 				zExtra_CollectA1s[sample] 		= zExtra
# 				yCapture_CollectA1s[sample] 	= yCapture
# 				yMissed_CollectA1s[sample] 		= yMissed
# 				yExtra_CollectA1s[sample] 		= yExtra
# 				pJoint_Diff_CollectA1s[sample] 	= pJoint_approx - pJoint_trueApprox
# 				#
# 				# Compute confusion matrix between inferred z's (zHyp) and true z's (z). (see below for details ...)
# 				grid = np.ix_(zAdded,zSubed)
# 				inferCA_ConfusionA1s[grid]+=1						# add to off-diagonal (i,j) for mix-ups
# 				inferCA_ConfusionA1s[( zIntersect,zIntersect )]+=1 	# add to diagonal (i,i) for correct inference
# 				zInferSampledA1s[list(zHyp_A1s)]+=1 				# keeping track of how many times each cell assembly was active in an inference sample.
# 				#




# 		if flg_do_greedy_inference:
# 			# Do Inference using the greedy inference algorithm with full Joint calculation. 
# 			if verbose_inference:
# 				print(' - - - - - - - - - - -')
# 				print('Full Greedy Inference:')

# 			zHyp, yHyp = rc.MAP_inferZ_greedy(y, N, z, M, q_init, ri_init, ria_init, approx=False, verbose=True)
# 			Y_hyp = rc.set2boolVec(yHyp,N)
# 			Z_hyp = rc.set2boolVec(zHyp,M)
# 			pjt, Q_part, Pi_part, Pia_part, mix_part, mix_approx, x = rc.LMAP(q_init, ri_init, ria_init, Y_true, Z_hyp, verbose=False)
# 			pJoint_infrd = pjt.sum()

# 			#
# 			# Compute statistics on inferred Cell / Cell Assembly activity comparing them to true y and z.  
# 			# NOTE: Will have to adapt this when true z-vector is not known.
# 			zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, yUnion, yIntersect, yDiff, ySubed, yAdded, yCapture, yMissed, yExtra, Kinf, KinfDiff = rc.compute_inference_statistics(z, zHyp, M, y, yHyp, N, verbose=verbose_inference)
# 			#
# 			Kinf_Collect[sample] 		= Kinf
# 			KinfDiff_Collect[sample]	= KinfDiff
# 			zCapture_Collect[sample] 	= zCapture
# 			zMissed_Collect[sample] 	= zMissed 
# 			zExtra_Collect[sample] 		= zExtra
# 			yCapture_Collect[sample] 	= yCapture
# 			yMissed_Collect[sample] 	= yMissed
# 			yExtra_Collect[sample] 		= yExtra
# 			pJoint_Diff_Collect[sample] = pJoint_infrd - pJoint_true
# 			if verbose_inference:
# 				print('pjoint greedy = ',pJoint_infrd,' & Difference = ',pJoint_Diff_Collect[sample])



# 			# Compute confusion matrix between inferred z's (zHyp) and true z's (z).
# 			# This will be an asymmetric matrix with:
# 			#	(1). Entry i,j indicates that assembly i is not found and assembly j is found instead.
# 			# 	(2). Entries on the diagonal i,i indicate correctly inferred CAs.
# 			#	(3). Entries in i,M indicate
# 			#	(4). Entries in M,j indicate
# 			#
# 			grid = np.ix_(zAdded,zSubed)
# 			inferCA_Confusion[grid]+=1						# add to off-diagonal (i,j) for mix-ups
# 			inferCA_Confusion[( zIntersect,zIntersect )]+=1 # add to diagonal (i,i) for correct inference
# 			zInferSampledE[list(zHyp)]+=1 						# keeping track of how many times each cell assembly was active in an inference sample.
# 			#




# 			if flg_do_approx_inference:
# 				# Do Inference using the "linear"? approximation.  That is, assuming that log(x-1) = log(x), which is valid for large x.
# 				if verbose_inference:
# 					print('Approximate Greedy Inference')
# 				#
# 				zHypA, yHypA = rc.MAP_inferZ_greedy(y, N, M, q_init, ri_init, ria_init, approx=True, verbose=False) #verbose_inference)
# 				Y_hypA = rc.set2boolVec(yHypA,N)
# 				Z_hypA = rc.set2boolVec(zHypA,M) 
# 				pjt, Q_part, Pi_part, Pia_part, mix_part, mix_approx, x = rc.LMAP(q_init, ri_init, ria_init, Y_true, Z_hypA) # , pJ_approx_err, _
# 				pJoint_approx = (pjt - mix_part).sum()
# 				#
# 				# Compute statistics on inferred Cell / Cell Assembly activity comparing them to true y and z.  
# 				# NOTE: Will have to adapt this when true z-vector is not known.
# 				zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, yUnion, yIntersect, yDiff, ySubed, yAdded, yCapture, yMissed, yExtra, Kinf, KinfDiff = rc.compute_inference_statistics(z, zHypA, M, y, yHypA, N, verbose=verbose_inference)
# 				#
# 				Kinf_CollectA[sample] 		= Kinf
# 				KinfDiff_CollectA[sample] 	= KinfDiff
# 				zCapture_CollectA[sample] 	= zCapture
# 				zMissed_CollectA[sample]	= zMissed 
# 				zExtra_CollectA[sample] 	= zExtra
# 				yCapture_CollectA[sample] 	= yCapture
# 				yMissed_CollectA[sample] 	= yMissed
# 				yExtra_CollectA[sample] 	= yExtra
# 				pJoint_Diff_CollectA[sample]= pJoint_approx - pJoint_trueApprox
# 				if verbose_inference:
# 					print('pjoint approx = ',pJoint_approx,' & Difference = ',pJoint_Diff_CollectA[sample]) # ' & "Error" = ',pJ_approx_err,
# 				#
# 				# Compute confusion matrix between inferred z's (zHyp) and true z's (z). (see below for details ...)
# 				grid = np.ix_(zAdded,zSubed)
# 				inferCA_ConfusionA[grid]+=1							# add to off-diagonal (i,j) for mix-ups
# 				inferCA_ConfusionA[( zIntersect,zIntersect )]+=1 	# add to diagonal (i,i) for correct inference
# 				zInferSampledA[list(zHypA)]+=1 						# keeping track of how many times each cell assembly was active in an inference sample.
# 				#

		
# 		if flg_do_1stOrdr_inference:
# 			# Do Inference using newest Fritz approximation.  See 20180803_144422.jpg for writeup, or comment better if it actually works.
# 			if verbose_inference:
# 				print('1st Order Inference Approximation')


# 			# # Doing new 1stOrdr inference here.
# 			pJoint_O1, Alpha_O1, Beta_O1, CinZ_O1 = rc.LMAP_1stOrdr_approx(q_init, ri_init, ria_init, Y_true, N, M, verbose=False)
# 			#
# 			#alpha_indep_vals = np.unique(Alpha)[::-1] 
# 			pJoint_O1 = pJoint_O1.round(decimals=pjt_tol)
# 			pj_O1_vals = np.unique(pJoint_O1)[::-1] 
# 			CA_O1_sets = list()
# 			pjFull_O1_vals = np.zeros_like(pj_O1_vals)


# 			#Compute full pJoint for each 1-hot z-vector - to use it in the MAP_inferZ_Comb function below.
# 			pJoint_1hot = np.zeros(M)
# 			Y = rc.set2boolVec(y,N)
# 			zHyp = set()
# 			for a in range(M):
# 				zHyp.add(a)	
# 				Z = rc.set2boolVec(zHyp,M)
# 				pjt, Q_part, Pi_part, Pia_part, mix_part, mix_approx, x = rc.LMAP(q_init, ri_init, ria_init, Y, Z, verbose=False)
# 				pJoint_1hot[a] = pjt
# 				#print('z = ',zHyp,' :: pjt = ',pjt)
# 				zHyp.remove(a)

# 			# Add the z=0 vector to the end too as a possibility.
# 			zHyp = set()
# 			Z = rc.set2boolVec(zHyp,M)
# 			pjt, Q_part, Pi_part, Pia_part, mix_part, mix_approx, x = rc.LMAP(q_init, ri_init, ria_init, Y, Z, verbose=False)
# 			pJoint_1hot[M] = pjt




# 			pJoint_1hot = pJoint_1hot.round(decimals=pjt_tol)	

# 			#
# 			for i in range(pj_O1_vals.size):
# 				CA_O1_sets.append( set(  np.where( pJoint_O1 == pj_O1_vals[i])[0] ) ) #indPJ[]
# 				#print(CA_O1_sets[i],'  ::  ',pj_O1_vals[i],'  ::  ',pJoint_1hot[list(CA_O1_sets[i])])
# 				pjFull_O1_vals[i] = pJoint_1hot[list(CA_O1_sets[i])].mean()

# 			#
# 			if verbose_inference:
# 				#print('Z true = ',z,'  :::  Y true = ',y)
# 				print('Cell Assemblies Ordered by 1st-order approximate pJoint')
# 				Pia = rc.sig(ria_init)
# 				for i in range(pj_O1_vals.size):
# 					zList1 = list(CA_O1_sets[i])
# 					for j in range( len(zList1) ):
# 						yHyp1 = set( np.where( np.round(1-Pia[:,zList1[j]]) )[0] )
# 						intersect = len( set.intersection( y, yHyp1 ) )
# 						zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, yUnion, yIntersect, yDiff, ySubed, yAdded, yCapture, yMissed, yExtra, Kinf, KinfDiff = rc.compute_inference_statistics(z, set([zList1[j]]), M, y, yHyp1, N, verbose=False)
# 						print('pjO1 = ',pj_O1_vals[i],'   :   pjFull = ',pjFull_O1_vals[i],'   :   z = ',zList1[j],' ---> y = ',yHyp1,' ( Cap=',yCapture,', Extra=', yExtra,', Miss=',yMissed,' )')

# 			#
# 			numPosPjts = (pj_O1_vals>0).sum() # Hard coded in comb function.
# 			zHyp_O1, yHyp_O1 = rc.MAP_inferZ_Comb(y, N, M, q_init, ri_init, ria_init, CA_O1_sets, pjFull_O1_vals, numPosPjts, approx=False, verbose=False)
			
# 			if verbose_inference:
# 				print('After 1st-order Combination, zHyp_O1=',zHyp_O1,' and yHyp_O1=',yHyp_O1)



# 			#
# 			# Compute statistics on inferred Cell / Cell Assembly activity comparing them to true y and z.  
# 			# NOTE: Will have to adapt this when true z-vector is not known.
# 			zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, yUnion, yIntersect, yDiff, ySubed, yAdded, yCapture, yMissed, yExtra, Kinf, KinfDiff = rc.compute_inference_statistics(z, zHyp_O1, M, y, yHyp_O1, N, verbose=verbose_inference)
# 			#
# 			Kinf_Collect_Ordr1[sample] 		= Kinf
# 			KinfDiff_Collect_Ordr1[sample]	= KinfDiff
# 			zCapture_Collect_Ordr1[sample] 	= zCapture
# 			zMissed_Collect_Ordr1[sample] 	= zMissed 
# 			zExtra_Collect_Ordr1[sample] 	= zExtra
# 			yCapture_Collect_Ordr1[sample] 	= yCapture
# 			yMissed_Collect_Ordr1[sample] 	= yMissed
# 			yExtra_Collect_Ordr1[sample] 	= yExtra
# 			#
# 			# Compute confusion matrix between inferred z's (zHyp) and true z's (z). (see below for details ...)
# 			grid = np.ix_(zAdded,zSubed)
# 			inferCA_Confusion_Ordr1[grid]+=1							# add to off-diagonal (i,j) for mix-ups
# 			inferCA_Confusion_Ordr1[( zIntersect,zIntersect )]+=1 		# add to diagonal (i,i) for correct inference
# 			zInferSampled_Ordr1[list(zHyp_O1)]+=1 						# keeping track of how many times each cell assembly was active in an inference sample.
# 			#


# 			# Compute confusion matrix between inferred y's (yHyp) and observed y's (y). (see below for details ...)
# 			grid = np.ix_(yAdded,ySubed)
# 			inferCell_Confusion_Ordr1[grid]+=1							# add to off-diagonal (i,j) for mix-ups
# 			inferCell_Confusion_Ordr1[( yIntersect,yIntersect )]+=1 	# add to diagonal (i,i) for correct inference
# 			yInferSampled_Ordr1[list(yHyp_O1)]+=1 						# keeping track of how many times each cell assembly was active in an inference sample.
# 			#









# 		if flg_do_2ndOrdr_inference:
# 			# Do Inference using newest Fritz approximation.  See 20180803_144422.jpg for writeup, or comment better if it actually works.
# 			if verbose_inference:
# 				print('2nd Order Inference Approximation')


# 			# # Doing new 2nd Order inference here.
# 			pJoint2, Alpha2, Beta2, Gamma2, CinZ2 = rc.LMAP_2ndOrdr_approx(q_init, ri_init, ria_init, Y_true, N, M, verbose=False)


# 			if flg_do_Hopfield_inference:
# 				print('Doing Hopfield Relaxation Step.')
# 				Theta2 = Alpha2 + Beta2

# 				Z_hop = np.random.binomial(1,0.5,M)
# 				#Z_hop = np.zeros(M).astype(int)

# 				for i in range(100):
# 					flip = np.random.randint(0,M) 					# this is za
# 					#za = (1-2*Z_hop[flip])
# 					zap = np.where(Z_hop)[0]						# this is za'

					
											
					
# 					if ( (Gamma2[flip,zap]).sum() - Theta2[flip] ) > 0:
# 						Z_hop[flip] = (Z_hop[flip]+1)%2
# 					print(i,'   ',flip,'   ',Gamma2[flip,:].sum() - Theta2[flip],'   ',Z_hop)


# 			if True:	
# 				print(' . . . . .  without pairwise interaction terms - Gamma(z1,z2)')
# 				pJoint2_1 = Alpha2+Beta2+CinZ2

# 				pj_2_vals = np.unique(pJoint2_1)[::-1] 
# 				CA_2_sets = list()
# 				#
# 				for i in range(pj_2_vals.size):
# 					CA_2_sets.append( set(  np.where(pJoint2_1==pj_2_vals[i])[0] ) ) #indPJ[]
# 				#
# 				if verbose_inference:
# 					#print('Z true = ',z,'  :::  Y true = ',y)
# 					print('Cell Assemblies Ordered by 2nd-order approximate pJoint without Gamma term.')
# 					for i in range(pj_2_vals.size):
# 						zList2 = list(CA_2_sets[i])
# 						for j in range( len(zList2) ):
# 							yHyp2 = set( np.where( np.round(1-Pia[:,zList2[j]]) )[0] )
# 							intersect = len( set.intersection( y, yHyp2 ) )
# 							zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, yUnion, yIntersect, yDiff, ySubed, yAdded, yCapture, yMissed, yExtra, Kinf, KinfDiff = rc.compute_inference_statistics(z, set([zList2[j]]), M, y, yHyp2, N, verbose=False)
# 							print('pj = ',pj_2_vals[i],'   :   z = ',zList2[j],' ---> y = ',yHyp2,' ( Cap=',yCapture,', Extra=', yExtra,', Miss=',yMissed,' )')
# 				#
# 					# print(' . . . . .  considering pairwise interaction terms - Gamma(z1,z2)')
# 					# for i in z:
# 					# 	xx = np.argsort(pJoint2[i,:])[::-1]
# 					# 	#print('za = ',i)
# 					# 	#print('zb = ',xx[0])
# 					# 	print(' ------------------------------ ')
# 					# 	print('(1). p(za=',i,',zb=',xx[0],') = ',pJoint2[i,xx[0]])
# 					# 	print('(2). p(za=',i,',zb=',xx[1],') = ',pJoint2[i,xx[1]])
# 					# print('There are ',np.round( np.where([pJoint2>pJoint2_1.max()])[0].size/2 ), ' pairs of z''s with larger pJoint')	


# 				if False:
# 					plt.imshow( pJoint2*(pJoint2>0) )
# 					#plt.imshow( Gamma2*(Gamma2>0) )
# 					plt.colorbar()
# 					plt.title( str('True z = ' + str(z) + ' : True y = ' + str(y) ) )
# 					plt.xlabel('z1')
# 					plt.ylabel('z2')
# 					plt.show()

# 					#
# 			# Compute statistics on inferred Cell / Cell Assembly activity comparing them to true y and z.  
# 			# NOTE: Will have to adapt this when true z-vector is not known.
# 			zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, yUnion, yIntersect, yDiff, ySubed, yAdded, yCapture, yMissed, yExtra, Kinf, KinfDiff = rc.compute_inference_statistics(z, zHyp, M, y, yHyp, N, verbose=verbose_inference)
# 			#
# 			Kinf_Collect_Ordr2[sample] 		= Kinf
# 			KinfDiff_Collect_Ordr2[sample]	= KinfDiff
# 			zCapture_Collect_Ordr2[sample] 	= zCapture
# 			zMissed_Collect_Ordr2[sample] 	= zMissed 
# 			zExtra_Collect_Ordr2[sample] 	= zExtra
# 			yCapture_Collect_Ordr2[sample] 	= yCapture
# 			yMissed_Collect_Ordr2[sample] 	= yMissed
# 			yExtra_Collect_Ordr2[sample] 	= yExtra
# 			pJoint_Diff_Collect_Ordr2[sample] = pJoint_infrd - pJoint_true
# 					#
# 			# Compute confusion matrix between inferred z's (zHyp) and true z's (z). (see below for details ...)
# 			grid = np.ix_(zAdded,zSubed)
# 			inferCA_Confusion_Ordr2[grid]+=1							# add to off-diagonal (i,j) for mix-ups
# 			inferCA_Confusion_Ordr2[( zIntersect,zIntersect )]+=1 	# add to diagonal (i,i) for correct inference
# 			zInferSampled_Ordr2[list(zHyp2)]+=1 						# keeping track of how many times each cell assembly was active in an inference sample.
# 			#		





# 	# # NEW COMMENT HERE:  WHAT AM I DOING BELOW IN THIS PLOTTING BELOW ??
# 	# Compute similarity between all cell assemblies by computing their pairwise dot products / angles.
# 	# This may explain some of the confusion that happens with inference.
# 	if flg_plt_infer_performance_stats:
# 		#
# 		if flg_do_greedy_inference:
# 			pf.plot_CA_inference_performance(inferCA_Confusion, inferCell_Confusion_Ordr1, CA_ovl, CA_coactivity, zInferSampledE, zInferSampledT, Zassem_hist, yInferSampled_Ordr1, Kinf_Collect, KinfDiff_Collect_Ordr1, num_SWs, params_init, params_init_param, infer_figs_dir, approx=False, inferType='greedy')
# 			if flg_do_approx_inference:
# 				pf.plot_CA_inference_performance(inferCA_ConfusionA, inferCell_Confusion_Ordr1, CA_ovl, CA_coactivity, zInferSampledA, zInferSampledT, Zassem_hist, yInferSampled_Ordr1, Kinf_CollectA, KinfDiff_Collect_Ordr1, num_SWs, params_init, params_init_param, infer_figs_dir, approx=True, inferType='greedy')
# 		#
# 		if flg_do_1shot_inference:	
# 			pf.plot_CA_inference_performance(inferCA_Confusion1s, inferCell_Confusion_Ordr1, CA_ovl, CA_coactivity, zInferSampledE1s, zInferSampledT, Zassem_hist, yInferSampled_Ordr1, Kinf_Collect1s, KinfDiff_Collect_Ordr1, num_SWs, params_init, params_init_param, infer_figs_dir, approx=False, inferType='1shot')
# 			if flg_do_approx_inference:
# 				pf.plot_CA_inference_performance(inferCA_ConfusionA1s, inferCell_Confusion_Ordr1, CA_ovl, CA_coactivity, zInferSampledA1s, zInferSampledT, Zassem_hist, yInferSampled_Ordr1, Kinf_CollectA1s, KinfDiff_Collect_Ordr1, num_SWs, params_init, params_init_param, infer_figs_dir, approx=True, inferType='1shot')
# 		#
# 		if flg_do_1stOrdr_inference:
# 			pf.plot_CA_inference_performance(inferCA_Confusion_Ordr1, inferCell_Confusion_Ordr1, CA_ovl, CA_coactivity, zInferSampled_Ordr1, zInferSampledT, Zassem_hist, yInferSampled_Ordr1, Kinf_Collect_Ordr1, KinfDiff_Collect_Ordr1, num_SWs, params_init, params_init_param, infer_figs_dir, approx=False, inferType='1stOrdr')
# 		#	
# 		if flg_do_2ndOrdr_inference:	
# 			pf.plot_CA_inference_performance(inferCA_Confusion_Ordr2, inferCell_Confusion_Ordr1, CA_ovl, CA_coactivity, zInferSampled_Ordr2, zInferSampledT, Zassem_hist, yInferSampled_Ordr1, Kinf_Collect_Ordr2, KinfDiff_Collect_Ordr1, num_SWs, params_init, params_init_param, infer_figs_dir, approx=False, inferType='2ndOrdr')
			









# 	# Plot #Cells correct vs. dropped vs. added in observed y vector vs in Noisy Interpretation of thresholded Pia for active za's
# 	if flg_plot_compare_YinferVsTrueVsObs:
# 		pf.plot_compare_YinferVsTrueVsObs(numCellsCorrectInYobs, numCellsAddedInYobs, numCellsDroppedInYobs, numCellsTotalInYobs, yCapture_Collect_Ordr1, yExtra_Collect_Ordr1, yMissed_Collect_Ordr1, yCapture_binVobs, yExtra_binVobs, yMissed_binVobs, numInferenceSamples, Q, bernoulli_Pi, mu_Pi, sig_Pi, mu_Pia, sig_Pia, params_init, params_init_param, infer_figs_dir, inferType='1stOrdr')

# 		# # TO DO - MAYBE !!! THESE MAY BE USEFUL TO CONSIDER / VISUALIZE TOO AT SOME POINT !!! 
# 		# Kinf_binVobs
# 		# inferCell_Confusion_binVobs
# 		# yInferSampled_binVobs
# 		# #
# 		# xx = np.matmul(Pia,Pia.T) # Cell correlation matrix (in terms of number of cell assemblies they are both active in?)
# 		# f, ax = plt.subplots(1,2)
# 		# ax[0].imshow(xx)
# 		# #
# 		# ax[1].imshow(inferCell_Confusion_binVobs)	
# 		# #plt.colorbar()
# 		# plt.show()







# 		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 		# (5).LEARN Pia AND Pi FROM Y & Z DATA - HAND CODING GRADIENTS MYSELF. 
# 		#
# 		# {Learning Experiments}: This code is a bit old and will have to be updated to be used (some functions have been modified).
# 		# Current learning code is in the full EM process above.
# 		#
# 		# --------------------------------------------
# 		#
# 		if flg_do_just_learning:
# 			print('Learning Model Parameters (q, ri & ria) from data: ',num_SWs,' Spike Words.')

# 			# Preallocate to error measures thru learning algorithm.
# 			pjoint  	= np.zeros( (numLearningSamples,5) )
# 			smp 		= np.zeros(numLearningSamples).astype(int) # which spike word gets sampled (should be uniform)
# 			#
# 			q_MSE   	= np.zeros(numLearningSamples) 
# 			ri_MSE 		= np.zeros( (numLearningSamples,3) ) # 3 allows room for mean, std, max of squared errors.
# 			ria_MSE 	= np.zeros( (numLearningSamples,3) )
# 			#
# 			q_deriv		= np.zeros(numLearningSamples) 
# 			ri_deriv 	= np.zeros( (numLearningSamples,7) ) # 7 allows room for mean, std, max, etc of derivatives.
# 			ria_deriv 	= np.zeros( (numLearningSamples,7) )

# 			for t in range(numLearningSamples):

# 				print('Iteration: ',t) 

# 				smp[t] = int(np.random.uniform(low=0, high=num_SWs-1))

# 				y = rc.set2boolVec(Y_list[smp[t]],N)
# 				z = rc.set2boolVec(Z_list[smp[t]],M)

# 				qp, rip, riap, pjoint1, q_MSE1, ri_MSE1, ria_MSE1, q_deriv1, ri_deriv1, ria_deriv1 = rc.learn_model_params( qp, rip, riap, q, ri, ria, y, z, M, learning_rate, verbose=verbose_learning)

# 				# Collect up changes in parameters and learning statistics for each sample. For plots later.
# 				pjoint[t]  		= pjoint1
# 				#
# 				q_MSE[t]   		= q_MSE1 
# 				ri_MSE[t,:]		= ri_MSE1
# 				ria_MSE[t,:] 	= ria_MSE1
# 				#
# 				q_deriv[t] 		= q_deriv1
# 				ri_deriv[t,:] 	= ri_deriv1
# 				ria_deriv[t,:] 	= ria_deriv1

# 			# ALSO PLOT HOW SMP's ARE SAMPLED FROM SPIKE WORDS. histogram up smp & num_SWs (MAYBE IN A SUBPLOT OF THE ERROR PLOT)
# 			if learning_SWsamps_hist_flg:
# 				pf.hist_SWsampling4learning_stats(smp, num_SWs, numLearningSamples, ria, ri, M, N, C, Cmin, Cmax, plt_save_learn_dir)

# 			# Plot error measures as a function of Learning iteration.
# 			if plt_learning_MSE_flg:
# 				pf.plot_params_MSE_during_learning(q_MSE, ri_MSE, ria_MSE, numLearningSamples, N, M, learning_rate, params_init, params_init_param, plt_save_learn_dir)

# 			# Plot of Derivative of LMAP w.r.t. each parameter at each iteration.
# 			# Pi & Pia are: [ np.sign(dri.mean())*np.abs(dri).mean(), 	np.abs(dri).std(), 	np.abs(dri).max(),
# 			#						dri.mean(), 	dri.std(),		dri.max(), 		dri.min()	]
# 			#
# 			if plt_learning_derivs_flg:
# 				pf.plot_params_derivs_during_learning(q_deriv, ri_deriv, ria_deriv, numLearningSamples, N, M, learning_rate, params_init, params_init_param, plt_save_learn_dir)

# 			# Plot 3 versions of model parameters to visualize errors 
# 			#	(1). True params
# 			#	(2). Initialized params
# 			#	(3). Learned params
# 			if plt_learning_params_init_n_final:
# 				pf.plot_params_init_n_learned(q, ri, ria, qp, rip, riap, q_init, ri_init, ria_init, numLearningSamples, N, M, learning_rate, params_init, params_init_param, plt_save_learn_dir)