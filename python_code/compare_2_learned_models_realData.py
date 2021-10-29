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



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (0). Set up directories and create dirs if necessary.
dirHome, dirScratch = dm.set_dir_tree()
EM_learning_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Models_learned_EM/')
Infer_postLrn_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Rasters_trial_vs_time_zInferred/')
EM_learnStats_Dir	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/InferStats_from_EM_learning/')

EM_figs_dir 		= str( dirScratch + 'figs/PGM_analysis/EM_Algorithm/Greg_retinal_data/')
SW_extracted_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/SpikeWordsExtracted/')
#
# Make directories for output data if  not already there.
if not os.path.exists( SW_extracted_Dir ):
	os.makedirs( SW_extracted_Dir )


flg_make_match_models_plots = False

flg_save_meanCS_forAllCAs = True
flg_plot_meanCS_forAllCAs = False


figSaveFileType = 'png' # 'png' or 'pdf'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (1). Load in npz file and extract data from it.

# # Parameters we can loop over.
# cell_types = ['[offBriskTransient]','[onBriskTransient]','[offBriskSustained]',\
# 			'[offBriskTransient,offBriskSustained]','[offBriskTransient,onBriskTransient]']
# Ns = [55, 39, 43, 98, 94]


cell_types = ['[offBriskTransient,onBriskTransient]']#, '[offBriskTransient,offBriskSustained]','[offBriskTransient]']			
Ns = [94]#, 98, 55]
stims = ['NatMov']# 'Wnoise',


model_CA_overcompleteness = [1] #[1,2] 	# how many times more cell assemblies we have than cells (1 means complete - N=M, 2 means 2x overcomplete)
SW_bins = [2]# , 1]			# ms. Build up spikewords from groups of cells in the same trial that fire within SW_bins of eachother.

yLo_Vals 		= [0] #[1,4] 	# If |y|<=yLo, then we force the z=0 inference solution and change Pi. This defines cells assemblies to be more than 1 cell.
yHi_Vals 		= [1000] 		# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution and change Pia.
yMinSWs 		= [1]#,3]		# DOING BELOW THING WITH YYY inside pgm functions. --> (set to 0, so does nothing) 
								# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<yLo.
#
pct_xVal_train = 0.5

# Learning rates for EM algorithm
learning_rates 	= [0.5]#, 0.1] 	 	# Learning rates to loop through
lRateScaleS = [ [1., 0.1, 0.1]]#, [1., 0.1, 1.] ]  # Multiplicative scaling to Pia,Pi,Q learning rates. Can set to zero and take Pi taken out of model essentially.

num_train_rands = 3

xValSWs_or_allSWs = 'allSWs' # 'xValSWs' or 'allSWs' - whether to use inference postlearning on test data set or all spikewords.

flg_EgalitarianPriorS = [True,False] 
sample_longSWs_1stS = ['Dont', 'Prob'] 


GLMtag = '_GLMind' # '_GLMind' for simulated independent GLM or '' for real data.

# Parameter initializations for EM algorithm to learn model parameters
params_init 	= 'NoisyConst'	# Options: {'True', 'RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise) }
#
Z_hot 			= 5 			# initialization for Q value (how many 1's expected in binary z-vector)
C_noise_ri 		= 1.
C_noise_ria 	= 1.
#
sigQ_init 		= 0.01			# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigPi_init 		= 0.05			# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sigPia_init 	= 0.05			# STD on Q initialization from true values. Used for 'NoisyTrue' and 'NoisyConst' initializations.
sig_init 		= np.array([ sigQ_init, sigPi_init, sigPia_init ])
params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )



# Flags for the EM (inference & learning) algorithm.
flg_include_Zeq0_infer  = True
verbose  				= False

if flg_include_Zeq0_infer:
	z0_tag='_zeq0'
else:
	z0_tag='_zneq0'



maxSamps = np.nan #50000 # np.nan if you want to use all the SWs for samples or a scalar to use only a certain value


if not np.isnan( maxSamps ):
	maxSampTag = str( '_'+str( int(maxSamps ) )+'Samps' )
else:
	maxSampTag = '_allSamps'

maxTrialTag = '_allTrials'

minTms = 0 
maxTms = 6000

GTtag_plot_pairwise_model_CA_match = '$\Delta Init$'

# Loop through different combinations of parameters and generate plots of results.
for ct,cell_type in enumerate(cell_types):
	N = Ns[ct]
	#
	#print(cell_type, N)
	#
	for flg_EgalitarianPrior in flg_EgalitarianPriorS:
		#
		if flg_EgalitarianPrior:	
			priorType = 'EgalQ' 
		else:
			priorType = 'BinomQ'
		#	
		for sample_longSWs_1st in sample_longSWs_1stS:
			#
			for i in range(len(stims)):
				stim=stims[i]
				#
				# num_SWs=num_dataGen_Samples[i]
				#
				# for samps in num_train_Samples:
					#
				for overcomp in model_CA_overcompleteness:
					M = overcomp*N
					#
					for yLo in yLo_Vals:
						#
						for yHi in yHi_Vals:
							#
							for yMinSW in yMinSWs:
								#
								for learning_rate in learning_rates:
									#
									for lRateScale in lRateScaleS:
										#
										for SW_bin in SW_bins:
											msBins = 1+2*SW_bin
											#
											#
											data 	= list() # To collect up model data and filenames over 
											rast 	= list()
											fnames 	= list() # different rands and A,B model combinations.
											fparams = list() 
											fnameR 	= list() 

											#	
											for rand in range(num_train_rands):
						
												# try: # where did this except statement go??
												if True:


													# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
													# # (1).  Set up directory structure and filename. Load it in and extract variables.
													init_dir = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') + '_LRsc' + str(lRateScale) +'/' )
													#
													model_dir = str( cell_type + '_N' + str(N) +'_M' + str(M) + z0_tag +  '_ylims' + str(yLo) + '_' + str(yHi) + \
														'_yMinSW' + str(yMinSW) + '_' + sample_longSWs_1st + 'Smp1st_' + priorType + '/')
													#

		
													plt_save_dir = str(EM_figs_dir + init_dir + model_dir)
													if not os.path.exists( plt_save_dir ):
														os.makedirs( plt_save_dir )	
													#
													meanCS_save_dir = str(EM_learning_Dir + init_dir + 'CA_meanCosSim/')
													if not os.path.exists( meanCS_save_dir ):
														os.makedirs( meanCS_save_dir )		



													# Extracting number of training and test samples from the file name. 
													fnamesInDir = os.listdir( str(EM_learning_Dir + init_dir + model_dir ) ) 
													fname = [s for s in fnamesInDir if str('LearnedModel' + GLMtag + '_' + stim) in s \
															and str( str(pct_xVal_train).replace('.','pt') + 'trn_'  + str(msBins) + 'msBins' + maxSampTag + '_rand' + str(rand) ) in s ]
												
													print('')
													print(fname)
													print('')


													for f in fname:
														if f.find('B.npz') > 0:
															Btag = 'B'
														else:
															Btag = ''
														data1 = np.load( str( EM_learning_Dir + init_dir + model_dir + f)  )
														print( str( EM_learning_Dir + init_dir + model_dir + f)  )
														data.append( data1 )
														fnames.append( f )
														fparams.append( str( str(rand) + Btag ) )

													
														# Average conditional probability of all spike-words post learning: 
														# < p( yi | zvec, Mfixed ) > ( all SWs i )

														# Load in raster file and Extract conditional # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
														print('Loading in data file saved from post EM learning inference in raster_zs_inferred_allSWs_given_model.py')

														try: 
															fr = f.replace( 'LearnedModel_', str('rasterZ_'+xValSWs_or_allSWs+maxTrialTag+'_') )
															rasterZs_fname = str( Infer_postLrn_Dir + init_dir + model_dir + fr )
															rast1 = np.load( rasterZs_fname )
															rast.append( rast1 )
															fnameR.append( fr )
														except:
															rast.append( None )
															fnameR.append( None )




															

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PLOTS!
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

											


											# # Load in PSTH data files so that we can compute mean cosine similarity across temporal activations of CAs
											# temporalCSofCAs_dir 	= str( EM_learning_Dir + init_dir + 'CA_temporalCosSim/')
											# temporalCSofCAs_fname 	= str( cellSubTypes + '_' + stim + '_' + str(msBins) + 'msBins_' + \
											# 	 str(bsPSTH) + 'msPSTHbins_'+ sample_longSWs_1st + 'Smp1st_' + priorType + '_rand' + str(rand) + Btag + '.npz')
											
											# # Load in npz file with temporal CA PSTH info from binned rasters. Then compare them across different models learned
											# # and maybe even across different SW samplings and priors too...
											# #
											# tPSTH = np.load( str(temporalCSofCAs_dir+temporalCSofCAs_fname) )
											# #
											# psthZ_accum = tPSTH['psthZ_accum']
											# binsPSTH = tPSTH['binsPSTH']
											# TcosSimMat = tPSTH['TcosSimMat'] 
											# cmk_resort = tPSTH['cmk_resort']

											# psthZ_sum = psthZ_accum.sum(axis=1)
											# psthZ_normed = psthZ_accum/psthZ_sum[:,None]

											# ScosSimMat = cosine_similarity(1-Pia.T,1-Pia.T)	# spatial cosine similairty - within one model.













											
											# For each cell assembly in each model, get average cosSim between it and all
											# CAs in other models to which it has been matched.
											if flg_save_meanCS_forAllCAs and len(data)>0:

												mean_CS_accModels = np.zeros( (len(data), M) ) # -1 # note: -1 to subtract self match.
												ZmatchCS_accModels = np.zeros( (len(data), len(data), M) )

												PiaBox = np.zeros( (len(data), N, M) )

												# Was a plot for the development of this idea. 
												if flg_plot_meanCS_forAllCAs:
													plt.rc('font', weight='bold', size=18)
													f, ax = plt.subplots( len(data)+1,1, figsize=(20,10) )
													#plt.subplots_adjust(hspace=0.4)

												for i in range(len(data)):
													#
													Pia1 		= rc.sig(data[i]['riap'])
													Pia1_init 	= rc.sig(data[i]['ria_snapshots'][0])
													#
													# Get rid of CAs in each model that are not significantly different from their initializations
													ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-Pia1, B=1-Pia1_init)
													x1 = Pia1[:,np.where(csNM<0.9)[0]]

													PiaBox[i] = Pia1

													for j in range(len(data)):
														#
														print('Models ',i,' and ', j)
														#
														Pia2 		= rc.sig(data[j]['riap'])
														Pia2_init 	= rc.sig(data[j]['ria_snapshots'][0])
														#
														# Get rid of CAs in each model that are not significantly different from their initializations
														ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-Pia2, B=1-Pia2_init)
														x2 = Pia2[:,np.where(csNM<0.9)[0]]
														#
														# #
														#
														# Find match between each pair of CAs.
														ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-x1, B=1-x2)

														if not i==j: 				# dont want to include self-match of cs=1 in there because its a bias.
															mean_CS_accModels[i][HungRowCol[1]] += cos_simHM 
															#xxx

														ZmatchCS_accModels[i][j][HungRowCol[1]] = HungRowCol[0] # which za in other models matches with this za in this model.



														# f,ax = plt.subplots(2,1)
														# ax[0].imshow(1-x1[:,HungRowCol[1]],vmin=0,vmax=1)
														# ax[1].imshow(1-x2[:,HungRowCol[0]],vmin=0,vmax=1)
														# plt.show()


													if flg_plot_meanCS_forAllCAs:
														srt = np.argsort( mean_CS_accModels[i] )[::-1]
														#
														ax[i].imshow(1-x1[srt], vmin=0, vmax=1 )
														ax[i].set_aspect('auto')
														ax[i].set_ylabel( str( r'$P_{ia}^'+str(i)+r'$' ) )
														#
														ax[len(data)].plot( mean_CS_accModels[i][srt]/(len(data)-1), label=str(r'$P_{ia}^'+str(i)+r'$') ) 
														# Note: (len(data)-1) because we subtracted off self match up above.

												if flg_plot_meanCS_forAllCAs:
													ax[len(data)].axis([0, M, 0, 1])
													#ax[len(data)].set_ylim([0,1])
													ax[len(data)].grid()
													ax[len(data)].legend(title=r'$m_1$', fontsize=8)
													ax[len(data)].set_ylabel(str(r'$<cs(z_a^{m_1},m_2)>$'))
													#
													plt.show()


												# Average cosine overlap between each CA in each model and the CAs in the other
												# k-1 models that that CA was matched with using the Hungarian Method.
												mean_CS_accModels = mean_CS_accModels/(len(data)-1) # Turns sum into mean.
												







												# aaa = 0 # model 1st.
												# for iii in range(25,30): # index into model aaa CA
												# 	#
												# 	print( mean_CS_accModels[aaa,iii] )
												# 	plt.plot( 1- PiaBox[aaa,:,iii], 'k', linewidth=4, alpha=0.8 )
												# 	#
												# 	for bbb in range(len(data)): # model 2nd
												# 		jjj = int(ZmatchCS_accModels[aaa,bbb,iii]) # index into model bbb CA
												# 		plt.plot( 1 -PiaBox[bbb,:,jjj], alpha=0.6, label=str('M'+str(bbb)+', z'+str(jjj)) )

												# 	plt.legend()
												# 	plt.title( str('M#'+str(aaa)+', z#'+str(iii)+', <cs> = '+str(mean_CS_accModels[aaa,iii].round(2))) )
												# 	plt.show()	

												# xxx	






												meanCS_fname = str( cell_type + GLMtag + '_' + stim + '_' + str(msBins) + 'msBins_' + sample_longSWs_1st + 'Smp1st_' + priorType + '_' + str(len(data)) + 'rands.npz')
												#
												np.savez( str(meanCS_save_dir+meanCS_fname), mean_CS_accModels=mean_CS_accModels, ZmatchCS_accModels=ZmatchCS_accModels, PiaBox=PiaBox, fnames=fnames, fnameR=fnameR, fparams=fparams )	
												
												# mean_CST_accModels=mean_CST_accModels,





											# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
											#
											# Plot translatePermute output for all pairs of models.
											#
											# NOTE: Should be lined up with the for loop inside where data, fparams, fnames are inititiated.
											delFromInit = np.zeros((len(data),1))
											meanLogCond = np.zeros((len(data),1))

											if flg_make_match_models_plots:

												for i in range(len(data)):
													#
													fname_tag = str( str(msBins) + 'ms_' + stim + GLMtag )
													#
													Pia1 		= rc.sig(data[i]['riap'])
													Pia1_init 	= rc.sig(data[i]['ria_snapshots'][0])
													#
													# Get rid of CAs in each model that are not significantly different from their initializations
													ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-Pia1, B=1-Pia1_init)
													x1 = Pia1[:,np.where(csNM<0.9)[0]]
													delFromInit[i] = 1 - csNM.mean()
													#
													pf.visualize_matchModels_cosSim( A=1-Pia1, Atag=fparams[i], B=1-Pia1_init, Btag=str(fparams[i]+'init'), ind=HungRowCol, cos_sim=cos_simHM, len_dif=lenDif, \
														cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=plt_save_dir, fname_tag=str(fname_tag+'_HungMethd'), figSaveFileType=figSaveFileType )
															



													# compute average conditional over all spike words with fixed model after training finished.
													if rast[i]:
														cond_inferred_allSWs = rast[i]['cond_inferred_allSWs'] 
														# 
														condsum = 0
														condcnt = 0
														for t in range(len(cond_inferred_allSWs)):
															condsum += np.array(cond_inferred_allSWs[t]).sum()
															condcnt += len(cond_inferred_allSWs[t])
														meanLogCond[i] = condsum/condcnt	
													else:
														meanLogCond[i] = np.nan





													for j in range(len(data)):
														if j > i:
															#
															# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
															#
															# (4a). Find Model1 CAs that best match each Model2 CA.
															#
															#
															
															Pia2 		= rc.sig(data[j]['riap'])
															Pia2_init 	= rc.sig(data[j]['ria_snapshots'][0])

															ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-Pia2, B=1-Pia2_init)
															x2 = Pia2[:,np.where(csNM<0.9)[0]]
															#
															pf.visualize_matchModels_cosSim( A=1-Pia2, Atag=fparams[j], B=1-Pia2_init, Btag=str(fparams[j]+'init'), ind=HungRowCol, cos_sim=cos_simHM, len_dif=lenDif, \
																cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=plt_save_dir, fname_tag=str(fname_tag+'_HungMethd'), figSaveFileType=figSaveFileType )

															#
															ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-x1, B=1-x2)
															#
															

															#
															if False:
																pf.visualize_matchModels_cosSim( A=1-Pia1, Atag=fparams[i], B=1-Pia2, Btag=fparams[j], ind=ind, cos_sim=cosSim, len_dif=lenDif, \
																	cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=plt_save_dir, fname_tag=fname_tag, figSaveFileType=figSaveFileType )
															#
															pf.visualize_matchModels_cosSim( A=1-x1, Atag=fparams[i], B=1-x2, Btag=fparams[j], ind=HungRowCol, cos_sim=cos_simHM, len_dif=lenDif, \
																cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=plt_save_dir, fname_tag=str(fname_tag+'_HungMethd'), figSaveFileType=figSaveFileType )


															#
															# # (4b). Find Model2 CAs that best match each Model1 CA.
															# ind, cosSim, lenDif, cosSimMat, lenDifMat = rc.matchModels_AtoB_cosSim(A=1-Pia2, B=1-Pia1, verbose=False)
															# #
															# pf.visualize_matchModels_cosSim( A=1-Pia2, Atag='Mod2', B=1-Pia1, Btag='Mod1', ind=ind, cos_sim=cosSim, len_dif=lenDif, \
															# 				cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir, fname_tag=str(rand) )

											else:

												for i in range(len(data)):
													#
													Pia1 		= rc.sig(data[i]['riap'])
													Pia1_init 	= rc.sig(data[i]['ria_snapshots'][0])
													#
													# Get rid of CAs in each model that are not significantly different from their initializations
													ind, cosSim, csNM, lenDif, ldNM, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-Pia1, B=1-Pia1_init)
													delFromInit[i] = 1 - csNM.mean()


													# compute average conditional over all spike words with fixed model after training finished.
													if rast[i]:
														cond_inferred_allSWs = rast[i]['cond_inferred_allSWs'] 
														# 
														condsum = 0
														condcnt = 0
														for t in range(len(cond_inferred_allSWs)):
															condsum += np.array(cond_inferred_allSWs[t]).sum()
															condcnt += len(cond_inferred_allSWs[t])
														meanLogCond[i] = condsum/condcnt	
													else:
														meanLogCond[i] = np.nan


											# print('meanLogCond : ', fparams)
											# print(meanLogCond.round(2))


											# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
											#
											# Collect up matrices of how many matching CAs models have (on main diagonal)
											# and share (on off diagonals) across A and B models and different rands.
											#
											# NOTE: Should be lined up with the for loop inside where data, fparams, fnames are inititiated.
											if data:

												#
												matchMod1toMod2_cosSim, matchMod1toMod2_lenDif, matchMod1toMod2_csHM, \
												matchMod2toMod1_cosSim, matchMod2toMod1_lenDif, matchMod2toMod1_csHM, _, _, _, _, _, _, \
												matchMod1toMod2_csNM, 	matchMod1toMod2_ldNM, 	matchMod2toMod1_csNM, matchMod2toMod1_ldNM, _, _, _, _ \
												= rc.compute_pairwise_model_CA_match(data) 
												#


												#
												# Plot cosine similarity for learned model and null model. (essentially learned model without matching up CAs.)
												fname_save = str( fnames[0][:fnames[0].find('_rand')].replace('LearnedModel_','CA_match_cosSim_' ) + '_' + str(len(fnames)) + 'files' )
												if False:
													pf.plot_pairwise_model_CA_match(matchMod1toMod2_cosSim, matchMod1toMod2_csNM, delFromInit, np.zeros((len(data),1)), meanLogCond, \
														GTtag_plot_pairwise_model_CA_match, fnames, fparams, plt_save_dir, fname_save, 'Cosine Similarity', figSaveFileType)

												# can use: matchMod1toMod2_cosSim or matchMod1toMod2_csHM.
												#	
												pf.plot_pairwise_model_CA_match(matchMod1toMod2_csHM, matchMod1toMod2_csNM, delFromInit, np.zeros((len(data),1)), meanLogCond, \
													GTtag_plot_pairwise_model_CA_match, fnames, fparams, plt_save_dir, str(fname_save+'_HungMethd'), 'Cosine Similarity Hungarian Sort', figSaveFileType)
												
													
												#
												# Plot vector length difference for learned model and null model. (essentially learned model without matching up CAs.)
												if False: # Len Difference not so useful.
													fname_save = str( fnames[0][:fnames[0].find('_rand')].replace('LearnedModel_','CA_match_LenDif_' ) + '_' + str(len(fnames)) + 'files' )
													pf.plot_pairwise_model_CA_match(matchMod1toMod2_lenDif, matchMod1toMod2_ldNM, delFromInit, np.zeros((len(data),1)), meanLogCond, \
														GTtag_plot_pairwise_model_CA_match, fnames, fparams, plt_save_dir, fname_save, 'Vector Length Similarity', figSaveFileType)



											# rand
										# lRateScale
									# learning_rate
								# SW_bins
							# yMinSW				
						# yHi
					# yLo
				# overcomp
			# stim
		# sample_longSWs_1st
	# Egal Prior
# cell type	




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# THIS IS GOOD. DO NOT DELETE AND MAYBE EVEN SAVE SOME OF THESE PLOTS !!!
# Compute cosine similarity and length diff for some interesting null model cases.
#
#		(1). Two random normal distributed mostly quiet values.
# 		(2). Two binomial drawn mostly quiet with some on values.
#		(3). Two models learned on random draws from same data.
#		(4). Two models learned on different data. (Not done.)
# 		(5). Model learned on data and binomial random model.
#		(6). Model learned on data and normal random quiet model.
#
if False:
	Pia1 = rc.sig(data[0]['riap']) 				# Two different learned models (unsorted / unmatched)
	#Pia2 = rc.sig(data[2]['riap'])

	# Pia1 = np.random.normal(1,0.1,(N,M)) 		# Two random normal matrices.
	# Pia2 = np.random.normal(1,0.1,(N,M))

	# Pia1 = 1-np.random.binomial(1,5/M,(N,M)) 	# Two binary bernoulli distributed matrices.
	Pia2 = 1-np.random.binomial(1,5/M,(N,M))

	ind, cosSim, csNM1, lenDif, ldNM1, cosSimMat, lenDifMat, HungRowCol, cos_simHM = rc.matchModels_AtoB_cosSim(A=1-Pia1, B=1-Pia2)

	# # Null Models: CosSim & LenDif of unordered matrices.
	# csNM1 = np.mean( np.diag(cosSimMat) ) 
	csNM2 = np.mean( cosSimMat ) 
	# ldNM1 = np.mean( np.diag(lenDifMat) )
	ldNM2 = np.mean( lenDifMat )

	print('cosSim Sort: ', np.nanmean(cosSim[0]).round(2) )
	print('cosSim Null: ', np.nanmean(csNM1).round(2) , csNM2.round(2) )
	print('lenDif Sort: ', np.nanmean(lenDif[0]).round(2) )
	print('cosSim Null: ', np.nanmean(ldNM1).round(2) , ldNM2.round(2) )

	saveDir = './'
	pf.visualize_matchModels_cosSim( A=1-Pia2, Atag='Mod1', B=1-Pia1, Btag='Mod2', ind=ind, cos_sim=cosSim, len_dif=lenDif, \
				cosSimMat=cosSimMat, lenDifMat=lenDifMat, numSamps=0, r=0, plt_save_dir=saveDir, fname_tag=str(rand) )







self_love = True
print('Self Love = ',self_love)
