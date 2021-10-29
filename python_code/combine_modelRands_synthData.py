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
EM_learning_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/Models_learned_EM/')
Infer_postLrn_Dir 	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/inference_postLrn/')
EM_learnStats_Dir	= str( dirScratch + 'data/python_data/PGM_analysis/synthData/inferStats_from_EM_learning/')

EM_figs_dir 		= str( dirScratch + 'figs/PGM_analysis/EM_Algorithm//')






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # (1). Load in npz file and extract data from it.

# Parameters we can loop over.
num_SWs_tot = [100000] #[50000, 100000] # [10000, 50000, 100000]
num_Cells = [100] #[50,100] #[50, 100] 					# Looping over N values
num_CAs_true = [100] #[50,100]# [50, 100] #,200			# Looping over M values used to build the model
model_CA_overcompleteness = [1] #,2		# how many times more cell assemblies the model assumes than are in true model (1 means complete - M_mod=M, 2 means 2x overcomplete)
learning_rates = [0.1] # [1.0, 0.5, 0.1] 	# Learning rates to loop through
#
ds_fctr_snapshots 	= 1000 	# Take a sample of model parameters every *ds_fctr* time steps to compute and plot MSE after we determine permutation of learned CA's to True CA's
pct_xVal_train 		= 0.5 	# percentage of spikewords (total data) to use as training data. Set the rest aside for test data for cross validation.
pct_xVal_train_prev	= 0.5
num_EM_rands		= 1 	# number of times to randomize samples and rerun EM on same data generated from single synthetic model.
#
# train_2nd_modelS = [True,False]



# Synthetic Model Construction Parameters
Ks 				= [2] # , 0] #[2, 2, 2, 2] 		# Number of	cell assemblies active, given a Bernoulli distribution ("hotness" of Z-vector)
Kmins 			= [0] # , 0] #[0, 0, 1, 1]		# Max & Min number of cell assemblies active 
Kmaxs 			= [2] # , 0] #[2, 2, 3, 3]		# 
#
Cs 				= [2] # , 2] #[2, 3, 2, 3]			# number of cells participating in cell assemblies. ("hotness" of each row in Pia)
Cmins 			= [2] # , 2] #[2, 2, 2, 2] 		# Max & Min number of cell active to call it a cell assembly
Cmaxs 			= [6] # , 6] #[6, 6, 6, 6] 		# 
#
yLo_Vals 		= [0] 		# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<=yLo.
yHi_Vals 		= [1000] 	# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution.
yMinSWs 		= [1] # [1, 3] # [1,2,3]
#
mu_Pia			= 0.0   		# Binary values in Pia matrix made continuous by sampling from Gaussian centered at 1-mu_Pia or 0+mu_Pia.
sig_Pia			= 0.1 		# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
#
bernoulli_Pi	= 1.0   	# Each value in binary Pi vector randomly sampled from bernoulli with p(Pi=1) = bernoulli_Pi.	
mu_Pi			= 0.0   	# Binary values in Pi vector made continuous by sampling from Gaussian centered at 1-mu_Pi or 0+mu_Pi.
sig_Pi			= 0.1 		# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.


num_test_samps_4xValS 	= [1] #[1, 10, 100] 

# Parameter initializations for EM algorithm to learn model parameters
params_initS 	= ['DiagonalPia'] #, 'NoisyConst'] 	# Options: {'True', 'RandomUniform', 'NoisyTrue' (w/ sig_init), 'NoisyConst' (w/ sig_init & C_noise) }
sigQ_init 		= 0.01			# STD on Q initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
sigPi_init 		= 0.05			# STD on Pi initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
sigPia_init 	= 0.05			# STD on Pia initialization from true or mean values in 'NoisyTrue' and 'NoisyConst' respectively.
Z_hot 			= 2 			# Mean of initialization for Q value (how many 1's expected in binary z-vector)
C_noise_ri 		= 1.0			# Mean of initialization of Pi values (1 means mostly silent) with variability defined by sigPia_init
C_noise_ria 	= 1.0			# Mean of initialization of Pia values (1 means mostly silent) with variability defined by sigPia_init

sig_init 		= np.array([ sigQ_init, sigPi_init, sigPia_init ])
params_init_str = str( str(Z_hot)+'hot_mPi' + str(C_noise_ri) + '_mPia' + str(C_noise_ria) + '_sI' + str(sig_init) )




# Learning rates for EM algorithm
lRateScale_Pi = 1.0 # Multiplicative scaling to Pi learning rate. If set to zero, Pi taken out of model essentially.




# Flags for the EM (inference & learning) algorithm.
flg_include_Zeq0_infer = True
if flg_include_Zeq0_infer:
	z0_tag='_zeq0'
else:
	z0_tag='_zneq0'



for num_SWs in num_SWs_tot:
	#
	for num_test_samps_4xVal in num_test_samps_4xValS:
		#
		for params_init in params_initS:
			#
			for learning_rate in learning_rates:
				#
				for xyz in range( len(Ks) ):
					#
					K 	 = Ks[xyz]
					Kmin = Kmins[xyz]
					Kmax = Kmaxs[xyz]
					C 	 = Cs[xyz]
					Cmin = Cmins[xyz]
					Cmax = Cmaxs[xyz]
					#
					for abc in range(len(num_Cells)):
						N = num_Cells[abc]
						M = num_CAs_true[abc]
						#
						for yMinSW in yMinSWs:
								#
								for yLo in yLo_Vals:
									#
									for yHi in yHi_Vals:
										#
										for overcomp in model_CA_overcompleteness:
											M_mod = int(overcomp*M)
											#
											data 	= list() # To collect up model data and filenames over 
											fnames 	= list() # different rands and A,B model combinations.
											dataPL 	= list() # To collect up model data and filenames over 
											#
											for rand in range(num_EM_rands):

												print('Random Sampling of Spike Words #',rand)
												if rand==0:
													rsTag = '_origModnSWs'
												else:
													rsTag = str( '_resampR0trn'+ str(pct_xVal_train_prev).replace('.','pt') )
												#


												# try:
												if True:


													# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
													# # (1).  Set up directory structure and filename. Load it in and extract variables.
													init_dir = str( 'Init_' + params_init + '_' + params_init_str + '_LR' + str(learning_rate).replace('.','pt') \
															+ '_LRpi' + str(lRateScale_Pi).replace('.','pt') +'/' )
													#
													model_dir = str( 'SyntheticData_N' + str(N) + '_M' + str(M) + '_Mmod' + str(M_mod) + '_K' + str(K) + 
															'_' + str(Kmin) + '_' + str(Kmax) + '_C' + str(C) + '_' + str(Cmin)+ '_' + str(Cmax) + 
															'_mPia' + str(mu_Pia) + '_sPia' + str(sig_Pia) + '_bPi' + str(bernoulli_Pi) + '_mPi' + 
															str(mu_Pi) + '_sPi' + str(sig_Pi) + z0_tag + '_gnu2/' )




													# Make a directories for output plots if they are not already there.
													if not os.path.exists( str(EM_figs_dir + init_dir + model_dir) ):
														os.makedirs( str(EM_figs_dir + init_dir + model_dir) )
													#
													if not os.path.exists( str(EM_learnStats_Dir + init_dir + model_dir) ):
														os.makedirs( str(EM_learnStats_Dir + init_dir + model_dir) )
													#
													if not os.path.exists( str(EM_learning_Dir + init_dir + model_dir) ):
														os.makedirs( str(EM_learning_Dir + init_dir + model_dir) )	
													#
													if not os.path.exists( str(Infer_postLrn_Dir + init_dir + model_dir) ):
														os.makedirs( str(Infer_postLrn_Dir + init_dir + model_dir) )	




													# Build up file names.
													fname_EMlrn = str('EM_model_data_' + str(num_SWs) + 'SWs_trn' + str(pct_xVal_train).replace('.','pt') + '_ylims' + str(yLo) \
																	+ '_' + str(yHi) + '_yMinSW' + str(yMinSW) + '_rand' + str(rand) + rsTag + '.npz' )

													fname_EMlrnB = str(fname_EMlrn[:-4]+'B.npz')
													#
													fnames.append( fname_EMlrn )
													fnames.append( fname_EMlrnB )


													# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
													#
													print('Loading in matching data files saved from EM learning algorithm in pgmCA_realData.py')
													print( fname_EMlrn )
													#
													data1 = np.load( str(EM_learning_Dir + init_dir + model_dir +fname_EMlrn) )
													data.append( data1 )
													#
													data2 = np.load( str(EM_learning_Dir + init_dir + model_dir +fname_EMlrnB) )
													data.append(data2)


													data3 = np.load( str(Infer_postLrn_Dir + init_dir + model_dir +fname_EMlrn.replace('EM_model_data_','SWs_inferred_postLrn_')) )
													dataPL.append(data2)


											# Concatenate models and get rid of columns in binary combined Pia that are the same.
											if data:
												print('# CAs before combination: ',len(data)*data[0]['riap'].shape[1])
												#
												Z_train = data[0]['Z_train']
												Y_train = data[0]['Y_train'] 
												#
												Z_test 	= data[0]['Z_test']
												Y_test 	= data[0]['Y_test'] 

												print( len(Y_train), len(Y_test) )

												#
												riapC = 20*np.ones_like( data[0]['riap'] ) 						# hack to make concatenate dims work. I kill it later.
												ripC = 20*np.ones_like( data[0]['rip'] ) 
												qpC = np.zeros( len(data) ) 
												for i in range(len(data)):
													riapC = np.concatenate( (riapC, data[i]['riap']), axis=1 ) 	# Concatenate all models A&B & different rands.
													ripC = np.vstack( (ripC, data[i]['rip']) )
													qpC[i] = data[i]['qp'][0]
												#
												ripC = ripC[1:]
												PiapC = 1-rc.sig(riapC)	
												PipC = 1-rc.sig(ripC)	
												QpC = rc.sig(qpC)	
												#
												card = (PiapC>0.5).sum(axis=0)		# Cardinality of Piap columns
												indCard = np.where(card>1)[0]		# Single cell CAs
												PiapC = PiapC[:,indCard]			# Take only CAs that are not single cell
												riapC = riapC[:,indCard]			# Take only CAs that are not single cell
												#
												# Find 1's in normalized binary overlap matrix.
												PiapCB = (PiapC>0.5).astype(int)	# Binarize.
												CardMn = np.sqrt( np.outer(card[indCard], card[indCard].T) ) 	# normalize with sqrt( |CA|_i * |CA|_j )
												BinOvl = np.matmul(PiapCB.T, PiapCB)							# number of shared 1's in columns of Piap
												BinOvlNorm = BinOvl/CardMn 										# normalized binary overlap
												#
												# # # # # # # # # # # # # # # # # #
												if False:
													plt.imshow( np.tril( BinOvlNorm, k=-1 ) )
													plt.colorbar()
													plt.show()
												#
												# #
												#
												Ovls = np.where( np.tril( BinOvlNorm, k=-1 ) == 1)
												NonOvls = set( np.arange( len(indCard) ) )
												for i in range(Ovls[0].shape[0]):
													# print( i, Ovls[0][i], Ovls[1][i], np.where(Ovls[0]==i)[0], np.where(Ovls[1]==i)[0] )
													if Ovls[0][i] in NonOvls:
														NonOvls.remove(Ovls[0][i])
												#
												PiapC = PiapC[:,list(NonOvls)] 
												riapC = riapC[:,list(NonOvls)] 
												#
												print('# CAs after combination: ',PiapC.shape[1])		
												#
												# # # # # # # # # # # # # # # # # #
												if False:
													plt.imshow(PiapC, vmin=0, vmax=1)
													plt.colorbar()
													plt.show()



												# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
												#
												# Make a conglomeration model.
												#
												riapCong = riapC
												ripCong  = rc.inv_sig((rc.sig(ripC)).mean(axis=0)) # Complicated but necessary. Justified below.
												qpCong 	 = qpC.mean()



												# Plot to justify how we compute ripCong.
												if False:
													plt.plot(PipC.mean(axis=0),color='k')
													plt.plot((1-rc.sig(ripC)).mean(axis=0),color='r')
													plt.plot(1-(rc.sig(ripC)).mean(axis=0),color='g')
													plt.plot(1-(rc.sig(ripC.mean(axis=0))),color='b')
													plt.plot(1-rc.sig(ripCong), color='m')
													plt.show()



											
												# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
												#
												# Run inference on all Spike Words / Data (Training and Test)
												# using the fixed conglomerated model after training.
												#
												if False:
													print('Inferring all data with post-learning model.')
													t0 = time.time()
													#
													num_PL_SWs = 1
													print('Inferring ',len(Y_train),' training spike words.')
													Z_inferred_train_postLrn, Y_inferred_train_postLrn, pj_inferred_train_postLrn = rc.inferZ_allSWs( \
															[Y_train[range(num_PL_SWs)]], [np.ones_like(Y_train[range(num_PL_SWs)])], [Z_train[range(num_PL_SWs)]], 1, 0, 2, \
															rc.inv_sig(np.array([1])), ripCong, riapCong, flg_include_Zeq0_infer, yLo, yHi, approx=False, verbose=True ) # replace qpCong !!!
													

													for i in range( len(dataPL) ):
														dataPL[i]['pjoint_train'].mean()
														dataPL[i]['pjoint_test'].mean()
													print( np.nanmean( np.array(pj_inferred_train_postLrn[0]) ) )


													#
													# print('Inferring ',len(Y_test),' test spike words.')
													# Z_inferred_test_postLrn, Y_inferred_test_postLrn, pj_inferred_test_postLrn   = rc.inferZ_allSWs( \
													# 		[Y_test],  [np.ones_like(Y_test)],  1, 0, 2, qp, rip, riap, flg_include_Zeq0_infer, yLo, yHi )
													#
													t1 = time.time()
													print('Done with inferring all data with post-learning model: time = ',t1-t0)












# # # # BELOW HERE I AM DIGGING INTO THE INFERENCE PROCEDURE. # # # # # # # #




verbose = False

q = data[0]['q']
ri = data[0]['ri']
ria = data[0]['ria']
#
Q = rc.sig(q)
Pi = rc.sig(ri)
Pia = rc.sig(ria)

#ri_in = 20*np.ones_like(ri) 	# or input 5*ri to have a quieter one or can make noisier neurons.
ri_in= 5*ri						# It changes things significantly. 
#ri_in= ri	

num_PL_SWs = 100
print('Inferring ',len(Y_train),' training spike words.')
Z_inferred_train_postLrn, Y_inferred_train_postLrn, pj_inferred_train_postLrn = rc.inferZ_allSWs( \
		[Y_train[range(num_PL_SWs)]], [np.ones_like(Y_train[range(num_PL_SWs)])], [Z_train[range(num_PL_SWs)]], 1, 0, 2, \
		q, ri_in, ria, flg_include_Zeq0_infer, yLo, yHi, approx=False, verbose=verbose ) 




# Preallocate memory to keep track of number 
zTru = np.zeros(num_PL_SWs).astype(int)
zInf = np.zeros(num_PL_SWs).astype(int)
zCap = np.zeros(num_PL_SWs).astype(int)
zExt = np.zeros(num_PL_SWs).astype(int)
zMis = np.zeros(num_PL_SWs).astype(int)
#
yTru = np.zeros(num_PL_SWs).astype(int)
yInf = np.zeros(num_PL_SWs).astype(int)
yCap = np.zeros(num_PL_SWs).astype(int)
yExt = np.zeros(num_PL_SWs).astype(int)
yMis = np.zeros(num_PL_SWs).astype(int)
#
yGenOut = np.zeros(num_PL_SWs).astype(int)
yGenIn = np.zeros(num_PL_SWs).astype(int)
yGenDrp = np.zeros(num_PL_SWs).astype(int)
#
# # 
#
for i in range(num_PL_SWs): # Can loop thru i for all training samples.

	# # Compute Statistics on data generation (Observed/NotObs Y's In/Out of Active Z's)
	#
	CellsInCAs = set(np.where(1-Pia[:,list(Z_train[i])]>0.5)[0]) 	# Ys that should be activated by active Zs.
	CellsActvOutCAs = Y_train[i].difference(CellsInCAs) 			# Active Ys not activated by active Zs.
	CellsActvInCAs = Y_train[i].intersection(CellsInCAs)			# Active Ys activated by active Zs.
	CellsInactvInCAs = CellsInCAs.difference(Y_train[i]) 			# Silent Ys that should be activated by active Zs.
	#
	# Mean Pi values for Noisy Cells vs all other cells.
	if verbose:
		print( 'Mean Pi Noisy Cells = ',Pi[list(CellsActvOutCAs)].mean() )						# Mean Pi of Noisy cells. 
		NotCellsActvOut = set(range(N)).difference(CellsActvOutCAs) # Non-noisy cells: All cells other than ones active outside CAs.
		print( 'Mean Pi Noisy Cells = ', Pi[list(NotCellsActvOut)].mean() )						# Mean Pi of Non-noisy cells.


	#
	if verbose:
		print('------------------------------------------------------------------------')
		print( 'Data Generation #', i  )
		print('Active Ys       : ', Y_train[i] )
		print('Active Zs       : ', Z_train[i],'   --->     Activate Ys: ', CellsInCAs  )
		print('Active Ys !inZ  : ', CellsActvOutCAs)
		print('Active Ys in Z  : ', CellsActvInCAs)
		print('!active Ys in Z : ', CellsInactvInCAs)
		print('')







	# # Compute Statistics on Inferred vs Observed Ys and Zs.
	#
	zUnion, zIntersect, zDiff, zSubed, zAdded, zCapture, zMissed, zExtra, yUnion, yIntersect, \
	yDiff, ySubed, yAdded, yCapture, yMissed, yExtra, Kinf, KinfDiff = rc.compute_inference_statistics_single(\
		Z_train[i], set(Z_inferred_train_postLrn[0][i]), M, Y_train[i], set(Y_inferred_train_postLrn[0][i]), N, verbose=verbose)


	zTru[i] 	= len(Z_train[i])
	zInf[i] 	= len(Z_inferred_train_postLrn[0][i])
	zCap[i] 	= zCapture
	zExt[i] 	= zExtra
	zMis[i] 	= zMissed
	#
	yTru[i] 	= len(Y_train[i])
	yInf[i] 	= len(Y_inferred_train_postLrn[0][i])
	yCap[i] 	= yCapture
	yExt[i] 	= yExtra
	yMis[i] 	= yMissed
	#
	yGenOut[i] 	= len(CellsActvOutCAs)
	yGenIn[i]  	= len(CellsActvInCAs)
	yGenDrp[i] 	= len(CellsInactvInCAs)

	if verbose:
		print('------------------------------------------------------------------------')
		print(i)
		print( 'Z. :tru.',zTru[i],' :inf.',zInf[i],' :cap.',zCap[i],' :ext.',zExt[i],' :mis.',zMis[i] )
		print( 'Y. :tru.',yTru[i],' :inf.',yInf[i],' :cap.',yCap[i],' :ext.',yExt[i],' :mis.',yMis[i] )


#xx = (1-Pia[ :, list(Z_train[i]) ].prod(axis=1)*Pi)



print( 'Z. :cap.',zCap.mean(),' :ext.',zExt.mean(),' :mis.',zMis.mean(),' // tru.',zTru.mean())
print( 'Y. :cap.',yCap.mean(),' :ext.',yExt.mean(),' :mis.',yMis.mean(),' // tru.',yTru.mean())
print( 'Y from :CA.',yGenIn.mean(), ' :Nois.',yGenOut.mean()  )

