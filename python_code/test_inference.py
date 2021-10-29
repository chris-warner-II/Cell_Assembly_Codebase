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
flg_include_Zeq0_infer = True
yLo 			= 0
yHi 			= 1000

N 				= 50 
M 				= 50
M_mod 			= M
#
C 				= 2
Cmin 			= 2
Cmax 			= 6
#
K 				= 2
Kmin 			= 0
Kmax 			= 2
#
bernoulli_Pi 	= 1.0
mu_Pi 			= 0.0
sig_Pi 			= 0.01
#
mu_Pia 			= 0.0
sig_Pia 		= 0.1

num_PL_SWs = 1000

verbose = False

q = None #ri = None #ria = None #ria_mod = None
while q==None: # while statement to resynthesize model if any cells participate in 0 assemblies.
	q, ri, ria, ria_mod = rc.synthesize_model(N,M,M_mod, C,K, Cmin,Cmax, bernoulli_Pi,mu_Pi,sig_Pi, mu_Pia,sig_Pia, verbose=False)
#
Y_train, Z_train = rc.generate_data(num_PL_SWs, q, ri, ria, Cmin, Cmax, Kmin, Kmax)


Pia = rc.sig(ria)
Pi = rc.sig(ri)
Q = rc.sig(q)



Pi_shifts = [-1., -0.5, -0.1, 0, 0.1, 0.5, 1.] # negative makes noisier neurons. Choose values between 0 and 1. Will push values up against rails.
			 # i.e., Pi_shift = +1 means completely silent (other than in CAs). Pi_shift = -1 means all neurons fire all the time. 

# Preallocate memory
pyi_gvnZ_stats_infZ_accPiShift 	= np.zeros( (len(Pi_shifts), 6, 3) )
pyi_gvnZ_stats_truZ_accPiShift 	= np.zeros( (len(Pi_shifts), 6, 3) )
pyi_gvnZ_auc_accPiShift 		= np.zeros( (len(Pi_shifts), 4, 3) )	

zTru = np.zeros( (num_PL_SWs, len(Pi_shifts)) ).astype(int)
zInf = np.zeros( (num_PL_SWs, len(Pi_shifts)) ).astype(int)
yTru = np.zeros( (num_PL_SWs, len(Pi_shifts)) ).astype(int)


for j,Pi_shift in enumerate(Pi_shifts): # Can loop thru j for different values of Pi_shift.

	print(j, Pi_shift)

	Pi_in = rc.sig(ri) + Pi_shift
	ri_in = rc.inv_sig( np.clip(Pi_in, 1e-9, 1-1e-9) )	


	print('Inferring ',len(Y_train),' training spike words.')
	Z_inferred_train_postLrn, Y_inferred_train_postLrn, pj_inferred_train_postLrn = rc.inferZ_allSWs( \
			[Y_train[:num_PL_SWs]], [np.ones_like(Y_train[:num_PL_SWs])], [Z_train[:num_PL_SWs]], 1, 0, 2, \
			q, ri_in, ria, flg_include_Zeq0_infer, yLo, yHi, approx=False, verbose=verbose ) 


	# Preallocate memory
	pyi_gvnZ_stats_infZ = np.zeros( (num_PL_SWs,6) )
	pyi_gvnZ_stats_truZ = np.zeros( (num_PL_SWs,6) )
	pyi_gvnZ_auc 		= np.zeros( (num_PL_SWs,4) )	
	#
	for i in range(num_PL_SWs): # Can loop thru i for all training samples.

		zTru[i,j] 	= len(Z_train[i])
		zInf[i,j] 	= len(Z_inferred_train_postLrn[0][i])
		yTru[i,j] 	= len(Y_train[i])

		# Compute p(yi=1|z) and p(yi=0|z) on inferred z-vector and true z-vector. Also, compute the area under the ROC curve.
		THs = np.linspace(0,1,11) 
		pyi_gvnZ_stats_inf, pyi_gvnZ_auc_inf, ROC_vals_inf = rc.compute_pyi_gvnZ(Z_inferred_train_postLrn[0][i], Y_train[i], ri, ria, THs)
		pyi_gvnZ_stats_tru, pyi_gvnZ_auc_tru, ROC_vals_tru = rc.compute_pyi_gvnZ(Z_train[i], Y_train[i], ri, ria, THs)

		pyi_gvnZ_stats_infZ[i] 	= pyi_gvnZ_stats_inf
		pyi_gvnZ_stats_truZ[i] 	= pyi_gvnZ_stats_tru
		pyi_gvnZ_auc[i]			= pyi_gvnZ_auc_inf + pyi_gvnZ_auc_tru


	
	# Taking mean, std, median across all 
	pyi_gvnZ_stats_infZ_accPiShift[j,:,0] 	= np.nanmean(pyi_gvnZ_stats_infZ,axis=0)
	pyi_gvnZ_stats_infZ_accPiShift[j,:,1] 	= np.nanstd(pyi_gvnZ_stats_infZ,axis=0)
	pyi_gvnZ_stats_infZ_accPiShift[j,:,2] 	= np.nanmedian(pyi_gvnZ_stats_infZ,axis=0) 
	#
	pyi_gvnZ_stats_truZ_accPiShift[j,:,0] 	= np.nanmean(pyi_gvnZ_stats_truZ,axis=0)
	pyi_gvnZ_stats_truZ_accPiShift[j,:,1] 	= np.nanstd(pyi_gvnZ_stats_truZ,axis=0)
	pyi_gvnZ_stats_truZ_accPiShift[j,:,2] 	= np.nanmedian(pyi_gvnZ_stats_truZ,axis=0) 
	#
	pyi_gvnZ_auc_accPiShift[j,:,0] 			= np.nanmean(pyi_gvnZ_auc,axis=0)
	pyi_gvnZ_auc_accPiShift[j,:,1] 			= np.nanstd(pyi_gvnZ_auc,axis=0)
	pyi_gvnZ_auc_accPiShift[j,:,2] 			= np.nanmedian(pyi_gvnZ_auc,axis=0) 




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Plot histograms of Z_inf at each Pi_shift value vs. Z_tru
if True:
	for k in range(len(Pi_shifts)): # (1,len(Pi_shifts)-1):
		PiS = Pi_shifts[k]
		a,b = np.histogram( zInf[:,k], bins=np.arange(zInf[:,k].max()+2) )
		plt.plot( b[:-1]+0.1*(k-1), np.cumsum(a), label=PiS )
		#
	a,b = np.histogram( zTru[:,0], bins=np.arange(zTru[:,0].max()+2) )
	plt.plot( b[:-1], np.cumsum(a) , 'k--', label='Tru')	
	plt.xlabel('|z|')
	plt.ylabel( str('counts/'+str(num_PL_SWs) ) )
	plt.title( 'Cumulative Histograms of inferred |z| vs. Pi-shift (neg=noisier)' )
	plt.legend()
	plt.grid()
	plt.show()




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Plot errorbars of AUC across different Pi_shift values.
#
# 	 pyi_gvnZ_auc_accPiShift
# 		---------------
# 		size = (7,4,3) 
#		  Dim.7 = Pi_shifts
# 		  Dim.4 = AUC {0=TN&FN:yEq0:w/Zinf, 1=TP&FP:yEq1:w/Zinf, 2=:yEq0:w/Zgen, 3=yEq1:w/Zgen}
#		  Dim.3 = 0=mean,1=std,2=median across all num_PL_SWs samples.
#	
if True:
	#
	# for yi's=1 - True Positives & False Positives with Generation Z.
	plt.errorbar(np.arange(len(Pi_shifts)), pyi_gvnZ_auc_accPiShift[:,3,0], yerr=pyi_gvnZ_auc_accPiShift[:,3,1], color='black', \
		barsabove=True, linestyle='solid', capsize=5, label=str('$\mu$, $\sigma$ genZ' ) ) # Mean & STD (acc SWs) of mean (acc yis).
	plt.scatter(np.arange(len(Pi_shifts)), pyi_gvnZ_auc_accPiShift[:,3,2], color='magenta', marker='d', label='median genZ' )
	#
	# for yi's=1 - True Positives & False Positives with Inference Z.
	plt.errorbar(np.arange(len(Pi_shifts)), pyi_gvnZ_auc_accPiShift[:,1,0], yerr=pyi_gvnZ_auc_accPiShift[:,1,1], color='blue', \
		barsabove=True, linestyle='solid', capsize=5, label=str('$\mu$, $\sigma$ infZ' ) ) # Mean & STD (acc SWs) of mean (acc yis).
	plt.scatter(np.arange(len(Pi_shifts)), pyi_gvnZ_auc_accPiShift[:,1,2], color='red', marker='o', label='median infZ' )
	#
	# for yi's=0 - True Negatives & False Negatives with Inference Z.
	if False: # NOTE: This is same as yi=1. Maybe has to be mathematically.
		plt.errorbar(np.arange(len(Pi_shifts)), pyi_gvnZ_auc_accPiShift[:,0,0], yerr=pyi_gvnZ_auc_accPiShift[:,0,1], color='green', \
			barsabove=True, linestyle='dashed', capsize=5, label=str('$\mu$, $\sigma$ y=0 infZ' ) ) # Mean & STD (acc SWs) of AUC.
		plt.scatter(np.arange(len(Pi_shifts)), pyi_gvnZ_auc_accPiShift[:,0,2], color='cyan', marker='x', label='median y=0 infZ' )
	#
	plt.xticks( np.arange(len(Pi_shifts)), Pi_shifts )
	plt.ylabel( str('Area Under ROC Curve') )
	plt.xlabel( str('Noisier <-- Pi Shift --> Quieter') )
	plt.title( "Statistics across SW samples of mean across $y_i$'s=1 \n Note: y_i's=0 are the same." )
	plt.legend()
	plt.grid()
	plt.show()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Plot errorbars of STATISTICS ACROSS SAMPLED SPIKE WORDS for 
# mean across cells in y-vector for differentPi_shift values.
#
# (1). pyi_gvnZ_stats_infZ_accPiShift
# (2). pyi_gvnZ_stats_truZ_accPiShift (meh?)
# 		---------------
# 		size = (7,6,3) 
#		  7 = Pi_shifts
# 		  6 = mean,std,median across cells on or off. {0-2 for y=0 and 3-5 for y=1}
#		  3 = mean,std,median across all num_PL_SWs samples.
#	
if True:
	#
	# for yi's=1 - True Positives & False Positives with Generation Z.
	plt.errorbar(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_truZ_accPiShift[:,3,0], yerr=pyi_gvnZ_stats_truZ_accPiShift[:,3,1], color='black', \
		barsabove=True, linestyle='solid', capsize=5, alpha=0.5, label=str('$\mu$, $\sigma$ y=1 genZ' ) ) # Mean & STD (acc SWs) of mean (acc yis).
	plt.scatter(np.arange(len(Pi_shifts)), pyi_gvnZ_auc_accPiShift[:,3,2], color='magenta', marker='d', label='median y=1 genZ' )
	#
	# for yi's=0 - True Positives & False Positives with Generation Z.
	plt.errorbar(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_truZ_accPiShift[:,0,0], yerr=pyi_gvnZ_stats_truZ_accPiShift[:,0,1], color='yellow', \
		barsabove=True, linestyle='solid', capsize=5, alpha=0.8, label=str('$\mu$ & $\sigma$ y=0 genZ' ) ) # Mean & STD (acc SWs) of mean (acc yis).
	plt.scatter(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_truZ_accPiShift[:,0,2], color='yellow', marker='o', label='median y=0 genZ' )
	#
	# for yi's=1 - True Positives & False Positives with Inference Z.
	plt.errorbar(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_infZ_accPiShift[:,3,0], yerr=pyi_gvnZ_stats_infZ_accPiShift[:,3,1], color='blue', \
		barsabove=True, linestyle='solid', capsize=5, alpha=0.5, label=str('$\mu$, $\sigma$ y=1 infZ' ) ) # Mean & STD (acc SWs) of mean (acc yis).
	plt.scatter(np.arange(len(Pi_shifts)), pyi_gvnZ_auc_accPiShift[:,3,2], color='red', marker='d', label='median y=1 infZ' )
	#
	# for yi's=0 - True Negatives & False Negatives with Inference Z.
	plt.errorbar(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_infZ_accPiShift[:,0,0], yerr=pyi_gvnZ_stats_infZ_accPiShift[:,0,1], color='green', \
		barsabove=True, linestyle='dashed', capsize=5, alpha=0.5, label=str('$\mu$ & $\sigma$ y=0 infZ' ) ) # Mean & STD (acc SWs) of mean (acc yis).
	plt.scatter(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_infZ_accPiShift[:,0,2], color='cyan', marker='x', label='median y=0 infZ' )
	#
	plt.xticks( np.arange(len(Pi_shifts)), Pi_shifts )
	plt.ylabel( str('p($y_i={0,1}|z$)') )
	plt.xlabel( str('Noisier <-- Pi Shift --> Quieter') )
	plt.title( "STATISTICS ACROSS SW samples of mean across $y_i$'s" )
	plt.legend()
	plt.grid()
	plt.show()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Plot errorbars of STATISTICS ACROSS CELLS IN Y-VECTOR for 
# mean across sampled spike words for different Pi_shift values.
#
# (1). pyi_gvnZ_stats_infZ_accPiShift
# (2). pyi_gvnZ_stats_truZ_accPiShift (meh?)
# 		---------------
# 		size = (7,6,3) 
#		  7 = Pi_shifts
# 		  6 = mean,std,median across cells on or off. {0-2 for y=0 and 3-5 for y=1}
#		  3 = mean,std,median across all num_PL_SWs samples.
#	
if True:
	#
	# for yi's=1 - True Positives & False Positives with Generation Z.
	plt.errorbar(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_truZ_accPiShift[:,3,0], yerr=pyi_gvnZ_stats_truZ_accPiShift[:,4,0], color='black', \
		barsabove=True, linestyle='solid', capsize=5, alpha=0.5, label=str('$\mu$, $\sigma$ y=1 genZ' ) ) # Mean & STD (acc SWs) of mean (acc yis).
	plt.scatter(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_truZ_accPiShift[:,5,0], color='magenta', marker='d', label='median y=1 genZ' )
	#
	# for yi's=0 - True Negatives & False Negatives with Generation Z.
	plt.errorbar(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_truZ_accPiShift[:,0,0], yerr=pyi_gvnZ_stats_truZ_accPiShift[:,1,0], color='yellow', \
		barsabove=True, linestyle='dashed', capsize=5, alpha=0.8, label=str('$\mu$ & $\sigma$ y=0 genZ' ) ) # Mean & STD (acc SWs) of mean (acc yis).
	plt.scatter(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_truZ_accPiShift[:,2,0], color='yellow', marker='x', label='median y=0 genZ' )
	#
	# for yi's=1 - True Positives & False Positives with Inference Z.
	plt.errorbar(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_infZ_accPiShift[:,3,0], yerr=pyi_gvnZ_stats_infZ_accPiShift[:,4,0], color='blue', \
		barsabove=True, linestyle='solid', capsize=5, alpha=0.5, label=str('$\mu$ & $\sigma$ y=1 infZ' ) ) # Mean & STD (acc SWs) of mean (acc yis).
	plt.scatter(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_infZ_accPiShift[:,5,0], color='red', marker='o', label='median y=1 infZ' )
	#
	# for yi's=0 - True Negatives & False Negatives with Inference Z.
	plt.errorbar(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_infZ_accPiShift[:,0,0], yerr=pyi_gvnZ_stats_infZ_accPiShift[:,1,0], color='green', \
		barsabove=True, linestyle='dashed', capsize=5, alpha=0.5, label=str('$\mu$ & $\sigma$ y=0 infZ' ) ) # Mean & STD (acc SWs) of mean (acc yis).
	plt.scatter(np.arange(len(Pi_shifts)), pyi_gvnZ_stats_infZ_accPiShift[:,2,0], color='cyan', marker='x', label='median y=0 infZ' )
	#
	plt.xticks( np.arange(len(Pi_shifts)), Pi_shifts )
	plt.ylabel( str('p($y_i={0,1}|z$)') )
	plt.xlabel( str('Noisier <-- Pi Shift --> Quieter') )
	plt.title( "Mean across SW samples of STATISTICS ACROSS $Y_i$'s" )
	plt.legend()
	plt.grid()
	plt.show()	