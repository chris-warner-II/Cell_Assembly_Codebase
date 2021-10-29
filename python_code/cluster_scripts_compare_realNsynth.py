import numpy as np
import os
import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.sbatch_scripts as ss



#
dirHomeLoc, dirScratch = dm.set_dir_tree()



# Params shared by both Real and Synthetic Data.
#
yLo			= 0		# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<=yLo.
yMinSW 		= 1 # [1, 3] # [1,2,3]

flg_include_Zeq0_infer = True
verbose = False

synthSamps = 50000


# # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # 
#
# Parameters for Real Data and Synthetic Model
#
cellSubTypes = [ '[offBriskTransient]','[offBriskSustained]','[onBriskTransient]', \
	 	'[offBriskTransient,offBriskSustained]', '[offBriskTransient,onBriskTransient]']


cellSubTypes = ['[offBriskTransient,offBriskSustained]', '[offBriskTransient,onBriskTransient]']

Ns = [98, 94]#[55, 43, 39, 98, 94]
Ms = Ns
numSWs_MovS = [809134, 717525]#[571276, 559505, 353174, 809134, 717525]
numSWs_WnzS = [852944, 861660]#[740267, 422947, 440264, 852944, 861660]
#
SW_binS =  [2,1,0] 	# ms. Build up spikewords from groups of cells in the same trial that fire within SW_bins of eachother.

# 
# #
#

yHiR = 300
yHi  = 1000 	# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution.
#
K_iter 			= [1,2,3,1,2,3,4,1,2,3,4,5]	# Number of	cell assemblies active, given a Bernoulli distribution ("hotness" of Z-vector)
Kmin_iter 		= [0,0,0,0,0,0,0,0,0,0,0,0]	# Max & Min number of cell assemblies active 
Kmax_iter 		= [3,3,3,4,4,4,4,5,5,5,5,5]
#
C_iter 			= [2,3,4,5,6]	# number of cells participating in cell assemblies. ("hotness" of each row in Pia)
Cmin_iter		= [2,2,2,2,2]	# Max & Min number of cell active to call it a cell assembly
Cmax_iter		= [6,6,6,6,6]
#
mu_Pia_iter		= [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1, 0.0]   	# Binary values in Pia matrix made continuous by sampling from Gaussian centered at 1-mu_Pia or 0+mu_Pia.
sig_Pia_iter	= [0.05, 0.1] 		# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
#
bernoulli_Pi	= 1.0   		# Each value in binary Pi vector randomly sampled from bernoulli with p(Pi=1) = bernoulli_Pi.	
mu_Pi_iter		= [0.02, 0.04]   	# Binary values in Pi vector made continuous by sampling from Gaussian centered at 1-mu_Pi or 0+mu_Pi.
sig_Pi_iter		= [0.01, 0.02] 		# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.


print('On each job, we are grid searching this many parameter combinations.')
print( len(K_iter)*len(C_iter)*len(mu_Pia_iter)*len(sig_Pia_iter)*len(mu_Pi_iter)*len(sig_Pi_iter) )
# tops should be like 5000 parameter combos I think. Because:
# 70sec*5000/3600/24 = 4 days and 30sec*5000/3600/24 = 2 days




what_to_run = 'cRvS'

# Flags for running on clusters (nersc or cortex)
whichClust 	= 'cortex' 		# 'cortex' or 'nersc'.
n_cores 	= 1 			# number of CPU's to run on.
time 		= '50-00:00:00' 	# dd-hhhh:mm:ss

mem = False# 7500 	# False # amount of memory (in MB) to allocate per CPU for each job submitted to cluster. pgmR needs more memory when running for 500k EM samples.
			# set to False to not specify amount of memory to use.





# Set up directory structure and make dirs if they dont exist already.
if whichClust=='nersc':
	homeDir = '/global/homes/w/warner/'
if whichClust=='cortex':
	homeDir = '/global/home/users/cwarner/'
cluster_path = str( homeDir + 'Projects/G_Field_Retinal_Data/cluster_scripts/compare_realNsynth/')
#	
base_path = str(dirHomeLoc + 'cluster_scripts/compare_realNsynth/')
if not os.path.exists(base_path):
	os.makedirs(base_path)



# Loop through all parameters and write sbatch scripts to submit jobs to the cluster.
wrapper = open( str(base_path + what_to_run + '.wrap') , 'w') # open wrapper file to call all the sbatch scripts written in the loop.

i=0

for SW_bin in SW_binS:
	#
	for indReal in range(len(cellSubTypes)):
		CST = cellSubTypes[indReal]
		N = Ns[indReal]
		M = Ms[indReal]
		M_mod	= M
		numSWs_Mov = numSWs_MovS[indReal]
		numSWs_Wnz = numSWs_WnzS[indReal]
		#
		print('Real Data:  ', CST, str(1+2*SW_bin), 'msBins')

		i+=1
		fname = str( what_to_run + str(i) )
		script_path = str(base_path    + fname + '.s')
		output_path = str(cluster_path + 'out_files/' + fname + '.o')
		error_path  = str(cluster_path + 'out_files/' + fname + '.e')

		#
		# write each sbatch call in wrapper file that I can just call once. NOTE: directory here is dir on NERSC cluster.
		wrapper.write( str( 'sbatch ' + cluster_path + fname + '.s \n') )	
		#
		ss.write_sbatch_script_compare_SWdists_realNsynth(script_path=script_path, output_path=output_path, error_path=error_path, 
			whichClust=whichClust, n_cores=n_cores, mem=mem, time=time, job_name=fname, CST=CST, N=N, M=M, M_mod=M_mod, 
			numSWs_Mov=numSWs_Mov, numSWs_Wnz=numSWs_Wnz, SW_bin=SW_bin, synthSamps=synthSamps, K_iter=K_iter, Kmin_iter=Kmin_iter, 
			Kmax_iter=Kmax_iter, C_iter=C_iter, Cmin_iter=Cmin_iter, Cmax_iter=Cmax_iter, mu_Pia_iter=mu_Pia_iter, sig_Pia_iter=sig_Pia_iter, 
			mu_Pi_iter=mu_Pi_iter, sig_Pi_iter=sig_Pi_iter, bernoulli_Pi=bernoulli_Pi, yHiR=yHiR, yHi=yHi, yLo=yLo, yMinSW=yMinSW, 
			flg_include_Zeq0_infer=flg_include_Zeq0_infer, verbose=verbose)
		

		
wrapper.close()	# close wrapper file				
print( str('made ' + str(i) + ' sbatch job scripts'  ) )		
print('Each job grid searches over',len(K_iter)*len(C_iter)*len(mu_Pia_iter)*len(sig_Pia_iter)*len(mu_Pi_iter)*len(sig_Pi_iter), ' parameter combinations.')













# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# #
# # # Params shared by both Real and Synthetic Data.
# # #
# # yLo			= 0		# Only look at spikewords that have more active cells than yLo for EM learning. That is, ignore |y|<=yLo.
# # yMinSW 		= 1 # [1, 3] # [1,2,3]

# # flg_include_Zeq0_infer = True
# # if flg_include_Zeq0_infer:
# # 	z0_tag='_zeq0'
# # else:
# # 	z0_tag='_zneq0'

# # plot_distributions = False
# # verbose = True

# # samps = 50000


# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # #
# # # Parameters for Real Data and Synthetic Model
# # #
# # cellSubTypes = [ '[offBriskTransient]','[offBriskSustained]','[onBriskTransient]', \
# # 	 	'[offBriskTransient,offBriskSustained]', '[offBriskTransient,onBriskTransient]']
# # Ns = [55, 43, 39, 98, 94]
# # Ms = Ns
# # numSWs_MovS = [571276, 559505, 353174, 809134, 717525]
# # numSWs_WnzS = [740267, 422947, 440264, 852944, 861660]
# # #
# # SW_binS =  [2,1,0] 	# ms. Build up spikewords from groups of cells in the same trial that fire within SW_bins of eachother.

# # # 
# # # #
# # #

# # yHiR = 300
# # yHi 	= 1000 	# If |y|>=yHi, then we assume at least 1 CA is on and disallow z=0 inference solution.
# # #
# # K_iter 			= [1,2,3,2,3,4]	# Number of	cell assemblies active, given a Bernoulli distribution ("hotness" of Z-vector)
# # Kmin_iter 		= [0,0,0,0,0,0]	# Max & Min number of cell assemblies active 
# # Kmax_iter 		= [3,3,3,4,4,4]
# # #
# # C_iter 			= [2,3,4,5,6]	# number of cells participating in cell assemblies. ("hotness" of each row in Pia)
# # Cmin_iter		= [2,2,2,2,2]	# Max & Min number of cell active to call it a cell assembly
# # Cmax_iter		= [6,6,6,6,6]
# # #
# # mu_Pia_iter		= [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]   	# Binary values in Pia matrix made continuous by sampling from Gaussian centered at 1-mu_Pia or 0+mu_Pia.
# # sig_Pia_iter	= [0.05] 		# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
# # #
# # bernoulli_Pi	= 1.0   	# Each value in binary Pi vector randomly sampled from bernoulli with p(Pi=1) = bernoulli_Pi.	
# # mu_Pi_iter		= [0.02]   	# Binary values in Pi vector made continuous by sampling from Gaussian centered at 1-mu_Pi or 0+mu_Pi.
# # sig_Pi_iter		= [0.01] # [0.01, 0.02, 0.03] 		# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.






# # Loop through different real data files and for each of them loop through all the synth parameter sweeps.
# for SW_bin in SW_binS:
# 	#
# 	for indReal in range(2,len(cellSubTypes)):
# 		CST = cellSubTypes[indReal]
# 		N = Ns[indReal]
# 		M = Ms[indReal]
# 		M_mod	= M
# 		numSWs_Mov = numSWs_MovS[indReal]
# 		numSWs_Wnz = numSWs_WnzS[indReal]
# 		#
# 		print('Real Data:  ', CST, str(1+2*SW_bin), 'msBins')

# 		# # # # # # # # # # # # # # # # # # # # # # # # # 
# 		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 		# # # # # # # # # # # # # # # # # # # # # # # # # 
# 		#
# 		# Load in Spike Words Extracted from real retinal data.
# 		print('Load in Spike Words Extracted from real retinal data.')
# 		t0 = time.time()
# 		#
# 		try:
# 			# Try to load in stats directly that were computed on ALL spike words. From stats files in SpikeWords Extracted directory
# 			fname_SWstats_Wnz = str( realData_Dir + str(CST) + '_Wnoise_' + str(1+2*SW_bin) + 'msBins_yMin' + str(yMinSW) + '_SWstats.npz' )
# 			data_Wnz = np.load(fname_SWstats_Wnz)
# 			nYr_Wnz = data_Wnz['nY'] 
# 			Ycell_histR_Wnz = data_Wnz['Ycell_hist'] 
# 			Cell_coactivityR_Wnz = data_Wnz['Cell_coactivity']
# 			del data_Wnz
# 		except:
# 			# Try to load in stats directly that were computed on ALL spike words. From rasterZ_allSWs files.
# 			fname_raster_Wnz = str( realRaster_dir + 'Init_DiagonalPia_2hot_mPi1.0_mPia1.0_sI[0.01 0.05 0.05]_LR0pt1_LRpi1pt0/' + \
# 				str(CST) + '_N' + str(N) +'_M' + str(M) + z0_tag + '_ylims' + str(yLo) + '_' + str(yHiR) + '_yMinSW' + str(yMinSW) + '/' + \
# 				'rasterZ_allSWs_Wnoise_' + str(numSWs_Wnz) + 'SWs_0pt5trn_' + str(1+2*SW_bin) + 'msBins_rand0.npz')
# 			data_RasWnz = np.load(fname_raster_Wnz)
# 			nYr_Wnz = data_RasWnz['nY_allSWs'] 
# 			Ycell_histR_Wnz = data_RasWnz['Ycell_hist_allSWs'] 
# 			Cell_coactivityR_Wnz = data_RasWnz['Cell_coactivity_allSWs']
# 			del data_RasWnz
# 		#
# 		# # #
# 		#
# 		try:
# 			fname_SWstats_Mov = str( realData_Dir + str(CST) + '_NatMov_' + str(1+2*SW_bin) + 'msBins_yMin' + str(yMinSW) + '_SWstats.npz' )
# 			data_Mov = np.load(fname_SWstats_Mov)
# 			nYr_Mov = data_Mov['nY'] 
# 			Ycell_histR_Mov = data_Mov['Ycell_hist'] 
# 			Cell_coactivityR_Mov = data_Mov['Cell_coactivity']
# 			del data_Mov
# 		except:
# 			fname_raster_Mov = str( realRaster_dir + 'Init_DiagonalPia_2hot_mPi1.0_mPia1.0_sI[0.01 0.05 0.05]_LR0pt1_LRpi1pt0/' + \
# 				str(CST) + '_N' + str(N) +'_M' + str(M) + z0_tag + '_ylims' + str(yLo) + '_' + str(yHiR) + '_yMinSW' + str(yMinSW) + '/' + \
# 				'rasterZ_allSWs_NatMov_' + str(numSWs_Mov) + 'SWs_0pt5trn_' + str(1+2*SW_bin) + 'msBins_rand0.npz')
# 			data_RasMov = np.load(fname_raster_Mov)
# 			nYr_Mov = data_RasMov['nY_allSWs'] 
# 			Ycell_histR_Mov = data_RasMov['Ycell_hist_allSWs'] 
# 			Cell_coactivityR_Mov = data_RasMov['Cell_coactivity_allSWs']
# 			del data_RasMov
# 		#
# 		t1 = time.time()
# 		print('Time: ',t1-t0)







# 		# Multidimensional matrices to hold results of distance from unity line on QQ-plot.
# 		QQ_yc_sm = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for |y| distributions for synth and movie
# 		QQ_yc_sn = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for |y| distributions for synth and noise
# 		#
# 		QQ_y1_sm = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for single y activity distributions for synth and movie
# 		QQ_y1_sn = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for single y activity distributions for synth and noise
# 		#
# 		QQ_y2_sm = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for pairwise y coactivity distributions for synth and movie
# 		QQ_y2_sn = np.zeros( [len(K_iter), len(C_iter), len(mu_Pia_iter), len(sig_Pia_iter), len(mu_Pi_iter), len(sig_Pi_iter)] ) # to hold KS test from QQ plots for pairwise y coactivity distributions for synth and noise

# 		print('Number of tests to run is: ',QQ_yc_sm.size)




# 		# # # # # # # # # # # # # # # # # # # # # # # # # 
# 		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 		# # # # # # # # # # # # # # # # # # # # # # # # # 
# 		#
# 		# Loop through the different parameters for synthetic model data generation.
# 		#
# 		#
# 		for Kind in range(len(K_iter)):	# Number of	cell assemblies active, given a Bernoulli distribution ("hotness" of Z-vector)
# 			K = K_iter[Kind]
# 			Kmin = Kmin_iter[Kind]
# 			Kmax = Kmax_iter[Kind]
# 			#
# 			for Cind in range(len(C_iter)):	# Number of cells participating in cell assemblies. ("hotness" of each row in Pia)
# 				C = C_iter[Cind]
# 				Cmin = Cmin_iter[Cind]
# 				Cmax = Cmax_iter[Cind]
# 				#
# 				for a,mu_Pia in enumerate(mu_Pia_iter): 	# Binary values in Pia matrix made continuous by sampling from Gaussian centered at 1-mu_Pia or 0+mu_Pia.
# 					for b,sig_Pia in enumerate(sig_Pia_iter):	# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.
# 						#
# 						for c,mu_Pi in enumerate(mu_Pi_iter): 	# Binary values in Pi vector made continuous by sampling from Gaussian centered at 1-mu_Pi or 0+mu_Pi.
# 							for d,sig_Pi in enumerate(sig_Pi_iter): 	# STD of Gaussian mentioned above. Note that values <0 or >1 are resampled because Pia values are probabilities.


# 								# # # # # # # # # # # # # # # # # # # # # # # # # 
# 								# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 								# # # # # # # # # # # # # # # # # # # # # # # # # 
# 								#
# 								# Construct synthetic model and generate spike words data.
# 								print('Construct synthetic model.')
# 								#
# 								t0 = time.time()
# 								q = None #ri = None #ria = None #ria_mod = None
# 								while q==None: # while statement to resynthesize model if any cells participate in 0 assemblies.
# 									q, ri, ria, ria_mod = rc.synthesize_model(N,M,M_mod, C,K, Cmin,Cmax, bernoulli_Pi,mu_Pi,sig_Pi, mu_Pia,sig_Pia, verbose=False)
# 								#
# 								t1 = time.time()
# 								print('Time: ',t1-t0)
# 								#
# 								print('Generate spike words data synth model')
# 								t0 = time.time()
# 								Y_synth, Z_synth = rc.generate_data(samps, q, ri, ria, Cmin, Cmax, Kmin, Kmax)
# 								t1 = time.time()
# 								print('Time: ',t1-t0)

# 								print('compute_dataGen_Histograms on synth data.')
# 								t0 = time.time()
# 								Ycell_histS, Zassem_histS, nYs, nZs, CA_coactivityS, Cell_coactivityS = \
# 									rc.compute_dataGen_Histograms( Y_synth, Z_synth, M, N )
# 								t1 = time.time()
# 								print('Time: ',t1-t0)


# 								# # # # # # # # # # # # # # # # # # # # # # # # # 
# 								# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 								# # # # # # # # # # # # # # # # # # # # # # # # # 
# 								#
# 								# Histogram distributions and compute KL-divergence similarity measure. 
# 								#
# 								print('compute KL divergence between Synth, Movie, Noise, |y|, single activity, and pairwise coactivity.')
# 								t0 = time.time()


# 								yc_sm, yc_sn, yc_nm, y1_sm, y1_sn, y1_nm, y2_sm, y2_sn, y2_nm, \
# 								mx1, nx1, sx1, mx2, nx2, sx2, mx3, nx3, sx3, \
# 								m1, n1, s1, m2, n2, s2, m3, n3, s3 = rc.compute_QQ_diff_metric( \
# 									nYr_Mov, nYr_Wnz, nYs, Ycell_histR_Mov[:-1], Ycell_histR_Wnz[:-1], Ycell_histS,  \
# 									Cell_coactivityR_Mov, Cell_coactivityR_Wnz, Cell_coactivityS)
# 								#
# 								QQ_yc_sm[Kind,Cind,a,b,c,d] = yc_sm
# 								QQ_yc_sn[Kind,Cind,a,b,c,d] = yc_sn
# 								#
# 								QQ_y1_sm[Kind,Cind,a,b,c,d] = y1_sm
# 								QQ_y1_sn[Kind,Cind,a,b,c,d] = y1_sn
# 								#
# 								QQ_y2_sm[Kind,Cind,a,b,c,d] = y2_sm
# 								QQ_y2_sn[Kind,Cind,a,b,c,d] = y2_sn
# 								#
# 								t1 = time.time()
# 								print('Time: ',t1-t0)

							
# 								# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 								# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 								# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 								#
# 								# QQ - plots. (These are now done for best fitting parameters in 
# 								# 					plot_SWdists_bestFit_SynthParams2Real.py)
# 								#
# 								plot_CDFs_n_PDFs = False
# 								if plot_CDFs_n_PDFs:
# 									#
# 									# input m1, n1, s1, m2, n2, s2, m3, n3, s3,
# 									#		mx1, nx1, sx1, mx2, nx2, sx2, mx3, nx3, sx3,


# 									f = plt.figure( figsize=(20,5) ) # size units in inches   
# 									plt.rc('font', weight='bold', size=10)
									

# 									# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 									#
# 									ax1 = plt.subplot2grid((2,3),(0,0)) 
# 									ax1.plot( np.array([0, 1]),np.array([0, 1]), 'k--', linewidth=2,  )
# 									ax1.plot( mx1, nx1, 'co-', linewidth=2, label='Mov v Wnz' )
# 									ax1.plot( sx1, nx1,	'gx-', linewidth=2, label='Syn v Wnz' )
# 									ax1.plot( sx1, mx1, 'bx-', linewidth=2, label='Syn v Mov' )
# 									ax1.text( 1, 0.0, str(QQ_yc_sm[Kind,Cind,a,b,c,d].round(3)), color='blue', horizontalalignment='right', verticalalignment='bottom' )	
# 									ax1.text( 1, 0.05, str(QQ_yc_sn[Kind,Cind,a,b,c,d].round(3)), color='green', horizontalalignment='right', verticalalignment='bottom' )	
# 									#ax1.text( 0.2, 1, str(QQ_yc_nm[Kind,Cind,a,b,c,d].round(3)), color='cyan', horizontalalignment='right', verticalalignment='bottom' )	
# 									ax1.set_title( '|y| CDFs' )
# 									#
# 									ax1b = plt.subplot2grid((2,3),(1,0)) 
# 									ax1b.plot( m1[1][1:], m1[0]/m1[0].sum(), 'bo-', linewidth=2, label='Mov' )
# 									ax1b.plot( n1[1][1:], n1[0]/n1[0].sum(), 'gx-', linewidth=2, label='Wnz' )
# 									ax1b.plot( s1[1][1:], s1[0]/s1[0].sum(), 'rx-', linewidth=2, label='Syn' )
# 									ax1b.legend()
# 									ax1b.set_title( '|y| PDFs' )
# 									#



# 									# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 									#
# 									ax2 = plt.subplot2grid((2,3),(0,1)) 
# 									ax2.plot( np.array([0, 1]),np.array([0, 1]), 'k--', linewidth=2,  )
# 									ax2.plot( mx2, nx2, 'co-', linewidth=2, label='Mov v Wnz' )
# 									ax2.plot( sx2, nx2,	'gx-', linewidth=2, label='Syn v Wnz' )
# 									ax2.plot( sx2, mx2, 'bx-', linewidth=2, label='Syn v Mov' )
# 									# ax2.plot( np.cumsum(Ycell_histR_Mov[indSortYR_Mov]/Ycell_histR_Mov.sum()), 	np.cumsum(Ycell_histR_Wnz[indSortYR_Wnz]/Ycell_histR_Wnz.sum()), 	'ro-', linewidth=2, label='NatMov vs Wnoise' )
# 									# ax2.plot( np.cumsum(Ycell_histS[indSortYS]/Ycell_histS.sum()), 				np.cumsum(Ycell_histR_Wnz[indSortYR_Wnz]/Ycell_histR_Wnz.sum()),	'gx-', linewidth=2, label='Synth vs Wnoise' )
# 									# ax2.plot( np.cumsum(Ycell_histR_Mov[indSortYR_Mov]/Ycell_histR_Mov.sum()), 	np.cumsum(Ycell_histS[indSortYS]/Ycell_histS.sum()),				'bx-', linewidth=2, label='NatMov vs Synth' )
# 									ax2.text( 1, 0.0, str(QQ_y1_sm[Kind,Cind,a,b,c,d].round(3)), color='blue', horizontalalignment='right', verticalalignment='bottom' )	
# 									ax2.text( 1, 0.05, str(QQ_y1_sn[Kind,Cind,a,b,c,d].round(3)), color='green', horizontalalignment='right', verticalalignment='bottom' )	
# 									#ax2.text( 0.2, 1, str(QQ_yc_nm[Kind,Cind,a,b,c,d].round(3)), color='red', horizontalalignment='right', verticalalignment='bottom' )	
# 									ax2.set_title( 'single y activity CDFs' )
# 									ax2.legend()
# 									#
# 									ax2b = plt.subplot2grid((2,3),(1,1)) 
# 									ax2b.plot( np.arange(N), m2[0]/m2[0].sum(), 'bo-', linewidth=2, label='Mov' )
# 									ax2b.plot( np.arange(N), n2[0]/n2[0].sum(), 'gx-', linewidth=2, label='Wnz' )
# 									ax2b.plot( np.arange(N), s2[0]/s2[0].sum(), 'rx-', linewidth=2, label='Syn' )
# 									ax2b.legend()
# 									ax2b.set_title( 'single y activity PDFs' )
# 									#


# 									# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 									#
# 									ax3 = plt.subplot2grid((2,3),(0,2)) 
# 									ax3.plot( np.array([0, 1]),np.array([0, 1]), 'k--', linewidth=2,  )
# 									ax3.plot( np.cumsum(m3[0]/(N**2 - N)), np.cumsum(n3[0]/(N**2 - N)), 'ro-', linewidth=2, label='NatMov vs Wnoise' )
# 									ax3.plot( np.cumsum(s3[0]/(N**2 - N)), np.cumsum(n3[0]/(N**2 - N)),	'gx-', linewidth=2, label='Synth vs Wnoise' )
# 									ax3.plot( np.cumsum(m3[0]/(N**2 - N)), np.cumsum(s3[0]/(N**2 - N)),	'bx-', linewidth=2, label='NatMov vs Synth' )
# 									ax3.text( 1, 0.0, str(QQ_y2_sm[Kind,Cind,a,b,c,d].round(3)), color='blue', horizontalalignment='right', verticalalignment='bottom' )	
# 									ax3.text( 1, 0.05, str(QQ_y2_sn[Kind,Cind,a,b,c,d].round(3)), color='green', horizontalalignment='right', verticalalignment='bottom' )	
# 									#ax3.text( 0.2, 1, str(QQ_yc_nm[Kind,Cind,a,b,c,d].round(3)), color='red', horizontalalignment='right', verticalalignment='bottom' )	
# 									ax3.set_title( 'pairwise y coactivity CDFs' )
# 									#
# 									ax3b = plt.subplot2grid((2,3),(1,2)) 
# 									ax3b.plot( m3[1][1:], m3[0]/m3[0].sum(), 'bo-', linewidth=2, label='Mov' )
# 									ax3b.plot( n3[1][1:], n3[0]/n3[0].sum(), 'gx-', linewidth=2, label='Wnz' )
# 									ax3b.plot( s3[1][1:], s3[0]/s3[0].sum(), 'rx-', linewidth=2, label='Syn' )
# 									ax3b.legend()
# 									ax3b.set_title( 'pairwise y coactivity PDFs' )
# 									#

# 									#
# 									plt.show()




# 		# # #
# 		#
# 		# Save a file for each real data combo.
# 		#
# 		fname_str = str( CST+'_'+str(1+2*SW_bin)+'msBins_'+str(len(K_iter))+'Ks_'+str(len(C_iter))+'Cs_'+str(len(mu_Pia_iter)) \
# 			+'m'+str(len(sig_Pia_iter))+'sPiaS_'+str(len(mu_Pi_iter))+'m'+str(len(sig_Pi_iter))+'sPiS_'+str(samps)+'SWs' )
# 		#
# 		np.savez( str(dat_save_dir+fname_str), 
# 			QQ_yc_sm=QQ_yc_sm, QQ_yc_sn=QQ_yc_sn, 
# 			QQ_y1_sm=QQ_y1_sm, QQ_y1_sn=QQ_y1_sn, 
# 			QQ_y2_sm=QQ_y2_sm, QQ_y2_sn=QQ_y2_sn, 
# 			K_iter=K_iter, Kmin_iter=Kmin_iter, Kmax_iter=Kmax_iter, 
# 			C_iter=C_iter, Cmin_iter=Cmin_iter, Cmax_iter=Cmax_iter,
# 			mu_Pia_iter=mu_Pia_iter, sig_Pia_iter=sig_Pia_iter, 
# 			mu_Pi_iter=mu_Pi_iter, sig_Pi_iter=sig_Pi_iter)
								
			



# 		# # #
# 		#
# 		# Sort through matrices of QQ measurements and grab the top couple.
# 		# 		(This are now done for best fitting parameters in 
# 		# 			plot_SWdists_bestFit_SynthParams2Real.py)
# 		if False:
# 			w = np.array([1., 1., 1.])
# 			measure = w[0]*QQ_yc_sm + w[1]*QQ_y1_sm + w[2]*QQ_y2_sm

# 			xx = np.sort(measure, axis=None)
# 			nBest = 3
# 			for ii in range(nBest): 
# 				indx = np.where(measure == xx[ii])
# 				#
# 				print('K = {', K_iter[ indx[0][0] ], ',', Kmin_iter[ indx[0][0] ], ',', Kmax_iter[ indx[0][0] ],'}', \
# 					'  C = {', C_iter[ indx[1][0] ], ',', Cmin_iter[ indx[1][0] ], ',', Cmax_iter[ indx[1][0] ],'}', \
# 					'  Pia (m,s) = {', mu_Pia_iter[ indx[2][0] ], ',', sig_Pia_iter[ indx[3][0] ], '}', \
# 					'  Pi (m,s) = {', mu_Pi_iter[ indx[4][0] ], ', ', sig_Pi_iter[ indx[5][0] ], '}')



