#script_to_find_bad_npz.py

dtype = 'synth' # 'real' or 'synth'

initSpecifiers = 'NoisyConst_5hot_mPi1.0_mPia1.0_sI' #[0.01 0.05 0.05]_LR0pt5_LRsc[1.0, 0.1, 0.1]'



import os
import fnmatch
import numpy as np

import utils.data_manipulation as dm # utils is a package I am putting together of useful functions

dirHome, dirScratch = dm.set_dir_tree()

if dtype=='synth':
	EM_data_dir = str( dirScratch + 'data/python_data/PGM_analysis/synthData/Models_learned_EM/' )
	#


	synthKeys = ['q', 'ri', 'ria', 'ria_mod', 'qp', 'rip', 'riap', 'q_init', 'ri_init', 'ria_init', \
		#
		'Q_SE', 'Pi_SE', 'Pi_AE', 'Pia_AE', 'PiaOn_SE', 'PiaOn_AE', 'PiaOff_SE', 'PiaOff_AE', 'num_dpSig_snaps', \
		'q_deriv', 'ri_deriv', 'ria_deriv', 'ria_snapshots', 'ri_snapshots', 'q_snapshots', 'ds_fctr_snapshots', \
		#
		'Z_inferred_train', 'pyiEq1_gvnZ_train', 'Y_train', 'Z_train', \
		'Z_inferred_test', 'pyiEq1_gvnZ_test', 'Y_test', 'Z_test', \
		#
		'pj_zHyp_train', 'pj_zHyp_test', 'pj_zTru_Batch', 'pj_zTrain_Batch', 'pj_zTest_Batch', \
		'cond_zHyp_train', 'cond_zHyp_test', 'cond_zTru_Batch', 'cond_zTrain_Batch', 'cond_zTest_Batch', \
		#
		'ind_matchGT2Mod', 'cosSim_matchGT2Mod', 'lenDif_matchGT2Mod', 'cosSimMat_matchGT2Mod', 'lenDifMat_matchGT2Mod', \
		'ind_matchMod2GT', 'cosSim_matchMod2GT', 'lenDif_matchMod2GT', 'cosSimMat_matchMod2GT', 'lenDifMat_matchMod2GT', \
		'xVal_snapshot', 'xVal_batchSize', 'zActivationHistory', 'argsRec']

	if all (k in data.keys() for k in synthKeys):
		print('Good file: ',fn)	



elif dtype=='real':
	EM_data_dir = str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/Models_learned_EM/' )
else:
	print('error w dtype')






InitDirs = fnmatch.filter(os.listdir(EM_data_dir),str('Init*'+initSpecifiers+'*') )

for ID in InitDirs:

	if dtype=='synth':
		SynDDir = fnmatch.filter(os.listdir( str(EM_data_dir + ID)  ),'SyntheticData*')
	elif dtype=='real':
		SynDDir = fnmatch.filter(os.listdir( str(EM_data_dir + ID)  ),'[*')
	else:
		print('error w dtype')

	for SD in SynDDir:

		print(ID, SD)

		if dtype=='synth':
			fnames = fnmatch.filter(os.listdir( str(EM_data_dir + ID + '/' + SD)  ),'EM_model_data*')
		elif dtype=='real':
			fnames = fnmatch.filter(os.listdir( str(EM_data_dir + ID + '/' + SD)  ),'LearnedModel*')
		else:
			print('error w dtype')

		for fn in fnames:

			data = np.load( str(EM_data_dir + ID + '/' + SD + '/' + fn ) )

			if all (k in data.keys() for k in synthKeys):
				print('Good file: ',fn)

			else:
				print('')
				# print('old file: ',fn)
				# print('removing file')
				#os.remove( str(EM_data_dir + ID + '/' + SD + '/' + fn ) )

