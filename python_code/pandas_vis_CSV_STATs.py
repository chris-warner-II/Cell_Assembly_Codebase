import argparse
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sb
import itertools as it

from scipy import io as io
import matplotlib.pyplot as plt
import time 						# to time operations for code analysis
import os.path 						# to check if a file or directory exists already
import sys

import utils.data_manipulation as dm # utils is a package I am putting together of useful functions
import utils.plot_functions as pf
import utils.retina_computation as rc




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def plot_comparison_2D(df, ThingsCompared, casesToCompare, xAxToCompare, yAxToCompare, zAxToCompare):
	#
	# df, 				- pandas data frame
	# ThingsCompared, 	- list
	# casesToCompare, 	- list of lists
	# xAxToCompare, 	- list
	# yAxToCompare, 	- list

	# colors = ['red', 'blue', 'orange', 'green', 'magenta', 'black']
	alf=0.5

	elems = df.shape[0]

	# get all combinations of casesToCompare
	# if len(casesToCompare)>1:
	combins = list( it.product(*casesToCompare) )
	# else:
	# 	combins = casesToCompare[0]

	maxx = -1000
	miny =  1000
	max_sz = 200

	if len(zAxToCompare)>0:
		maxZ = np.array(df[zAxToCompare]).max()
		#print('maxZ = ',maxZ)


	for i in range( len(combins) ):
		result = np.ones(elems).astype(bool)
		#
		for j in range( len(ThingsCompared) ):
			result = np.bitwise_and( df[ThingsCompared[j]]==combins[i][j], result)
			#
		try:	
			x = np.array([ df[result][XX] for XX in xAxToCompare ]).sum(axis=0) # x axis
			y = np.array([ df[result][YY] for YY in yAxToCompare ]).sum(axis=0) # y axis
		except:
			print('meh')
		

		if len(x)>0 and len(y)>0:
			# print(maxx)
			# print(x)
			# print(x.max())
			maxx = np.max([maxx,x.max()])
			miny = np.min([miny,y.min()])

			if len(zAxToCompare)>0:
				#print([ df[result][ZZ] for ZZ in zAxToCompare ] )
				z = max_sz*np.array([ df[result][ZZ] for ZZ in zAxToCompare ]).sum(axis=0) # scatter marker size
			else:
				z = max_sz

			#print('z',z)	

			#
			plt.scatter(x,y, s=z, alpha=alf, label=str(combins[i]) ) #, color=colors[i] )
			try:
				plt.scatter( x.mean(), y.mean(), s=z.mean(), facecolors='none', edgecolors='black', linewidth=2, alpha=alf ) #, color=colors[i] )
			except:
				plt.scatter( x.mean(), y.mean(), s=max_sz, facecolors='none', edgecolors='black', linewidth=2, alpha=alf ) #, color=colors[i] )
			
			#
			try:
				xerr = np.std(x) # sp.stats.sem(x) # or 
				yerr = np.std(y) # sp.stats.sem(y) # or 
				plt.errorbar( x.mean(), y.mean(), yerr=yerr, xerr=xerr, linewidth=2, alpha=alf ) #, color=colors[i] )
			except:
				print('meh')
		
	if len(zAxToCompare)>0:
		plt.scatter( 0.8*maxx,1.2*miny, s=max_sz, facecolors='none', edgecolors='black', linewidth=2, alpha=alf )
		plt.scatter( 0.8*maxx,1.2*miny, s=.25*max_sz, facecolors='none', edgecolors='black', linewidth=2, alpha=alf )
		plt.text( 0.82*maxx,1.18*miny, str(zAxToCompare), fontsize=8 )
		plt.text( 0.82*maxx,1.20*miny, str( str(np.round(.25*maxZ,2))+'%'), fontsize=8 )
		plt.text( 0.82*maxx,1.22*miny, str( str(np.round(maxZ,2))+'%'), fontsize=8 )



	plt.title( str('Compare different ' + str(ThingsCompared) ) )
	plt.xlabel( str(xAxToCompare) )
	plt.ylabel( str(yAxToCompare) )
	plt.legend(fontsize=8) #, loc='lower left')
	plt.grid()
	#
	plt.show()




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# def hist_comparison_1D(df, ThingsCompared, casesToCompare, xAxToCompare, binns):

# 	# colors = ['red', 'blue', 'orange', 'green', 'magenta', 'black']
# 	alf=0.5

# 	elems = df.shape[0]

# 	# get all combinations of casesToCompare
# 	combins = list( it.product(*casesToCompare) )
# 	for i in range( len(combins) ):
# 		result = np.ones(elems).astype(bool)
# 		for j in range( len(ThingsCompared) ):
# 			result = np.bitwise_and( df[ThingsCompared[j]]==combins[i][j], result)

# 		x = np.array([ df[result][XX] for XX in xAxToCompare ]).sum(axis=0)
# 		#y = np.array([ df[result][YY] for YY in yAxToCompare ]).sum(axis=0)
# 		#
# 		plt.hist(x, alpha=alf, label=str(combins[i]) ) #, color=colors[i] )
# 		#plt.errorbar( x.mean(), y.mean(), yerr=y.std(), xerr=x.std(), alpha=alf ) #, color=colors[i] )
		

# 	plt.title( str('Compare different ' + str(ThingsCompared) ) )
# 	plt.xlabel(xAxToCompare)
# 	plt.legend()
# 	plt.grid()
# 	#
# 	plt.show()











# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
dirHome, dirScratch = dm.set_dir_tree()

dataType	= 'synth' # {'real' OR 'synth'}

if dataType=='synth':
	CSV_dir = str( dirScratch + 'data/python_data/PGM_analysis/synthData/InferStats_from_EM_learning/' )
elif dataType=='real':
	CSV_dir = str( dirScratch + 'data/python_data/PGM_analysis/realData/G_Field/InferStats_from_EM_learning/' )
else:
	print('data type not understood. Cant find CSV file.')	

XtraTag 	= '' #'_master'
CSV_fname = str( 'STATs_' + dataType + 'Data' + XtraTag + '.csv' )


df = pd.read_csv( str(CSV_dir + CSV_fname) )

print(df.keys())






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# Add some new keys to the data frame that are M*%|CAs| so
# that they turn %|CAs| into #|CAs|.
#
df['#|CAs|=0'] 		= (df['M']*df['%|CAs|=0']).astype(int)
df['#|CAs|=1'] 		= (df['M']*df['%|CAs|=1']).astype(int)
df['#|CAs|=2'] 		= (df['M']*df['%|CAs|=2']).astype(int)
df['#|CAs|>2&<6'] 	= (df['M']*df['%|CAs|>2&<6']).astype(int)
df['#|CAs|>=6'] 	= (df['M']*df['%|CAs|>=6']).astype(int)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # KEYS IN SYNTHETIC DATA CSV FILE. 			IN REAL DATA CSV FILE.
# # -------------------------------- 			----------------------
# -------------------------------------------------------------
# 'C', 										
# 'Cmax', 									'cell type',
# 'Cmin', 									'stim',
# 'K',										'msBins',
# 'Kmax', 									 
# 'Kmin', 
#
# -------------------------------------------------------------
# '# EM samps',								'# EM samps',   
# '# SWs', 									'# SWs',
# 'M', 										'M',
# 'N', 										'N',
# 'yHi', 		(MAYBE,NOT NECESSARY?)		'yHi',
# 'yLo', 		(MAYBE,NOT NECESSARY?)		'yLo',  
# 'init',  									'init',
# 'Pi mean init',							'Pi mean init',
# 'Pia mean init',							'Pia mean init',
# 'Pi std init', 							'Pi std init',
# 'Pia std init', 							'Pia std init',
# 'Q std init',								'Q std init',
# 'Zhot init',								'Zhot init',
# 'LR', 									'LR',
# 'LRxPi' 									'LRxPi'
# '% pj not nan', 							'% pj not nan',
# 'mean pj postLrn', 						'mean pj postLrn',
#
# -------------------------------------------------------------
# '%Pi>0.1',								'%Pi>0.1',
# '%|CAs|=0',								'%|CAs|=0', 
# '%|CAs|=1',								'%|CAs|=1', 
# '%|CAs|=2',								'%|CAs|=2', 
# '%|CAs|>2&<6', 							'%|CAs|>2&<6',
# '%|CAs|>=6'	 							'%|CAs|>=6',  
# '|CAs| max'	 							'|CAs| max'
# 'Q',										'Q',
# '2nd model', 	<-(implement?)				'2nd model', 
# 'max CA ovl'								'max CA ovl', 
# 'std CA ovl',								'std CA ovl', 
# 'mean CA ovl',							'mean CA ovl',
# '%times y=1 obs allSWs',					'%times y=1 obs allSWs',
# '%times z=0 inf allSWs', 					'%times z=0 inf allSWs', 
# '%times y=1 obs train', 					'%times y=1 obs train', 
# '%times z=0 inf train',					'%times z=0 inf train',
#
# -------------------------------------------------------------
# '# y Total Test',							'# y Total Test', 						
# '# y Total Train',						'# y Total Train',
# '# y Total postLrn', 						'# y Total postLrn',
# '% y Captured Test',						'% y Captured Test',
# '% y Captured Train', 					'% y Captured Train', 
# '% y Captured postLrn', 					'% y Captured postLrn', 
# '% y Extra Test',							'% y Extra Test',
# '% y Extra Train', 						'% y Extra Train', 
# '% y Extra postLrn', 						'% y Extra postLrn', 
# '% y Missed Test', 						'% y Missed Test', 						
# '% y Missed Train', 						'% y Missed Train', 
# '% y Missed postLrn', 					'% y Missed postLrn', 
# '|Y| overinferred mean', 					'|Y| overinferred mean',
# '|Y| overinferred std',					'|Y| overinferred std'
# '|Y| overinferred skew',					'|Y| overinferred skew'
# '|Y| inf mean',							'|Y| inf mean',
# '|Y| inf std',							'|Y| inf std',
# '|Y| inf skew',							'|Y| inf skew',
# '|Y| obs mean',							'|Y| obs mean',
# '|Y| obs std',							'|Y| obs std',
# '|Y| obs skew',							'|Y| obs skew',
# '|Z| inf mean',							'|Z| inf mean',
# '|Z| inf std',							'|Z| inf std',
# '|Z| inf skew',							'|Z| inf skew',
#
# -------------------------------------------------------------
# '# z Total Test', 
# '# z Total Train',
# '# z Total postLrn', 
# '% z Captured Test',						 ^
# '% z Captured Train', 					 |
# '% z Captured postLrn', 					N/A
# '% z Extra Test',							 |
# '% z Extra Train', 						 v
# '% z Extra postLrn', 
# '% z Missed Test',
# '% z Missed Train', 
# '% z Missed postLrn', 
# '|Z| overinferred mean', 
# '|Z| overinferred std'
# '|Y| overinferred skew',
#  |Z| obs mean
#  |Z| obs std
#  |Z| obs skew
#
# -------------------------------------------------------------







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Filtering out models that don't meet some minimal performance criteria: 
# Here is how to filter out different things I dont want to look at, like with thresholds.
#
#
filter_names = ['zEq0<99%', '% of 2<=|CA|<6']
full_filter = np.ones(df.shape[0]).astype(bool)
#
# (1). 'zEq0<99%' -- Take only models where z=0 wasnt inferred almost 100% of the time
indx_learned = df['%times z=0 inf allSWs']<0.99
full_filter = np.bitwise_and(full_filter, indx_learned)
print('Number of models never learned because z=0 always inferred: ', df.shape[0] - indx_learned.sum())
# 
# (2). '2<=|CA|<6' -- Take only models where some percentage of CAs are reasonably sized.
# 		NOTE: Maybe should change this to some NUMBER of CAs are reasonably sized since M is based on N.
indx_CAsize = df['%|CAs|=2']+df['%|CAs|>2&<6']>0.3
#indx_CAsize = df['#|CAs|=2']+df['#|CAs|>2&<6']>20
full_filter = np.bitwise_and(full_filter, indx_CAsize)
print('Number of models that learned bad CAs: ', df.shape[0] - indx_CAsize.sum())
#
# Display the things that pass the filters.
print('Data passing the filters', filter_names ,' is ',full_filter.sum(),' / ',df.shape[0]) 
if dataType=='real':
	print( df[full_filter].loc[:, ['cell type', 'stim', 'msBins', 'yMinSW', '2nd model', '# EM samps', '%times z=0 inf allSWs', '#|CAs|=2', '#|CAs|>2&<6', '% y Captured postLrn', 'mean pj postLrn'] ] )
#	
if dataType=='synth':
	print( df[full_filter].loc[:, ['init','N', 'M', '# SWs', '# EM samps', 'LR', '%times z=0 inf allSWs', '#|CAs|=2', '#|CAs|>2&<6', '% y Captured postLrn', 'mean pj postLrn'] ] )
#
#print( df[full_filter].loc[:, ['%times z=0 inf allSWs', '%|CAs|=2', '%|CAs|>2&<6', '% y Captured postLrn', 'mean pj postLrn'] ] ) # .sort_values(by='% y Captured postLrn') # HOW TO DISPLAY MULTIPLE COLUMNS OF DATA FRAME !!



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Analysis of Synthetic Data.
#
#
if dataType=='synth':



	# # # scratch
	# ThingsCompared = ['init','N','M'] 					
	# casesToCompare = [ ['DiagonalPia'], [50,100], [50,100] ] 
	# zAxToCompare=['%|CAs|=2','%|CAs|>2&<6']
	# yAxToCompare=['% y Extra postLrn']
	# xAxToCompare=['% y Captured postLrn']
	# #
	# print(ThingsCompared)
	# #
	# plot_comparison_2D(df[full_filter], \
	# 	ThingsCompared=ThingsCompared, \
	# 	casesToCompare=casesToCompare, \
	# 	xAxToCompare=xAxToCompare, \
	# 	yAxToCompare=yAxToCompare, \
	# 	zAxToCompare=zAxToCompare)





	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (0). Learning rate. (Set N=50, M=50, #SWs=100k - best values)
	#
	# 		More EM samples --> No clear trends actually.
	#							(SHOULD IMPLEMENT CROSS-VALIDATION TO STOP EM ALGORITHM!!)
	#
	ThingsCompared = ['N','K'] 					
	casesToCompare = [ [50], [2,0] ] 
	xAxToCompare=['%|CAs|=2','%|CAs|>2&<6']
	yAxToCompare=['mean pj postLrn']
	zAxToCompare=['% y Captured postLrn']
	#
	print(ThingsCompared) # [full_filter]
	#
	plot_comparison_2D(df, \
		ThingsCompared=ThingsCompared, \
		casesToCompare=casesToCompare, \
		xAxToCompare=xAxToCompare, \
		yAxToCompare=yAxToCompare, \
		zAxToCompare=zAxToCompare)








	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (1). Different size networks. Vary N & M.
	#
	#		Take Aways:
	#		----------
	#		Smaller N 		--> larger 'mean pj postLrn'.
	# 						--> larger '% y Captured postLrn'
	#		Smaller M 		--> larger '%|CAs|=2','%|CAs|>2&<6' 	
	#								(probably just because it is percent of M)
	#								(and 'DiagonalPia' init isnt good for overcomplete.)
	#		overcomp (M>N)	--> smaller '% y Captured postLrn'
	# 							smaller '%|CAs|=2','%|CAs|>2&<6' 
	# 		 		depends on N, but does worse for overcomplete too.
	#
	ThingsCompared = ['init','N','M'] 					
	casesToCompare = [ ['DiagonalPia'], [50,100], [50,100] ] 
	zAxToCompare=['%|CAs|=2','%|CAs|>2&<6']
	yAxToCompare=['mean pj postLrn']
	xAxToCompare=['% y Captured postLrn']
	#
	print(ThingsCompared)
	#
	plot_comparison_2D(df[full_filter], \
		ThingsCompared=ThingsCompared, \
		casesToCompare=casesToCompare, \
		xAxToCompare=xAxToCompare, \
		yAxToCompare=yAxToCompare, \
		zAxToCompare=zAxToCompare)



	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (2). Different number of Spike Words. (Set N=50, M=50 - best values)
	#
	# 		More spike words --> larger '%|CAs|=2','%|CAs|>2&<6'
	#						 --> smaller std on 'mean pj postLrn'
	# 							 (All have same mean 'mean pj postLrn')
	#
	ThingsCompared = ['init','N','M','# SWs'] 					
	casesToCompare = [ ['DiagonalPia'], [50], [50], [10000,50000,100000] ] 
	xAxToCompare=['%|CAs|=2','%|CAs|>2&<6']
	yAxToCompare=['mean pj postLrn']
	zAxToCompare=['% y Captured postLrn']
	#
	print(ThingsCompared)
	#
	plot_comparison_2D(df[full_filter], \
		ThingsCompared=ThingsCompared, \
		casesToCompare=casesToCompare, \
		xAxToCompare=xAxToCompare, \
		yAxToCompare=yAxToCompare, \
		zAxToCompare=zAxToCompare)




	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (3). Different number of EM samples. (Set N=50, M=50, #SWs=100k - best values)
	#
	# 		More EM samples --> No clear trends actually.
	#							(SHOULD IMPLEMENT CROSS-VALIDATION TO STOP EM ALGORITHM!!)
	#
	nSW = 100000
	ThingsCompared = ['init','N','M','# SWs','# EM samps'] 					
	casesToCompare = [ ['DiagonalPia'], [50], [50], [nSW], list( (np.array([1/5, 1/2, 1, 2])*nSW).astype(int) ) ] 
	xAxToCompare=['%|CAs|=2','%|CAs|>2&<6']
	yAxToCompare=['mean pj postLrn']
	zAxToCompare=['% y Captured postLrn']
	#
	print(ThingsCompared)
	#
	plot_comparison_2D(df[full_filter], \
		ThingsCompared=ThingsCompared, \
		casesToCompare=casesToCompare, \
		xAxToCompare=xAxToCompare, \
		yAxToCompare=yAxToCompare, \
		zAxToCompare=zAxToCompare)







# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Analysis of Real Data.
#
#
if dataType=='real':

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (1). Different Cell Types shown different Stim.
	#
	# 		--> No clear trends.
	#							
	#
	allCellTypes = [ '[offBriskTransient]', '[offBriskSustained]', '[onBriskTransient]', \
		'[offBriskTransient,offBriskSustained]', '[offBriskTransient,onBriskTransient]' ]
	#
	for i in range(len(allCellTypes)):
		ThingsCompared = ['cell type', 'stim'] 		# ,'yMinSW'			
		casesToCompare = [ [ allCellTypes[i] ], ['NatMov','Wnoise'] ] # , [1,2,3]   
		xAxToCompare=['%|CAs|=2','%|CAs|>2&<6']
		yAxToCompare=['mean pj postLrn']
		zAxToCompare=['% y Captured postLrn']
		#
		plot_comparison_2D(df[full_filter], \
			ThingsCompared=ThingsCompared, \
			casesToCompare=casesToCompare, \
			xAxToCompare=xAxToCompare, \
			yAxToCompare=yAxToCompare, \
			zAxToCompare=zAxToCompare)


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (2). yMinSW
	#
	# 		--> No clear trends.
	#							
	#
	ThingsCompared = ['yMinSW']		
	casesToCompare = [ [ 1,2,3 ] ] 
	xAxToCompare=['%|CAs|=2','%|CAs|>2&<6']
	zAxToCompare=[]# ['mean pj postLrn']
	yAxToCompare=['% y Captured postLrn']
	#
	plot_comparison_2D(df[full_filter], \
		ThingsCompared=ThingsCompared, \
		casesToCompare=casesToCompare, \
		xAxToCompare=xAxToCompare, \
		yAxToCompare=yAxToCompare, \
		zAxToCompare=zAxToCompare)		



	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (3). msBins
	#
	# 		Bigger binning on SWs --> More reasonable CA's but smaller '% y Captured postLrn'
	# 			(Makes sense because larger msBins means longer SWs which would have both effects).
	#							
	#
	ThingsCompared = ['msBins']		
	casesToCompare = [ [ 1,3,5 ] ] 
	xAxToCompare=['%|CAs|=2','%|CAs|>2&<6']
	zAxToCompare=[]# ['mean pj postLrn']
	yAxToCompare=['% y Captured postLrn']
	#
	plot_comparison_2D(df[full_filter], \
		ThingsCompared=ThingsCompared, \
		casesToCompare=casesToCompare, \
		xAxToCompare=xAxToCompare, \
		yAxToCompare=yAxToCompare, \
		zAxToCompare=zAxToCompare)	










	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# (X). xxx
	#
	# 
	if False:
		print('Note: Cool way to index into a key to see what all its values are. Useful for some Keys.')
		xxx = df.keys()
		for i,x in enumerate(xxx):
			print(i,'	: 	Key = ',x, ' 	:	 Vals = ',list(np.unique(df[x].values)))
			print(' ')





	# ThingsCompared = ['cell type', 'stim'] 					
	# casesToCompare = [ ['[offBriskTransient]','[onBriskTransient]'], ['NatMov','Wnoise'] ]  # ['[offBriskTransient]'], 
	# plot_comparison_2D(df_learned, \
	# 	ThingsCompared=ThingsCompared, \
	# 	casesToCompare=casesToCompare, \
	# 	xAxToCompare=['% y Captured postLrn'], \
	# 	yAxToCompare=['|Z| inf std'], \
	# 	zAxToCompare=['%|CAs|=2','%|CAs|>2&<6'])



	# ThingsCompared = ['msBins'] 					
	# casesToCompare = [ [1,3,5] ] 
	# plot_comparison_2D(df_learned, ThingsCompared=ThingsCompared, casesToCompare=casesToCompare, \
	# 	xAxToCompare=['% y Captured postLrn'], yAxToCompare=['%|CAs|=2','%|CAs|>2&<6'])

	# ThingsCompared = ['stim'] 					
	# casesToCompare = [ ['NatMov','Wnoise']  ] 
	# plot_comparison_2D(df_learned, ThingsCompared=ThingsCompared, casesToCompare=casesToCompare, \
	# 	xAxToCompare=['% y Captured postLrn'], yAxToCompare=['%|CAs|=2','%|CAs|>2&<6'])


	# ThingsCompared = ['cell type'] 					
	# casesToCompare = [ ['[offBriskTransient]', '[offBriskSustained]', '[onBriskTransient]', \
	# 		'[offBriskTransient,offBriskSustained]', '[offBriskTransient,onBriskTransient]' ] ]
	# plot_comparison_2D(df_learned, ThingsCompared=ThingsCompared, casesToCompare=casesToCompare, \
	# 	xAxToCompare=['% y Captured postLrn'], yAxToCompare=['%|CAs|=2','%|CAs|>2&<6'])





# (2). Look at correlations using Seaborn pairplot function.

pp_cols = ['% y Captured postLrn', '% y Extra postLrn', '% z Captured postLrn', '% z Extra postLrn', \
		'% y Captured Train', '% y Extra Train', '% z Captured Train', '% z Extra Train']










