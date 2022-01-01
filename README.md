# Cell Assembly Codebase

Welcome. This code repo contains the 2nd half of my thesis work. Most of the development was done in Python with raw data receiving and preprocessing done in matlab. Raw data was provided by experimental collaborators in mat files. Additionally, some bash scripting to interface with a computer cluster using sbatch.



The paper stemming from this work, entitled "A probabilistic latent variable model to detect noisy patterns in binary data", can be found at arxiv.org/xyz











## Python Functions:

### Utils packages that are imported and called from various functions:

(1). **retina_computation.py** - contains many functions that are used in several places in the code (for example, in code referencing real retinal data, synthetic data and GLM generated data). In here is code to synthesize a model, generate synthetic data, perform MAP estimation for model parameters, infer a latent vector given a model and observed vector, initialize model parameters, learn model parameters, run the expectation maximization algorithm, compare learned cell assemblies to ground truth ones, and more ...

(2). **sbatch_scripts.py** - contains functions to write sbatch script text files which call python functions with command line parameter input arguments. These utils functions are called inside nested for loops in cluster_scripts_\*.py functions to generate individual sbatch jobs for each parameter setting in a grid search and a script which allows me to submit all those jobs to the cluster scheduler with a single command.

(3). **data_manipulation.py** - small group of functions that deal with paths and directories and manipulation of data and data files, mostly.

(4). **plot_functions.py** - collection of many of the plotting functions used in the development of this project, analysis of performance along the way, as well as the final figures contained in the paper.



<!--- COMMENT OUT THIS
write_sbatch_script_pgmCA_realData > pgmCA_realData.py
write_sbatch_script_pgmCA_GLMsimData > pgmCA_GLMsimData.py
write_sbatch_script_pgmCA_synthData > pgmCA_synthData.py
write_sbatch_script_infer_postLrn_synthData > infer_postLrn_synthData.py
write_sbatch_script_pgmCA_and_infPL_synthData > pgmCA_synthData.py & infer_postLrn_synthData.py

write_sbatch_script_rasterZ_realData > raster_zs_inferred_allSWs_given_model.py
write_sbatch_script_rasterZ_GLMsimData > raster_zs_inferred_allSWs_given_model.py
write_sbatch_script_rasterZ_xVal_realData > raster_zs_inferred_xValSWs_given_model.py

write_sbatch_script_StatsInfPL_realData > StatsInfPL_realData.py
write_sbatch_script_compare_SWdists_realNsynth > compare_SWdists_realNsynthData.py
--->










	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



(5). **pgmCA_realData.py** - 

(5). **pgmCA_synthData.py** - 

(5). **pgmCA_GLMsimData.py** - 


	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


(8). **raster_zs_inferred_allSWs_given_model.py** - compute rasters for cell assembly activity (like is done for cell activity) for all spike words. Save data in an npz file. Then plots are made from that npz data file using plot_raster_PSTH_zs.py

(9). **raster_zs_inferred_xValSWs_given_model.py** - compute rasters for cell assembly activity (like is done for cell activity) for spike words in the test/validation dataset. Save data in an npz file. Then plots are made from that npz data file using plot_raster_PSTH_zs.py


	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

(6). **plot_raster_PSTH_zs.py** - code to produce many different plots from Cell Assembly rasters, based on user input flags at the top of this function. Can loop through a grid of hyperparameter value to produce output plots for each parameter value combination. Many of the figures in the paper are generated with this function.







	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

(6). **cluster_scripts_realData.py** - 

(6). **cluster_scripts_synthData.py** - 

(6). **cluster_scripts_GLMsimData.py** - 




	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

### Functions to fit model parameters to spike-word statistics in retinal data.

(1). **compare_SWdists_realNsynthData.py** - function that constructs a model with a set of user-input parameter values, generates spike-words from that model, computes moments on the observed spike-words and then compares those moments from synthetic data to observable moments in real retinal data via QQ plots to be able to quantitatively say that a synthetic model with certain parameter values reproduces responses similar to a retina responding to natural movie stimulus, for example. This is discussed in the paper section, "Fitting Model parameters to spike-word statistics". This function is called from cluster_scripts_compare_realNsynth.py to be run in parallel on computer cluster with each run saving spike-word distribution moments stats to a file. Single run results can be combined and the best model parameters determined in plot_SWdists_bestFit_SynthParams2Real.py.

(2). **cluster_scripts_compare_realNsynth.py** - function that will write sbatch text files to run many jobs on computer cluster with one command-line call. Runs "compare_SWdists_realNsynthData.py" and saves plot and data file for each parameter value combination.
	
(3). **plot_SWdists_bestFit_SynthParams2Real.py** - function to make plots of distributions for spike-word statistics in order to compare observed spike-word statistics for synthetic data generated from models with various parameter values to those spike-word statistics in real retinal data. This is discussed in the section "Fitting Model parameters to spike-word statistics" of the paper and displayed in Fig. 2, "Fitting synthetic model to spike-word moments". This allows us to choose the best model parameters to synthesize data based on QQ-plots comparing spike-word stats between real and synthesized data.			



	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


StatsInfPL_realData.py
StatsInfPL_synthData.py






combine_modelRands_synthData.py					

compare_2_learned_models_realData.py				
compare_2_learned_models_synthData.py	
			
scratch_pad_COSYNE_poster.py




	
test_inference.py


compare_p_of_ys_PGM2nulls.py	

				
compute_GLM_p_of_y.py						
compute_GLM_p_of_y_Better.py					

explore_retina_data_UEA.py					

infer_postLrn_synthData.py					

pandas_vis_CSV_STATs.py	

vis_learned_pgmCA_realData.py
vis_learned_pgmCA_synthData.py
vis_model_snapshots_realData.py					
vis_model_snapshots_synthData.py


















## Matlab Functions:

(1). edit_all_DataRetina.m - function which opens various functions I have played with and written over the course of this project. It acts as a historical document and a table of contents. Some of these functions are deprecated and no longer used.

(2). addProjPaths.m - for the purposes of this project, the function adds directory structure to the matlab path for downloaded matlab packages

	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

(3). explore_data.m - transforms raw data provided in a mat file into format friendly to python and saves it in a mat file. Also makes ellipse plots of retinal ganglion cell receptive fields.

(4). explore_data_GLM_NM.m - Takes data provided by collaborators from GLM simulation of retinal ganglion cells responding to same stimulus as was presented to actual cells and transforms that data into a python friendly format.

	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

(5). find_GaussRF_PBs.m - this function was an exploration in applying image segmentation algorithm to natural movie input. It was not particularly successful.

(6). findRateCodeAssemblies.m - an early exploration of retina data. Replaced by explore_data.m

(7). sumSpikeRastersOverTrials.m - constructs spike rasters in a 3D binary data cube (#cells x time_ms x num_trials) for both white noise and natural movie stimulus

(8). CA_STRFs.m - function that explored fitting Spatio-Temporal Receptive Fields (STRFs) to cell assemblies using the STRFlab matlab package from Theunissen and Gallant labs and GLM tools from the J. Pillow Lab.

(9). STRF_fit.m - function to fit 2D gaussian spatio-temporal receptive field for a cell with a 3rd dimension time component from its spike responses to white noise stimulus. Fit STRF for each cell to both dark and light responses and compare them.

(10). tile_RFs.m - compare the dark and light responses of each cell's STRF to quantify if they are different.

(11). warnerSTC.m - some edits I made to J. Pillow's simpleSTC function

	





