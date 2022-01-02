# Cell Assembly Codebase

Welcome. This code repo contains the 2nd half of my thesis work. Most of the development was done in Python with raw data receiving and preprocessing done in matlab. Raw data was provided by experimental collaborators in mat files. Raw data consisted of input stimulus videos (white noise and natural movie) as well as cell id's and spike times provided in matlab structures. In matlab, we computed receptive fields (using STRFlab) and converted data formats to transfer spike times over to python. In python, we did the vast majority of the development - including synthetic model generation, data synthesis, model training, inference and model analysis and comparison. In order to speed up development, we parallelized the search over model hyperparameters by sbatch scripting and submitting python jobs to a computer cluster.


The paper stemming from this work, entitled "A probabilistic latent variable model to detect noisy patterns in binary data", can be found at arxiv.org/xyz


	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

## Python Functions:

#### Utils packages:

(1). **retina_computation.py** - contains many functions that are used in several places in the code (for example, in code referencing real retinal data, synthetic data and GLM generated data). In here is code to synthesize a model, generate synthetic data, perform MAP estimation for model parameters, infer a latent vector given a model and observed vector, initialize model parameters, learn model parameters, run the expectation maximization algorithm, compare learned cell assemblies to ground truth ones, and more ...

(2). **sbatch_scripts.py** - contains functions to write sbatch script text files which call python functions with command line parameter input arguments. These utils functions are called inside nested for loops in cluster_scripts_\*.py functions to generate individual sbatch jobs for each parameter setting in a grid search and a script which allows me to submit all those jobs to the cluster scheduler with a single command.

(3). **data_manipulation.py** - small group of functions that deal with paths and directories and manipulation of data and data files, mostly.

(4). **plot_functions.py** - collection of many of the plotting functions used in the development of this project, analysis of performance along the way, as well as the final figures contained in the paper.

	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#### Main functions to run EM algorithm to learn model from data

(1). **pgmCA_realData.py** - function that will load in real retina data, split it into test and train sets, initialize model parameters and run EM algorithm to learn model parameters to fit observed responses in that data. Will save model and training statistics to an npz file. Will also train a 2nd model on the "test" data for cross-validation purposes to compare the two models against one another. This function can be run for a single set of model hyperparamters on the cluster as part of a grid search by using cluster_scripts_realData.py with what_to_run flag = 'pgmR'.

(2). **pgmCA_synthData.py** - function that will load in synthesized data from npz file or synthesize a model and generate data and save it if it does not exist. Then, function will initialize a model, split the synthesized data into test and train sets, initialize and learn two models using the two different splits of the data and the EM algorithm. Since this is synthesized data and we have the ground truth model it was synthesized from, we can compare the learned model to the ground truth, and we can compare difference between those during the learning procedure. This function can be run for a single set of model hyperparamters on the cluster as part of a grid search by using cluster_scripts_synthData.py with what_to_run flag = 'pgmS'.

(3). **pgmCA_GLMsimData.py** - function that will load in data from a GLM simulation based on real retina responses provided by our collaborator. It will then split the data into test and train sets, initialize model parameters and run EM algorithm to learn model parameters to fit observed responses in that data. Will save model and training statistics to an npz file. Will also train a 2nd model on the "test" data for cross-validation purposes to compare the two models against one another. This function can be run for a single set of model hyperparamters on the cluster as part of a grid search by using cluster_scripts_GLMsimData.py with what_to_run flag = 'pgmG'.

	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#### Functions to infer latent variable (z) activity with model fixed, post learning.

(1). **raster_zs_inferred_allSWs_given_model.py** - compute rasters for cell assembly activity (like is done for cell activity) for all spike words. Save data in an npz file. Then plots are made from that npz data file using plot_raster_PSTH_zs.py. This can be run as part of a hyperparameter grid search from cluster_scripts_realData.py with what_to_run flag = 'rasZ' or from cluster_scripts_GLMsimData.py with what_to_run flag = 'rasG'.

(2). **raster_zs_inferred_xValSWs_given_model.py** - compute rasters for cell assembly activity (like is done for cell activity) for spike words in the test/validation dataset. Save data in an npz file. Then plots are made from that npz data file using plot_raster_PSTH_zs.py, This can be run as part of a hyperparameter grid search from cluster_scripts_realData.py with what_to_run flag = 'rasX'. 

(3). **infer_postLrn_synthData.py** - function to infer latent unit (z) activity for model learned on synthetic data. Since the ground truth z activity is known, we compare inferred z's to ground truth z's and also compute statistics on inference step. This can be run  as part of a hyperparameter grid search from cluster_scripts_synthData.py with what_to_run flag = 'infPL'. 

(4). **StatsInfPL_realData.py** - From data saved from inference functions 1 or 2 in this section, this function computes a whole battery of statistics that are used to quantify performance and to compare different learned models to one another. For inference from model learned on real retina data. This can be run as part of a hyperparameter grid search from cluster_scripts_realData.py with what_to_run flag = 'statI'.

(5). **StatsInfPL_synthData.py** - From data saved from inference function 3 in this section, this function computes a whole battery of statistics that are used to quantify performance and to compare different learned models to one another. For inference from model learned on synth data. This may be unfinished ...

	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#### Functions to grid search hyperparameters on computer cluster

(1). **cluster_scripts_realData.py** - For real retina data, write sbatch script text files for each set of hyperparameter values in a grid search that can be run on computer cluster. With the right setting of the what_to_run flag, we can run the EM algorithm, infer latent unit activity (z's) with a fixed model after learning, learn a model and then infer z's in sequence, or compute statistics on the inference step. This references functions defined in the sbatch_scripts.py util file.

(2). **cluster_scripts_synthData.py** -  For synthetic data, write sbatch script text files for each set of hyperparameter values in a grid search that can be run on computer cluster. With the right setting of the what_to_run flag, we can run the EM algorithm, infer latent unit activity (z's) with a fixed model after learning, or learn a model and then infer z's in sequence. This references functions defined in the sbatch_scripts.py util file.

(3). **cluster_scripts_GLMsimData.py** -  For data from the GLM model fit to retina data, write sbatch script text files for each set of hyperparameter values in a grid search that can be run on computer cluster. With the right setting of the what_to_run flag, we can run the EM algorithm or infer latent unit activity (z's) with a fixed model after learning. This references functions defined in the sbatch_scripts.py util file.

(4). **cluster_scripts_compare_realNsynth.py** - function that will write sbatch text files to run many jobs on computer cluster with one command-line call. Runs "compare_SWdists_realNsynthData.py" and saves plot and data file for each parameter value combination.

	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#### Functions to fit model parameters to spike-word statistics in retinal data.

(1). **compare_SWdists_realNsynthData.py** - function that constructs a model with a set of user-input parameter values, generates spike-words from that model, computes moments on the observed spike-words and then compares those moments from synthetic data to observable moments in real retinal data via QQ plots to be able to quantitatively say that a synthetic model with certain parameter values reproduces responses similar to a retina responding to natural movie stimulus, for example. This is discussed in the paper section, "Fitting Model parameters to spike-word statistics". This function is called from cluster_scripts_compare_realNsynth.py to be run in parallel on computer cluster with each run saving spike-word distribution moments stats to a file. Single run results can be combined and the best model parameters determined in plot_SWdists_bestFit_SynthParams2Real.py.
	
(2). **plot_SWdists_bestFit_SynthParams2Real.py** - function to make plots of distributions for spike-word statistics in order to compare observed spike-word statistics for synthetic data generated from models with various parameter values to those spike-word statistics in real retinal data. This is discussed in the section "Fitting Model parameters to spike-word statistics" of the paper and displayed in Fig. 2, "Fitting synthetic model to spike-word moments". This allows us to choose the best model parameters to synthesize data based on QQ-plots comparing spike-word stats between real and synthesized data.			

	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#### Function to make plots

(1). **plot_raster_PSTH_zs.py** - function to produce many different plots from Cell Assembly rasters, based on user input flags at the top of this function. Can loop through a grid of hyperparameter value to produce output plots for each parameter value combination. Many of the figures in the paper are generated with this function.

(2). **vis_learned_pgmCA_realData.py** - function to gather up model and inference performance with real data. Loops through hyperparameter values. Loads in npz data file containing model and inference performance with fixed model and also inference during learning. Also makes a number of diagnostic plots along the way. Appends all info about parameter values and inference statistics into CSV file.

(3). **vis_learned_pgmCA_synthData.py** - does very similar thing as vis_learned_pgmCA_realData.py just with synth data. So hyperparameters are a little different. Also, compares learned structure to known structure in ground truth. Makes lots of plots. Saves CSV file and some npz files too.

(4). **vis_model_snapshots_realData.py** - During model training, snapshots of model and inference procedure were taken. This function will make plots of model and inference stats from different snapshots in the learning procedure. Can show two models run on 50/50% test-train cross validation splits side by side. Does so for model trained on real retina data.	
				
(5). **vis_model_snapshots_synthData.py** - Does same as vis_model_snapshots_realData.py, but for model trained on synth data. And since there is ground truth, can also show model during training as it approaches the ground truth model.

(6). **pandas_vis_CSV_STATs.py** - function to use pandas and seaborn plotting to explore differences between model hyperparameter values and how they effect the inference step after the model has been learned. The CSV data files that this function analyzes were made in vis_learned_pgmCA_realData.py and vis_learned_pgmCA_synthData.py. Can run for synth data or real data.

	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#### Functions to compare models to one another after learning

(1). **compare_2_learned_models_realData.py** - function to perform some post-hoc analysis after two models have been learned on real data to compare them. Compute cosine similarity between pairs of cell assemblies in models and match up CA's found in both models.
				
(2). **compare_2_learned_models_synthData.py** - function to compare a pair of learned models trained on synthetic data to one another to see if they learned the same structure, and also to compare the structure learned in each model to the ground truth structure. Figures 3,4 and 5 and the associated analysis in the section of the paper entitled "Performance assessment and results on synthetic data" were produced using this code.	

	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#### Functions to compare models to GLM & LNP null models
				
(1). **compute_GLM_p_of_y.py** - compute individual cell firing rate/probability for real retina cells under the GLM and LNP models for a given stimulus. Run with and without spike history. Data from GLM/LNP models delivered by our collaborators.

					
(2). **compute_GLM_p_of_y_Better.py** - function that does the same as compute_GLM_p_of_y.py, but in addition it investigates if there is a systematic offset between when a synchronous firing pattern is most likely under the GLM and when we observe it using our model. 

(3). **compare_p_of_ys_PGM2nulls.py** - gathering up model predicted joint probability p(y,z), GLM predicted firing probability p(y) and LNP predicted firing probability for each cell, stimulus and trial. These are saved into data structures and plotted. 	

	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#### Exploratory functions

(1). **combine_modelRands_synthData.py** - An attempt with models learned on synthetic data to combine them together to see if the agglomerated model was better than either one alone.					

(2). **explore_retina_data_UEA.py** - Early function exploring Sonja Gruen's Unitary Events Analysis (UEA) framework and her Elephant toolbox to analyze retina spike trains.					

	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

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

	





