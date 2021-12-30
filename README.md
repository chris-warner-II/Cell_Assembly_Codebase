# Cell Assembly Codebase

Welcome. This code repo contains the 2nd half of my thesis work. Most of the development was done in Python with raw data receiving and preprocessing done in matlab. Raw data was provided by experimental collaborators in mat files. Additionally, some bash scripting to interface with a computer cluster using sbatch.



The paper stemming from this work, entitled "A probabilistic latent variable model to detect noisy patterns in binary data", can be found at arxiv.org/xyz











## Python Functions:

Utils packages that are imported and called from various functions:

(1). **data_manipulation.py** - 

(2). **retina_computation.py** -

(3). **plot_functions.py** - 

(4). **sbatch_scripts.py** - contains functions to write sbatch script text files which call python functions with command line parameter input arguments. These utils functions are called inside nested for loops in cluster_scripts_\*.py functions to generate individual sbatch jobs for each parameter setting in a grid search and a script which allows me to submit all those jobs to the cluster scheduler with a single command.



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

(6). **plot_raster_PSTH_zs.py**

(7). **other_plot_functions**



	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

(6). **cluster_scripts_realData.py** - 

(6). **cluster_scripts_synthData.py** - 





(6). **cluster_scripts_GLMsimData.py**

(6). **cluster_scripts_compare_realNsynth.py**



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

	





