% Script to add paths to subdirectories in Project Folder.

curr = pwd;   % Run from Projects directory both on Mac & Cluster
addpath(curr);

addpath(genpath([curr,'/CodeDownloads']));

addpath(genpath([curr,'/BSDS_code_full'])); % Code to interact with Berkeley Segmentation images.
% addpath(genpath([curr,'/BSDS_bench_jailbreak'])); % Benchmark (quality accessing) code that came with BSDS images

% addpath(genpath([curr,'/MaxEnt'])); % Code from Chris Hillar using MaxEnt distribution on graph given node degrees as Null Model
addpath(genpath([curr,'/ImageSeg']));
% addpath(genpath([curr,'/NoTopo_Nets']));
% addpath(genpath([curr,'/NoTopo_Nets_Old']));
addpath(genpath([curr,'/Kuramoto_DS']));

addpath(genpath([curr,'/StyleTransfer_Subproject']));

addpath(genpath([curr,'images/BSDS_images/BSR/bench']));
%addpath(genpath([curr,'images/BSDS_images/BSR/segbench']));


addpath([curr,'/graph_clustering_code/']);

% addpath([curr,'/S_Palmer_Data/Code/']);
addpath(genpath([curr,'/S_Palmer_Data/Code/SpikeSortEdit']));


addpath(genpath([curr,'/G_Field_Retinal_Data/home/G_Field_Retinal_Data/Chris_working_code_2019/matlab_code']));



% % Code Downloaded from Internet we are not using currently (it is in different directory now.)
% addpath([curr,'/Ncut_ShiMalik']); % Code from Jianbo Shi's website: (http://www.cis.upenn.edu/~jshi/software/)
% addpath([curr,'/CannyEdge']); % Code downloaded from internet to find edges with canny filter (prob not used here).

% addpath(genpath([curr,'/Retinal_Data']));

clear