%% Exploratory Code to analyze Retina Data sent by Greg Field with white noise and natural stimulus.




%% Variables included in Kiersten Ruda's GLM simulations for cell subsets with natural movie stimulus.

whichPop = 'fullpop'; % (1). 'subpop' - spiketrains from a pairwise and indpenendent GLM simulation of selected small subsets of cells
                      % (2). 'fullpop' - spiketrains from an indpenendent GLM simulation of all cells of {offBT, onBT, offBS} types - 137 cells.

fname = ['glm_cat_sim_',whichPop,'_v2']; 

%       in 'glm_cat_sim_subpop.mat'
%
% cat_simulation_triggers - times for trial starts in data simulations  
% -----------------------
%
% Mixed_group - 9 {offBriskTransient, offBriskSustained} cells
% -----------
%   cat_spikes_indpt: Spike Trains from Independent GLM model on natural movie stim.
%     cat_spikes_cpl: Spike Trains from pairwise coupled GLM model on natural movie stim.
%     matlab_indices: [38 39 40 42 36 71 83 85 86]
%
% Off_BT_group - 8 offBriskTransient cells
% ------------
%   cat_spikes_indpt: Spike Trains from Independent GLM model on natural movie stim.
%     cat_spikes_cpl: Spike Trains from pairwise coupled GLM model on natural movie stim.
%     matlab_indices: [1 2 5 37 49 51 8 36]
%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%
%       in 'glm_cat_sim_fullpop.mat'
%
% cat_sim_triggers - times for trial starts in data simulations  
% -------------
%
% cat_indpt_spike_preds - spike times for trial starts in data simulations  
% ------------------
% 
% cat_psth_indpt_perf - some performance measure for independent model
% ----------------
%
% cat_indpt_inst_fr - instantaneous firing rate in ms bins.
% --------------





%% Setup some directory structure
projsDir  = pwd; %RUN THIS FROM PROJECTS DIRECTORY. Must call addProjPaths before running this.
codeDir   = [projsDir,'/G_Field_Retinal_Data/home/G_Field_Retinal_Data/Chris_working_code_2018/matlab_code/'];	% Directory where my code resides
inputDir = [projsDir,'/G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/data/GField_data/'];                                      % Directory where Greg's raw data sits.
outDataDir = [projsDir,'/G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/data/matlab_data/'];                                 % Directory to save output data from this.
outFigsDir = [projsDir,'/G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/figs/older_elephant/'];                               % Directory to save output figures from this.


if ~exist(outDataDir,'dir')
    mkdir(outDataDir)
end

if ~exist(outFigsDir,'dir')
    mkdir(outFigsDir)
end


% load this older file for two variables on CellTypes and CellTypeIDs to put them in the output mat file from this function also.
load([outDataDir,'allCells_spikeTrains_CellXTrial.mat'])
%cellTypeIDs
%allCellTypes



%% The basics (load file. get number of cells. etc.)

disp(['Load raw retina data mat file: ',fname])
load([inputDir,fname,'.mat'])  % mat file which includes data 




%% (1). Split spike data up into a [cell_num x trial_num] data cell array. 
%       This is used as input to python code!
if exist([outDataDir,'GLM_sim_',whichPop,'_spikeTrains_CellXTrial.mat'],'file')
    load([outDataDir,'GLM_sim_',whichPop,'_spikeTrains_CellXTrial.mat'])

else
    
    
    
    
   %
    % Pack spikes into a cell which becomes an NDARRAY in python.
    %
    if strfind(whichPop, 'subpop')

        trigs = cat_simulation_triggers;
        num_trials = numel(trigs);
        trigs(num_trials+1) = 1e16;  % because last trigger is beginning of the last trial
        
        % SAVE 4 files
        
        %
        %
        % (1). For Independent offBriskTransient Cells
        % ----------------------------------        
        num_cells = numel(Off_BT_group.matlab_indices);
        spikes_GLM = Off_BT_group.cat_spikes_indpt;
        matlab_inds = Off_BT_group.matlab_indices;
        %
        for j = 1:num_trials
            j
            for i = 1:num_cells
                %
                indCat = find( (spikes_GLM{i}>=trigs(j)) & (spikes_GLM{i}<trigs(j+1)) );
                spikes{i,j} = ceil(1000.*(spikes_GLM{i}(indCat) - trigs(j))');
                %
            end
        end
        %
        save([outDataDir,'GLM_sim_',whichPop,'_offBT_ind_spikeTrains_CellXTrial.mat'],...
            'spikes', 'matlab_inds', 'trigs', 'allCellTypes', 'cellTypeIDs')
        
        % 
        %
        % (2). For Coupled offBriskTransient Cells
        % -------------------------------        
        num_cells = numel(Off_BT_group.matlab_indices);
        spikes_GLM = Off_BT_group.cat_spikes_cpl;
        matlab_inds = Off_BT_group.matlab_indices;
        %
        for j = 1:num_trials
            j
            for i = 1:num_cells
                %
                indCat = find( (spikes_GLM{i}>=trigs(j)) & (spikes_GLM{i}<trigs(j+1)) );
                spikes{i,j} = ceil(1000.*(spikes_GLM{i}(indCat) - trigs(j))');
                %
            end
        end
        %
        save([outDataDir,'GLM_sim_',whichPop,'_offBT_cpl_spikeTrains_CellXTrial.mat'],...
            'spikes', 'matlab_inds', 'trigs', 'allCellTypes', 'cellTypeIDs')
        
        
        
        

        % 
        %
        % (3). For Independent  {offBriskTransient,offBriskSustained}  Cells
        % ---------------------------------------------------        
        num_cells = numel(Mixed_group.matlab_indices);
        spikes_GLM = Mixed_group.cat_spikes_indpt;
        matlab_inds = Mixed_group.matlab_indices; 
        %
        for j = 1:num_trials
            j
            for i = 1:num_cells
                %
                indCat = find( (spikes_GLM{i}>=trigs(j)) & (spikes_GLM{i}<trigs(j+1)) );
                spikes{i,j} = ceil(1000.*(spikes_GLM{i}(indCat) - trigs(j))');
                %
            end
        end
        %
        save([outDataDir,'GLM_sim_',whichPop,'_offBT_offBS_ind_spikeTrains_CellXTrial.mat'],...
            'spikes',  'matlab_inds', 'trigs', 'allCellTypes', 'cellTypeIDs')
        
        
  
        
        
        
        
        % 
        %
        % (4). For Coupled  {offBriskTransient,offBriskSustained}  Cells
        % ---------------------------------------------------        
        num_cells = numel(Mixed_group.matlab_indices);
        spikes_GLM = Mixed_group.cat_spikes_cpl;
        matlab_inds = Mixed_group.matlab_indices; 
        %
        for j = 1:num_trials
            j
            for i = 1:num_cells
                %
                indCat = find( (spikes_GLM{i}>=trigs(j)) & (spikes_GLM{i}<trigs(j+1)) );
                spikes{i,j} = ceil(1000.*(spikes_GLM{i}(indCat) - trigs(j))');
                %
            end
        end
        %
        save([outDataDir,'GLM_sim_',whichPop,'_offBT_offBS_cpl_spikeTrains_CellXTrial.mat'],...
            'spikes',  'matlab_inds', 'trigs', 'allCellTypes', 'cellTypeIDs')
        
        
    elseif strfind(whichPop,'fullpop')         
        
            % From Independent {offBriskTransient, offBriskSustained, onBriskTransient} All Cells - cell types in that order.
            %                           55 offBriskTransient, 43 offBriskSustained, 39 onBriskTransient
            % ----------------------------------------------------------------
            % SAVE 3 files (redundant, but files are small and it eases processing in python)
            % (1). offBT
            % (2). offBT_offBS
            % (3). offBT_onBT
         
        trigs = cat_sim_triggers;
        num_trials = numel(trigs);
        trigs(num_trials+1) = 1e16;  % because last trigger is beginning of the last trial
        %
        % 

        
        
        %
        %
        % (1). 55 offBriskTransient Cells
        % -------------------------------  
        num_cells = 55;
        spikes_GLM = cat_indpt_spike_preds(1:num_cells);
        matlab_inds =  cellTypeIDs(1,:);
        %
        for j = 1:num_trials
            j
            for i = 1:num_cells
                %
                indCat = find( (spikes_GLM{i}>=trigs(j)) & (spikes_GLM{i}<trigs(j+1)) );
                spikes{i,j} = ceil(1000.*(spikes_GLM{i}(indCat) - trigs(j))');
                %
            end
        end
        %
        %
        % % % %
        %
        save([outDataDir,'GLM_sim_',whichPop,'_offBT_ind_spikeTrains_CellXTrial.mat'],...
            'spikes',  'matlab_inds', 'trigs', 'allCellTypes', 'cellTypeIDs')
        
        
        %
        %
        % (2). 55 offBriskTransient Cells and 43 offBriskSustained Cells
        % ------------------------------- ----------------
        num_cells = 55+43;
        spikes_GLM = cat_indpt_spike_preds(1:num_cells);
        matlab_inds =  [ cellTypeIDs(1,: ),  cellTypeIDs(2,1:43)  ];
        %
        for j = 1:num_trials
            j
            for i = 1:num_cells
                %
                indCat = find( (spikes_GLM{i}>=trigs(j)) & (spikes_GLM{i}<trigs(j+1)) );
                spikes{i,j} = ceil(1000.*(spikes_GLM{i}(indCat) - trigs(j))');
                %
            end
        end
        %
        %
        % % % %
        %
        save([outDataDir,'GLM_sim_',whichPop,'_offBT_offBS_ind_spikeTrains_CellXTrial.mat'],...
            'spikes',  'matlab_inds', 'trigs', 'allCellTypes', 'cellTypeIDs')
        
        
        
        %
        %
        % (3). 55 offBriskTransient Cells and 39 onBriskTransient Cells
        % ------------------------------- -----------------
        num_cells = 55+39;
        spikes_GLM = cat_indpt_spike_preds([1:55,end-39+1:end]);
        matlab_inds =  [ cellTypeIDs(1,:), cellTypeIDs(5,1:39 )  ];
        %
        for j = 1:num_trials
            j
            for i = 1:num_cells
                %
                indCat = find( (spikes_GLM{i}>=trigs(j)) & (spikes_GLM{i}<trigs(j+1)) );
                spikes{i,j} = ceil(1000.*(spikes_GLM{i}(indCat) - trigs(j))');
                %
            end
        end
        %
        %trigs = round(1000.*trigs(1:num_trials)); % get rid of added entry at beginning of this section.
        %
        % % % %
        %
        save([outDataDir,'GLM_sim_',whichPop,'_offBT_onBT_ind_spikeTrains_CellXTrial.mat'],...
            'spikes',  'matlab_inds', 'trigs', 'allCellTypes', 'cellTypeIDs')
        
         
    else
        
        disp(['Dont understand whichPop = ',whichPop])
        
        
    end
    
    
    
    
end





