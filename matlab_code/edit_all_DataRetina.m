% Script to quickly open / edit all the functions that are useful for the
% Retinal Spike Train Analysis Project.


%% Add Paths to Directories Containing Code
addpath(genpath(['./CodeDownloads']));
addpath(genpath(['./RetinaData/Chris_working_2018']));

%% edit this file
edit edit_all_DataRetina




%% Most recent MATLAB files used in Cell Assembly Project.
if (1)
    edit explore_data.m
    edit explore_data_GLM_NM.m
    %
    edit find_GaussRF_PBs.m
    edit findRateCodeAssemblies.m
    edit STRF_fit.m
    edit sumSpikeRastersOverTrials.m
    edit tile_RFs.m
    edit warnerSTC.m
    edit CA_STRFs.m







%% edit Spike Train Generation function (Ground Truth for Sanity Checking)
if (0)
    edit genSpikeTrain % synthetic data.







%% The various functions and scripts I have written as part of processing chain.
if(0)
    edit calc_Rate_ISI                           % Calculates some statistics on spike trains.
    %
    % Look For Overlap of High Activity Windows in Pairs of Spike Trains from Pairs of Neurons.
    edit HAW_analysis                            % Original exploratory scripts that does all of analysis.  Is broken up in others below.
    edit write_sbatch_HAWSample                  % write series of scripts for cluster as well as call to copy and paste
    edit HAWOverlap_DataGen                      % called repeatedly to calculate HAWOvL and sample from null model for subset of cells.
    edit Sample_HAW_OvL_NMs                      % sample user specified number of times from HAW OvL Null Model (called form DataGen)
    edit visualize_HAWOvL                        % plot HAWs from both cells as well as Overlap.
    edit combine_HAWOvL_stats                    % recombine subMatrices of subResults from each run of DataGen and do some analysis
    %
    % Within High Activity Windows, look for Overlap of spikes on finer time-scale.
    edit spikeInHAWOvL_DataGen                   %
    edit spikeWithinHAW_analysis                 %
    edit write_sbatch_SpikeInHAWSample           %
    % edit Sample_SpikeInSpindOvL_NMs            %
    edit Sample_SpikeInHAWOvL_NMsB               %
    edit visualize_SpikeInHAWOvL                 %
    edit combine_SpikeInHAWOvL_stats             %
    %
    edit write_sbatch_analyze_SpikeInHAW_Plots
    edit combine_GaussFit_LagVsXcor
    %
    edit XcorrVanilla_SpikeTrains
    edit write_sbatch_XcorrVanilla
    edit combine_XcorrVanilla_ST
end




%% edit Chronux analysis functions ... Jarvis & Mitra
if(0)
   edit exploreDataChronux % wrapper function I wrote to implement Chronux Analysis
   edit mtspectrumpt         % frequency spectrum for point process times
   edit mtspectrumsegpt      % f spectrum split into segments and averaged
   edit mtspectrumsegpt_loop % allow segmented spectrum calculation for multiple channels.
   edit mtspecgrampt         % (time)x(freq)x(chan) spectrogram
   edit mtspecgramtrigpt     % event triggered spectrogram
   %
   edit cohgrampt        %
   edit cohgramcpt       %
   edit coherencypt      % Figure out how to use and ...
   edit coherencycpt     %
   edit coherencysegpt   % Explain these!
   edit coherencysegcpt  % 
   edit cohmatrixpt      %
   edit CrossSpecMatpt   %
   %
   edit mtdspectrumpt    % derivative of 1D frequency spectrum
   edit mtdspecgrampt    % derivative of 2d time-freq spectrogram
end

%% edit Oscillation Score analysis functions ... Ovidiu Jurjut (orig. Raul Muresan)
if(0)
    % edit exploreDataOscillationScore
    edit exploreDataOscillationScoreB
    edit OScoreSpikes
    edit OScoreACpeak
    %
    edit OS_max_timeseries
    edit OSmargAnalNorm
    edit OS_simple_stats
end
%


%% Functions Ive written or edited to do adminastrative things.  Dont need to view usually.
if(0)
    % edit function to determine if on Cortex cluster and set home dir accordingly
    edit onCluster

    % edit function to save nice image using Ghostscript
    edit saveGoodImg
end