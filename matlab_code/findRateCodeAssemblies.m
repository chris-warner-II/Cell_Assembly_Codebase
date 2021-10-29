function [mostActive, params] = findRateCodeAssemblies(timeWin, numActive)

% THIS IS FROM THE FIRST TIME WE LOOKED AT GREG FIELD'S DATA. SHOWN WHITE
% NOISE. OLD STUFF. BETTER AND NEWER IS EXPLORE_DATA.M
%
% File: datarun.mat Not raw data.  It has been processed to include spike 
% times and a functional classification of the neurons that we recorded from 
% (there are ~10 cell types we can distinguish). Contains 1 hour of recording
% (the 2nd hour of recording)
% 
% datarun.triggers has the stimulus triggers.  After 100 frames that are 
% displayed, a trigger is sent to make sure a frame is not dropped.  
% So there should be ~1.6 seconds between triggers (stimulus refresh is at 
% 60 Hz).  The visual stimulus was spatial-temporal white noise.
% 
% datarun.spikes is a Cell Array containing the spike times for every 
% neuron that we recorded in this piece of retina.  There are 429 neurons.  
% 
% datarun.cell_ids is a list of ID numbers for every neuron.
% 
% datarun.cell_types contains information about which neurons belong to 
% which cell type ? the neurons are tracked by the ID.
% 
% For example, datarun.cell_types{8} gives you the type name (off t3) and 
% cell IDs for cells of this type.  The first 7 entries in data 
% run.cell_types are empty for reasons that are not worth explaining right 
% now.  But if you start with the 8th entry in the cell array, everything 
% makes sense.  
% 
% I encourage you to analyze first the following example cells:
% 
% cell type: on t1
% 	cells IDs:  1507, 3483, 9085
% 
% cell type: on t2
% 	cell IDs: 948, 2672, 6782
% 
% cell type: off t5
% 	cell IDs: 17, 1952, 4669
% 
% cell type: off t2
% 	cell IDs: 603, 2537, 4022
%
% Note: sampling rate in Hz (20 kHz)


%% The basics (load file. get number of cells. etc.)
load Field_data % mat file which includes data
recTime = 3600; % length of recording (in seconds) - 1hr.

if ~exist('timeWin','var')
    timeWin = 1; % (in seconds) - sliding window thru hour long window
end
numCells = numel(datarun.spikes);
cellSearch = 1:numCells; % look at all cells. Could look at subset too.

spikesInHr = zeros(1,numCells); % How many spikes each cell fired in hour of data

% find average spike rate across population over whole recording time
% of neurons to choose sparseness of binary Spike event matrix.
sumSpikes = 0; % initialize counter for total number of spikes across population.
for i = 1:numCells % Loop through the 429 cells collected from.
    sumSpikes = sumSpikes + numel(datarun.spikes{i});
    spikesInHr(i) = numel(datarun.spikes{i});
end


slideWin = 0:timeWin:3600; % construct a vector of beg and end vals to step thru entire data.
% numTimeSteps = numel(slideWin)-1;
numTimeSteps = 500; % can look at a subset of time steps and not entire hour.

% record most active (with firing rates above their mean) neurons in time bin
mostActiveRel = zeros(numTimeSteps,numActive);
mostActiveAbs = zeros(numTimeSteps,numActive);


for k = 1:numTimeSteps

    beg = slideWin(k); % seconds of collection to begin analysis
    fin = slideWin(k+1); % seconds of collection to finish analysis
    
    disp(['Processing data: ',num2str(beg),'-',num2str(fin),' seconds. (Total = ',num2str(recTime), ' seconds)'])

    spikesInWin = zeros(1,numCells); % How many spikes each cell fired in window
    
    sparseness = sumSpikes/(recTime*1000*numCells);

    % make a sparse matrix to hold binary spike words for all cells recorded
    ms_bins = round((fin-beg)*1000); % convert seconds into milliseconds
    binSpikes = spalloc(numCells, ms_bins, ceil(5*sparseness));


    %% Convert spike times into binary words of whether cells spiked or not in a 1ms time bin.
    for i = 1:numCells % Loop through the 429 cells collected from.
        spikeTimes = datarun.spikes{i} - 3600; % subtract off 1 hour (because data from 2nd hr of recording)
        sTinWin = find( (spikeTimes > beg) & (spikeTimes < fin) );
        spikesInWin(i) = numel(sTinWin);

       for j = 1:numel(sTinWin) % Step through spikeTimes found in time segment.
           binSpikes( i, ceil( (spikeTimes(sTinWin(j))-beg)*1000 ) ) = 1;
       end

    %    keyboard
       clear x spikeTimes

    end
    
    %% Plot Spikes within time window (like Rate Code)
    meanfire = spikesInHr*timeWin/3600; % mean firing rate normalized to window size
    rateVmean = spikesInWin-meanfire;        %

    % Plot Avg Firing Rate (over hour), rate over time window, and their difference
    if(0)
        figure, hold on 
        subplot(311), stem(meanfire), title('Normalized Average Spike Rate (Across Hour)')
        axis([0 numCells+1 0 2*max(meanfire(:))])
        subplot(312), stem(spikesInWin), title(['Spike Rate from ',num2str(beg),'-',num2str(fin),' seconds.'])
        axis([0 numCells+1 0 2*max(meanfire(:))])
        subplot(313), stem(rateVmean,'k'), title(['Spike Rate & Mean (over Hour) Difference.'])
        axis([0 numCells+1 -max(meanfire(:)) max(meanfire(:))])
        set(gcf,'Units','normalized'), set(gcf,'Position',[0 0 1 1])
    end

    % Looking for assemblies of activity. Find a most active cells in time bin.
    [rateRel,indRel]= sort(rateVmean,'descend');
    mostActiveRel(k,:) = indRel(1:numActive); % Active relative to mean firing rate (over hour long recording)
    [rateAbs,indAbs]= sort(spikesInWin,'descend');
    mostActiveAbs(k,:) = indAbs(1:numActive); % Active means absolute firing rate (over short time window)
        
end

params = 1; % fill this with something useful at some point.


%% Convert Most Active Array to be a binary one to display which neurons are most active.
% Could put this in a separate function called display RateCodedAssemblies.
if(1)
    
mostActiveRelRaster = spalloc(numTimeSteps,numCells, numActive*numTimeSteps);
mostActiveAbsRaster = spalloc(numTimeSteps,numCells, numActive*numTimeSteps);

for k = 1:numTimeSteps
    
    mostActiveRelRaster(k,mostActiveRel(k,:)) = 1;
    mostActiveAbsRaster(k,mostActiveAbs(k,:)) = 1;
    
end

figure, 
subplot(211), imagesc(1 - mostActiveRelRaster'), % to show spikes in black
title(['Top ',num2str(numActive),' Most Active Cells (relative to mean rate over entire hour)'])
% xlabel(['Time Bin = ',num2str(timeWin),' sec'])
ylabel('Neuron #')
set(gca,'XtickLabel',[slideWin(numTimeSteps+1)/10:10:slideWin(numTimeSteps+1)]);
%
subplot(212), imagesc(1 - mostActiveAbsRaster')
title(['Top ',num2str(numActive),' Most Active Cells (absolute firing rate in time window)'])
xlabel(['Time in Seconds (Bin Size = ',num2str(timeWin),' sec)'])
ylabel('Neuron #')
set(gca,'XtickLabel',[slideWin(numTimeSteps+1)/10:10:slideWin(numTimeSteps+1)]);
%
colormap('bone')
set(gcf,'Units','normalized'), set(gcf,'Position',[0 0 1 1])

end


keyboard

