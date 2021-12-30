%% Exploratory Code to analyze Retina Data sent by Greg Field with white noise and natural stimulus.
%
%   This code takes in raw data in mat file provided by Greg Field, does some preprocessing and
%   reshapes data structures into a form that will be friendly with python and then saves them 
%   to mat files. Also, this code makes ellipse plots of cell receptive fields for the different
%   cell types.
%
% - - - - - - - Variables included in Greg Field's Data Files. - - - - - - - - - - - 
%
% triggers_wnrep            - timestamps when each of the 200 trials from repeated whitenoise stimulus began.
% spikes_wnrep              - spiketimes relative to stimulus beginning for each of cell (is a matlab cell of size num_cells)
% movie_wnrep               - movie of the actual stimulus presented during this experiment.
% 
% triggers_catrep           - natural movie stimulus (cat cam)
% spikes_catrep             - (same as above)
% movie_catrep              - (same as above)
% 
% triggers_wn               - Non-repeated white noise stimulus (what does trigger mean in this context?)
% spikes_wn                 - (same as above)
% movie_wn                  - (same as above)
% 
% stas                      - spatio-temporal receptive fields gotten by spike triggered averaging on white noise movie repeats.
% refresh_time              - monitor refresh rate.
% indices_for_typed_cells   - index of cells included in each cell type
%
%                   Cell Types
%     offBriskTransient: 55 cells
%     offBriskSustained: 43 cells
%          offExpanding: 13 cells
%          offTransient:  4 cells
%      onBriskTransient: 39 cells
%      onBriskSustained:  6 cells
%           onTransient:  7 cells
%           dsOnoffDown:  7 cells
%          dsOnoffRight:  3 cells
%           dsOnoffLeft:  3 cells
%             dsOnoffUp:  2 cells



%% Setup some directory structure
projsDir  = pwd; %RUN THIS FROM PROJECTS DIRECTORY. Must call addProjPaths before running this.
codeDir   = [projsDir,'/G_Field_Retinal_Data/home/G_Field_Retinal_Data/Chris_working_code_2018/matlab_code/'];  % Directory where my code resides
inputDir = [projsDir,'/G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/data/GField_data/'];                   % Directory where Greg's raw data sits.
outDataDir = [projsDir,'/G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/data/matlab_data/']; % Directory to save output data from this.
outFigsDir = [projsDir,'/G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/figs/older_elephant/']; % Directory to save output figures from this.


if ~exist(outDataDir,'dir')
    mkdir(outDataDir)
end

if ~exist(outFigsDir,'dir')
    mkdir(outFigsDir)
end


%% The basics (load file. get number of cells. etc.)
fname = 'all_cells_2017_01_16_0'; % [[ Note: retina_data_2018 and retina_data2_2018 are subsets of this data set!! ]]


disp(['Load raw retina data mat file: ',fname])
load([inputDir,fname,'.mat'])  % mat file which includes data 




stims = {'White Noise','Natural Movie'};

num_cells = numel(stas);
num_trials = numel(triggers_catrep);
YdimImg = size(stas{1},1);
XdimImg = size(stas{1},2);

cellType = 'allCells';


plotInfo.plotFlag = 1; % show and save plots when making determining receptive fields.
plotInfo.outputDir = outFigsDir;


%% (1). Split spike data up into a [cell_num x trial_num] data cell array. 
%       This is used as input to python code!
if(1)
    if exist([outDataDir,cellType,'_spikeTrains_CellXTrial.mat'],'file')
        load([outDataDir,cellType,'_spikeTrains_CellXTrial.mat'])
    else
        %
        % Extract Cell Types and Indices of cells belonging to each category.
        allCellTypes = fieldnames(indices_for_typed_cells);
        maxCells=0;
        for i = 1:numel(allCellTypes)
            eval(['numCells=numel(indices_for_typed_cells.',allCellTypes{i},');'])
            maxCells = max(numCells,maxCells);
        end
        %
        cellTypeIDs = zeros( numel(allCellTypes), maxCells );
        for i = 1:numel(allCellTypes)
            eval(['numCells=numel(indices_for_typed_cells.',allCellTypes{i},');'])
            eval(['cellTypeIDs(i,1:numCells)=indices_for_typed_cells.',allCellTypes{i},';'])
        end
        %
        %
        % Pack spikes into a cell which becomes an NDARRAY in python.
        triggers_wnrep(num_trials+1) = 1e16;
        triggers_catrep(num_trials+1) = 1e16; % because last trigger is beginning of the last trial
        %
        for j = 1:num_trials
            j
            for i = 1:num_cells
                indWn = find( (spikes_wnrep{i}>=triggers_wnrep(j)) & (spikes_wnrep{i}<triggers_wnrep(j+1)) );
                spikesInWn{i,j} = ceil(1000.*(spikes_wnrep{i}(indWn) - triggers_wnrep(j))');
                %
                indCat = find( (spikes_catrep{i}>=triggers_catrep(j)) & (spikes_catrep{i}<triggers_catrep(j+1)) );
                spikesInCat{i,j} = ceil(1000.*(spikes_catrep{i}(indCat) - triggers_catrep(j))');
            end
        end
        %
        triggers_wnrep = round(1000.*triggers_wnrep(1:num_trials));
        triggers_catrep = round(1000.*triggers_catrep(1:num_trials)); % get rid of added entry at beginning of this section.
        %
        save([outDataDir,cellType,'_spikeTrains_CellXTrial.mat'],...
                    'spikesInWn', 'spikesInCat', 'triggers_wnrep', 'triggers_catrep','cellTypeIDs','allCellTypes')
    end
end










%% (2). Split spike data up into a [cell_num x trial_num] data cell array in order to do some
%       Spike Triggered Averaging on the White Noise input. 
%       This is used as input to python code!
if(0)
    if exist([outDataDir,cellType,'_spikeTrains_CellXTrial_Wn4STA.mat'],'file')
        load([outDataDir,cellType,'_spikeTrains_CellXTrial_Wn4STA.mat'])
    else
        num_trials = numel(triggers_wn);
        %
        % Extract Cell Types and Indices of cells belonging to each category.
        allCellTypes = fieldnames(indices_for_typed_cells);
        maxCells=0;
        for i = 1:numel(allCellTypes)
            eval(['numCells=numel(indices_for_typed_cells.',allCellTypes{i},');'])
            maxCells = max(numCells,maxCells);
        end
        %
        cellTypeIDs = zeros( numel(allCellTypes), maxCells );
        for i = 1:numel(allCellTypes)
            eval(['numCells=numel(indices_for_typed_cells.',allCellTypes{i},');'])
            eval(['cellTypeIDs(i,1:numCells)=indices_for_typed_cells.',allCellTypes{i},';'])
        end
        %
        %
        % Pack spikes into a cell which becomes an NDARRAY in python.
        triggers_wn(num_trials+1) = 1e16; % because last trigger is beginning of the last trial
        %
        for j = 1:num_trials
            j
            for i = 1:num_cells
                indWn = find( (spikes_wn{i}>=triggers_wn(j)) & (spikes_wn{i}<triggers_wn(j+1)) );
                spikesInWn_4STA{i,j} = ceil(1000.*(spikes_wn{i}(indWn) - triggers_wn(j))');
            end
        end
        %
        triggers_wn = round(1000.*triggers_wn(1:num_trials)); % get rid of added entry at beginning of this section.
        %
        save([outDataDir,cellType,'_spikeTrains_CellXTrial_Wn4STA.mat'],...
                    'spikesInWn_4STA', 'triggers_wn', 'cellTypeIDs','allCellTypes')
    end
end




%% Visualize movies of the STAs for the different cell types.
if(0)
    
    % 'offBriskTransient'
    % 'offBriskSustained'
    % 'offExpanding'
    % 'offTransient'
    % 'onBriskTransient'
    % 'onBriskSustained'
    % 'onTransient'
    % 'dsOnoffDown'
    % 'dsOnoffRight'
    % 'dsOnoffLeft'
    % 'dsOnoffUp'
    cellTypesForSTAmovies = {'offBriskSustained'};
    
    for k = 1:numel(cellTypesForSTAmovies)
        c=0;
        h=figure;
        for i = indices_for_typed_cells.(cellTypesForSTAmovies{k})
            c = c+1;
            disp([cellTypesForSTAmovies{k},num2str(c),'_cell',num2str(i)])
            matrix2video(stas{i})
            title([cellTypesForSTAmovies{k},num2str(c),'_cell',num2str(i)])
            pause
        end
    end
 
end




%% Plot temporal responses of cells of each type.
figure, 
subplot(131)
plot(STRF_TimeParamsL(indices_for_typed_cells.onBriskTransient,:)')
ax = gca;
ax.FontSize = 16; 
title('onBT','fontsize',20,'fontweight','bold')
ylabel('Cell Temporal Responses','fontsize',16,'fontweight','bold')
xlabel('STA movie "frame" ','fontsize',16,'fontweight','bold')
grid('on')
%
subplot(132)
plot(STRF_TimeParamsD(indices_for_typed_cells.offBriskTransient,:)')
ax = gca;
ax.FontSize = 16; 
title('offBT','fontsize',20,'fontweight','bold')
grid('on')
%
subplot(133)
plot(STRF_TimeParamsD(indices_for_typed_cells.offBriskSustained,:)')
ax = gca;
ax.FontSize = 16; 
title('offBS','fontsize',20,'fontweight','bold')
grid('on')





%% (2). Compute parameters of RFs (center position, STD, orientation angle) seperately for both ON & OFF parts of Biphasic response
%       This is used as input to python code!
if(0)
    if exist([outDataDir,cellType,'_STRF_fits_',num2str(num_cells),'cells.mat'],'file')
        % ~exist('STRF_GaussParams', 'var')
        load([outDataDir,cellType,'_STRF_fits_',num2str(num_cells),'cells.mat'])
    else
        disp('Computing STRF Fit Parameters')
        STRF_GaussParams = zeros(num_cells,6);
        STRF_TimeParams = zeros(num_cells,30);
        %
        for k = 1:numel(allCellTypes)
            c=0;
            for i = indices_for_typed_cells.(allCellTypes{k})  %1:num_cells
                c = c+1;
                cell_sta = stas{i};
                plotInfo.cellID = [allCellTypes{k},num2str(c),'_cell',num2str(i)];
                [xDLM, tDLM] = STRF_fit(cell_sta, plotInfo);
                %
                % Only grab Dark Fits (since these are OffBriskTransient cells) and
                % turn cell into a matrix (6xN for Gaussian fit parameters; 30xN for RF
                % temporal profile fit
                STRF_GaussParamsD(i,:) = xDLM(1,:);
                STRF_GaussParamsL(i,:) = xDLM(2,:);
                STRF_GaussParamsM(i,:) = xDLM(3,:);
                STRF_TimeParamsD(i,:) = tDLM(1,:);
                STRF_TimeParamsL(i,:) = tDLM(2,:);
                STRF_TimeParamsM(i,:) = tDLM(3,:);
            end
        
        end
        %
        % SAVE STRF PARAMS HERE! (dont save structs because they are difficult to deal with in Python)
        save([outDataDir,cellType,'_STRF_fits_',num2str(num_cells),'cells'],'STRF_GaussParamsD','STRF_GaussParamsL','STRF_GaussParamsM',...
            'STRF_TimeParamsD','STRF_TimeParamsL','STRF_TimeParamsM')
    end
end












































%
%
%
%
%
%
%% PLOTS BELOW HERE. AND STUFF I AM NOT PASSING FORWARD INTO PYTHON.  DONT NEED NECESSARILY. 
%       It is all marked if(0) anyway.
%
%
%
%






%% (1). number of spikes for each cell across all trials for each stimulus
if(0)
    num_spikes = zeros(2,num_cells);
    for i = 1:num_cells
        num_spikes(1,i) = numel(spikes_wnrep{i});
        num_spikes(2,i) = numel(spikes_catrep{i});
    end
end





%% (3). Histogram spike times across all trials for each cell and for all cells.
if(0)
    disp('plot Histogram spike times across all trials for each cell and for all cells.')
    clines = lines(num_cells);
    h=figure;
    subplot(211), hold on, 
    for i = 1:num_cells
        [N,X] = hist([spikesInWn{i,:}],5000); plot(X,N./sum(N),'Color',clines(i,:),'LineWidth',0.5) 
    end
    [N,X] = hist([spikesInWn{:}],5000); plot(X,N./sum(N),'k','LineWidth',3)
    xlim([0 max([diff(triggers_catrep);diff(triggers_wnrep) ]) ])
    title(['Stimulus Locked Spikes in ',cellType,' w/ White Noise'],'FontSize',20,'FontWeight','Bold')
    %xlabel('Time (msec)','FontSize',18,'FontWeight','Bold')
    ylabel('(normalized by total # spikes)','FontSize',18,'FontWeight','Bold')
    set(gca,'FontSize',16,'FontWeight','Bold')
    % 
    subplot(212),hold on,
    for i = 1:num_cells
        [N,X] = hist([spikesInCat{i,:}],5000); 
        plot(X,N./sum(N),'Color',clines(i,:),'LineWidth',0.5)
    end
    [N,X] = hist([spikesInCat{:}],5000); 
    plot(X,N./sum(N),'k','LineWidth',3)
    text(3500,0.025, {'Colors = Individual Cells', 'Black = Avg all ',num_cells,' Cells'},'FontSize',16,'FontWeight','Bold')
    
    xlim([0 max([diff(triggers_catrep);diff(triggers_wnrep) ]) ])
    title(['w/ Natural Movie'],'FontSize',20,'FontWeight','Bold')
    xlabel('Time (msec)','FontSize',18,'FontWeight','Bold')
    ylabel('P(spike)','FontSize',18,'FontWeight','Bold')
    set(gca,'FontSize',16,'FontWeight','Bold')
    saveGoodImg(h,[outputDir,'spike_train_plots/',cellType,'_StimLockedSpiking'],[0 0 1 1])
    close(h)
end



%% (3). Compute spike overlaps from trial-to-trial in each single cell that are stimulus locked (note: this is to msec precision).
if(0)
    disp('Compute spike overlaps from trial-to-trial in each single cell that are stimulus locked.')
    if exist([outputDir,'matlab_data/',cellType,'_PairedSpikes_in_TrialPairs.mat'],'file')
        load([outputDir,'matlab_data/',cellType,'_PairedSpikes_in_TrialPairs.mat'])
    else
        num_trial_pairs = num_trials*(num_trials-1)/2;
        %
        WhatAreThese.overview = 'Trying to find spikes shared in each single cell across trial pairs meaning they are stimulus locked (with msec precision)';
        WhatAreThese.trialPairs = '(200*199)/2 each unique pair of trials';
        WhatAreThese.pairedSpikeTimes = 'The time in msec (relative to stim onset) of each spike shared. {Wn=white noise, Cat=natural movie}';
        WhatAreThese.numPairedSpikes = 'Number of shared spikes between trial pairs (conting up # of pairedSpikeTimes) {Wn=white noise, Cat=natural movie}';
        %
        trialPairs = zeros(2,num_trial_pairs);
        pairedSpikeTimes.Wn = cell(num_cells, num_trial_pairs);
        pairedSpikeTimes.Cat = cell(num_cells, num_trial_pairs);
        numPairedSpikes.Wn = zeros(num_cells, num_trial_pairs);
        numPairedSpikes.Cat = zeros(num_cells, num_trial_pairs);
        cntr = 0;
        %
        for j = 1:num_trials-1
            for k = j+1:num_trials
                cntr = cntr+1;
                cntr/num_trial_pairs
                StimLocked.trialPairs(:,cntr) = [j;k];
                for i = 1:num_cells
                    indPairC = find( ismember( round( spikesInCat{i,j}.*1000 ), round( spikesInCat{i,k}.*1000 ) ) );
                    pairedSpikeTimes.Cat{i,cntr} = spikesInCat{i,j}(indPairC);
                    numPairedSpikes.Cat(i,cntr) = numel(indPairC);
                    %
                    indPairW = find( ismember( round( spikesInWn{i,j}.*1000 ), round( spikesInWn{i,k}.*1000 ) ) );
                    pairedSpikeTimes.Wn{i,cntr} = spikesInWn{i,j}(indPairW);
                    numPairedSpikes.Wn(i,cntr) = numel(indPairW);
                end
            end
        end
        %
        save([outputDir,'matlab_data/',cellType,'_PairedSpikes_in_TrialPairs'], ...
                'WhatAreThese', 'trialPairs', 'pairedSpikeTimes', 'numPairedSpikes')

    end
end



%% (4). Stimulus-Locked Spike Reliability: Find/Plot spike times that are shared across trials - colorcoded for number of trials spike repeats in.
if(0) % NOT SURE THIS IS ALL THAT USEFUL.  LOOK AT FIGURE AGAIN TO UNDERSTAND IT BEFORE WORKING ON IT.
    for i = 1:num_cells

        h=figure; 
        %
        % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
        %
        % (0). Determine colorbar and colormap based on mean, standard deviations, max.
        sharedWn = [spikesInWn{i,:}]; 
        sharedT_Wn = unique(sharedWn); % Spike times shared across at least 2 trials in White Noise Stimulus.
        numSharedSpikeTime_Wn = zeros(size(sharedT_Wn));
        for j = 1:numel(sharedT_Wn)
            j./numel(sharedT_Wn)
            numSharedSpikeTime_Wn(j) = numel(find(ismember(sharedWn , sharedT_Wn(j))));
        end

        sharedCat = [spikesInCat{i,:}];
        sharedT_Cat = unique(sharedCat); % Spike times shared across at least 2 trials in White Noise Stimulus.
        numSharedSpikeTime_Cat = zeros(size(sharedT_Cat));
        for j = 1:numel(sharedT_Cat)
            j./numel(sharedT_Cat)
            numSharedSpikeTime_Cat(j) = numel(find(ismember(sharedCat , sharedT_Cat(j))));
        end
        %
        numShared = [numSharedSpikeTime_Cat,numSharedSpikeTime_Wn];
        cbar_tick = round([0, mean(numShared)+[0,1,2]*std(numShared), max(numShared) ]);
        maxC = numel(cbar_tick); % which entry in cbar_tick to set colormap to. (0, mn, mn+std, mn+2std, max)
        cmap = jet( cbar_tick(maxC) ); % set colormap so that range of colors
        %
        % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
        % 
        % (1). Plot spike raster with White Noise stimulus.
        sp1= subplot(2,10,[1:7]); hold on,
        for j = 1:num_trials
            j
            scatter( spikesInWn{i,j}, repmat(j,size(spikesInWn{i,j}) ),15, 'k.' ) % Plot all Spikes in all trials for Cell #i in black.
            %
            A = (StimLocked.trialPairs==j);
            inds = find( A(1,:)|A(2,:) );
            shared = [StimLocked.spikeTimes.Wn{i,inds}]; % spike times when trial #j that are shared with other trials. Colorcode by #trials it overlaps with.
            sharedSpikesInTrialWn(j) = numel(shared);
            numShared = histc( shared , unique(shared) );
            scatter( unique(shared), repmat(j,size(numShared) ), 15, cmap(min([numShared;repmat(cbar_tick(maxC),size(numShared))]),:), 'Marker','.' )
            %keyboard
        end
        %
        title(['Stimulus Locked Spikes in ',cellType,' Cell#',num2str(i),' w/ White Noise'],'FontSize',20,'FontWeight','Bold')
        %xlabel('Time (msec)','FontSize',18,'FontWeight','Bold')
        %ylabel('Trial #','FontSize',18,'FontWeight','Bold')
        set(gca,'FontSize',16,'FontWeight','Bold')
        axis([0 5500 1 200])
        %
        % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
        %
        % (2). Plot spike raster with Cat Natural Movie stimulus.
        sp2= subplot(2,10,[11:17]); hold on,
        for j = 1:num_trials
            j
            scatter( spikesInCat{i,j}, repmat(j,size(spikesInCat{i,j}) ),15, 'k.' ) % Plot all Spikes in all trials for Cell #i in black.
            %
            A = (StimLocked.trialPairs==j);
            inds = find( A(1,:)|A(2,:) );
            shared = [StimLocked.spikeTimes.Cat{i,inds}]; % spike times when trial #j that are shared with other trials. Colorcode by #trials it overlaps with.
            sharedSpikesInTrialCat(j) = numel(shared);
            numShared = histc( shared , unique(shared) );
            scatter( unique(shared), repmat(j,size(numShared) ), 15, cmap(min([numShared;repmat(cbar_tick(maxC),size(numShared))]),:), 'Marker','.' )
            %keyboard
        end
        %
        title(['w/ Natural Movie'],'FontSize',20,'FontWeight','Bold')
        xlabel('Time (msec)','FontSize',18,'FontWeight','Bold')
        ylabel('Trial #','FontSize',18,'FontWeight','Bold')
        set(gca,'FontSize',16,'FontWeight','Bold')
        axis([0 5500 1 200])
        cbh=colorbar('East');
        set( cbh,'YTick',linspace(0,1,maxC),'YTickLabel',cbar_tick(1:maxC),'FontSize',16,'FontWeight','Bold' )
        ylabel(cbh,'#trials spike reliable')
        %
        % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
        %
        % (3). Plot histograms of shared spikes for all trials for white noise stimulus & Cat Natural Movie.
        [Xw,Yw] = hist(numSharedSpikeTime_Wn,num_trials/10);
        [Xc,Yc] = hist(numSharedSpikeTime_Cat,num_trials/10);
        sp3 = subplot(2,10,[8:10,18:20]); hold on,
        plot(Yw./num_trials,Xw,'b','LineWidth',2)
        plot(Yc./num_trials,Xc,'r','LineWidth',2)
        legend({'white noise','natural movie'})
        set(gca,'FontSize',16,'FontWeight','Bold','YAxisLocation','right'); %'XTick',cbar_tick); % ,'XTickLabel',{'0',['\mu = ',num2str(cbar_tick(2))], ['\mu+\std = ',num2str(cbar_tick(3))], ['\mu+2\std = ',num2str(cbar_tick(4))], num2str(cbar_tick(5))}
        title('Reliable Spikes Histogram','FontSize',18,'FontWeight','Bold')
        xlabel('% trials spike reoccurs','FontSize',18,'FontWeight','Bold')
        ylabel('# spikes','FontSize',18,'FontWeight','Bold')
        grid on
        axis tight
        saveGoodImg(h,[outputDir,'spike_train_plots/singleCell_allTrials/StimLockedSpikeRaster_',cellType,num2str(i),'_colorcode_wnoiseVnatmov'],[0 0 1 1])
        close(h)
    end
end




%% Make Stimulus movies (Already Saved them)
if(0)
    filename = [outputDir,'stim_movies/movie_catrep.avi'];         % (1). Natural Scenes Cat
    matrix2avi(movie_catrep,'file',filename,'map','bone','fps',10)

    filename = [outputDir,'stim_movies/movie_wnrep.avi'];          % (2). Binary White Noise
    matrix2avi(round(movie_wnrep),'file',filename,'map','bone','fps',10)
end











%% Plot all RF ellipses on one figure.
if(1)
    if ~exist('STRF_GaussParamsD', 'var')
        load([outDataDir,cellType,'_STRF_fits_',num2str(num_cells),'cells'])
    end
    %
    for k = 1:numel(allCellTypes)
        %
        % Plot Dark Response RF ellipses.
        hf=figure; hold on, axis square
        a=0;
        for i = indices_for_typed_cells.(allCellTypes{k})
            a=a+1;
            x = STRF_GaussParamsD(i,:);
            el=ellipse( x(3), x(5), 2*pi-x(6), x(2), x(4),'k'); % h=ellipse(ra,rb,ang,x0,y0,C,Nb)
            set(el,'LineWidth',2);
            text(x(2),x(4),num2str(a),'FontSize',20,'FontWeight','Bold','VerticalAlignment','Middle','HorizontalAlignment','Center','Color','r')
        end
        axis ij
        grid on
        title([num2str(numel(allCellTypes{k})),' ',allCellTypes{k},' Receptive Fields Tile Space (Dark Response)'],'FontSize',20,'FontWeight','Bold')
        %
        saveGoodImg(hf,[outFigsDir,'STRF_fit_plots/',allCellTypes{k},'_RFsDark_tile_space'],[0 0 0.7 0.8])
        close(hf)
        % 
        % %
        %
        % Plot Light Response RF ellipses.
        hf=figure; hold on, axis square
        a=0;
        for i = indices_for_typed_cells.(allCellTypes{k})
            a=a+1;
            x = STRF_GaussParamsL(i,:);
            el=ellipse( x(3), x(5), 2*pi-x(6), x(2), x(4),'k'); % h=ellipse(ra,rb,ang,x0,y0,C,Nb)
            set(el,'LineWidth',2);
            text(x(2),x(4),num2str(a),'FontSize',20,'FontWeight','Bold','VerticalAlignment','Middle','HorizontalAlignment','Center','Color','r')
        end
        axis ij
        grid on
        title([num2str(numel(allCellTypes{k})),' ',allCellTypes{k},' Receptive Fields Tile Space (Light Response)'],'FontSize',20,'FontWeight','Bold')
        %
        saveGoodImg(hf,[outFigsDir,'STRF_fit_plots/',allCellTypes{k},'_RFsLight_tile_space'],[0 0 0.7 0.8])
        close(hf)
                % 
        % %
        %
        % Plot Light Response RF ellipses.
        hf=figure; hold on, axis square
        a=0;
        for i = indices_for_typed_cells.(allCellTypes{k})
            a=a+1;
            x = STRF_GaussParamsM(i,:);
            el=ellipse( x(3), x(5), 2*pi-x(6), x(2), x(4),'k'); % h=ellipse(ra,rb,ang,x0,y0,C,Nb)
            set(el,'LineWidth',2);
            text(x(2),x(4),num2str(a),'FontSize',20,'FontWeight','Bold','VerticalAlignment','Middle','HorizontalAlignment','Center','Color','r')
        end
        axis ij
        grid on
        title([num2str(numel(allCellTypes{k})),' ',allCellTypes{k},' Receptive Fields Tile Space (Mean Response)'],'FontSize',20,'FontWeight','Bold')
        %
        saveGoodImg(hf,[outFigsDir,'STRF_fit_plots/',allCellTypes{k},'_RFsMean_tile_space'],[0 0 0.7 0.8])
        close(hf)
    end
end
    



%% Plot all RF ellipses on one figure for {offBriskTransient,onBriskTransient}.
%                           and also for {offBriskTransient,offBriskSustained}
if(1)
    if ~exist('STRF_GaussParamsD', 'var')
        load([outDataDir,cellType,'_STRF_fits_',num2str(num_cells),'cells'])
    end
    %
    %
    % Plot offBriskTransient Dark Response and onBriskTransient Light Response RF ellipses.
    hf=figure; hold on, axis square
    a=0;
    for i = indices_for_typed_cells.offBriskTransient
        a=a+1;
        x = STRF_GaussParamsD(i,:);
        el=ellipse( x(3), x(5), 2*pi-x(6), x(2), x(4),'r'); % h=ellipse(ra,rb,ang,x0,y0,C,Nb)
        set(el,'LineWidth',2);
        text(x(2),x(4),num2str(a),'FontSize',10,'FontWeight','Bold','VerticalAlignment','Middle','HorizontalAlignment','Center','Color','r')
    end
    %
    b=0;
    for i = indices_for_typed_cells.onBriskTransient
        b=b+1;
        x = STRF_GaussParamsL(i,:);
        el=ellipse( x(3), x(5), 2*pi-x(6), x(2), x(4),'b'); % h=ellipse(ra,rb,ang,x0,y0,C,Nb)
        set(el,'LineWidth',2);
        text(x(2),x(4),num2str(b),'FontSize',10,'FontWeight','Bold','VerticalAlignment','Middle','HorizontalAlignment','Center','Color','b')
    end
    axis ij
    grid on
    
    %title(['\fontsize{16}black {\color{magenta}magenta ','\color[rgb]{0 .5 .5}teal \color{red}red} black again'],'interpreter','tex')
    
    title(['{\color{red}',num2str(a),' offBriskTransient} & {\color{blue}',num2str(b),' onBriskTransient} Receptive Fields Tile Space'],'FontSize',20,'FontWeight','Bold','interpreter','tex')
    %
    saveGoodImg(hf,[outFigsDir,'STRF_fit_plots/offBriskTransient_onBriskTransient_RFs_tile_space'],[0 0 0.7 0.8])
    close(hf)
    
    
    % Plot offBriskTransient Dark Response and offBriskSustained Dark Response RF ellipses.
    hf=figure; hold on, axis square
    a=0;
    for i = indices_for_typed_cells.offBriskTransient
        a=a+1;
        x = STRF_GaussParamsD(i,:);
        el=ellipse( x(3), x(5), 2*pi-x(6), x(2), x(4),'r'); % h=ellipse(ra,rb,ang,x0,y0,C,Nb)
        set(el,'LineWidth',2);
        text(x(2),x(4),num2str(a),'FontSize',10,'FontWeight','Bold','VerticalAlignment','Middle','HorizontalAlignment','Center','Color','r')
    end
    %
    b=0;
    for i = indices_for_typed_cells.offBriskSustained
        b=b+1;
        x = STRF_GaussParamsD(i,:);
        el=ellipse( x(3), x(5), 2*pi-x(6), x(2), x(4),'b'); % h=ellipse(ra,rb,ang,x0,y0,C,Nb)
        set(el,'LineWidth',2);
        text(x(2),x(4),num2str(b),'FontSize',10,'FontWeight','Bold','VerticalAlignment','Middle','HorizontalAlignment','Center','Color','b')
    end
    axis ij
    grid on
    
    %title(['\fontsize{16}black {\color{magenta}magenta ','\color[rgb]{0 .5 .5}teal \color{red}red} black again'],'interpreter','tex')
    
    title(['{\color{red}',num2str(a),' offBriskTransient} & {\color{blue}',num2str(b),' offBriskSustained} Receptive Fields Tile Space'],'FontSize',20,'FontWeight','Bold','interpreter','tex')
    %
    saveGoodImg(hf,[outFigsDir,'STRF_fit_plots/offBriskTransient_offBriskSustained_RFs_tile_space'],[0 0 0.7 0.8])
    close(hf)
    
    
    
end








%% Compare gross spiking activity (#spikes for all cells) in the two types of stimuli.
    
if(0)
    [x,ind] = sort(num_spikes(1,:));
    h=figure; hold on
    plot(num_spikes(1,ind)'./1000,'b','LineWidth',2)
    plot(num_spikes(2,ind)'./1000,'r','LineWidth',2)
    
    %errorbar
    
    legend(stims,'Location','SouthEast')
    title(['Total #Spikes of 200 Repeats of 5 sec of stimulus. ',cellType,' cells'],'FontSize',20,'FontWeight','Bold')
    ylabel('#Spikes per Second','FontSize',18,'FontWeight','Bold')
    xlabel('Cell # (sorted by Wn)','FontSize',18,'FontWeight','Bold')
    set(gca,'FontSize',16,'FontWeight','Bold')
    grid on
    %
    saveGoodImg(h,[outputDir,'spike_train_plots/',cellType,'_numspikes_wnoise_v_natmov'],[0 0 1 1])
    close(h)
end







%% Plot all RF ellipses on one figure for each stim type with lines colorcoded by num_spikes.
if(0)
    if ~exist('STRF_GaussParams', 'var')
        load([outputDir,'matlab_data/STRF_fits_',num2str(num_cells),cellType,'cells'])
    end
    %    
    colors = colormap(jet);
    num_col = size(colors,1);
    val_col = ceil((num_col-1).*num_spikes./max(num_spikes(:)));
    %
    hf=figure; 
    for j = 1:numel(stims)
        subplot(1,2,j), hold on
        for i = 1:num_cells
            x = STRF_GaussParams(i,:);
            el=ellipse( x(3), x(5), 2*pi-x(6), x(2), x(4),'k');
            set(el,'LineWidth',4);
            e2=ellipse( x(3), x(5), 2*pi-x(6), x(2), x(4),'k');
            set(e2,'LineWidth',2,'Color',colors(val_col(j,i),:));
            text(x(2),x(4),num2str(i),'FontSize',20,'FontWeight','Bold','VerticalAlignment','Middle','HorizontalAlignment','Center','Color','k')
        end
        title([stims{j}],'FontSize',20,'FontWeight','Bold') % num2str(num_cells),' ',cellType,' 
        axis([-5 XdimImg 1 YdimImg])
        axis square
    end
    %
    annotation('textbox', [0 0.9 1 0.1],'String', [num2str(num_cells),' ',cellType,' Cell RFs & Spiking Activity'], 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center','FontSize',24,'FontWeight','Bold')   
    cb=colorbar('Location','South');
    set(cb,'XTick',[1,num_col],'XTickLabel',[min(num_spikes(:)), max(num_spikes(:))],'FontSize',16,'FontWeight','Bold')
    set(get(cb,'title'),'string','# spikes','FontSize',18,'FontWeight','Bold');  
    saveGoodImg(hf,[outputDir,'STRF_fit_plots/',cellType,'_RFs_tile_space_wSpikeRate'],[0 0 1 0.8])
    close(hf)
end






%% How regular are spiketrains (single cell, many trials, both stimuli) based on stimulus.  
%  Look at spike activity over stim repeats for same cell.
if(0)
    maxTrigTime = max( [diff(triggers_wnrep); diff(triggers_catrep)] ); % to standardize plot time axes
    args.LineFormat.Color='k';
    args.LineFormat.LineWidth=2; % for spike train plotting function.
    %    
    for i = 1:num_cells
        for j = 1:num_trials-1
            indWn = find( (spikes_wnrep{i}>=triggers_wnrep(j)) & (spikes_wnrep{i}<triggers_wnrep(j+1)) );
            spikesInWn_cells{j} = (spikes_wnrep{i}(indWn) - triggers_wnrep(j))';
            %
            indCat = find( (spikes_catrep{i}>=triggers_catrep(j)) & (spikes_catrep{i}<triggers_catrep(j+1)) );
            spikesInCat_cells{j} = (spikes_catrep{i}(indCat) - triggers_catrep(j))';
        end
        
        % Plot spike raster for each individual cell over 200 repeats of each stimulus (white noise & natural movie)
        %
        h1 = figure; %hold on
        sp1 = subplot(211);
        [xPointsWn, yPointsWn] = plotSpikeRaster(spikesInWn_cells,args);
        xlim([0 maxTrigTime])
        title([cellType,' Cell #',num2str(i),' : Stim = ',stims{1}],'FontSize',20,'FontWeight','Bold')
        set(gca,'FontSize',16,'FontWeight','Bold','YTick',[0,round(num_trials/2),num_trials],'XTick',[0:5])
        %
        sp2=subplot(212);
        [xPointsCat, yPointsCat] = plotSpikeRaster(spikesInCat_cells,args);
        xlim([0 maxTrigTime])
        xlabel('Time (seconds)','FontSize',18,'FontWeight','Bold')
        ylabel('Trial #','FontSize',18,'FontWeight','Bold')
        title(['Stim = ',stims{2}],'FontSize',20,'FontWeight','Bold')
        set(gca,'FontSize',16,'FontWeight','Bold','YTick',[0,round(num_trials/2),num_trials],'XTick',[0:5])
        %
        saveGoodImg(h1,[outputDir,'spike_train_plots/singleCell_allTrials/',cellType,'_cell',num2str(i),'_spikeRaster_wnoise_v_natmov'],[0 0 1 0.8])
        close(h1)
    end
end






%% Population Spiking - Look at spiking of all cells in single trials. Save Data too.
if(0)
    plotFlag = 1;
    [ popCodeCat_sumTrials, popCodeWn_sumTrials ] = sumSpikeRastersOverTrials( spikesInCat, spikesInWn, stims, cellType, outDataDir, plotFlag );
end



%% Plot all cell RF time courses
if(0)
    clines = lines(num_cells);
    numT = size(STRF_TimeParams,2);
    h=figure; hold on
    for i = 1:num_cells
        plot(fliplr(STRF_TimeParams(i,:)),'Color',clines(i,:),'LineWidth',2)
    end
    xlabel('Time before spike (msec)','FontSize',18,'FontWeight','Bold')
    ylabel('pixel intensity in RF (a.u.)','FontSize',18,'FontWeight','Bold')
    title(['Temporal Response of ',num2str(num_cells),' ',cellType,' RGC''s'],'FontSize',20,'FontWeight','Bold')
    set(gca,'Xtick',[0:5:numT],'XtickLabel',[numT:-5:0],'FontSize',16,'FontWeight','Bold')
    %
    saveGoodImg(h,[outFigsDir,'STRF_fit_plots/',cellType,'_RFs_temporalResponse'],[0 0 1 1])
    close(h)
end


%% Make a movie of the stimulus with RFs overlayed and spiking across trials shown in color or ellipse thickness or something.

if(0)
    if ~exist('STRF_GaussParams', 'var')
        load([outDataDir,'STRF_fits_',cellType,'_',num2str(num_cells),'cells'])
    end
    %
    
    if ~exist('popCodeCat_sumTrials', 'var')
        load([outDataDir,cellType,'_booleanSpikes_cellVtimeVtrial'])
    end
    
    %    
    stim = 'Cat';
    makeMovie
    
    stim = 'Wn';
    makeMovie
end




%% Construct a #cell x #cell x #timePts tensor that shows connectivity between cell pairs at different time delays
%  Build it up from spike correlations on the millisecond timescale.
if(0)
    spikeTrains2timeDelayNetwork;
end

