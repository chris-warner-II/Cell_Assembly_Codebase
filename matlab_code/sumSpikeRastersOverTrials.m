function [ popCodeCat_sumTrials, popCodeWn_sumTrials  ] = sumSpikeRastersOverTrials...
            ( spikesInCat, spikesInWn, stims, cellType, outputDir, plotFlag )

% comments
% Constructing a spike raster that is #cells x time_ms x num_trials for both the 
% white noise stimulus and for natural movie stimulus
%

disp('Population Spiking - Look at spiking of all 55 cells in all trials')



num_cells =  size(spikesInCat,1);
num_trials =  size(spikesInCat,2);

popCodeWn = false(num_cells,5500,num_trials-1);  % note: 5500ms is enough time to capture ~5sec trials.
popCodeCat = false(num_cells,5500,num_trials-1); % note: boolean tensors.


for j = 1:num_trials-1
    j
    for i=1:num_cells
        popCodeWn(i, spikesInWn{i,j}+1, j) = true;
        popCodeCat(i, spikesInCat{i,j}+1, j) = true; % add 1 to spike time because sometimes spike occurs at exactly 0ms.
    end
end
%
popCodeWn_sumTrials = sum(popCodeWn,3);
popCodeCat_sumTrials = sum(popCodeCat,3);
maxC = max([ popCodeWn_sumTrials(:); popCodeCat_sumTrials(:) ]);
%
save( [outputDir,cellType,'_booleanSpikes_cellVtimeVtrial'], 'popCodeWn', 'popCodeCat', ...
                           'popCodeWn_sumTrials', 'popCodeCat_sumTrials', 'num_cells', 'num_trials' )

                       
                       
                       
                       
                       
if(plotFlag)
    
    h=figure; 
    subplot(211), imagesc(popCodeWn_sumTrials)
    ylabel('Cell #','FontSize',18,'FontWeight','Bold')
    title([cellType,'Population Spiking across ',num2str(num_trials),' trials : Stim = ',stims{1}],'FontSize',20,'FontWeight','Bold')
    set(gca,'FontSize',16,'FontWeight','Bold')
    caxis([0 maxC])
    cbh=colorbar('East');
    set( cbh,'FontSize',16,'FontWeight','Bold','YColor','w' )
    %set(get(cbh,'Color')
    ylabel(cbh,'#trials spike reliable','Color','w')
    %
    subplot(212), imagesc(popCodeCat_sumTrials)
    xlabel('Time (msec bins)','FontSize',18,'FontWeight','Bold')
    ylabel('Cell #','FontSize',18,'FontWeight','Bold')
    title(['Stim = ',stims{2}],'FontSize',20,'FontWeight','Bold')
    set(gca,'FontSize',16,'FontWeight','Bold')
    caxis([0 maxC])
    %
    saveGoodImg(h,[outputDir,'../../figs/older_elephant/spike_train_plots/allCells_allTrials/',cellType,'_popCode_sumTrials_wnVcat'],[0 0 1 0.8])
    close(h)
    
end
