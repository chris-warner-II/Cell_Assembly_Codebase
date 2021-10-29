
% Using STRFlab from Theunissen and Gallant labs.
global globDat % set up global variables for STRFlab to use.


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%
%% (0). STRF_flgs: Which method to use to compute Spatio Temporal Receptive Fields for Cell Assemblies
STRF_PreProc = 'RTAC'; % options for training in preprocessing directory: 
                       % 'RTAC'     - for response triggered Avg & Covariance.
                       % 'Wavelets' - uses Gabors
                       % 'Spectra'  - ???
                       % 'Stim2d'   - ??? 
                           
STRF_trnAlgo = 'SCG';  % options for training in layer_3optim directory: 
                       % 'SCG'        - I honestly dont know what this does. Doesnt really work well as is either.
                       % 'GradDesc'   -
                       % 'GDLineSrch' - 
                       % 'PF'         -
                       % 'LARS'       -
                       % 'DirectFit'  - 'Direct Fit Method'.
                       % 'Backslash'  -
                         
spatialDSfctr = 1; % Downsample movie spatially by this factor
CAbinToStimTime = 1; % Flag to downsample activations of CAs from 1ms to 16.667 ms (stimulus refresh rate)


% Params for the datafiles output from my Python processing.
cellType = '[offBriskTransient]';
cTid = 1; % hard coding this for now. Location of cellType in 'allCellTypes' cell array.
msBins = 1;
whichStim = 'Wnz_4STA'; % 'Wnoise' or 'NatMov' or 'Wnz_4STA'
responseType = 'Cells'; % Options: 'Cells' or 'CAs' (can also take cell spikes from G Field data - not doing now except for Wnz_4STA)



disp(['Finding Spatio-Temporal Receptive Fields for ',cellType,' ',responseType,' responding to ',whichStim])

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%
% (0).  Set up directories to data and whatnot..
data_dir = '/Users/chriswarner/Desktop/Grad_School/Berkeley/Work/Fritz_Work/Projects/G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/data/';






% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%
%% (1). Load in movie_catrep and reshape that into a 2D Stim matrix where t is one dimension and x&y are rastered out into a single dimension.
disp(['Loading in the stimulus and spike response for ',whichStim])
load([data_dir,'GField_data/all_cells_2017_01_16_0.mat'])
%
switch whichStim
    %
    case 'Wnoise'
        Stim = movie_wnrep; 
        crop_mov = 0;        % No border on the wnrep movie.   
    %    
    case 'Wnz_4STA'          % 60 minutes of white noise stimulus with no repeat. 
        Stim = movie_wn;  
        crop_mov = 0;        % No border on the wn movie.   
        spatialDSfctr = 1;   % This stimulus is already spatially small. (40x53 pixels) vs (600x795 pixels) for others.linspace(trigs(1),trigs(2),100)
        trigs = triggers_wn;
        %
        stim_time_stamps = zeros(1,size(Stim,3));
        dist_betwn_trigs = 100;
        for i = 0:numel(trigs)-1
            stim_time_stamps(1+ dist_betwn_trigs*i:dist_betwn_trigs*(i+1)) = linspace(trigs(i+1),trigs(i+2),dist_betwn_trigs);
        end
        %
        load([data_dir,'matlab_data/allCells_spikeTrains_CellXTrial.mat']) 
        sp = spikes_wn;
        sp = sp( :,cellTypeIDs(cTid,:) );
    %  
    case 'NatMov'
        load([data_dir,'GField_data/correct_movie_catrep.mat']) % movie_catrep in all_cells_2017_01_16_0.mat is shifted incorrectly.
        Stim = movie_catrep;
        crop_mov = 1;        % Crop catrep movie because there is a border on it.   
    %    
    otherwise
        disp(['The stimulus type, ',whichStim,', is not understood.'])
end
%
clear triggers_wnrep spikes_wnrep movie_catrep movie_wn  spikes_wn  movie_wnrep triggers_wn spikes_catrep triggers_catrep
% keeping: stas, refresh_time


% TO DO: If I want to look at spike trains extracted from original GField data for catrep or wnrep.
if(0)
    load([data_dir,'matlab_data/allCells_spikeTrains_CellXTrial.mat']) 
    % Can make sp using spikesInWn, allCellTypes, cellTypeIDs, spikesInCat
    %
end


% Compute relationship between stimulus time bins and spike times.
stim_sample_dur  = (16+2/3)/1000; %stim_time_dur/tdim;  % Duration of each stimulus frame in ms. (same as refresh_time*1000)
                           % NOTE: something wrong with the precision of  refresh_time. So I hardcode it.
stim_time_dur    = stim_sample_dur*size(Stim,3); % time of stimulus in ms (5 seconds for wnrep and catrep or 1hr for wn)
spk_sample_dur   = 1;                            % time for each bin in spike train in ms (1ms)
t_SampRatio      = stim_sample_dur/spk_sample_dur; 





% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%
%% (2). Load in a 2D cell of CellOrCA spiketimes (Trial X Cell x TimeOfActivityInMS) and for each unit, reshape it into a single vector of spiketimes by throwing away trial information.
if ~strcmp(whichStim,'Wnz_4STA')
    load([data_dir,'/matlab_data/CA_raster_times/',cellType,'_',whichStim,'_',num2str(msBins),'msBins.mat'])
    switch responseType
        case 'CAs'
            sp = raster_Z_inferred_allSWs; % Activations from CAs / Cell Assemblies
        case 'Cells'
            sp = raster_allSWs;            % Spikes from Cells. From our code (should match Greg's matlab stuff.
            %
            if(1)
                load([data_dir,'matlab_data/allCells_spikeTrains_CellXTrial.mat']) 
                sp = spikesInWn;
            end
            
        otherwise
            disp(['The response type, ',responseType,', is not understood.'])
    end
end

%
numTrials = size(sp,1); % 200 for catrep, wnrep and 1 for wn.
numUnits  = size(sp,2); % number CAs or Cells

        % % % % % %     % % % % % %     % % % % % %     % % % % % %     






        % % % % % %     % % % % % %     % % % % % %     % % % % % %    


%
% Crop catrep movie spatially because there is a weird, useless boundary around it.
% No need to crop the white noise movie.
if(crop_mov)
    yLims = [61,540]; % These values were pulled out by eye from visualizing a frame of the stim.
    xLims = [1,640];
    Stim = Stim(yLims(1):yLims(2),xLims(1):xLims(2),:);
end


        % % % % % %     % % % % % %     % % % % % %     % % % % % %    


%
% Downsample stimulus movie spatially to more quickly fit STRFs.
if(spatialDSfctr~=1)
    ydim = size(Stim,1);
    xdim = size(Stim,2); 
    tdim = size(Stim,3); 
    StimDS = zeros(round(ydim*spatialDSfctr),round(xdim*spatialDSfctr),tdim);
    for i = 1:tdim  % NOTE: imresize has LOTS OF options about HOW you downsize !
        StimDS(:,:,i) = imresize( Stim(:,:,i), [size(StimDS,1),size(StimDS,2)] );
    end
    Stim = StimDS;
    clear StimDS
end
%
ydim = size(Stim,1);
xdim = size(Stim,2); 
tdim = size(Stim,3); 

        % % % % % %     % % % % % %     % % % % % %     % % % % % %    






% %% Preprocess movie stimulus for STA and STC
% % Options: preprocRTAC, preprocWavelets3d, preprocSpectra
% disp(['Preprocessing Stimulus with ',STRF_PreProc])
% tic
% switch STRF_PreProc
%     
%     case 'RTAC' 
%     % (1). Simplest Response-Triggered Average / Covariance    
%         params.RTAC = [1 0]; % [DoSTA, DoSTC]
%         [stimProc,params] = preprocRTAC(Stim, params); 
%    
%     case 'Wavelets'
%     % (2). Gabor Filters: Wavelet Basis [LINEARIZATION]
%     %      Lets make a complex cell with a preferred direction and a certain temporal frequency
%     %      [GABORS GOOD FOR EDGES. LOOK FOR LARGE AND LOW FREQ OVALS OR EDGES?]
%         params = preprocWavelets3d;
%         [stimProc,params] = preprocWavelets3d(Stim, params);
%     otherwise
%         disp('Preprocessing not understood. Options are RTAC, Wavelets & Spectra')
% end
% toc







% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%
%% Loop over all units: Cells or Cell Assemblies.
for i = 5:5%numUnits
    
    

    % Group together activations of a Cell Assembly, throwing away trial information.
    % This is right to do for repeated stimuli (wnrep and catrep). And wn has only 1 trial.
    activs = sp{1,i}; 
    for j = 2:numTrials
        activs = horzcat(activs, sp{j,i});
    end
    
    
    % FIX THIS: (HOW? - UPSAMPLE STIMULUS?) Bin CA activations within stimulus movie frame. 
    % Throw away ms  information on finer timescale than that. (Wrong thing to do!)
    if(CAbinToStimTime)
        activsB = activs/t_SampRatio;
        resp = histogram(activs,tdim);
        resp = resp.Values;
        
    else
        disp('Otherwise, need to intelligently upsample the stimulus to the response timescale.')
    end
    
    
%     % Since it is possible to have more than one spike per frame, let's try
%     % a little more information preserving threshhold, with a high threshhold 
%     % for 2 spikes and a little bit lower threshhold for 1 spike
%     resp(resp<150) = 0;
%     resp(resp>=300) = 2;
%     resp(resp>=150) = 1;
    



    tdim = 100;
    numSpikes = sum(resp(1:tdim));
    
    t_strf = 21;
    h = zeros( ydim, xdim); %, t_strf+1 );
    %
    figure
    for t = 1:tdim
        try
            h = h + resp(t) .* ( Stim(:,:,t-t_strf) - 0.5 );
        catch
            disp(['t=',num2str(t),' and t_back=',num2str(t_strf)])
            continue % if t_back brings you past the first stim sample.
        end
        imagesc(h), colormap('bone'), colorbar, title(t)
        pause(0.05)
    end



%     % JUST SIMPLE STA I HAVE HAND CODED. IF THIS DOESNT WORK FOR 300 FRAMES
%     % OF WHITE NOISE STIMULUS, WE HAVE NO HOPE WITH 300 FRAMES OF NATURAL
%     % MOVIE STIM.
%     %
%     
%     tdim = 100000;
%     numSpikes = sum(resp(1:tdim));
%     
%     t_strf = 29;
%     h = zeros( ydim, xdim, t_strf+1 );
%     %
%     for t = 1:tdim
%         disp(['t=',num2str(t)])
%         for t_back = 0:t_strf
%             try
%                 h(:, :, 1+t_back) = h(:, :, 1+t_back) + ( resp(t) .* ( Stim(:, :, t-t_back) - mean(mean(Stim(:,:,t-t_back))) ) );
%             catch
%                 disp(['t=',num2str(t),' and t_back=',num2str(t_back)])
%                 continue % if t_back brings you past the first stim sample.
%             end
%         end
%         
%         % Visualize learned STRF for this many time points.
%         if mod(t,10000)==0
%             numSpikes = sum(resp(1:t));
%             STA = h / numSpikes;
%             for i = 1:(t_strf+1)
%                 subplot(6,5,i), imagesc(STA(:,:,i)), title(['t=-',num2str(i-1)])
%             end
%             pause
%             
%         end
%         
%     end
%     %
    
    
    % Use J. Pillow
    tic
    t_strf = 10;
%     xxx = Stim(end-19:end,21:40,:);     % Crop stim because it is too large and errors. Cheating.
%     yyy = reshape(xxx,[20*20,tdim])';   % (Stim into warnerSTC or simpleSTC must be 2D: tbins X xbins*ybins)
    yyy = reshape(Stim,[xdim*ydim,tdim])' - mean(mean(mean(Stim)));   % (Stim into warnerSTC or simpleSTC must be 2D: tbins X xbins*ybins)
    
    
    % Check that I reshaped stim correctly that it matches original stim if I reshape it back.
    for j = 1:10
        if any(any(Stim(:,:,1) - reshape( yyy(1,:), ydim, xdim) ))
            disp('Something wrong with reshaping stim to enter into simpleSTC')
            break
        end
    end
    
    
    
    
    
    
    % STA = warnerSTC( yyy, activs, 1, t_strf);
    
    STA = simpleSTC( yyy, activs, t_strf );
    
    toc
    
%     ydim = size(Stim,1);
%     xdim = size(Stim,2);
    
%     ydim=20;
%     xdim=20;

    x = 5;
    y = 2;
    for i = 1:t_strf
        subplot(y,x,i), imagesc( reshape(STA(i,:),ydim,xdim ) ), title(['t=-',num2str(i-1)])
    end
    
    disp(['Max STA ',num2str( max(max(max(STA))) )])
    disp(['Min STA ',num2str( min(min(min(STA))) )])
    
 
    %figure, imagesc(STC), colorbar



    
    

    
    
    

%     
%     
%     % Initialize a new strf - A GLM
%     strf_tDelays = [0:8];
%     strf = linInit(size(stimProc,2), strf_tDelays); % Create a GLM.
%     strf.b1 = mean(resp);    % mean
%     strf.params = params;    % params from stimulus preprocessing
%     groupIdx = [1:globDat.nSample];
%     strfData(stimProc,resp, groupIdx)  % globalize the data
%     
% 
% 
%     % set options and index
%     % [USE: Change in filters to determine convergence.]
%     disp(['Training STRFs using ',STRF_trnAlgo])
%     switch STRF_trnAlgo
%         
%         case 'SCG'
%         % (1). Scaled Conjugate Gradient Optimization
%         
%             options=trnSCG; 
%             options.display=10;
%             options.maxIter = 300000;
%             trainingIdx = [1:globDat.nSample];
%             strfTrained=strfOpt(strf,trainingIdx,options);
%             
%         
%         case 'GradDesc'
%         % (2). Gradient Descent [REGULARIZATION]
%             
%             options = trnGradDesc;      % set the options to defaults for trnGradDesc
%             options.earlyStop = 0;      % turn on early stopping
%             options.display = 5;        % graphically display every iteration
%             options.coorDesc = 1;       % if 1,coordinate gradient descent. Move in single direction of largest gradient ("boosting").
%             options.nDispTruncate = 0;  % don't truncate first steps
%             options.maxIter = 300000;   % Make large so that we can use all the data.
%             options.stepSize = 1e-3;    % Stepsize can be important and you should tune it !
%                                         % Set a step size to something small enough error doesn't
%                                         % oscilate on the training set with each step and doesn't
%                                         % oscilate too much on the early stopping set
%             trainingIdx = [1:floor(.8*globDat.nSample)]; % We'll try using 80% of the data for the training set
%             stoppingIdx = [floor(.8*globDat.nSample)+1:globDat.nSample]; % And the remaining 20% for the stopping set
%             strfTrained=strfOpt(strf,trainingIdx,options,stoppingIdx); % Train and viusalize the STRF
%            
%         case 'GDLineSrch'
%         % (3). GDLineSrch 
%             disp('GDLineSrch')
%             
%             
%         case 'DirectFit'
%         % (4). Direct Fit method   
%             options = trnDirectFit();
%             trainingIdx = [1:globDat.nSample];
%             strfTrained=strfOpt(strf,trainingIdx,options);
%             
%         otherwise
%             disp(['Dont understand the optimization algorithm',STRF_trnAlgo])
%             disp('Implemented Options are: SCG, GradDesc')
%             disp('Not implemented Options are: GradDescLineSearch, PF, LARS, DirectFit, Backslash')
%     end
%         
%  
%         
%         
%         
% 
%         
%         
%         %% Visualize the strf
%         %
%         switch STRF_PreProc
%             
%             case 'RTAC'
%             % (1). JUST VANILLA RESPONSE-TRIGGERED AVG/COVAR
%             preprocRTAC_vis(strfTrained,ydim,xdim);
%         
%             
%             case 'Wavelets' % LINEARIZATION
%             % (2). GABOR WAVELET BASIS: preprocessing the stimuli with a pyramid of 3D Gabor filters.
%             %      Params: scale, spatial frequency, temporal frequency, orientation and position
%             preprocWavelets3dVis(strfTrained);
%             
%         end
%         
%         
%         
%         
%         
% %     %% [CROSS-VALIDATION] Use the trained STRF to predict the response to data. 
% %     strfData(stimVal,respVal)
% %     testingIdx = [1:globDat.nSample];
% %     [strfTrained,predresp]=strfFwd(strfTrained,testingIdx); % Predict response on heldout data using STRF learned.
%         
%     
%     %% [RESAMPLING] -  BOOTSTRAPPING, JACKKNIFING, CROSSVALIDATION.
%     if(0)
%         options = resampBootstrap; % See also: resampJackknife, resampCrossVal
%         options.optimOpt = trnGradDesc;
%         options.optimOpt.coorDesc = 1;
%         options.optimOpt.earlyStop = 1;
%         options.optimOpt.nDispTruncate = 0;
%         options.optimOpt.display = -1;
%         options.optimOpt.stepSize =  1e-03;
%         options.testFrac = 0.20;
%         options.nResamp = 5;
%         %
%         strfData(stim,resp)
%         %
%         trainingIdx = [1:globDat.nSample];
%         strfTrained_tmp=strfOpt(strf,trainingIdx,options);
%         strfTrained2 = strfTrained_tmp(1);
%         strfTrained2.b1 = mean(cat(2, strfTrained_tmp.b1), 2);
%         strfTrained2.w1 = mean(cat(4, strfTrained_tmp.w1), 4);
% 
%         % NOTE: have to define Validation data. stimVal and respVal
%         strfData(stimVal,respVal)
%         testingIdx = [1:globDat.nSample];
%         [strfTrained2,predresp]=strfFwd(strfTrained2,testingIdx); % Predict response on heldout data using STRF learned.
%         %
%         nonnanidx = find(~isnan(predresp));                       % Remove any NaN's from the prediction
%         predcorr2 = corr(predresp(nonnanidx),respVal(nonnanidx)); % correlation between prediciton and response
%     end
%     
%         
    
    
end



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%
% (3). Whats next ...
