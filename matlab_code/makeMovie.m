%function makeMovie(dirOut,)


% THIS IS RUNNING LIKE A SCRIPT. LOAD THE APPROPRIATE MAT FILE 
% (RETINA_DATA.MAT). AND RUN THIS. EVENTUALLY, TURN THIS INTO A FUNCTION.
%
% NOTE: ONCLUSTER IS JUST A SHORT FUNCTION I WROTE TO SET DIRECTORIES. YOU
% SHOULD EDIT IT FOR YOUR DIRECTORY STRUCTURE. 
%
%

data_dir_base = '/Users/chriswarner/Desktop/Grad_School/Berkeley/Work/Fritz_Work/Projects/G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/data/';

outputDir = './G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/figs/older_elephant/stim_movies/';


cellTypes = {'offBriskSustained','offBriskTransient'}; %{'offBriskSustained','offBriskTransient','onBriskTransient'};
stim = 'Cat'; % either 'Wn' for white noise or 'Cat' for natural movie


%colors = {'Reds9','Blues9'};


cTstr = cellTypes{1};
for i = 2:numel(cellTypes)
    cTstr = [cTstr,'_',cellTypes{i}];
end


%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%



% Most of the data we got from Greg Field - with spikes, triggers, stas (for 3 celltypes), stim (movie_catrep is shifted and incorrect so we load it below)
load([data_dir_base,'GField_data/all_cells_2017_01_16_0.mat']) 
% I only want 'indices_for_typed_cells'. Clear the rest.

    
%
% May 2019: the correction to movie_catrep - and the _wPBs one has the probabalistic boundaries from Independent Sensors Gaussian RF blurring at sigma=1
    % load([data_dir_base,'GField_data/correct_movie_catrep.mat'])
load([data_dir_base,'matlab_data/correct_movie_catrep_wPBs.mat']) % contains 4 movies:  movie_catrep=im, imG, grad_im, grad_imG



%
%
% Some binary boolean spike trains I made for Off Brisk Transient Cells (Cell x msTime x Trial)
load([data_dir_base,'matlab_data/allCells_booleanSpikes_cellVtimeVtrial.mat'])



%
% STRF fits for all 329 cells. Made using STRF_fit.m - fits an oval to the spatial RF and also a time course.
load([data_dir_base,'matlab_data/allCells_STRF_fits_329cells.mat'])


switch stim
    case 'Wn'
        movie_stim = movie_wnrep;   % White noise movie.
        popCode_sumTrials = popCodeWn_sumTrials;
        stimFname = 'wnStim';
        stimTitle = 'White Noise';
        crop_cat_mov = 0;
    case 'Cat'
%         movie_stim = movie_catrep;  % Original Cat Cam Natural movie.
%         fname_filter_add=''
        % 
        % movie_stim = imG;           % GaussRF filtered movie
        % fname_filter_add='_Gauss'
        % 
        % movie_stim = 1-grad_im;       % 2D spatial deriv of orig movie.
        % fname_filter_add='_grad'
        % 
        movie_stim = 1-grad_imG;      % 2D spatial deriv of GaussRF filtered movie.
        fname_filter_add = '_gradGauss';
        %
        popCode_sumTrials = popCodeCat_sumTrials;
        stimFname = 'catStim';
        stimTitle = 'Natural Movie';
        crop_cat_mov = 0;
    otherwise
        disp('uuuuh, huh?')
        keyboard
end


% % HERE, I WAS DISPLAYING CELL NUMBERS TRYING TO MATCH UP NUMBER IN MATLAB, PTYHON, AND CELL ID NUMBER FOR KIERSTEN.
if(0)
    disp('offBriskTransient: Python, Matlab, Cell Id')
    for i = 1:numel(indices_for_typed_cells.offBriskTransient)
        disp( [num2str(i-1),'   ',num2str(i),'   ',num2str(indices_for_typed_cells.offBriskTransient(i))] )
    end


    disp('offBriskSustained: Python, Matlab solo, Matlab Pair Cell Id')
    for i = 1:numel(indices_for_typed_cells.offBriskSustained)
        disp( [num2str(i-1+numel(indices_for_typed_cells.offBriskTransient)),'   ',num2str(i),'   ',num2str(i+numel(indices_for_typed_cells.offBriskTransient)),...
            '   ',num2str(indices_for_typed_cells.offBriskSustained(i))] )
    end
end



clear movie_wnrep movie_wn triggers_catrep triggers_wnrep ...
      triggers_wn stas spikes_catrep spikes_wnrep spikes_wn ...
      refresh_time movie_catrep imG grad_im grad_imG popCodeWn ...
      popCodeWn_sumTrials popCodeCat popCodeCat_sumTrials
  
  
if(crop_cat_mov) % NOT WORKING RIGHT NOW !!
    % These values were pulled out by eye from visualizing a frame of the stim.
    yLims = [60,540];
    xLims = [1,640];
    movie_stim = movie_stim(yLims(1):yLims(2), xLims(1):xLims(2), :);
else
    yLims = [1,size(movie_stim,1)];
    xLims = [1,size(movie_stim,2)];
end  


XdimImg = size(movie_stim,2);
YdimImg = size(movie_stim,1);

timeStim = size(movie_stim,3); % 300 frames in 5 sec of stim 
timeSpikes = size(popCode_sumTrials,2); % 5500 msec. Each trial is 5sec.
spatialUS = 15;  % number of pixels that are in the movie image for each pixel in the RF.
timeUS = 16+2/3; % number of time points in the movie (16.6667ms per frame)


if(0)
    timePerMovie= 300;
    movie_tRange(:,1) = 1:timePerMovie:timeSpikes;
    movie_tRange(:,2) = [1:timePerMovie:timeSpikes]+timePerMovie-1;
    movie_tRange = min(movie_tRange,timeSpikes);
end




% explicitly set movie_tRange here.
movie_tRange =  ...
                [1600, 1800; ...
                 900, 1100; ...
                3300, 3400; ...
                4400, 4500; ...
                1300, 1400; ...
                4700, 4800; ...
                2900, 3000]; % [ a,b ; c,d ] means from ta to tb. from tc to td. etc.






for k = 1:size(movie_tRange,1)





    


    for c = 1:numel(cellTypes)
        cellType = cellTypes{c};
        
        
        switch cellType
            case  'offBriskSustained'
                STRF_GaussParams = STRF_GaussParamsD;
            case 'offBriskTransient'
                STRF_GaussParams = STRF_GaussParamsD;
            case 'onBriskTransient'
                STRF_GaussParams = STRF_GaussParamsL;
        end
        
        
        % Set up and save a movie of time evolution of coupled oscillator system
        movName=[outputDir,'/',stimFname,'_',cellType,'RFs',fname_filter_add,'_spikesTrialAvgd_t',num2str(movie_tRange(k,1)),'-',num2str(movie_tRange(k,2)),'ms'];

        if exist([movName,'.avi'])
            continue
        end

        writerObj = VideoWriter([movName,'.avi']);
        writerObj.FrameRate = 15;
        open(writerObj);
        %
        %
        
        
        for i = movie_tRange(k,1):movie_tRange(k,2)
            i
            pp = figure;
            imagesc(movie_stim(:,:,ceil(i/timeUS))), colormap(bone),





                %
                maxC = max(max(popCode_sumTrials( indices_for_typed_cells.(cellType), : ))) + 1;
                color = jet(maxC);% othercolor(colors{c},maxC);
                %
                for j = indices_for_typed_cells.(cellType)
                    x = STRF_GaussParams(j,:);
                    x(2:end-1) = spatialUS.*x(2:end-1);
                    e2=ellipse( x(3), x(5), 2*pi-x(6), x(2)+xLims(1), x(4)+yLims(1),'k');
                    set(e2,'LineWidth',1.5,'Color',color(popCode_sumTrials(j,i)+1,:));
                    %text(x(2),x(4),num2str(i),'FontSize',12,'FontWeight','Bold','VerticalAlignment','Top','HorizontalAlignment','Left','Color','r')
                    % MAYBE ALSO SCALE ELLIPSE SIZE BY popCodeCat_sumTrials
                end



                title([num2str(numel(indices_for_typed_cells.(cellType))),' ',cellType,' RGC''s.', num2str(num_trials),' trials.'],...
                    'FontSize',20,'FontWeight','Bold') %*spatialUs
                xlabel(['Time = ',num2str(i),' msec'],'FontSize',20,'FontWeight','Bold')
                text(0.99*XdimImg,0.01*YdimImg,{'# trials','spiked on',['\color{blue}0'],...
                    ['\color{green}',num2str(round(maxC/2))],['\color{red}',num2str(maxC)]},'HorizontalAlignment',...
                    'Right','VerticalAlignment','Top','FontSize',20,'FontWeight','Bold')
                set(gca,'Xtick',[],'Ytick',[])
                axis tight



            set(pp,'units','normalized','position', [0 0 YdimImg/XdimImg 1],'Color', 'white'); 

            % Save Frame as part of the movie file
            mov = getframe(pp);

            writeVideo(writerObj,mov);
            close(pp)


        end
        
        close(writerObj);
        
    end

end