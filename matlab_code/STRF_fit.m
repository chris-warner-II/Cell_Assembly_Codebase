function [xDLM, temporal, minInSTAcube, maxInSTAcube] = STRF_fit(cellSTA, plotInfo)

%% Fit a 2D gaussian function to Greg Field Retina data


%% --- Stuff ADDED BY CHRIS TO FIT A STA DATA POINT -------------------------------

%
% Look for MINIMUM value in 3D STRF. This is the peak of the Dark Response.
[minVal,minLoc] = min(cellSTA(:));
[minX,minY,minZ] = ind2sub(size(cellSTA),minLoc);
minInSTAcube = [minVal, minX, minY, minZ];
Z = abs(cellSTA(:,:,minZ)); % {+ if max, - if min} : required for 2D Gauss fit to work.
Z = padarray(Z,[diff(size(Z)), 0],'post');
ZDark = medfilt2(double(Z));



%
% Look for MAXIMUM value in 3D STRF. This is the peak of the Light Response.
[maxVal,maxLoc] = max(cellSTA(:));
[maxX,maxY,maxZ] = ind2sub(size(cellSTA),maxLoc);
maxInSTAcube = [maxVal, maxX, maxY, maxZ];
Z = abs(cellSTA(:,:,maxZ)); % {+ if max, - if min} : required for 2D Gauss fit to work.
Z = padarray(Z,[diff(size(Z)), 0],'post');
ZLight = medfilt2(double(Z));


% Look for MEAN value for all timepoints in 3D STRF.
Z = mean(abs(cellSTA),3); 
Z(Z<mean(Z(:))) = 0;% remove values less than the mean because Gauss fit function needs values to go to zero with distance
                    % and there is a non-zero value because I am taking the mean of the absolute value.


[maxMnVal,maxMnLoc] = max(Z(:));
[maxMnX,maxMnY,maxMnZ] = ind2sub(size(Z),maxMnLoc);
maxInSTAmn = [maxMnVal, maxMnX, maxMnY, maxMnZ];
Z = padarray(Z,[diff(size(Z)), 0],'post');
ZMean = medfilt2(double(Z));


% Necessary for LQSCURVEFIT function (some sort of 2D ramp in both X & Y)
MdataSize = size(Z,1);
[X,Y] = meshgrid(1:MdataSize);
xdata = zeros(size(X,1),size(Y,2),2);
xdata(:,:,1) = X;
xdata(:,:,2) = Y;
noise = 0;

% parameters are: [Amplitude, x0, sigmax, y0, sigmay, angle(in rad)]
%x0 = [0.05, size(Z,1)/2, 0.5, size(Z,1)/2, 0.5, +0.02*2*pi]; %Inital guess parameters
InterpolationMethod = 'nearest'; % 'nearest','linear','spline','cubic'

%x = [2, 2.2, 7, 3.4, 4.5, +0.02*2*pi]; % centroid parameters (1st guess I think)
%xin = x; 


%% --- Fit---------------------
% define lower and upper bounds [Amp,xo,wx,yo,wy,fi].

%
x0= double([abs(minInSTAcube(1)), minInSTAcube(2), 2, minInSTAcube(3), 2, +0.02*2*pi]);
lb = [                0,          1,              0,          1,              0, -pi/4];
ub = [realmax('double'),  MdataSize,             10,  MdataSize,             10,  pi/4];
xDark = lsqcurvefit(@D2GaussFunctionRot,x0,xdata,ZDark,lb,ub);
%
x0= double([maxInSTAcube(1), maxInSTAcube(2), 2, maxInSTAcube(3), 2, +0.02*2*pi]);
lb = [                0,          1,              0,          1,              0, -pi/4];
ub = [realmax('double'),  MdataSize,             10,  MdataSize,             10,  pi/4];
xLight = lsqcurvefit(@D2GaussFunctionRot,x0,xdata,ZLight,lb,ub);
%
x0= double([maxInSTAmn(1), maxInSTAmn(2), 2, maxInSTAmn(3), 2, +0.02*2*pi]);
lb = [                0,          1,              0,          1,              0, -pi/4];
ub = [realmax('double'),  MdataSize,             10,  MdataSize,             10,  pi/4];
opt.TolFun=1e-16;
opt.TolX=1e-10;
xMean = lsqcurvefit(@D2GaussFunctionRot,x0,xdata,ZMean,lb,ub,opt);



% 
%     sigG = 2; % found to be optimal std of gaussian blurring RF in the image seg project.
%     [kern] = construct_gaussian_kernel(sigG);
%     imG = imfilter(ZLight,kern,'symmetric');
%     
%     figure, 
%     subplot(121), surf(ZLight)
%     subplot(122), surf(imG)
    
    


% % Compute mean of parameters from dark & light RF fits.
% x_dl_mn = zeros(1,6);
% X_dl_mn([2,4]) = mean([xDark([2,4]);xLight([2,4])]); % average Light & Dark RF center
% X_dl_mn(6) = circ_mean([xDark(6);xLight(6)]);        % average of tilt angles or RF ellipses

xDLM = [xDark; xLight; xMean];

xSTAmax = size(cellSTA,2);
ySTAmax = size(cellSTA,1);
xind = max( min( xSTAmax,round(xDark(2)) ), 0);
yind = max( min( ySTAmax,round(xDark(4)) ), 0);
temporal(1,:) = flipud( squeeze( cellSTA(yind,xind,:) ) );
xind = max( min( xSTAmax,round(xLight(2)) ), 0);
yind = max( min( ySTAmax,round(xLight(4)) ), 0);
temporal(2,:) = flipud( squeeze( cellSTA(yind,xind,:) ) );
xind = max( min( xSTAmax,round(xMean(2)) ), 0);
yind = max( min( ySTAmax,round(xMean(4)) ), 0);
temporal(3,:) = flipud( squeeze( cellSTA(yind,xind,:) ) );


% Fit mean (across time) spatial RF with parameter initializations based on
% average of Light and Dark ones. Maybe not.




%% -----Plot profiles----------------
if plotInfo.plotFlag
    Biphasic = {'Dark','Light','Mean'};
    hf2 = figure;
    for i = 1:numel(Biphasic)
        
        eval(['x = x',Biphasic{i},';'])
        eval(['Z = Z',Biphasic{i},';'])
        
        %% (1). Plot Receptive Field at max activation.
        alpha(0)
        switch i
            case 1
                subplot(4,12, [13:15,25:27,37:39]) 
            case 2
                subplot(4,12, [17:19,29:31,41:43])
            case 3
                subplot(4,12, [21:23,33:35,45:47])
        end
        imagesc(X(1,:),Y(:,1)',Z), hold on
        el=ellipse( x(3), x(5), 2*pi-x(6), x(2), x(4),'w');
        set(el,'LineWidth',2);
        e2=ellipse( 2*x(3), 2*x(5), 2*pi-x(6), x(2), x(4),'w');
        set(e2,'LineWidth',2,'LineStyle','--');
        set(gca,'YDir','reverse')
        colormap('jet')
        colorbar('south')
        %
        string1 = ['       Amplitude','       X-Coord', '    X-Width','       Y-Coord','    Y-Width','     Angle'];
        string3 = ['Fit      ',num2str(x(1), '% 100.3f'),'             ',num2str(x(2), '% 100.3f'),'         ',...
            num2str(x(3), '% 100.3f'),'         ',num2str(x(4), '% 100.3f'),'        ',num2str(x(5), '% 100.3f'),...
            '     ',num2str((mod(x(6),2*pi)),'% 100.3f')];
        text(2,+MdataSize*1.09,string1,'Color','red','FontSize',10,'FontWeight','Bold')
        text(2,+MdataSize*1.13,string3,'Color','red','FontSize',10,'FontWeight','Bold')

        % Plot cross sections
        % generate points along horizontal axis
        m = -tan(x(6));% Point slope formula
        b = (-m*x(2) + x(4));
        xvh = 1:MdataSize;
        yvh = xvh*m + b;
        hPoints = interp2(X,Y,Z,xvh,yvh,InterpolationMethod);
        % generate points along vertical axis
        mrot = -m;
        brot = (mrot*x(4) - x(2));
        yvv = 1:MdataSize;
        xvv = yvv*mrot - brot;
        vPoints = interp2(X,Y,Z,xvv,yvv,InterpolationMethod);

        hold on % Indicate major and minor axis on plot

        % % plot points 
        % plot(xvh,yvh,'r.') 
        % plot(xvv,yvv,'g.')

        % plot lines 
        plot([xvh(1) xvh(size(xvh))],[yvh(1) yvh(size(yvh))],'r','LineWidth',2) 
        plot([xvv(1) xvv(size(xvv))],[yvv(1) yvv(size(yvv))],'g','LineWidth',2) 

        hold off
        axis([1-0.5 MdataSize+0.5 -1-0.5 MdataSize+0.5])
        % % % % %
        ymin = - noise * x(1);
        ymax = x(1)*(1+noise);
        xdatafit = linspace(1-0.5,MdataSize+0.5,300);
        hdatafit = x(1)*exp(-(xdatafit-x(2)).^2/(2*x(3)^2));
        vdatafit = x(1)*exp(-(xdatafit-x(4)).^2/(2*x(5)^2));
        
        %% (2). Plot Cross Sections
        switch i
            case 1
                subplot(4,12, [1:3]) 
            case 2
                subplot(4,12, [5:7])
            case 3
                subplot(4,12, [9:11])
        end
        xposh = (xvh-x(2))/cos(x(6))+x(2);% correct for the longer diagonal if fi~=0
        plot(xposh,hPoints,'r.',xdatafit,hdatafit,'black','LineWidth',2)
        axis([1-0.5 MdataSize+0.5 ymin*1.1 ymax*1.1])
        title(Biphasic{i},'FontSize',18,'FontWeight','Bold')
        %
        % %
        %
        switch i
            case 1
                subplot(4,12, [16,28,40]) 
            case 2
                subplot(4,12, [20,32,44])
            case 3
                subplot(4,12, [24,36,48])
        end
        xposv = (yvv-x(4))/cos(x(6))+x(4);% correct for the longer diagonal if fi~=0
        plot(vPoints,xposv,'g.',vdatafit,xdatafit,'black','LineWidth',2)
        axis([ymin*1.1 ymax*1.1 1-0.5 MdataSize+0.5])
        set(gca,'YDir','reverse')
        figure(gcf) % bring current figure to front

        

        %% (3). Plot time course.
        switch i
            case 1
                subplot(4,12,4) 
            case 2
                subplot(4,12,8)
            case 3
                subplot(4,12,12)
        end
        plot( temporal(i,:) ,'LineWidth',2)
        title('Temporal','FontSize',18,'FontWeight','Bold')

    end
    
    % Put a title on.
    subplot(4,12, [17:19,29:31,41:43])
    title([plotInfo.cellID],'FontSize',20,'FontWeight','Bold')
    
    % Save plot.
    saveGoodImg(hf2,[plotInfo.outputDir,'STRF_fit_plots/',plotInfo.cellID],[0 0 1 0.5])
    close(hf2)
    
end