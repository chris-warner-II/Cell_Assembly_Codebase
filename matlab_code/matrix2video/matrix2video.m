function matrix2video(mtrx, frm_rt, tm_dm, lp, trnsps)

%Simple function that plays a 3-D matrix as a movie to help visualize the values of the matrix. 
%Much like movie but can be used to directly on raw data instead of converting it to 
%movie frames 
%
%mtrx is the 3-D matrix
%frm_rt is the Frame Rate (default is 25)
%tm_dm is the time dimension of the matrix (default is 3rd dimension)
%lp is the number of times the video is to be played (default is 1)
%trnps should be set to 1, if the image has to be transposed (default 0)

%Suresh E Joel Feb 14,2003

switch(nargin)
case 0,
    disp('Too few arguements for playvideo');
    return;
case 1,
    if(length(size(mtrx))~=3),
        disp('Video dimensions must be 3');
        return;
    end;
    frm_rt=25;
    tm_dm=3;
    lp=1;
    trnsps=0;
case 2,
    if(length(size(mtrx))~=3),
        disp('Video dimensions must be 3');
        return;
    end;
    tm_dm=3;
    lp=1;
    trnsps=0;
case 3
    if(length(size(mtrx))~=3),
        disp('Video dimensions must be 3');
        return;
    end;
    lp=1;
    trnsps=0;
case 4
    if(length(size(mtrx))~=3),
        disp('Video dimensions must be 3');
        return;
    end;
    trnsps=0;
case 5
    if(length(size(mtrx))~=3),
        disp('Video dimensions must be 3');
        return;
    end;
end;

mx_vd=max(max(max(mtrx)));
mn_vd=min(min(min(mtrx)));

switch(tm_dm)
case 1,
    mtrx=permute(mtrx,[2 3 1]);
case 2,
    mtrx=permute(mtrx,[3 1 2]);
case 3,
    %do nothing;
end;

h_img=imagesc(mtrx(:,:,1),[mn_vd,mx_vd]);
set(h_img,'EraseMode','none');

%If an image file is shown please uncomment the line below
%axis off image 

for j=1:lp,
    for i=1:size(mtrx,3),
        set(h_img,'CData',mtrx(:,:,i));
        wt=1/frm_rt - 0.0177;    %approximately takes 0.0177 secs for the rest of the loop in my computer (1.5GHz) 
        if wt<0, wt==0; end;
        pause(wt);  
    end;
end;