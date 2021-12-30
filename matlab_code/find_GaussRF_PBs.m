% This function was an exploration in mixing my two projects - image segmentation and retinal image processing
% via cell assembly activity. In the image segmentation project, we found that probabilistic boundaries (pb's)
% computed from spatial gradients of both raw image pixels and of image pixels filtered by gaussian that mimic
% receptive fields of retinal ganglion cells do provide information about object boundaries. While an interesting
% hypothesis, we were unable to show any quantitative result here because the input "catrep" movie was limited
% for 3 reasons. First, it was very short, only 5 seconds long and consisting of only 150 very correlated frames.
% Second, there were not really any objects or well defined boundaries in the movie - it consisted only of a leafy
% forest floor with the camera at the eye level of a cat. Thirdly, even for the rare and weak boundaries that did
% exist in the natual movie stimulus, we had no ground truth and the process of constructing that ground truth 
% would have been costly, subject to experimenter bias and not worth the effort given the stimulus.
%

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%
% (0).  Set up directories to data and whatnot..
data_dir = '/Users/chriswarner/Desktop/Grad_School/Berkeley/Work/Fritz_Work/Projects/G_Field_Retinal_Data/scratch/G_Field_Retinal_Data/data/';

load([data_dir,'GField_data/correct_movie_catrep.mat']) % movie_catrep is natural movie of size 600ypix X 795xpix X 300tbins.

fname_save = [data_dir,'matlab_data/correct_movie_catrep_wPBs.mat'];

ydim = size(movie_catrep,1);
xdim = size(movie_catrep,2);
tdim = size(movie_catrep,3);

imG  = zeros(ydim,xdim,tdim);
grad_im  = zeros(ydim,xdim,tdim);
grad_imG = zeros(ydim,xdim,tdim); 

for t = 1:tdim

    im = movie_catrep(:,:,t); % load in a frame for the natural movie stim

    sigG = 1; % found to be optimal std of gaussian blurring RF in the image seg project.
    [kern] = construct_gaussian_kernel(sigG);
    imG(:,:,t) = imfilter(im,kern,'symmetric');
    
    grad_im(:,:,t) = compute_Spatial_Gradient(im, 0);
    grad_imG(:,:,t) = compute_Spatial_Gradient(imG(:,:,t), 0);
    
    % Display the frames. They look fine.
    if(1)
       figure;
       subplot(2,2,1), imagesc(im), colormap(bone), title('im') 
       subplot(2,2,2), imagesc(imG(:,:,t)), colormap(bone), title('imG')
       subplot(2,2,3), imagesc(1-grad_im(:,:,t)), colormap(bone), title('grad im') 
       subplot(2,2,4), imagesc(1-grad_imG(:,:,t)), colormap(bone), title('grad imG')
       pause
    end

end


% Can save movie_catrep, imG, grad_im, grad_imG, sigG.
save( fname_save, 'movie_catrep', 'imG', 'grad_im', 'grad_imG', 'sigG' );