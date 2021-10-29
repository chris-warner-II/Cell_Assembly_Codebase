function [RF_params, h] = tile_RFs(STRF_params)


% Is the angle between the Light RF and Dark RF much different from pi/2 or 90 deg?
for i = 1:numel(STRF_params)
    diff([abs(diff([mod(STRF_params{i}.fitGauss_lite(6),2*pi),mod(STRF_params{i}.fitGauss_dark(6),2*pi)])), pi/2])
end
keyboard