% Author : Viktor-Adam Koropecky
% Year : 2021
% Email : koropvik@fel.cvut.cz
%
% This file is used to fill in experiment parameters for the different
% algorithms used in this work.

data = "Pics/hologram.png";
bgd_data = "Pics/electrodes.png";

dx = 1.55e-6; %Pixel size
n = 1.45; %Refractive index of medium
n_o = 1.59; %Refractive index of beads (unused)
lambda = 525e-9; %Light wavelength
z_m = 2.75e-3; %The main image plane used for reconstruction

x = 2048; %Width of the reconstructed image
y = x; %Height of the reconstructed image

x_o = 0; % Offset on the x-axis (2nd dimension) TODO
y_o = 8; % Offset of the y-axis (1st dimension) TODO

% Used algorithm for phase reconstruction
% "iterative" - Algorithm from [1], is not currently in a working order and
% will possibly be abandoned in the future
% "inverse" - Algorithm from [2], not yet implemented
% "fienup" - The famous Fienup's algorithm [3]
used_algorithm = "fienup";

%Additional parameters used for the iterative algorithm
if used_algorithm == "iterative"
    z = linspace(2.0e-3, 3.5e-3, 7);
    threshold = 0.02; % threshold for masking
    dilation = 5; % the amount of pixels by which to dilate the calculated mask
    lpfilter = 11; % parameter for gaussian filter by which the mask is 
end

%Additional parameters used for the Fienup's algorithm
if used_algorithm == "fienup"
    r_constr = [-1,1];
    i_constr = [-1,0];
end

iter = 10; %Maximum number of iterations