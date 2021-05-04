% Author : Viktor-Adam Koropecky
% Year : 2021
% Email : koropvik@fel.cvut.cz
%
% This file is used to fill in experiment parameters for the different
% algorithms used in this work.

data = "Pics/hologram.png";
bgd_data = "Pics/electrodes.png";
mask = "Pics/electrode_mask.png";

dx = 1.55e-6; %Pixel size
n = 1.45; %Refractive index of medium
n_o = 1.59; %Refractive index of beads (unused)
lambda = 525e-9; %Light wavelength
z_m = 2.2e-3; %The main image plane used for reconstruction

x = 2048; %Width of the reconstructed image
y = 2048; %Height of the reconstructed image

x_o = 0; % Offset on the x-axis (2nd dimension)
y_o = 0; % Offset of the y-axis (1st dimension)

reduce_background = false;
simulation = false;

% Used algorithm for phase reconstruction
% "iterative" - Algorithm from [1], is not currently in a working order and
% will possibly be abandoned in the future
% "inverse" - Algorithm from [2], not yet implemented
% "fienup" - The famous Fienup's algorithm [3]
% "multi" - Multilayer version of the inverse algorithm
used_algorithm = "inverse";

%Additional parameters used for the iterative algorithm
if used_algorithm == "iterative"
    z = linspace(2.0e-3, 3.5e-3, 7);
    threshold = 0.02; % threshold for masking
    dilation = 5; % the amount of pixels by which to dilate the calculated mask
    lpfilter = 11; % parameter for gaussian filter by which the mask is 
elseif used_algorithm == "multi"
    r_constr = [-1 0;0 0];
    i_constr = [-1 -1; 1 1];
    z = [2.3e-3; 2.4e-3];
else
    r_constr = [0,0.5];
    i_constr = [-1,1];
end

% Additional parameters used for the basic Momey's inverse algorithm
if used_algorithm == "inverse" || used_algorithm == "multi"
    mu = 0.04;
    t = 0.03;
end

iter = 9; %Maximum number of iterations