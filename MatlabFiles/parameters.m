% Author : Viktor-Adam Koropecky
% Year : 2021
% Email : koropvik@fel.cvut.cz
%
% This file is used to fill in experiment parameters for the different
% algorithms used in this work.

simulation = false;
print_extra = true;
record_results = true;

data = "Pics/hologramsklo.png";
ResultFolder = strcat('Results/' , strrep(datestr(datetime), ':', '_'), '/');
if record_results == true
    mkdir(ResultFolder);
end

dx = 1.55e-6; %Pixel size
n = 1.45; %Refractive index of medium
n_o = 1.59; %Refractive index of beads (unused)
lambda = 515e-9; %Light wavelength
z_m = 2.347e-3; %The main image plane used for reconstruction

x = 2048; %Width of the reconstructed image
y = 2048; %Height of the reconstructed image

x_o = 0; % Offset on the x-axis (2nd dimension)
y_o = 0; % Offset of the y-axis (1st dimension)

% Parameters for FISTA
mu = 0.2; %Soft-thresholding hyperparameter
t = 0.2; %Constant gradient step
iter = 5; %Maximum number of iterations