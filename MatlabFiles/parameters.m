% Author : Viktor-Adam Koropecky
% Year : 2021
% Email : koropvik@fel.cvut.cz
%
% This file is used to fill in experiment parameters for the different
% algorithms used in this work.

data = "Pics/hologram.png";
ResultFolder = strcat('Results/' , strrep(datestr(datetime), ':', '_'), '/');
mkdir(ResultFolder);

dx = 1.55e-6; %Pixel size
n = 1.45; %Refractive index of medium
n_o = 1.59; %Refractive index of beads (unused)
lambda = 515e-9; %Light wavelength
z_m = 2.4e-3; %The main image plane used for reconstruction

x = 2048; %Width of the reconstructed image
y = 2048; %Height of the reconstructed image

x_o = 0; % Offset on the x-axis (2nd dimension)
y_o = 0; % Offset of the y-axis (1st dimension)

simulation = false;
print_extra = false;
record_results = true;

% Parameters for FISTA
r_constr = [-2,0]; %Real Bounds
i_constr = [-1,1]; %Imaginary bounds
mu = 0.06; %Soft-thresholding hyperparameter
t = 0.25; %Constant gradient step
iter = 4; %Maximum number of iterations