clc
close all

run("parameters.m");

%% Loading electrode image 

image = im2double(imread(data));
tmp_background = im2double(imread(bgd_data));

%Sliding the background to fit the hologram with particles
background = zeros(2048,2048);
background(9:end,:) = tmp_background(1:end-8,:);

%Subtracting background from the foreground
hologram = image - background;
hologram = hologram - min(hologram(:));
hologram = hologram / max(hologram(:));
hologram = 1-sqrt(hologram);

%% Finding the appropriate offset 
%  (with the used images it was determined to be equal to 8)
%tmp_holo = background(6:end-5,6:end-5);
%autoc = xcorr2(image, background);

%% Running the selected algorithm

disp("Reconstruction algorithm started!");
Hq_m = rs_backprop(z_m,y,x,dx,n,lambda);
if used_algorithm == "iterative"
    Hq = complex(zeros(y,x,numel(z)),zeros(y,x,numel(z)));
    for i = 1:numel(z)
        Hq(:,:,i) = rs_backprop(z(i),y,x,dx,n,lambda);
    end
    h_reconstruction = iterative(hologram, Hq, x, y, z, ...
        iter, u_threshold, l_threshold, dilation, lpfilter, false);
end

%% Printing the results

%Calculating the twin-image-free final result from the reconstructed hologram
i_reconstruction = c_norm(ifft2(fft2(h_reconstruction).*Hq_m));

%Just a simple backpropagation for comparison with the selected algorithm
i_simple = c_norm(ifft2(Hq_m.*fft2(hologram)));

figure();
subplot(1,2,1);
title("Reconstruction result");
imshow(abs(i_reconstruction));
subplot(1,2,2);
imshow(abs(i_simple));
title("Simple backpropagation");