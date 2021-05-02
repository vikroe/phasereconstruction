clc
close all

run("parameters.m");

%% Loading electrode image 

if simulation == false

    image = im2double(imread(data));
    tmp_background = im2double(imread(bgd_data));
    im_mask = imread(mask);

    if reduce_background == true
        %Sliding the background to fit the hologram with particles
        background = zeros(size(tmp_background));
        background(9:end,:) = tmp_background(1:end-8,:);

        %Subtracting background from the foreground
        hologram = image - background + mean(hologram,'all');
        hologram = imgaussfilt(hologram,2,'FilterSize',5);
        m = mean(hologram, 'all');
        hologram = hologram / m;
    else
        hologram = image;
        hologram = imgaussfilt(hologram,2,'FilterSize',5);
        m = mean(hologram, 'all');
        hologram = hologram / m;
    end
    
    if (size(hologram,1) >= (y + y_o) && size(hologram, 2) >= (x + x_o))
        hologram = hologram(1+y_o:y+y_o, 1+x_o:x+x_o);
    else
        disp("The set hologram dimensions are larger than the actual hologram!");
    end
    
    
    
else
    
    H = RS_propagator(-z_m,y,x,dx,n,lambda);
    [hologram, ground_truth] = simulated_hologram(y,x,H,0.01);
    
end

%% Finding the appropriate offset 
%  (with the used images it was determined to be equal to 8)
%tmp_holo = background(6:end-5,6:end-5);
%autoc = xcorr2(image, background);

%% Running the selected algorithm

disp("Reconstruction algorithm started!");
Hq_m = RS_propagator(z_m,y,x,dx,n,lambda);
if used_algorithm == "iterative"
    Hq = complex(zeros(y,x,numel(z)),zeros(y,x,numel(z)));
    for i = 1:numel(z)
        Hq(:,:,i) = RS_propagator(z(i),y,x,dx,n,lambda);
    end
    h_reconstruction = iterative(hologram, Hq, x, y, z, ...
        iter, threshold, dilation, lpfilter, false);
    % Backpropagating the result as it is located in the hologram plane
    i_reconstruction = c_norm(propagation(h_reconstruction, Hq_m));
elseif used_algorithm == "fienup"
    i_reconstruction = fienup(hologram, Hq_m, iter, r_constr, i_constr);
elseif used_algorithm == "inverse"
    i_reconstruction = (fista(hologram, Hq_m, iter, mu, t, r_constr, i_constr, im_mask) + 1)*m;
elseif used_algorithm == "multi"
    Hq = complex(zeros(y,x,numel(z)),zeros(y,x,numel(z)));
    for i = 1:numel(z)
        Hq(:,:,i) = RS_propagator(z(i),y,x,dx,n,lambda);
    end
    i_reconstruction = multilayer_fista(hologram, Hq, iter, mu, t, r_constr, i_constr);
else
    disp("Chosen algorithm not recognized");
end

%% Printing the results

%Just a simple backpropagation for comparison with the selected algorithm
i_simple = abs(c_norm(propagation(hologram, Hq_m)));

figure(1);
if used_algorithm == "iterative"
    subplot(1,2,1);
    imshow(abs(i_reconstruction));
    title("Reconstruction result");
    subplot(1,2,2);
    imshow(abs(i_simple));
    title("Simple backpropagation");
elseif used_algorithm == "multi"
    modulus1 = abs((i_reconstruction(:,:,1)+1)*m);
    modulus2 = abs((i_reconstruction(:,:,2)+1)*m);
    subplot(2,3,1);
    imshow(modulus1);
    title("Reconstructed amplitude 1");
    subplot(2,3,4);
    imshow(modulus2);
    title("Reconstructed amplitude 2");
    subplot(2,3,2);
    p1 = angle(i_reconstruction(:,:,1)+1);
    imshow(p1, [-1,1]);
    title("Reconstructed phase 1");
    subplot(2,3,5);
    imshow(angle(i_reconstruction(:,:,2)+1), [-1,1]);
    title("Reconstructed phase 2");
    subplot(2,3,3);
    imshow(abs(i_simple), [-1,1]);
    title("Simple backpropagation");
    
else
    modulus = abs(i_reconstruction);
    subplot(1,2,1);
    imshow(modulus, [0,1]);
    title("Reconstructed amplitude");
    subplot(1,2,2);
    imshow(angle(i_reconstruction), [-1,1]);
    title("Reconstructed phase");
    %subplot(1,3,3);
    %imshow(abs(i_simple), [-1,1]);
    %title("Simple backpropagation");
end