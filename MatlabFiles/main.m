%clc
%close all

run("parameters.m");

%% Loading electrode image 

if simulation == false
    image = im2double(imread(data));
    Hq_m = RS_propagator(-z_m,y,x,dx,n,lambda);

    hologram = image;
    m = mean(hologram, 'all');
    hologram = hologram / m;
    
    if (size(hologram,1) >= (y + y_o) && size(hologram, 2) >= (x + x_o))
        hologram = hologram(1+y_o:y+y_o, 1+x_o:x+x_o);
    else
        disp("The set hologram dimensions are larger than the actual hologram!");
    end
    
    i_simple = abs(propagation(hologram, Hq_m));
    
    ground_truth = 0;
    if print_extra
        figure(1);
        subplot(1,2,1);
        imshow(hologram, [0,2]);
        title("Reconstructed amplitude");
        subplot(1,2,2);
        imshow(abs(ground_truth), [0,1]);
        title("Reconstructed phase");
        if record_results
            imwrite(i_simple, "Results/backpropagation.png", "PNG");
            imwrite(hologram, "Results/simulation.png", "PNG");
        end
    end
    
else
    [hologram, ground_truth] = simulated_hologram(y,x,Hq_m,0.03);
    i_simple = abs(propagation(hologram, Hq_m));
    if print_extra
        figure(1);
        subplot(1,2,1);
        imshow(hologram, [0,2]);
        title("Reconstructed amplitude");
        subplot(1,2,2);
        imshow(abs(ground_truth), [0,1]);
        title("Reconstructed phase");
        if record_results
            imwrite(ground_truth, "Results/ground_truth.png", "PNG");
            imwrite(hologram, "Results/simulation.png", "PNG");
        end
    end
end

%% Running the selected algorithm

disp("Reconstruction algorithm started!");
i_reconstruction = fista(hologram, Hq_m, iter, mu, t, r_constr, i_constr, ...
    simulation, ground_truth) + conj(Hq_m(1,1));

%% Printing the results

figure(2);

modulus = abs(i_reconstruction)*m;
modulus = modulus/max(modulus(:));
phase = angle(i_reconstruction);
for i = 1:x
    for j = 1:y
        phase(i,j) = bound(0,2,phase(i,j));
    end
end

subplot(1,3,1);
imshow(modulus, [0,1]);
title("Reconstructed amplitude");

subplot(1,3,2);
imshow(r_norm(phase), [-0,1]);
title("Reconstructed phase");

subplot(1,3,3);
imshow(abs(i_simple)*m, [-1,1]);
title("Simple backpropagation");

if record_results
    imwrite(modulus, strcat(ResultFolder, 'modulus.png'), "PNG");
    imwrite(r_norm(phase), strcat(ResultFolder,'phase.png'), "PNG");
    imwrite(abs(i_simple)*m, strcat(ResultFolder,"backprop.png"), "PNG");
end