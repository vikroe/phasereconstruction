function h_finished = iterative(hologram, Hq, x, y, z, iter, thresh, dilation, lpf, verbose)
%% Initialization of variables
    M = zeros(y,x,numel(z));
    Hnq = conj(Hq);

%% Hologram reconstruction affected by twin-image and Construction of mask M
    disp("Calculating masks...");
    for i = 1:numel(z)
        a_real = r_norm(abs(propagation(hologram, Hq(:,:,i))));
        M(:,:,i) = mask(a_real, thresh, dilation, lpf);
    end
    disp("Masks calculated...");

%% Implementation of the iterative cleaning algorithm 

    a = hologram;
    for i = 1:iter
        disp(strcat("Iteration ", int2str(i)));
        a = c_norm(a);
        for idx = 1:numel(z)
            masking = M(:,:,idx) .* propagation(a,Hnq(:,:,idx));
            prop_mask = propagation(masking, Hq(:,:,idx));
            a = a - prop_mask / numel(z);
        end
    end

%% Print results if verbose
    
    h_finished = a;

    if verbose
        im = imag(h_finished);
        figure(3);
        subplot(1,3,1);
        imshow(hologram/max(hologram, [],'all'));
        title("Original Hologram");
        subplot(1,3,2);
        imshow(abs(a_real)/max(abs(a_real(:))));
        title("Reconstruction");
        subplot(1,3,3);
        pause(1)
    end

end



