function reconstruction = fista(image, H, iter, mu, t, flag_snr, ground_truth)
    guess = complex(image, zeros(size(image)));
    u = guess;
    s = 1;
    
    Hn = conj(H);
    
    height = size(image,1);
    width = size(image, 2);
    
    if flag_snr
        n = norm(ground_truth(:));
        snr = 20*log10(n/norm(guess(:) - ground_truth(:)));
        disp(strcat("Iteration 0, SNR:  ",  num2str(snr)));
    end
    
    for i = 1:iter
        model = 1 + propagation(u, H);
        Imodel = abs(model).^2;
        c = sum(Imodel(:).*image(:))/sum(Imodel(:).*Imodel(:));
        
        %Calculating cost
        residues = c*Imodel-image;
        cost = sum(residues(:).^2) + mu*sum(abs(guess(:)));
        disp(strcat("Iteration " , num2str(i-1), ", Cost:  ",  num2str(cost)));
        
        new_guess = u - 2*t*c*propagation(model.*residues, Hn);

        for j = 1:height*width
            %Positive bounding
            new_guess(j) = complex(min(0,real(new_guess(j))),imag(new_guess(j)));
            %Soft-thresholding bounding
            new_guess(j) = softthreshold(new_guess(j), mu, t);
        end

        s_new = 0.5*(1+sqrt(1+4*s^2)); %Accelerated gradient descent
        u = new_guess + (s-1)*(new_guess -guess)/s_new;
        
        s = s_new;
        guess = new_guess;
        
        if flag_snr 
            snr = 20*log10(n/norm(guess(:) - ground_truth(:)));
            disp(strcat("Iteration " , num2str(i), ", SNR:  ",  num2str(snr)));
        end
        
    end
    model = 1 + propagation(u, H);
    Imodel = abs(model).^2;
    c = sum(Imodel(:).*image(:))/sum(Imodel(:).*Imodel(:));

    %Calculating cost
    residues = c*Imodel-image;
    cost = sum(residues(:).^2) + mu*sum(abs(guess(:)));
    disp(strcat("Iteration " , num2str(iter), ", Cost:  ",  num2str(cost)));
    reconstruction = guess;
end