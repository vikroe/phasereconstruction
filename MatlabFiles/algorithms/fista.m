function reconstruction = fista(image, H, iter, mu, t, r_constr, i_constr, mask)
    guess = complex(image, zeros(size(image)));
    u = guess;
    s = 1;
    
    Hn = conj(H);
    
    height = size(image,1);
    width = size(image, 2);
    
    for i = 1:iter
        model = 1 + propagation(u, H);
        Imodel = abs(model).^2;
        c = sum(Imodel(:).*image(:))/sum(Imodel(:).*Imodel(:));
        
        %calculating cost
        cost_pixel = c*Imodel-image;
        cost = sum(cost_pixel(:).^2) + mu*sum(abs(guess(:)));
        disp(strcat("Iteration " , num2str(i-1), ", Cost:  ",  num2str(cost)));
        
        r = propagation(model.*(c*Imodel - image), Hn);
        new_guess = u - 2*t*c*r;
        
        %strict bounds

        %soft thresholding bounds
        for j = 1:height*width
            new_guess(j) = max(0, new_guess(j) - mu*t);
        end
        for j = 1:height*width
            if mask(j) ~= 0
                new_guess(j) = complex(0,...
                min([imag(new_guess(j)),i_constr(2)]));
                new_guess(j) = complex(0,...
                max([imag(new_guess(j)),i_constr(1)]));
            else
                new_guess(j) = complex(min([real(new_guess(j)),0]),...
                min([imag(new_guess(j)),i_constr(2)]));
                new_guess(j) = complex(max([real(new_guess(j)),-1]),...
                max([imag(new_guess(j)),i_constr(1)]));
            end
        end
        s_new = 0.5*(1+sqrt(1+4*s^2)); %Lines 21-24 are FISTA unique
        u = new_guess + (s-1)*(new_guess -guess)/s_new;
        
        s = s_new;
        guess = new_guess;
    end
    reconstruction = guess;
end