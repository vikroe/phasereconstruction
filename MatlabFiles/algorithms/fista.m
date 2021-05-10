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


        for j = 1:height*width
            new_guess(j) = complex(min([real(new_guess(j)),r_constr(2)]),...
            min([imag(new_guess(j)),i_constr(2)]));
            new_guess(j) = complex(max([real(new_guess(j)),r_constr(1)]),...
            max([imag(new_guess(j)),i_constr(1)]));
        %new_guess(j) = complex(real(new_guess(j)),...
        new_guess(j) = complex(sign(real(new_guess(j)))*max([0, abs(real(new_guess(j))) - mu]),...
        sign(imag(new_guess(j)))*max([0, abs(imag(new_guess(j)))-mu]));
        %imag(new_guess(j)));
        end
        %soft thresholding bounds
        s_new = 0.5*(1+sqrt(1+4*s^2)); %Lines 21-24 are FISTA unique
        u = new_guess + (s-1)*(new_guess -guess)/s_new;
        
        s = s_new;
        guess = new_guess;
    end
    reconstruction = guess;
end