function reconstruction = multilayer_fista(image, H, iter, mu, t, r_constr, i_constr)
    guess = complex(cat(3,image,image), zeros(size(image,1),size(image,2),2));
    u = guess;
    s = 1;
    
    Hn = conj(H);
    r = complex(zeros(size(image,1),size(image,2),2), zeros(size(image,1),size(image,2),2));
    
    height = size(image,1);
    width = size(image, 2);
    
    for i = 1:iter
        model = 1 + propagation(u(:,:,1), H(:,:,1)) + propagation(u(:,:,2), H(:,:,2));
        Imodel = abs(model).^2;
        
        c = sum(Imodel(:).*image(:))/sum(Imodel(:).*Imodel(:));
        
        %calculating cost
        cost_pixel = c*Imodel-image;

        cost = sum(cost_pixel(:).^2) + mu*sum(abs(guess(:)));
        disp(strcat("Cost:  ",  num2str(cost)));
        
        
        r(:,:,1) = propagation(model(:,:,1).*(c*Imodel - image), Hn(:,:,1));
        r(:,:,2) = propagation(model(:,:,1).*(c*Imodel - image), Hn(:,:,2));
        new_guess = u - 2*t*c*r;
        
        %strict bounds
        for k = 1:size(H,3)
            for j = 1:height
                for l = 1:width
                    new_guess(j,l,k) = complex(max(min([real(new_guess(j,l,k)),r_constr(2,k)]),r_constr(1,k)),...
                        max(min([imag(new_guess(j,l,k)),i_constr(2,k)]),i_constr(1,k)));
                     %new_guess(j,l,k) = max(0, new_guess(j,l,k) - mu*t);
                end
            end
        end
        s_new = 0.5*(1+sqrt(1+4*s^2)); %Lines 39-41 are FISTA unique
        u = new_guess + (s-1)*(new_guess -guess)/s_new;
        s = s_new;
        guess = new_guess;
    end
    reconstruction = guess;
end