function reconstruction = fista(image, H, iter, flag, mu, t)
    guess = complex(zeros(size(image)), zeros(size(image)));
    u = guess;
    s = 1;
    
    height = size(image,1);
    width = size(image, 2);
    
    for i = 1:iter
        model = 1 + propagation(u, H);
        c = sum(model.*image)/sum(model.*model);
        r = propagation(c*model - image, H);
        new_guess = u - 2*t*c*r;
        for j = height*width
            if flag == true %Check if positivity is enforced
                new_guess(j) = max(0, new_guess(j) - mu*t);
            else
                new_guess(j) = sign(new_guess(j))*max(0, abs(new_guess(j)) - mu*t);
            end
        end
        s_new = 0.5*(1+sqrt(1+4*s^2));
        u = new_guess + (s-1)*(new_guess -guess)/s_new;
        
        s = s_new;
        guess = new_guess;
    end
    reconstruction = guess;
end