function reconstruction = fienup(image, H, iter, i_constr, r_constr)
%FIENUP Summary of this function goes here
%   Detailed explanation goes here
    disp("Intializing the Fienup's algorithm");
    x = size(image, 2);
    y = size(image, 1);

    Hn = conj(H); %Propagation kernel to the oposite plane from H
    image = r_norm(image);
    
    guess = propagation(sqrt(image)-1, H);
    
    for i = 1:iter
        disp(strcat("Iteration ",int2str(i), "..."));
        a = 1 + propagation(guess, Hn);
        for qy = 1:y
            for qx = 1:x
                if abs(a(qy,qx)) ~= 0
                    a(qy,qx) = sqrt(image(qy,qx))*(a(qy,qx)/abs(a(qy,qx)));
                else
                    a(qy,qx) = 0;
                end
            end
        end
        guess = propagation(a-1,H);
        for qy = 1:y
            for qx = 1:x
                guess(qy,qx) = complex(max(min([real(guess(qy,qx)),r_constr(2)]),r_constr(1)),...
                    max(min([imag(guess(qy,qx)),i_constr(2)]),i_constr(1)));
            end
        end
    end
    reconstruction = guess;
end

