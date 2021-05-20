function [hologram,ground_truth] = simulated_hologram(y,x,H,sigma)
% Hologram simulation 

    image = complex(zeros(y,x),zeros(y,x));

    for i = 1:y
        for j = 1:x
            if (i-70)^2 + (j-70)^2 < 16^2
                image(i,j) = complex(real(image(i,j)) - 0.5, imag(image(i,j))+0.1);
            elseif (i-200)^2 + (j-120)^2 < 16^2
                image(i,j) = complex(real(image(i,j)) - 0.5, imag(image(i,j))-0.1);
            elseif (i-430)^2 + (j-450)^2 < 16^2
                image(i,j) = complex(real(image(i,j)) - 0.5, imag(image(i,j))+0.2);
            elseif (i-320)^2 + (j-400)^2 < 16^2
                image(i,j) = complex(real(image(i,j)) - 0.5, imag(image(i,j))+0.2);
            elseif (i-170)^2 + (j-220)^2 < 16^2
                image(i,j) = complex(real(image(i,j)) - 0.5, imag(image(i,j))+0.1);
            elseif (i-300)^2 + (j-350)^2 < 16^2
                image(i,j) = complex(real(image(i,j)) - 0.5, imag(image(i,j))-0.2);
            end
        end
    end

    sensor = propagation(image, H);
    hologram = abs(1 + sensor).^2 + sigma*randn(y,x);
    ground_truth = image;

end
