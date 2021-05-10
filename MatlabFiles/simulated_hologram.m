function [hologram,ground_truth] = simulated_hologram(y,x,H,sigma)
% Hologram simulation 

    image = complex(zeros(y,x),zeros(y,x));

    for i = 1:y
        for j = 1:x
            if (i-100)^2 + (j-100)^2 < 30^2
                image(i,j) = image(i,j) - 0.5;
            end
        end
    end

    sensor = propagation(image, H);
    hologram = abs(1 + sensor).^2 + sigma*randn(y,x);
    ground_truth = (1+image)/mean(1+image(:));

end
