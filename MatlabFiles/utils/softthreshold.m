function result = softthreshold(input, mu, t)
%SOFTTHRESHOLD Summary of this function goes here
%   Detailed explanation goes here
result = complex(sign(real(input))*max([0, abs(real(input)) - mu*t]),...
            sign(imag(input))*max([0, abs(imag(input))-mu*t]));
end

