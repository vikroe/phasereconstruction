function result = propagation(image,H)
%PROPAGATION An operator for propagation of an image by a propagator H.
% INPUTS:
%       image - 2D array to be propagated. No limit on input types, double
%       and complex are recommended.
%       
%       H - 2D array of the propagator. Expected to be of type complex.
%
% OUTPUTS:
%       result - 2D complex array of a propagated image. 
result = ifft2(fft2(image).*H);
end

