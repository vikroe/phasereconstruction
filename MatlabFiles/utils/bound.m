function result = bound(low, high, in)
%BOUND Summary of this function goes here
%   Detailed explanation goes here
    result = max(min(in, high), low);
end

