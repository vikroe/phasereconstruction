function normalized = c_norm(A)
    %C_NORM - scales the complex array to a unit circle by the absolute
    %values of its members
    
    a = abs(A);
    scale = max(a(:));
    normalized = A/scale;
end