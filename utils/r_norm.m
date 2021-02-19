function normalized = r_norm(A)
    %R_NORM - scales the input array to interval <0;1>
    
    normalized = A - min(A);
    normalized = normalized/max(normalized(:));
end