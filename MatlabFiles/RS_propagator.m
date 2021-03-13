function Hq = RS_propagator(z, Y, X, dx, n, lambda)
    Hq = complex(zeros(Y,X),zeros(Y,X));
    calc = 1/dx;
    pre = n/lambda;
    for i = 1:Y
        for j = 1:X
            FX = (i-1.5) * calc/X - calc/2.0;
            FY = (j-1.5) * calc/Y - calc/2.0;
            res = 2 * pi*z*pre * sqrt(1 - (FX/pre)^2 - (FY/pre)^2);
            temp = 0;
            if sqrt(FX^2 + FY^2) < pre
                temp = 1;
            end
            if temp == 1 
                Hq(i, j) = complex(cos(res),sin(res));
            else
                Hq(i, j) = 0;
            end
        end
    end
    Hq = fftshift(Hq);
end