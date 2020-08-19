dx = 1.55e-6;
lambda = 520e-9;
n = 1.45;
distances = linspace(2.2e-3, 5e-3, 7);
ndistances = -distances;
width = 511; % image size
height = 511;

Hnq = [];
Hq = [];
for i = 1:numel(distances)
    Hnq = cat(3,Hnq,our_fresnel_kernel(ndistances(i),width,height,dx,n,lambda));
    Hq = cat(3,Hq,our_fresnel_kernel(distances(i),width,height,dx,n,lambda));
end

%% Creating a simulated hologram with a single rectangle

r = 15;
original_image = ones(width,height);
temp_image = zeros(width,height);
for i=1:numel(temp_image(1,:))
    for j =1:numel(temp_image(:,1))
        if i > width/2-5 && i < width/2+5  && j < height/2+100 && j > height/2-100
            temp_image(i,j) = 1;
        end
    end
end

original_image = ones(width,height) - (original_image - temp_image);
fourier = fft2(original_image);
kernelized = Hnq(:,:,1).*fourier;
hologram = ifft2(kernelized);

actual_hologram = hologram;
hologram = real(hologram);
figure(5)
subplot(1,2,1)
imshow(hologram)
subplot(1,2,2)
imshow(original_image)

%% Loading image 

image = imread("Pics/Hologram_large.png");
%hologram8 = rgb2gray(image);
hologram = im2double(image);
figure(1)
hologram = hologram(end-510:end,end-510:end);
[minimum, maximum, hologram] = normalize(hologram);
imshow(hologram)

%% Hologram reconstruction affected by twin-image and Construction of mask M
M = [];
fourier2 = fft2(hologram);

for i = 1:numel(distances)
    kernelized = Hnq(:,:,i).*fourier2;
    reconstruction = abs(ifft2(kernelized));

    temp = (mask(reconstruction, 0.18, 4, 21));
    M = cat(3,M,temp);
end

mask_index = 1;
%M(:,:,mask_index) = temp_image;
%M(:,:,mask_index) = imdilate(M(:,:,mask_index),strel('disk',2,0));
%M(:,:,mask_index) = imgaussfilt(M(:,:,mask_index),2,'FilterSize',11);
figure(2)
subplot(1,3,1);
imshow(M(:,:,mask_index));
subplot(1,3,2);

kernelized = Hnq(:,:,mask_index).*fourier2;
reconstruction = abs(ifft2(kernelized));

imshow(reconstruction);
subplot(1,3,3);

kernelized = Hq(:,:,mask_index).*fourier2;
reconstruction = abs(ifft2(kernelized));

imshow(reconstruction+M(:,:,mask_index)*0.5);
%% Implementation of the iterative cleaning algorithm 

iterations = 25;

IH = hologram;

a = fft2(hologram);
for i = 1:iterations
    for j = 1:numel(distances)
        backpropagation = a.*Hq(:,:,j);
        masking = (M(:,:,j)) .* ifft2(backpropagation);
        propagation = (fft2(masking).*Hnq(:,:,j));
        anew = a - propagation/numel(distances);
        a = anew;
    end
    for j = 1:numel(distances)
        kernelized = Hnq(:,:,j).*a;
        tmp_propagation = abs(ifft2(kernelized));
        M(:,:,j) = (mask(reconstruction, 0.18/sqrt(1+i), 3, 11));
    end
end

    [~,~,finished] = normalize(abs(ifft2((a.*Hnq(:,:,mask_index)))));

    figure(3);
    subplot(2,4,1);
    imshow(hologram/max(hologram, [],'all'));
    title("Original Hologram");
    subplot(2,4,2);
    imshow(abs(ifft2(backpropagation)));
    title("Back-propagation");
    subplot(2,4,3);
    imshow(M(:,:,1));
    title("Masking");
    subplot(2,4,4);
    imshow(abs(ifft2(propagation))/(max(abs(ifft2(propagation)), [],'all')));
    title("Propagation");
    subplot(2,4,5);
    imshow(abs(a));
    title("New hologram");
    subplot(2,4,7);
    imshow(abs(reconstruction)/max(abs(reconstruction), [],'all'));
    title("Original reconstruction");
    subplot(2,4,8);
    imshow(finished);
    title("Cleaned");
    pause(1)





%% Functions

function Hq = our_fresnel_kernel(z, Y,X,dx,n,lambda)
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

function Hq = fresnel_kernel(z, Y,X,dx,lambda)
    pixel = 1e-6;
    Hq = complex(zeros(Y,X),zeros(Y,X));
    calc = 1/pixel;
    for i = 1:Y
        for j = 1:X
            FX = (i-1.5) * calc/X - calc/2.0;
            FY = (j-1.5) * calc/Y - calc/2.0;
            res = 2 * pi*z/lambda * sqrt(1 - (FX*lambda)^2 - (FY*lambda)^2);
            temp = 0;
            if sqrt(FX^2 + FY^2) < 1/lambda
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

function M = mask(image, threshold, dilation, lpfilter)
    M = ones(size(image,1),size(image,2));
    m = min(image, [], 'all');
    for i = 1:size(image,1)
        for j = 1:size(image,2)
            if image(i,j) > threshold
                M(i,j) = 0;
            end
        end
    end
    M = imdilate(M,strel('disk',dilation,0));
    M = imgaussfilt(M,2,'FilterSize',lpfilter);
    M = ones(size(image)) - M;
    M = imdilate(M,strel('disk',dilation+2,0));
    M = imgaussfilt(M,2,'FilterSize',lpfilter);
    M = ones(size(image)) - M;
end

function kernel = lowpass(side)
    kernel = zeros(side,side);
    for i = 1:side
        for j = 1:side
            if ((i-side/2)^2 + (j-side/2)^2) < side^2
                kernel(i,j) = 1;
            end
        end
    end
    kernel = kernel / sum(kernel, 'all');
    imshow(kernel)
end

function [minimum, maximum, normalized] = normalize(image)
    minimum = min(image,[],'all');
    maximum = max(image,[],'all');
    normalized = (image - abs(minimum))*(1/(abs(maximum)-abs(minimum)));
end