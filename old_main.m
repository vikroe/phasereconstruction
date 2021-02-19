dx = 1.55e-6;
lambda = 520e-9;
n = 1.45;
distances = linspace(1.9e-3, 3.1e-3, 12);
ndistances = -distances;
width = 511; % image size
height = 511;

simulation = 0;

Hnq = [];
Hq = [];
for i = 1:numel(distances)
    Hnq = cat(3,Hnq,fresnel_kernel(ndistances(i),width,height,dx,n,lambda));
    Hq = cat(3,Hq,fresnel_kernel(distances(i),width,height,dx,n,lambda));
end

%% Creating a simulated hologram with a single rectangle

if simulation == 1
    r = 5;
    original_image = ones(width,height);
    temp_image = zeros(width,height);
    for i=1:numel(temp_image(1,:))
        for j =1:numel(temp_image(:,1))
            if (i - width/2-5)^2 + (j - height/2+100)^2 < r^2
                temp_image(i,j) = 1;
            end
        end
    end

    original_image = original_image - temp_image;
    fourier = fft2(original_image);
    kernelized = Hq(:,:,4).*fourier;
    hologram = ifft2(kernelized);

    actual_hologram = hologram;
    [~,~,hologram] = normalize(abs(real(hologram)));
    figure(5)
    subplot(1,2,1)
    imshow(hologram)
    subplot(1,2,2)
    imshow(original_image)
end

%% Loading image 
if simulation == 0
    image = imread("Pics/Hologram_large.png");
    %hologram8 = rgb2gray(image);
    hologram = im2double(image);
    figure(1)
    hologram = hologram(end-510:end,end-510:end);
    [minimum, maximum, hologram] = normalize(hologram);
    imshow(hologram)
end
%% Hologram reconstruction affected by twin-image and Construction of mask M
Masks = [];
fourier2 = fft2(hologram);

for i = 1:numel(distances)
    kernelized = Hq(:,:,i).*fourier2;
    reconstruction = abs(ifft2(kernelized));

    temp = (mask(reconstruction, 0.18, 3, 11));
    Masks = cat(3,Masks,temp);
end


mask_index = 4;
if simulation == 1
    Masks(:,:,mask_index) = temp_image;
    %Masks(:,:,mask_index) = imdilate(Masks(:,:,mask_index),strel('disk',2,0));
    %Masks(:,:,mask_index) = imgaussfilt(Masks(:,:,mask_index),2,'FilterSize',11);
end
figure(2)
subplot(1,3,1);
imshow(Masks(:,:,mask_index));
subplot(1,3,2);

kernelized = Hq(:,:,mask_index).*fourier2;
reconstruction = abs(ifft2(kernelized));

imshow(reconstruction);
subplot(1,3,3);

kernelized = Hnq(:,:,mask_index).*fourier2;
reconstruction = abs(ifft2(kernelized));

imshow(reconstruction+Masks(:,:,mask_index)*0.5);
%% Implementation of the iterative cleaning algorithm 

iterations = 2;

IH = hologram;
M = Masks;

a = fft2(1-hologram);
for i = 1:iterations
    if simulation ~= 1
        for j = 1:numel(distances)
            backpropagation = a.*(Hq(:,:,j));
            masking = (M(:,:,j)) .* ifft2(backpropagation);
            propagation = (fft2(masking).*(Hnq(:,:,j)));
            a = a - propagation;
        end
        %for j = 1:numel(distances)
        %    kernelized = Hnq(:,:,j).*(1-a);
        %    tmp_propagation = abs(ifft2(kernelized));
        %    M(:,:,j) = (mask(reconstruction, 0.18/sqrt(i+1), 3, 11));
        %end
    else
        backpropagation = a.*Hq(:,:,mask_index);
        masking = M(:,:,mask_index) .* ifft2(backpropagation);
        propagation = fft2(masking) .* Hnq(:,:,mask_index);
        a = a - propagation;
    end
end

%% print pictures

    [~,~,finished] = normalize(abs(ifft2(a.*Hnq(:,:,4))));

    figure(3);
    subplot(2,4,1);
    imshow(hologram/max(hologram, [],'all'));
    title("Original Hologram");
    subplot(2,4,2);
    imshow(abs(ifft2(backpropagation)));
    title("Back-propagation");
    subplot(2,4,3);
    %imshow(original_image);
    %title("Original Object");
    %subplot(2,4,4);
    imshow(abs(ifft2(propagation))/(max(abs(ifft2(propagation)), [],'all')));
    title("");
    subplot(2,4,4);
    imshow(abs(hologram));
    title("Hologram");
    subplot(2,4,7);
    imshow(abs(reconstruction)/max(abs(reconstruction), [],'all'));
    title("Reconstruction");
    subplot(2,4,8);
    imshow(1-finished);
    title("Reconstruction + Phase retrieval");
    pause(1)





%% Functions

function Hq = fresnel_kernel(z, Y,X,dx,n,lambda)
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

function [minimum, maximum, normalized] = normalize(image)
    minimum = min(image,[],'all');
    maximum = max(image,[],'all');
    normalized = (image - abs(minimum))*(1/(abs(maximum)-abs(minimum)));
end