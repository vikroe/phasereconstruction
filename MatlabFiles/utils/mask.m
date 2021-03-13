function M = mask(image, thresh, dilation, lpfilter)
    M = ones(size(image,1),size(image,2));
    for id = 1:size(image,1)
        for j = 1:size(image,2)
            if image(id,j) > thresh
                M(id,j) = 0;
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