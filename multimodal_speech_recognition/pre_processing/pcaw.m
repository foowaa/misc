%http://blog.csdn.net/yutianzuijin/article/details/10823985
%https://my.oschina.net/findbill/blog/543485
clear all
clc
cd /home/user3/pic/pic

folder = '/home/user3/pic/pic/pic/';
files = dir('*.bmp');
num_images = length(files);
image_dims = [60 80];
for m=1:num_images
    file = files(m,1);
    img = imread(file.name);
    img = double(img);
    if m==1
        images = zeros(num_images, prod(image_dims));
    else
        images(m,:) = img(:);
    end
end


mean_face = mean(images ,2 );
shifted_images = images-repmat(mean_face, 1, prod(image_dims));
covariance = cov(shifted_images);
num_eigens = 100;
[eigenvec, eigenval] = eig(covariance);

eigenvec = eigenvec(:,1:num_eigens);
    
    
save eigenvec

