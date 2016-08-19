%try to build an image classifier based on neural network
pkg load image;

filepos = dir('img\');
fileneg = dir('neg\');

X = zeros(100,57600,'uint8');

for n=3:52
tmp = rgb2gray(imread(['img\',filepos(n).name]));
tmp = imresize(tmp,[180,320]);
tmp = tmp(:);
X(n-2,:) = tmp';
end;

for n=3:52
tmp = rgb2gray(imread(['neg\',fileneg(n).name]));
tmp = imresize(tmp,[180,320]);
tmp = tmp(:);
X(n+48,:) = tmp';
end;

imtest = X(20,:);
imtest = reshape(imtest(1:57600),180,320);
imshow(imtest);

