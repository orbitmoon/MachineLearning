clear ; close all; clc

pkg load image;

mnist = dir('tp\');

tmp = imread(['tp\',mnist(3).name]);
tmpre = imresize(tmp,[14,14]);
X = double(tmpre);
X = (X-127.5)/127.5;

y = [0 1];

m=1;
lambda = 1;

L1size = [5 3];
L2size = [1 3];
L3size = 2;

init_W1 = randInitializeKernals(L1size(1),L1size(2));
init_W2 = randInitializeKernals(L2size(1),L2size(2)); 
init_W3 = randInitializeWeights(5*5*3,2);

init_B1 = randInitializeBias(L1size(2));
init_B2 = randInitializeBias(L2size(2));
init_B3 = randInitializeBias(L3size);

initial_cnn_params = [init_W1(:);init_W2(:);init_W3(:);init_B1(:);init_B2(:);init_B3(:)];
cnn_params = initial_cnn_params;

startpoint = 1;
W1 = reshape(cnn_params(startpoint:(L1size(1)^2*L1size(2))),L1size(1),L1size(1),L1size(2));
startpoint += (L1size(1)^2*L1size(2));

W2 = reshape(cnn_params(startpoint:startpoint -1 + (L2size(1)^2*L2size(2))),L2size(1),L2size(2));
startpoint += (L2size(1)^2*L2size(2));

W3 = reshape(cnn_params(startpoint:startpoint -1 + 5*5*3*2),2,5*5*3);
startpoint += (5*5*3*2);

B1 = reshape(cnn_params(startpoint:startpoint + L1size(2)-1),L1size(2),1);
startpoint += L1size(2);

B2 = reshape(cnn_params(startpoint:startpoint + L2size(2)-1),L2size(2),1);
startpoint += L2size(2);

B3 = reshape(cnn_params(startpoint:startpoint + L3size-1),L3size,1);

z1 = zeros(10,10,3);
for i = 1:3
    c = flipud(fliplr(W1(:,:,i)));
    z1(:,:,i) = conv2(X,c,'valid') + B1(i);
end;
a1 = sigmoid(z1);

z2 = zeros(5,5,3);
for i = 1:3
    z2(:,:,i) = meanpool(a1(:,:,i),W2(i),B2(i));
end;













