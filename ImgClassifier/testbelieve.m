clear ; close all; clc

pkg load image;

mnist = dir('tp\');

tmp = imread(['tp\',mnist(3).name]);
tmpre = imresize(tmp,[14,14]);
X = double(tmpre);
X = (X-127.5)/127.5;

y = [1];

m=1;
lambda = 1;

L1size = [5 3];
L2size = [1 3];
L3size = [5 2];
L4size = 1;

init_W1 = randInitializeKernals(L1size(1),L1size(2));
init_W2 = randInitializeKernals(L2size(1),L2size(2)); 
init_W3 = randInitializeKernals(L3size(1),L3size(2));
init_W4 = randInitializeWeights(2,1);

init_B1 = randInitializeBias(L1size(2));
init_B2 = randInitializeBias(L2size(2));
init_B3 = randInitializeBias(L3size(2));
init_B4 = randInitializeBias(L4size);

initial_cnn_params = [init_W1(:);init_W2(:);init_W3(:);init_W4(:);init_B1(:);init_B2(:);init_B3(:);init_B4(:)];
cnn_params = initial_cnn_params;

startpoint = 1;
W1 = reshape(cnn_params(startpoint:(L1size(1)^2*L1size(2))),L1size(1),L1size(1),L1size(2));
startpoint += (L1size(1)^2*L1size(2));

W2 = reshape(cnn_params(startpoint:startpoint -1 + (L2size(1)^2*L2size(2))),L2size(1),L2size(2));
startpoint += (L2size(1)^2*L2size(2));

W3 = reshape(cnn_params(startpoint:startpoint -1 + (L3size(1)^2*L3size(2))),L3size(1),L3size(1),L3size(2));
startpoint += (L3size(1)^2*L3size(2));

W4 = reshape(cnn_params(startpoint:startpoint -1 + 2*1),1,2);
startpoint += (2*1);

B1 = reshape(cnn_params(startpoint:startpoint + L1size(2)-1),L1size(2),1);
startpoint += L1size(2);

B2 = reshape(cnn_params(startpoint:startpoint + L2size(2)-1),L2size(2),1);
startpoint += L2size(2);

B3 = reshape(cnn_params(startpoint:startpoint + L3size(2)-1),L3size(2),1);
startpoint += L3size(2);

B4 = reshape(cnn_params(startpoint:startpoint + L4size-1),1,L4size);

z2 = zeros(10,10,3);
for i = 1:3
    c = flipud(fliplr(W1(:,:,i)));
    z2(:,:,i) = conv2(X,c,'valid') + B1(i);
end;
a2 = sigmoid(z2);

z3 = zeros(5,5,3);
for i = 1:3
    z3(:,:,i) = meanpool(a2(:,:,i),W2(:,i),B2(i));
end;
a3 = sigmoid(z3);

z4 = zeros(1,1,2);
for i = 1:2
    c = flipud(fliplr(W3(:,:,i)));
    for j = 1:3
        z4(:,:,i)+=conv2(a3(:,:,j),c,'valid') + B3(i);
    end;
end;
a4 = sigmoid(z4);

a4_full = a4(:)';

z5 = a4*W4' + B4;
a5 = sigmoid(z5);

J =  -(1/m)*sum((y.*log(a5)+(1-y).*log(1-a5))(:));

delta5 = a5 - y;
delta4 = delta5*W4.*(a4_full.*(1-a4_full));

delta4_r = reshape(delta4,1,1,2);
delta3 = zeros(5,5,3);
for i=1:3
    for j=1:2
        delta3(:,:,i) += (a3(:,:,i).*(1-a3(:,:,i))).*conv2(delta4_r(:,:,j),rot90(rot90(W3(:,:,j))),'full');
    end;
end;



