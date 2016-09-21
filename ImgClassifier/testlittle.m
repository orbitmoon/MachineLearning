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

C3size = [5 6];
S4size = [2 6];
C5size = [5 12];
F6size = 10;
num_labels = 2; 
fprintf('\nLoading Saved Neural Network Parameters ...\n')

C3 = randInitializeKernals(C3size(1),C3size(2));
S4 = randInitializeKernals(S4size(1),S4size(2));
C5 = randInitializeKernals(C5size(1),C5size(2));
F6 = randInitializeWeights(C5size(2), F6size)';
softmax = randInitializeWeights(F6size, num_labels)';

B3 = randInitializeBias(C3size(2));
B4 = randInitializeBias(S4size(2));
B5 = randInitializeBias(C5size(2));
B6 = randInitializeBias(F6size);
BS = randInitializeBias(num_labels);

initial_cnn_params = [C3(:);S4(:);C5(:);F6(:);softmax(:);B3(:);B4(:);B5(:);B6(:);BS(:)];
cnn_params = initial_cnn_params;
startpoint = 1;
C3re = reshape(cnn_params(startpoint:startpoint + (C3size(1)^2*C3size(2))-1),C3size(1),C3size(1),C3size(2));
startpoint += (C3size(1)^2*C3size(2));

S4re = reshape(cnn_params(startpoint:startpoint + (S4size(1)^2*S4size(2))-1),S4size(1),S4size(1),S4size(2));
startpoint +=  (S4size(1)^2*S4size(2));

C5re = reshape(cnn_params(startpoint:startpoint + (C5size(1)^2*C5size(2))-1),C5size(1),C5size(1),C5size(2));
startpoint +=  (C5size(1)^2*C5size(2));

F6re = reshape(cnn_params(startpoint:startpoint + (C5size(2)*F6size)-1),C5size(2),F6size);
startpoint +=  (C5size(2)*F6size);

softmaxre = reshape(cnn_params(startpoint:startpoint + (F6size*num_labels)-1),F6size,num_labels);
startpoint += (F6size*num_labels);

B3re = reshape(cnn_params(startpoint:startpoint + C3size(2)-1),C3size(2),1);
startpoint += C3size(2);

B4re = reshape(cnn_params(startpoint:startpoint + S4size(2)-1),S4size(2),1);
startpoint += S4size(2);

B5re = reshape(cnn_params(startpoint:startpoint + C5size(2)-1),C5size(2),1);
startpoint += C5size(2);

B6re = reshape(cnn_params(startpoint:startpoint + F6size-1),F6size,1);
startpoint += F6size;

BSre = reshape(cnn_params(startpoint:startpoint + num_labels-1),num_labels,1);

featuremapL3 = zeros(10,10,6);
for i = 1:6
    W = flipud(fliplr(C3re(:,:,i)));
    featuremapL3(:,:,i) = conv2(X,W,'valid') + B3re(i);
end;
featuremapL3 = sigmoid(featuremapL3);

featuremapL4 = zeros(5,5,6);
for i = 1:6
    featuremapL4(:,:,i) = subSample(featuremapL3(:,:,i),S4re(:,:,i)) + B4re(i);
end;
featuremapL4 = sigmoid(featuremapL4);

featuremapL5 = zeros(1,1,12);
for i = 1:12
    W = flipud(fliplr(C5re(:,:,i)));
    for j = 1:6
        featuremapL5(:,:,i) += conv2(featuremapL4(:,:,j),W,'valid') + B5re(i);
    end;
end;
featuremapL5 = sigmoid(featuremapL5);

L5 = zeros(12,1);
for i = 1:12
    L5(i,1) = featuremapL5(1,1,i);
end;

Z6 = L5'*F6re + B6re';
A6 = sigmoid(Z6);

Z7 = A6*softmaxre + BSre';
hX = sigmoid(Z7);

J =  -(1/m)*sum((y.*log(hX)+(1-y).*log(1-hX))(:));
reg = (lambda/(2*m))*(sum((C3re.^2)(:))+sum((S4re.^2)(:))+sum((C5re.^2)(:))+sum((F6re.^2)(:))+sum((softmaxre.^2)(:)));
Jreg = J+reg;

delta7 = hX - y;                                              %1*2
delta6 = delta7*softmaxre'.*(A6.*(1-A6));     %1*10
delta5 = delta6*F6re'.*(L5'.*(1-L5'));              %1*12

W=zeros(150,12);
C5d=zeros(5,5);
for i=1:12
    C5d = C5re(:,:,i);
    C5d = C5d(:);
    for j=1:150
        iv=mod(j,25);
        if iv~=0
            W(j,i) = C5d(iv);
        else
            W(j,i) = C5d(25);
        end;
    end;
end;

featuremapL4line = featuremapL4(:);
delta4 = W*delta5'.*(featuremapL4line.*(1-featuremapL4line));

W2=zeros(150,600);
S4d = S4re(:)';

for i = 1:150
    iv = ceil(i/25);
    W2(i,(4*(i-1)+1):(4*i)) = S4d(1,(4*(iv-1)+1):(4*iv));
end;

featuremapL3line = featuremapL3(:);
delta3 = W2'*delta4.*(featuremapL3line.*featuremapL3line);

W3 = zeros(1176,600);


D6 = A6'*delta7;  %10*2
D5 = L5*delta6;    %12*10
D4a = featuremapL4(:)*delta5;
D4b = zeros(25,12);

for i = 1:12
    for j = 1:25
        D4b(j,i) = D4a(j,i)+D4a(j+25,i)+D4a(j+50,i)+D4a(j+75,i)+D4a(j+100,i)+D4a(j+125,i);
    end;
end;



