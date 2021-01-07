%{
Name   :  A 9-neuron discrete-time Hopfield network 
          as an associative memory of 3-by-3 digital images
Author :  Wang Yue
Date   :  2020.11.18  
%}
clear; clc; 
close all

% Original patterns to be stored
x1 = [-1;+1;+1;  -1;-1;-1;  -1;+1;+1]; 
x2 = [-1;-1;-1;  -1;+1;-1;  -1;+1;-1];
% caculate W---------------------
W1=x1*x1'-eye(9);
W2=x2*x2'-eye(9);
W=W1+W2;
% Noisy patterns to be used for retrieval // probe / key / cue
input1 = [-1;+1;-1;  -1;-1;-1;  -1;+1;+1;];
input2 = [-1;-1;-1;  -1;+1;-1;  -1;+1;+1;];
% Desired output / corresponding prototype patterns
output1 = W*input1;
output2 = W*input2;
% sgn -----------------
for i = 1:9
    if output1(i)>=0
        output1(i) = 1;
    else
        output1(i) = -1;
    end
end

for i = 1:9
    if output2(i)>=0
        output2(i) = 1;
    else
        output2(i) = -1;
    end
end

% plot ---------------------
%Original patterns figures
fig11=reshape(x1, 3,3); 
figure(1),subplot(1,2,1),imshow(255*uint8(fig11))
title('Original patterns 1');
fig12=reshape(x2, 3,3); 
figure(1),subplot(1,2,2),imshow(255*uint8(fig12));
title('Original patterns 2');

% Noisy patterns to be used for retrieval
fig21=reshape(input1, 3,3); 
figure(2),subplot(1,2,1),imshow(255*uint8(fig21));
title('Noisy patterns 1');
fig22=reshape(input2, 3,3); 
figure(2),subplot(1,2,2),imshow(255*uint8(fig22));
title('Noisy patterns 2');

% Corresponding prototype patterns
fig31=reshape(output1, 3,3); 
figure(3),subplot(1,2,1),imshow(255*uint8(fig31));
title('Corresponding prototype patterns 1');
fig32=reshape(output2, 3,3); 
figure(3),subplot(1,2,2),imshow(255*uint8(fig32));
title('Corresponding prototype patterns 2');



