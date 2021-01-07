%{
Name   :  A 5-neuron MAXNET with lateral connection weight of -0.15, 
            and external input vector of [0.1, 0.3, 0.5, 0.7, 0.9]
Author :  Wang Yue
Date   :  2020.11.16   
%}

clear; clc; 
close all

input_vector = [0.1, 0.3, 0.5, 0.7, 0.9];
x=input_vector;
w_ilc = -0.15; % weight of inhibitory lateral connections
[m,n]  = size(input_vector); % n is number of nodes

% build weight matrix
%(the weight of self excitatory connections is 1)->eye(n)
%(the weight of inhibitory lateral connections is -w)->ones(n,n)*w_ilc - eye(n)*w_ilc
w = eye(n) + ones(n,n)*w_ilc - eye(n)*w_ilc; 

t=0; 
flag=0;
while (flag==0)
    t=t+1;
    for i=1:n  
        u(i) = x * w(i,:)';
        v(i) = max (0, u(i)); %activation function
    end
    x=v;
    
    count = sum(v(:)== 0); % the number of zeros in v
    if count==n-1
        winner=find(v~=0);
        flag=1;
        break;
    end

end

v = input_vector(winner);
fprintf('Find the winner in iteration %g \n',t);
fprintf('The winner is : %g ( %f ) \n',winner,v);




