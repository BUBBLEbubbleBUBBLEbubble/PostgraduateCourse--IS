%{
Name   : A radial-basis function network with Gaussian function (newrb)
Author : Wang Yue
Date   : 2020.10.20
%}
clear;
dataset = [0 0 0;
           0 1 1;
           1 0 1;
           1 1 0];
x = dataset(:,1:2)';    
z = dataset(:,3)';
net = newrb(x, z);
Y = sim(net, x);
view(net);


