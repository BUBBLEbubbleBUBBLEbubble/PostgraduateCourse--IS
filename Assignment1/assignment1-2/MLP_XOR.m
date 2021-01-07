%{
Name   : A multilayer Perceptron with unipolar sigmoid activation functions
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

net = newff([0 1; 0 1],[2 1], {'logsig' 'logsig'}, 'trainlm', 'learngdm', 'mse');
net.trainParam.epochs = 100;
net.trainParam.goal = 0;
net.trainParam.lr=0.01;
% net.trainParam.mc=0.9;
% net.trainParam.show=25;
net = train(net,x,z);
Y = sim(net, x);
view(net);
