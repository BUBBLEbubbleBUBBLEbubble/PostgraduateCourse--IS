%{
Name   :  adaline - OR
Author :  Wang Yue
Date   :  2020.10.18   
%}
clear;

data = [0,0,1,-1;  %dataset(x1, x2, -threshold, y)
        0,1,1,1;
        1,0,1,1;
        1,1,1,1]; 
[d,n] = size(data); %d=datasize  n=#of x
x = data(:,1:3);    %data
z = data(:,4);      %ground truth
% w = rand(1,3);    %weight 
w = [0 0 1];        %weight
lr = 0.05;          %learning rate
E = 1;              %Error
E_threshold = 0.00001;
t = 0; 
iteration = 100;    %iteration
alpha = 1;   
while (E > E_threshold) && (t < iteration)
% for t=1:100
    t = t + 1; 
    %--------------
    for i=1:d 
        for j=1:d 
            y(j) = x(j,:)*w'*alpha; %Identity activition function
        end
        for k=1:d
            w = w + lr*(z(k)-y(k))*x(k,:);
        end
    end
    mse(t) = 1/d * ((y-z')*(y-z')'); %mean squared error
    sse(t) = ((y-z')*(y-z')'); %sum squared error
    E = mse(t); 
end

figure(1);%----------------------------------
%-----------line
X = -3:3; %x values for graph
Y = -(w(1,1)/w(1,2))*X-(w(1,3)/w(1,2)); %equation for graph
plot(X,Y);  axis([-1 2 -1 2]);
hold on;
%-------------spot
for i = 1:d
    if ( data(i,4)==1 )
        scatter(data(i,1),data(i,2),'b+');
        hold on;
    else
        scatter(data(i,1),data(i,2),'ro');
        hold on;
    end
end
title('The results of classification');
xlabel('x1') 
ylabel('x2') 
hold off;

figure(2);%----------------------------------------
a = 1:iteration;
b = mse;
c = sse;
subplot(2,1,1);plot(a,b, 'b*-'); 
title('mean squared error');
xlabel('t') ;
ylabel('mse') ;
hold on;
subplot(2,1,2);plot(a,c, 'b*-');
title('sum squared error');
xlabel('t') ;
ylabel('sse') ;
hold off;


