%{
Name   :  perceptron - OR
Author :  Wang Yue
Date   :  2020.10.18   
%}

clear;

data = [0,0,1,-1;  %dataset(x1, x2, -threshold, y)
        0,1,1,1;
        1,0,1,1;
        1,1,1,1]; 
[d,n] = size(data); %d=datasize  n=#of x
% w = rand(1,3);    %weight 
w = [0 0 0];        %weight 
lr = 0.005;         %learning rate
E = 1;              %Error
E_threshold = 0.0001;
t=0;                %Cumulative number of iterations
iteration = 100;    %Maximum iterations
mse = zeros(1,iteration);
sse = zeros(1,iteration);
while (E > E_threshold) && (t < iteration)
    t= t+1;
    %--------------
    for i=1:d
        u = data(i,1:3)*w';
        if u > 0    %transfer function
            y(i) = +1;
        else
            y(i) = -1;
        end
        w = w + lr*(data(i,4)-y(i))*data(i,1:3);
    end
    
    mse(t) = 1/d * ((y-data(:,4)')*(y-data(:,4)')'); %mean squared error
    sse(t) = ((y-data(:,4)')*(y-data(:,4)')'); %sum squared error
    E = mse(t);
    %----------------
 
end

figure(1);%----------------------------------------------------------------
%-----------line
X = -3:3; %x values for graph
Y = -(w(1,1)/w(1,2))*X-(w(1,3)/w(1,2)); %equation for graph
plot(X,Y); hold on;
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

figure(2);%----------------------------------------------------------------
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




