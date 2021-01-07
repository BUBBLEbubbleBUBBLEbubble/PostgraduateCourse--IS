%{
Name   : A radial-basis function network with Gaussian function (method2)
Author : Wang Yue
Date   : 2020.10.21
%}
clear
dataset = [0 0 0;
           0 1 1;
           1 0 1;
           1 1 0];
x = dataset(:,1:2);    
z = dataset(:,3);
hideNum=8;             %Number of hidden layer neurons
rho=rand(4,hideNum);   %The value of the radial basis function
y=rand(4,1);           %output
w=rand(1,hideNum);     %The weight of the ith neuron in the hidden layer and the output neuron
sf=rand(1,hideNum);    %The scaling factor of the distance between the sample and the center of the ith neuron
c=rand(hideNum,2);     %The center of the ith neuron in the hidden layer
t=0;                   %Cumulative number of iterations
sn=0;                  %Same cumulative number of error values
E_pre=0;               %The cumulative error of the previous iteration
lr=0.05;               %learning rate

while(1)
    t=t+1;
    E=0;
    %Calculate the value of the radial basis function for each sample------
    for i=1:4
        for j=1:hideNum
            p(i,j)=exp(-sf(j)*(x(i,:)-c(j,:))*(x(i,:)-c(j,:))');
        end
        y(i,t)=w*p(i,:)';
    end
    %Calculate cumulative error--------------------------------------------
    for i=1:4
        E=E+((y(i,t)-z(i))^2); %Calculate the mean square error
    end
    E=E/2; %accumulated error
    
    %delta w¡¢sf
    w_d=zeros(1,hideNum);
    sf_d=zeros(1,hideNum);
    for i=1:4
        for j=1:hideNum
            w_d(j)=w_d(j)+(y(i,t)-z(i))*p(i,j);
            sf_d(j)= sf_d(j)-(y(i,t)-z(i))*w(j)*(x(i,:)-c(j,:))*(x(i,:)-c(j,:))'*p(i,j);
        end
    end
    %update w¡¢sf----------------------------------------------------------
    w=w-lr*w_d/4;
    sf=sf-lr*sf_d/4;
    %Conditions for iteration termination----------------------------------
    if(abs(E_pre-E)<1e-10)
        sn=sn+1;
        if(sn==100)
            break;
        end
    else
        E_pre=E;
        sn=0;
    end
end

%plot
a = 1:t;  
b = y(1,:); 
c = y(2,:); 
d = y(3,:); 
e = y(4,:); 
plot(a, b, 'y*'); hold on;
plot(a, c, 'go'); hold on;
plot(a, d, 'r.'); hold on;
plot(a, e, 'b.'); hold on;
title('The fitting of four sample inputs');
legend('x1=0,x2=0','x1=0,x2=1','x1=1,x2=0','x1=1,x2=1');
xlabel('t') 
ylabel('target values') 
hold off;