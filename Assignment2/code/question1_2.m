%{
Name   :  A k-Winners-Take-All Model (kWTA) 
          where k = 1 with a Single State Variable (y) 
          and the Heaviside Step Activation Function 
Author :  Wang Yue
Date   :  2020.11.16   
%}

clear; clc; 
close all
input_vector = [0.1, 0.3, 0.5, 0.7, 0.9];

% u - input
% x - output
% y(t) - state variable
% beta - step size (the step size ¦Â is no bigger than the minimum difference of inputs)

u = input_vector;
[m,n]  = size(u); % n is number of nodes
beta = 0.05; %beta = 0.01/0.02/0.05/0.08
k = 1;
t = 1;
y(t) = 0;

%initialize x ------------------
for i= 1:n   
   if u(i)-y(t) >= 0  %activation function
       x(i) = 1;
   else
       x(i) = 0;
   end
end
%------------------------
flag=0;
while (flag==0)
    
    y(t+1) = y(t) + beta * (sum(x) - k);
    
    for i= 1:n
       if u(i)-y(t+1) >= 0  %activation function
           x(i) = 1;
       else
           x(i) = 0;
       end
    end

    count = sum(x(:)== 0); % the number of zeros in x
    if count==n-k   
        winner=find(x~=0);
        flag=1;
        break;
    end
    
  t = t+1; 
  
end

w = input_vector(winner);
fprintf('Find the winner in iteration %g \n',t);
fprintf('The winner is : %g ( %f ) \n',winner,w);





