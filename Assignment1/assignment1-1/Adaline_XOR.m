%{
Name   :  adaline - XOR
Author :  Wang Yue
Date   :  2020.10.22   
%}
clear
% Network initialization--------
dataset = [0 1 0 1;
           1 0 0 1;
           1 1 0 0]; 
X=dataset(1:2,:);
Y=dataset(3,:);
[l,c]=size(X); 
disp('The initial weights are randomly generated as follows£º');
tt=0;
% presion=input('Please enter the training error accuracy£º');  %0.00001
% speed1=input('Please enter your learning rate£º');  %0.1
presion=0.00001;
lr1=0.5;  
lr2=lr1;
w1=rands(2,2);     %Implicit layer weights initialization
w2=rands(1,2);     %Output layer weights initialization
b1=rands(2,1);
b2=rands(1);
iteration=10000;   %Maximum iteration number
t=1;               %Initialize the number of iterations
e=1;               %Initialization error
%------------------------------------------------------------------       
while(e>presion && t<iteration)   %Less than the error accuracy and the maximum number of iterations
    e=0; 
    for i=c*(t-1)+1:t*c         
        % Feedforward-----------------------------
        % The first layer
        x0=X(:,i-c*t+c);
        n1=w1*x0+b1;  
        y1=logsig(n1);   
        % The second layer
        n2=w2*y1+b2; 
        y2(i)=logsig(n2);  
        % Feedback algorithm----------------------
        e=e+(dataset(3,i-c*t+c)-y2(i))^2; 
        deta2=-2*dlogsig(n2,y2(i))*(dataset(3,i-c*t+c)-y2(i)); %Calculate deta2 for the output layer
        temp=zeros(size(y1,1)); 
        for j=1:size(y1,1)
            temp(j,j)=(1-y1(j))*y1(j); 
        end 
        deta1=temp*w2'*deta2; %Calculate the deta1 for the input layer
        %A weight iteration 
        w1=w1-lr1*deta1*x0';
        w2=w2-lr2*deta2*y1'; 
        b1=b1-lr1*deta1; 
        b2=b2-lr2*deta2;  
    end
    E(t)=0.5*e;
    t=t+1;
end
% Results output
 for n=1:1:t-1 
     p0(n)=y2(c*n-3); 
     p1(n)=y2(c*n-2); 
     p2(n)=y2(c*n-1); 
     p3(n)=y2(c*n); 
 end 
 if t<35000 
     tt=tt+1;
 end
disp('The ideal output is£º1 1 0 0')
fprintf('The actual output is£º%f,%f,%f,%f\n',p0(n),p1(n),p2(n),p3(n))
fprintf('The final iteration error is£º%f\n',e)
fprintf('Number of iterations is£º%d\n',t)

%plot----------------------------------------
figure(1);
plotpv(X,Y);      hold on %point
plotpc(w1,b1);    hold on %line
title('The result of classification');
xlabel('x1');  	
ylabel('x2'); 

figure(2);
plot(p0,'y*'); hold on;
plot(p1,'r.'); hold on;
plot(p2,'g*'); hold on;
plot(p3,'b.'); hold on;
legend('x1=0,x2=1','x1=1,x2=0','x1=0,x2=0','x1=1,x2=1');
title('The fitting of four sample inputs');
xlabel('t');    
ylabel('target values') 
hold off;