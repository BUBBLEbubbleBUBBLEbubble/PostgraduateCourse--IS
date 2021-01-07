%{
Name   :  perceptron - XOR
Author :  Wang Yue
Date   :  2020.10.19  
%}

clear;
x =[1 1;
    1 0;
    0 1;
    0 0];
t = [0 1 1 0]';
lr = 0.5; %learning rate
feature_number = size(x,2);
w_hidden_node_number = 2;
w_output_node_number= 1; % regression problem
w_hidden = rand(feature_number, w_hidden_node_number);
w_hidden_th = rand(1, w_hidden_node_number);
% w_hidden = [0.5 0.4; 0.3 0.1];
% w_hidden_th = [0.1 0.2];
w_output = rand(w_hidden_node_number, w_output_node_number);
w_output_th = rand(1, w_output_node_number);
% w_output = [0.1 ; -0.2];
y = zeros(4,1000);

for k = 1:1000
    for i = 1:size(t, 1)
        y_hidden = tanh(x(i,:) * w_hidden + w_hidden_th);
        y_output = logsig(y_hidden * w_output + w_output_th);
        y(i,k) = y_output;
        
        e = t(i) - y_output;
        
        d_output = e.* y_output .* (1 - y_output);
        d_hidden = (1-y_hidden.^2) .* (d_output * w_output');
        
        w_output = w_output + lr * y_hidden' * d_output;
        w_output_th = w_output_th + lr * d_output;
        w_hidden = w_hidden + lr * x (i, :)' * d_hidden;
        w_hidden_th = w_hidden_th +  lr * d_hidden;
    end
    
    [X1, X2] = meshgrid(-0.5:1.5);
    Y1 = w_hidden_th(1) + X1*w_hidden(1, 1) + X2 * w_hidden(2, 1);
    Y2 = w_hidden_th(2) + X1*w_hidden(1, 2) + X2 * w_hidden(2, 2);
    %Dynamic figure-start-------------
    for i = 1:4
        if ( t(i)==1 )
            scatter(x(i,1),x(i,2),'b+');
            hold on
        else
            scatter(x(i,1),x(i,2),'ro');
            hold on
        end
    end
    hold on;
    contour(X1, X2, Y1, [0,0], 'k');
    contour(X1, X2, Y2, [0,0], 'k');
    title(['Iteration: ' num2str(k)]);
    hold off
    drawnow;
    %Dynamic figure-end-------------
end

% %Static figure--------------
% for i = 1:4
%     if ( t(i)==1 )
%         scatter(x(i,1),x(i,2),'b+');
%         hold on
%     else
%         scatter(x(i,1),x(i,2),'ro');
%         hold on
%     end
% end

X1 = -2:2;
X2 = -2:2;
Y1 = -(w_hidden(1, 1)/w_hidden(2, 1)) * X1 - (w_hidden_th(1,1)/w_hidden(2, 1));
hold on;
Y2 = -(w_hidden(2, 1)/w_hidden(2, 1)) * X2 - (w_hidden_th(1,2)/w_hidden(2, 2));
hold on;
plot(X1,Y1);
plot(X2,Y2);
axis([-1 2 -1 2]);
title('The results of classification');
xlabel('x1') 
ylabel('x2') 
hold off;

%plot
figure(2);
a = 1:1000;  
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
