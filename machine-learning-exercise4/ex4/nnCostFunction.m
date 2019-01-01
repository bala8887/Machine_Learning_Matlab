function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

%one=[1,0,0,0,0,0,0,0,0,0];
%two=[0,1,0,0,0,0,0,0,0,0];
%three=[0,0,1,0,0,0,0,0,0,0];
%four=[0,0,0,1,0,0,0,0,0,0];
%five=[0,0,0,0,1,0,0,0,0,0];
%six=[0,0,0,0,0,1,0,0,0,0];
%seven=[0,0,0,0,0,0,1,0,0,0];
%eight=[0,0,0,0,0,0,0,1,0,0];
%nine=[0,0,0,0,0,0,0,0,1,0];
%zero=[0,0,0,0,0,0,0,0,0,1];

%element=size(y,1);

%Y=[];
%for i=1:element
%nmbr=y(i,:);

%switch(nmbr)
%   case 1
%     Y=[Y;one];
%   case 2
%     Y=[Y;two];
%   case 3
%     Y=[Y;three];     
%   case 4
%     Y=[Y;four];
%   case 5
%     Y=[Y;five];
%   case 6
%     Y=[Y;six];
%   case 7
%     Y=[Y;seven];
%   case 8
%     Y=[Y;eight];
%   case 9
%     Y=[Y;nine];
%   otherwise
%     Y=[Y;zero]; 
%end
%end

Y=[];
X=[ones(size(X,1),1) X];
a2=sigmoid(X*transpose(Theta1));
a2=[ones(size(a2,1),1) a2];
a3=sigmoid(a2*transpose(Theta2));

k = size(a3, 2);

tmp_y=[];

for i=1:k
    for j=1:k
        if (i==j)
            tmp_y=[tmp_y 1];
        else
            tmp_y=[tmp_y 0];
        end
    end
end

tmp_y=reshape(tmp_y,k,k);

iterations=size(y,1);

Y=[];

for i=1:iterations
   Y=[Y; tmp_y(y(i,:),:)];
end

J=(-1/m)*(sum(sum((Y.*log(a3))+((1-Y).*log(1-a3)))));
rglrztn=(lambda/(2*m))*(sum(sum((Theta1(:,2:end).^2)))+sum(sum(Theta2(:,2:end).^2)));
%J=((-1/m)*(sum(sum((Y.*log(a3))+((1-Y).*log(1-a3))))))+((lambda/(2*m))*((sum(sum(Theta1)))+(sum(sum(Theta2)))));
J=J+rglrztn;

% You need to return the following variables correctly 
%J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

%X=[ones(size(X,1),1) X];
%a2=sigmoid(X*transpose(Theta1));
%a2=[ones(size(a2,1),1) a2];
%a3=sigmoid(a2*transpose(Theta2));
%25,401 1,401 10 26
delta1=[]; delta2=[];
%for t=1:iterations
%    a1=X(t,:);
%    a2=sigmoid(a1*transpose(Theta1));
%    a2=[ones(size(a2,1),1) a2];
%    a3=sigmoid(a2*transpose(Theta2));
%    %fprintf('%d \n',size(a3));
%    error3=a3-Y(t,:); %1,10
%    %fprintf('%d \n',error3);
    %error2=(error3*Theta2).*sigmoidGradient(Theta1*a1); %1,26 25*1
%    tmp_error2=sigmoidGradient(Theta1*transpose(a1));
%    tmp_error2=[1; tmp_error2];
%    error2=(error3*Theta2).*transpose(tmp_error2);
    %fprintf('%d \n',size(error2)); %1,26
%    if t==1
%        delta1=transpose(a1)*error2(2:end);
%        delta2=transpose(a2)*error3;
%    else
%        delta1=delta1+(transpose(a1)*error2(2:end));
%        delta2=delta2+(transpose(a2)*error3);
%    end
%end

a2=sigmoid(X*transpose(Theta1));
a2=[ones(size(a2,1),1) a2];
a3=sigmoid(a2*transpose(Theta2));
disp("please check this");
size(a2) %5000 26
size(a3) %5000 10
size(Y) %5000 10
error3=a3-Y; %5000 10
tmp_error2=sigmoidGradient(Theta1*transpose(X)); % 25*401 5000 401 =5000*25
%tmp_error2=[transpose(ones(size(tmp_error2,2),1)); tmp_error2];
error2=(error3*Theta2(:,2:end)).*transpose(tmp_error2); %error3*Theta2(:,2:end) = 5000*25 
%size(error3) %5000,10 error3*Theta2=5000*26
%size(Theta2) %10,26 Theta1*X=5000*25
%size(Theta1) %25,401
%size(X) %5000,401
%size(a2) %5000,26
%size(a3) %5000,10
%delta1=transpose(X)*error2(:,2:end);
delta1=transpose(X)*error2; %5000*401 5000*25
delta2=transpose(a2)*error3; %26*10

size(delta1)
size(delta2)

delta1=transpose(delta1);
delta2=transpose(delta2);

theta1_fs=(1/m).*(delta1);
theta2_fs=(1/m).*(delta2);

theta1_sp=(lambda/m).*(Theta1);
theta2_sp=(lambda/m).*(Theta2);

theta1_tmp=theta1_fs+theta1_sp;
theta2_tmp=theta2_fs+theta2_sp;

Theta1_grad=[theta1_fs(:,1) theta1_tmp(:,2:end)];
Theta2_grad=[theta2_fs(:,1) theta2_tmp(:,2:end)];

%Theta1_grad=(1/m).*(delta1); %25,401
%Theta2_grad=(1/m).*(delta2); %10,26

%fprintf("bala");
%size(Theta1_grad)
%size(Theta2_grad)

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end