function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);

s = 0;
for i=1:m,
    s = s + ( -1*y(i)*log(h(i)) - (1-y(i))*log(1-h(i)) ) + lambda/(2*m)*sum(theta(2:length(theta)).^2);
end;
J = s / m;

% SECOND PART DOES NOT WORK
gs = zeros(size(theta) - [1 0]);
gs1 = 0
gs1 = gs1 + (h(1) - y(1))*X(1, 2:size(X,2));
for i=2:m,
    gs = gs + (h(i) - y(i))*X(i, 2:size(X,2))' + lambda*theta(2:length(theta));
end;
grad = ([gs1, gs] / m);


% =============================================================

end
