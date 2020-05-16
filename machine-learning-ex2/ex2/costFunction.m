function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

A = X * theta;
g = sigmoid(A);
lg = log(g);
lg = transpose(lg);
left = lg * (-y);

g_prime = 1 - g;
lg = log(g_prime);
lg = transpose(lg);
right = lg * (1 - y);
scaler = left - right;
J = scaler / m;

sub = g - y;
x_transpose = transpose(X);
grad = x_transpose * sub;
grad = grad / m;
% =============================================================

end
