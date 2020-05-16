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
unreqularized_cost = scaler / m;

theta(1) = 0;
theta_square = transpose(theta) * theta;
regularized_cost = 0.5 * (lambda / m) * theta_square;
J = unreqularized_cost + regularized_cost;

sub = g - y;
x_transpose = transpose(X);
grad = x_transpose * sub;
grad = grad / m;
reg = (lambda / m) * theta;
grad = grad + reg;
% =============================================================

end
