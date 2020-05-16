function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

A = X * theta;
error_vector = A - y;
A = error_vector.^2;
S = sum(A);
unregularized_cost = S/(2 * m);

theta(1) = 0;
square_sum = sum(theta.^2);
regularized_cost = 0.5 * (lambda/m) * square_sum;
J = unregularized_cost + regularized_cost;

transpose_x = transpose(X);
theta_change = transpose_x * error_vector;
unregularized_grad = theta_change / m;

regularized_grad = (lambda/m) * theta;
grad = unregularized_grad + regularized_grad;

% =========================================================================

grad = grad(:);

end
