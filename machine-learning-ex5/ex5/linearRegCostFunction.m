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

h = X* theta;
error_diff = h - y;
sq_err = error_diff .^ 2;
J  = sum(sq_err)/(2*m);





% Calculate gradient for linear regression with regularization
h = X * theta;
error_diff = h - y;
grad = (X' * error_diff)/m;

theta(1) = 0;
reg = sum(theta .^ 2)* lambda/ (2*m);
J = J + reg;

grad_reg = theta * lambda / (m);
grad = grad + grad_reg;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
