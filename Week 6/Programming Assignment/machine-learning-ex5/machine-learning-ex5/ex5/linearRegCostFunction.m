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

# ---- C O S T ------
regTermConstant = lambda / (2 * m) ;
regTermInner = sum( theta(2:end).^ 2 );
regTerm = regTermConstant * regTermInner;

JConstant = 1 / (2 * m) ;
Jhyp = X*theta;
Jfinal =(Jhyp - y).^2 ;

J = JConstant .* sum(Jfinal) + regTerm ;

# --- G R A D I E N T ------

regGrad1 = (lambda / m) * theta;
regGrad1(1) = 0;
hyp = X*theta;
error = hyp - y;
grad1 = (1/m) * X'*(error);
grad = grad1 + regGrad1 ;


% =========================================================================

grad = grad(:);

end


