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

% cost function J:
z = X*theta;
hX = sigmoid(z);
temp = y.*log(hX)+(1-y).*log(1-hX);
regtheta = theta.*theta;
sumtheta = sum(regtheta(:))-regtheta(1);
reg = (lambda/(2*m))*sumtheta;
J = -(1/m)*(sum(temp(:)))+reg;

% gradient of J:
reggrad = (lambda/m)*theta;
reggrad(1) = 0;
grad=(1/m)*(X'*(hX-y))+reggrad;



% =============================================================

end
