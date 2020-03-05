% CSCC11 - Introduction to Machine Learning, Fall 2018, Assignment 1
% D. Fleet, B. Chan
%
% [y] = EvalPolynomial(w, x)
%
% This function evaluates a polynomial with weights w at inputs x 
%
% w - weights (coefficients) for the polynomial model 
%     (as estimated by FitPolynomialRegression.m).  
%     Weights are ordered such that the j'th element of w is 
%     the linear coefficient of the j'th-order monomial, x^{j}
% x - 1-column vector which contains the inputs
%
% y - 1-column vector which is predicted by the polynomial model given 
%     by the estimated weights w and inputs x
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TO DO: Complete this function so that it evaluates the 
%         trained polynomial model on the given input values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [y] = EvalPolynomial(w, x)
  [row_w, ~] = size(w);
  [row_x, ~] = size(x);
  B = zeros(row_x, row_w);
  
  count = 0;
  for i = 1:row_x
      for j = 1:row_w
         B(i,j) = x(i)^count;
         count = count+1;
      end
      count = 0;
  end

  y = B*w;
  
  