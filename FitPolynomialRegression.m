% CSCC11 - Introduction to Machine Learning, Fall 2018, Assignment 1
% D. Fleet, B. Chan
%
% [w] = FitPolynomialRegression(K, x, y)
%
% This function finds optimal solves for the weights (and bias) for polynomial 
% regression given training data (x,y)
%
% The polynomial model is
%       y_{i} = sum_{k = 0}^{K} w_{k} * x_{i}^{k}
% where y_{i} is an observed value at x_{i}
%
% K - the degree of the polynomial, ranging from 1 to 10 
% x - 1-column vector that contains training inputs
% y - 1-column vector which contains training outputs for inputs x
%
% w - vector of length K+1  with estimated monomial coefficients 
%     for monomials x^0, x^1, ... , x^K
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TO DO: Complete this function to set up the regression
%        problem and solve for the weights w that correspond
%        to the least-squares estimate that fits the observed
%        data.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [w] = FitPolynomialRegression(K, x, y)

  [row, ~] = size(x);
  B = zeros(row, K+1);
  count = 0;
  for i = 1:row
      for j = 1:K+1
         B(i,j) = x(i)^count;
         count = count+1;
      end
      count = 0;
  end
 
  w = B \ y;


  