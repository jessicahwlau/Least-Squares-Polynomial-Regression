% CSCC11 - Introduction to Machine Learning, Fall 2018, Assignment 1
% D. Fleet, B. Chan
%
% [d, trainError, testError] = GetBestPolynomial(xTrain, yTrain, xTest, yTest, h) 
%
% This function takes as input a training dataset (xTrain, yTrain), a test 
% dataset (xTest, yTest), and the maximum degree of polynomials to consider.  
% It then computes the LS optimal weights for all polynomial regression models 
% from degree 1, 2, ... h.  It then computes residual errors for training 
% and testing data.  It then chooses the polynomial order of the best 
% model.
%
% xTrain - 1-column vector of training inputs
% yTrain - 1-column vector of training outputs for inputs xTrain
% xTest - 1-column vector of testing inputs
% yTest - 1-column vector of testing outputs for inputs xTest
% h - the maximum polynomial degree to consider 
% (Note: 1 <= h <= 10)
%
% d - polynomial degree that produces the best model
% eTrain - 1-column vector of length h containing the total squared error
%          on training data from fitted polynomials of degree 1 to h 
%          (in ascending order)
% eTest - 1-column vector of length h containining the total squared error
%         on test data from fitted polynomials of degree 1 to h 
%         (in ascending order)
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TO DO: Complete this function so that it computes the 
%         residual errors of the estimated weights for
%         multiple polynomial models, as well as the degree 
%         of the best polynomial model.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [d, trainError, testError] = GetBestPolynomial(xTrain, yTrain, xTest, yTest, h) 
  testError = zeros(h, 1);
  trainError = zeros(h, 1);

  function [Error, w] = GetError(x, y, k, w)
    if w == 0
        [w] = FitPolynomialRegression(k, x, y);
    end
    [Bw] = EvalPolynomial(w, x);
    Error = norm(y- Bw)^2;
  end
  
  for k = 1:h
    [Error, w] = GetError(xTrain, yTrain, k, 0);
    trainError(k) = Error;
    [Error, ~] = GetError(xTest, yTest, k, w);
    testError(k) = Error;
  end

  % Plot the residual errors
  plot(trainError, 'r');
  hold on
  plot(testError, 'b');
  title('The Residual errors of the models on training (red) and testing errors (blue)');
  
  [~, min_test] = min(testError);
  [~, min_train] = min(trainError);
  if min_test < min_train
      d = min_test;
  elseif min_train < min_test
      d = min_train;
  else
      d = min_test;    
  end
end
