
[spmatrix, tokenlist, category] = readMatrix('MATRIX.TEST');

testMatrix = full(spmatrix);
numTestDocs = size(testMatrix, 1);
numTokens = size(testMatrix, 2);

% Assume nb_train.m has just been executed, and all the parameters computed/needed
% by your classifier are in memory through that execution. You can also assume 
% that the columns in the test set are arranged in exactly the same way as for the
% training set (i.e., the j-th column represents the same token in the test data 
% matrix as in the original training data matrix).

% Write code below to classify each document in the test set (ie, each row
% in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

% Construct the (numTestDocs x 1) vector 'output' such that the i-th entry 
% of this vector is the predicted class (1/0) for the i-th  email (i-th row 
% in testMatrix) in the test set.
output = zeros(numTestDocs, 1);

%---------------
% YOUR CODE HERE

function p = posterior(wordString,prior,feat_prob_Matrix)
  p = prior + sum(wordString.*feat_prob_Matrix);
end

function val = judge_spam(wordString,p_spam,p_email,feat_prob_spam,feat_prob_email)
  spam = p_spam + wordString*feat_prob_spam;
  email = p_email + wordString*feat_prob_email;
  if spam > email
    val = 1;
  else
    val = 0;
  end
end

function out = solve(wordMatrix,p_spam,p_email,feat_prob_spam,feat_prob_email)
  out = zeros(size(wordMatrix,1),1);
  for ww = 1:size(wordMatrix,1)
    wordString = wordMatrix(ww,:);
    out(ww) = judge_spam(wordString,p_spam,p_email,feat_prob_spam,feat_prob_email);
  end
end

%---------------
output = solve(testMatrix,log_pspam,log_pemail,log_phi_spam,log_phi_email);

% Compute the error on the test set
y = full(category);
y = y(:);
error = sum(y ~= output) / numTestDocs;

%Print out the classification error on the test set
fprintf(1, 'Test error: %1.4f\n', error);

trains = [50 100 200 400 800 1400];
%errors = zeros(size(trains,1),1);
errors(2) = error;

%diff = log_phi_spam-log_phi_email;
%[sortedM,sortedI] = sort(diff,'descend');
%hits = sortedI(1:5);