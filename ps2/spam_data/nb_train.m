pkg load statistics

[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN.100');

trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.


% YOUR CODE HERE

%Build the denominator
V = size(trainMatrix,2);
emailMatrix = trainMatrix(find(trainCategory==0),:);
spamMatrix = trainMatrix(find(trainCategory==1),:);

emailWords = sum(sum(emailMatrix));
spamWords = sum(sum(spamMatrix));

log_pspam = log(size(spamMatrix,1)/numTrainDocs);
log_pemail = log(size(emailMatrix,1)/numTrainDocs);

log_phi_email = zeros(V,1);
log_phi_spam = zeros(V,1);

for k = 1:V
  log_phi_spam(k) = log((sum(spamMatrix(:,k))+1)/(spamWords+V));
  log_phi_email(k) = log((sum(emailMatrix(:,k))+1)/(emailWords+V));
end
