% Before using this method, set num_train to be the number of training
% examples you wish to read.

num_train = 400;

[sparseTrainMatrix, tokenlist, trainCategory] = ...
    readMatrix(sprintf('MATRIX.TRAIN.%d', num_train));

% Make y be a vector of +/-1 labels and X be a {0, 1} matrix.
ytrain = (2 * trainCategory - 1)';
Xtrain = 1.0 * (sparseTrainMatrix > 0);

numTrainDocs = size(Xtrain, 1);
numTokens = size(Xtrain, 2);

% Xtrain is a (numTrainDocs x numTokens) sparse matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents if the j-th token appears in
% email i.

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% For the SVM, we convert these to +1 and -1 to form the numTrainDocs x 1
% vector ytrain.

% This vector should be output by this method

%---------------
% YOUR CODE HERE
Xtrain(Xtrain>0) = 1.0;
%Entry (i,j) is X_i * X_j
X_dots = Xtrain*Xtrain'; %aka gram matrix
%Row (i) is X_i * X_i, or diagonal of gram
X_dot_self = sum(Xtrain.^2,2);
tau = 8;

%X_dot_self' repeated + X_dot_self'
%(i,j) is X_i*X_i+X_j*X_j
%To compute matrix of ||X-X||^2, expand into
%(X_i^2-2X_i*X_j+X_j^2)
KMat = full(exp(-(repmat(X_dot_self,1,numTrainDocs)+repmat(X_dot_self',numTrainDocs,1)-2*X_dots)/(2*tau^2)));

lambda = 1/(64*numTrainDocs);
average_alpha = zeros(numTrainDocs, 1);
alphas = zeros(numTrainDocs, 1);


for tt = 2:numTrainDocs*40
  ii = randi(numTrainDocs);
  step = 1/sqrt(tt);
  calc = ytrain(ii)*KMat(ii,:)*alphas;
  if calc < 1
    %grad = -1/numTrainDocs*ytrain(ii)*KMat(:,ii)+lambda*alphas(ii)*KMat(:,ii);
    grad = -1*ytrain(ii)*KMat(:,ii)+lambda*alphas(ii)*KMat(:,ii)*numTrainDocs;
  end
  alphas = alphas-grad*step;
  average_alpha += alphas;
end

average_alpha = average_alpha/(40*numTrainDocs);
%---------------
