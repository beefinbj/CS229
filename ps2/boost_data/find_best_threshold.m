function [ind, thresh] = find_best_threshold(X, y, p_dist)

% FIND_BEST_THRESHOLD Finds the best threshold for the given data
%
% [ind, thresh] = find_best_threshold(X, y, p_dist) returns a threshold
%   thresh and index ind that gives the best thresholded classifier for the
%   weights p_dist on the training data. That is, the returned index ind
%   and threshold thresh minimize
%
%    sum_{i = 1}^m p(i) * 1{sign(X(i, ind) - thresh) ~= y(i)}
%
%   OR
%
%    sum_{i = 1}^m p(i) * 1{sign(thresh - X(i, ind)) ~= y(i)}.
%
%   We must check both signed directions, as it is possible that the best
%   decision stump (coordinate threshold classifier) is of the form
%   sign(threshold - x_j) rather than sign(x_j - threshold).
%
%   The data matrix X is of size m-by-n, where m is the training set size
%   and n is the dimension.
%
%   The solution version uses efficient sorting and data structures to perform
%   this calculation in time O(n m log(m)), where the size of the data matrix
%   X is m-by-n.

[mm, nn] = size(X);
ind = 1;
thresh = 0;

% ------- Your code here -------- %
%
% A few hints: you should loop over each of the nn features in the X
% matrix. It may be useful (for efficiency reasons, though this is not
% necessary) to sort each coordinate of X as you iterate through the
% features.
best_error = inf;

  for dd = 1:nn
    [XSorted, indsSorted] = sort(X(:,dd),1,'descend');
    YSorted = y(indsSorted);
    pSorted = p_dist(indsSorted);
    thresholds = (XSorted+circshift(XSorted,1))/2; %thresholds are averages of consecutive values
    thresholds(1) = XSorted(1)+1; %First one is miscalculated. Should be a bit bigger than X(1)
    increments = pSorted.*YSorted; %Pos are from misclassing +1, negs are from misclassing -1
    increments = circshift(increments,1);
    increments(1) = 0; %Increment after you shift the index
    errors = ones(mm,1) * (pSorted' * (YSorted==1)); %Error if maximal threshold, all classified -1
    errors = errors - cumsum(increments);
    %Get rest of errors by deducting terms.
    %If increment is neg (misclass -1), gets added. If pos, gets deducted
    %This computes errors using phi_pos
    [low, lowInd] = min(errors); %best error from using threshold to label pos
    [hi, hiInd] = max(errors);
    %(All classified -1 error)-(Reward for correctly labeling +1) + (Error from -1 now labeled +1)
    %(1-(All classified -1 error)) = (All classified +1 error)
    %Want to +(Error from +1 now labeled -1)-(Reward for correctly labeling -1)
    hi = 1-hi;
    if (hi < low)
      err_dd = hi;
      err_ind = hiInd;
    else
      err_dd = low;
      err_ind = lowInd;
    end
    err_thresh = thresholds(err_ind);
    if (err_dd < best_error)
      best_error = err_dd;
      ind = dd;
      thresh = err_thresh;
    end
  end