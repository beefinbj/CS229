X = load('logistic_x.txt');
Y = load('logistic_y.txt');

function [theta, ll] = log_regression(X,Y,num_iters)

mm = size(X,1);
nn = size(X,2);

theta = zeros(nn,1);

ll = zeros(num_iters,1);

for ii = 1:num_iters
  power = Y .* (X * theta);
  ll(ii) = (1/mm) * sum(log(1+exp(-power))); %Compute J(theta)
  probs = 1 ./ (1+exp(power)); %Sigmoid-based probability
  grad = -(1/mm) * (X' * (probs .* Y)); %Compute gradient of J?
  H = (1/mm) * (X' * diag(probs .* (1 - probs)) * X); %Generate Hessian
  theta = theta - H \ grad; %Newton's method for higher dimensions
end;

end;

X = [ones(size(X, 1), 1) X]; %Account for intercept

[theta, ll] = log_regression(X, Y, 20);

m = size(X, 1);

figure;

hold on;

%Scatter
plot(X(Y < 0, 2), X(Y < 0, 3), 'rx', 'linewidth', 2);
plot(X(Y > 0, 2), X(Y > 0, 3), 'go', 'linewidth', 2);

%Decision Line
x1 = min(X(:,2)):.01:max(X(:,2));
x2 = -(theta(1) / theta(3)) - (theta(2) / theta(3)) * x1;

plot(x1,x2, 'linewidth', 2);

xlabel('x1');
ylabel('x2');