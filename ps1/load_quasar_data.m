% load_quasar_data
%
% Loads the data in the quasar data files
%
% Upon completion of this script, the matrices and data are as follows:
%
% lambdas - A length n = 450 vector of wavelengths {1150, ..., 1599}
% train_qso - A size m-by-n matrix, where m = 200 and n = 450, of noisy
%      observed quasar spectra for training.
% test_qso - A size m-by-n matrix, where m = 200 and n = 450, of noisy observed
%       quasar spectra for testing.

load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2:end, :);
load quasar_test.csv;
test_qso = quasar_test(2:end, :);

X = quasar_train(2,:);

inputs = [ones(size(lambdas,1),1) lambdas];
outputs = X';

%Part B(i)
theta_1 = (inputs' * inputs)\inputs' * outputs;

%figure;
%
%hold on;
%
%%Scatter
%scatter(lambdas,X,'+');
%
%%Regression line
%plot(lambdas,inputs*theta_1);
%
%xlim([min(lambdas) max(lambdas)]);
%
%xlabel('Wavelength');
%ylabel('Flux');

%Part B(ii)
theta_2 = zeros(2,size(lambdas,1));
output_2 = zeros(size(lambdas,1),1);

bands = [1 5 10 100 1000];
all_out_weight = zeros(size(lambdas,1),size(bands,2));

for k = 1:size(bands,2)
  for i = 1:size(lambdas,1)
    w = exp(-1*(lambdas(i)-lambdas).^2/(2*bands(k)^2));
    weights(:,i) = w';
    D = diag(w');
    th = (inputs' * D * inputs)\inputs' * D * outputs;
    theta_2(:,i) = th;
  end
  out_weight = diag(inputs*theta_2);
  all_out_weight(:,k) = out_weight;
end

%hold off;
%
%figure;
%
%hold all;
%
%scatter(lambdas,X,'+');
%
%%Regression line
%for j = 1:size(bands,2)
%  vals = all_out_weight(:,j)';
%  plot(lambdas,vals);
%end
%
%%plot(lambdas,output_2);
%
%xlim([min(lambdas) max(lambdas)]);
%ylim([-2 10]);
%
%xlabel('Wavelength');
%ylabel('Flux');
%legend('show');

%Part C(i,ii)
bands_C = [5];

smoothed = zeros(size(train_qso,1),size(train_qso,2));
smoothed(1,:) = lambdas;

for f = 1:size(train_qso,1)
  outputs_f = train_qso(f,:)';
  
  theta_f = zeros(2,size(lambdas,1));
  
  for k_f = 1:size(bands_C,2)
    for i_f = 1:size(lambdas,1)
      w_f = exp(-1*(lambdas(i_f)-lambdas).^2/(2*bands_C(k_f)^2));
      weights(:,i_f) = w_f';
      D_f = diag(w_f');
      th_f = (inputs' * D_f * inputs)\inputs' * D_f * outputs_f;
      theta_f(:,i_f) = th_f;
    end
  out_weight_f = diag(inputs*theta_f);
  smoothed(f,:) = out_weight_f;
  end
end

function d = func_dist(f1,f2)
  if size(f1) ~= size(f2)
    error('Inputs must be same size')
  end
  d = sum((f2-f1).^2);
end

function ke = ker(t)
  ke = max(1-t,0);
end

function [h, nei] = neighbk(func, k, all_func)
  dists = zeros(size(all_func,1),1);
  for j = 1:size(all_func,1)
    d = func_dist(all_func(j,:),func);
    dists(j,:) = d;
  end
  [sortedDist, sortedIndex] = sort(dists);
  nei = sortedIndex(1:k);
  h = sortedDist(end);
end

function fleft_hat = estimator(fright, k, all_fright, all_fleft)
  [h, nei] = neighbk(fright,k,all_fright);
  num_sum = 0;
  denom_sum = 0;
  for n = 1:size(nei,1)
    coeff = ker(func_dist(fright,all_fright(nei(n),:))/h);
    num_sum += coeff*all_fleft(nei(n),:);
    denom_sum += coeff;
  end
  fleft_hat = num_sum/denom_sum;
end

left_edge_index = 1200-min(lambdas);
right_edge_index = size(lambdas,1)-(max(lambdas)-1300);

train_left = smoothed(:,1:left_edge_index);
train_right = smoothed(:,right_edge_index:end);

k = 3;

total_dist = 0;

for rr = 1:size(smoothed,1)
  fright_matrix = train_right;
  fleft_matrix = train_left;
  estimated = estimator(train_right(rr,:),k,fright_matrix,fleft_matrix);
  est_dist = func_dist(estimated, train_left(rr,:));
  total_dist += est_dist;
end

avg_dist = total_dist/size(smoothed,1);

%Part C(iii)
%Smooth the test data
smoothed_test = zeros(size(test_qso,1),size(test_qso,2));
smoothed_test(1,:) = lambdas;

for ftest = 1:size(test_qso,1)
  outputs_ftest = test_qso(ftest,:)';
  
  theta_ftest = zeros(2,size(lambdas,1));
  
  for k_ftest = 1:size(bands_C,2)
    for i_ftest = 1:size(lambdas,1)
      w_ftest = exp(-1*(lambdas(i_ftest)-lambdas).^2/(2*bands_C(k_ftest)^2));
      weights_test(:,i_f) = w_f';
      D_ftest = diag(w_ftest');
      th_ftest = (inputs' * D_ftest * inputs)\inputs' * D_ftest * outputs_ftest;
      theta_ftest(:,i_ftest) = th_ftest;
    end
  out_weight_ftest = diag(inputs*theta_ftest);
  smoothed_test(ftest,:) = out_weight_ftest;
  end
end

fright_test = smoothed_test(:,right_edge_index:end);
fleft_test = smoothed_test(:,1:left_edge_index);

est_test = zeros(size(test_qso),left_edge_index);

total_dist_test = 0;

for ss = 1:size(test_qso,1)
  estimated_test = estimator(fright_test(ss,:),k,train_right,train_left);
  est_test(ss,:) = estimated_test;
  est_dist_test = func_dist(estimated_test, fleft_test(ss,:));
  total_dist_test += est_dist_test;
end

avg_dist_test = total_dist_test/size(test_qso,1);

figure;

hold all;
plot(lambdas,smoothed_test(1,:));
plot(lambdas(1:left_edge_index),est_test(1,:));

xlabel('Wavelength');
ylabel('Flux');
legend('show');