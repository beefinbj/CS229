k = 16;

img = double(imread('mandrill-small.tiff'));

centroids = zeros(k,3);

rounds = 50;

for mm = 1:k
  centroids(mm,:) = img(randi(size(img,1)),randi(size(img,2)),:);
end

for rr = 1:rounds
  assignsCounts = zeros(k,1);
  assignsRGB = zeros(k,3);
  for ii = 1:size(img,1)
    for jj = 1:size(img,2)
      pp = repmat(squeeze(img(ii,jj,:))',k,1);
      dists = centroids-pp;
      dists = sum(dists.^2,2);
      [M,I] = min(dists);
      assignsCounts(I) = assignsCounts(I)+1;
      assignsRGB(I,:) = assignsRGB(I,:) + squeeze(img(ii,jj,:))';
    end
  end
  
  for ss = 1:k
    centroids(ss,:) = assignsRGB(ss,:)/assignsCounts(ss);
    %clust = assigns==ss;
    %rr = img(:,:,1)(clust);
    %gg = img(:,:,2)(clust);
    %bb = img(:,:,3)(clust);
    %cc = cat(3,rr,gg,bb);
    %centroids(ss,:) = mean(mean(cc));
  end  
end

bigimg = double(imread('mandrill-large.tiff'));

bigass = zeros(size(img,1),size(img,2));
out = zeros(size(bigimg));

for ii = 1:size(bigimg,1)
  for jj = 1:size(bigimg,2)
    pp = repmat(squeeze(bigimg(ii,jj,:))',k,1);
    dists = sqrt(sum((centroids-pp).^2,2));
    [M,I] = min(dists);
    out(ii,jj,:) = centroids(I,:);
  end
end