function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

m = size(X,1);
distance = zeros(m, K);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for iter = 1:K
    D = bsxfun(@minus, X, centroids(iter, :));
    D = sum(D.^2,2);
    distance(:, iter) = D;
end

for i = 1:m
    min = distance(i, 1);
    pos = 1;
    for j = 1:K
        if distance(i, j) < min
            min = distance(i, j);
            pos = j;
        end
    end
    idx(i,1) = pos;
end

% =============================================================

end

