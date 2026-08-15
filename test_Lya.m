%% Verify the 7-by-7 counterexample in Section 2
clear;
clc;

n = 7;

% Lyapunov operator: L_A(X) = AX + XA'
A = [5  0 -2  0  5  1  0;
     0  5  5 -5 -2 -1 -1;
     5  0  2  0  2  0 -2;
     6  2  0  2 -4 -5  2;
    -1 -2  6  0  2  1 -1;
     2  7 -5 -1  1  1 -1;
    -4  4  2  6  2 -2 -2];
LA = @(X) A*X + X*A';

% A skew-symmetric matrix K,
% || L_A(K) ||_2 > || L_A(X) ||_2, for all symmetric X
K = [0 -16 -1  2  1  2  1;
    16   0 -3 -2  1  1 -1;
     1   3  0  9 -1  0 -1;
    -2   2 -9  0  1  0  2;
    -1  -1  1 -1  0 -5  1;
    -2  -1  0  0  5  0  0;
    -1   1  1 -2 -1  0  0];

% Basis for symmetric matrix space
% E_11, ..., E_77
ns    = n*(n+1)/2;
B     = zeros(n,n,ns);
for i = 1:n
    B(i,i,i) = 1;
end
% E_12 + E_21, ..., E_67 + E_76.
pairs = nchoosek(1:n,2);
for k = 1:size(pairs,1)
    i = pairs(k,1);
    j = pairs(k,2);
    B(i,j,n+k) = 1;
    B(j,i,n+k) = 1;
end

% R_S is the coordinate matrix of L_A(X) = AX + XA^T on S^7.
% idx(k) = pairs(k,1) + n*(pairs(k,2)-1)
idx = sub2ind([n,n],pairs(:,1),pairs(:,2));
RS = zeros(ns);
for j = 1:ns
    Y = A*B(:,:,j) + B(:,:,j)*A.';
    RS(:,j) = [diag(Y); Y(idx)];
end

% Section 2.1: verify M is positive definite.
W = diag([ones(n,1); 2*ones(size(pairs,1),1)]);
M = 309*W - RS.'*W*RS;

[~, p] = chol(M);
if p == 0
    fprintf('M is positive definite, therefore ||L_A|_{S^7}||_F^2 < 309.\n');
else
    fprintf('M is non positive definite! \n');
end

% Section 2.2: verify the skew-symmetric witness.
normLAK = norm(LA(K), 'F')/norm(K, 'F');

fprintf('||L_A|_{K^7}||_F^2 > || L_A(K) ||_F^2 / || K ||_F^2 = %.15f > 309.\n', normLAK^2);

%% Actually, we can verify L_A's norm on symmetric and skew symmetric matrices numerically
fprintf('Alternatively,\n');

% For symmetric matrix
% ||L_A|_{S^7}||_F = max ||sqrt(W)*R_S*z||_2 / ||sqrt(W)*z||_2
D = diag(sqrt(diag(W)));
fprintf('||L_A|_{S^7}||_F = max ||sqrt(W)*R_S*z||_2 / ||sqrt(W)*z||_2 = %.15f.\n', norm(D*RS/D,2)^2);

% For skew-symmetric matrix, we first construct R_K
% Basis for symmetric matrix space
% E_11, ..., E_77
nk    = n*(n-1)/2;
C     = zeros(n,n,nk);
% E_12 - E_21, ..., E_67 - E_76.
pairs = nchoosek(1:n,2);
for k = 1:size(pairs,1)
    i = pairs(k,1);
    j = pairs(k,2);
    C(i,j,k) = 1;
    C(j,i,k) = -1;
end
idx = sub2ind([n,n],pairs(:,1),pairs(:,2));
RK = zeros(nk);
for j = 1:nk
    Y = A*C(:,:,j) + C(:,:,j)*A.';
    RK(:,j) = [Y(idx)];
end

% And similarly
% ||L_A|_{K^7}||_F = max ||sqrt(W)*R_K*z||_2 / ||sqrt(W)*z||_2
fprintf('||L_A|_{K^7}||_F = max ||R_K*z||_2 / ||z||_2 = %.15f.\n', norm(RK,2)^2);