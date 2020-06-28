
addpath /home/alex/Courses/'Deep Learning in Data Science'/'Assignment One'/cifar-10-batches-mat/
filename_train = 'cifar-10-batches-mat/data_batch_1.mat';
filename_val = 'cifar-10-batches-mat/data_batch_2.mat';
filename_test = 'cifar-10-batches-mat/test_batch.mat';
[trainX, trainY, trainLabels] = LoadBatch(filename_train);
[testX, testY, testLabels] = LoadBatch(filename_test);

K = size(trainY, 1);
d = size(trainX, 1);
N = size(trainX, 2);
deviation = 0.01;  
W = deviation.*randn(K,d);
b = deviation.*randn(K,1);

n_batch = 100;
eta = 0.1;
n_epoch = 40;
lambda = 0.0;
randomAccuracy = ComputeAccuracy(trainX, trainLabels, W, b);


% Gradient checking 
P = EvaluateClassifier(trainX(:,1:n_batch), W, b);
%[ngrad_W, ngrad_b] = ComputeGradsNumSlow(trainX(:,1:n_batch), trainY(:,1:n_batch), W, b, lambda, 1e-6);
%[grad_W, grad_b] = ComputeGradients(trainX(:,1:n_batch), trainY(:,1:n_batch), P, W, lambda);

%absb = abs(grad_b - ngrad_b);
%absw = abs(grad_W  - ngrad_W);
%absoluteMaxW = max(absw(:,1))
%absolutMaxb = max(absb)
%averageAbsoluteW = sum(absw(:,1)) / size(absw, 2)
%averageAbsoluteb = sum(absb) / size(absb, 1)
%absDifb = abs(grad_b - ngrad_b) ./ max(1e-6, abs(grad_b) + abs(ngrad_b));
%absDifW = abs(grad_W - ngrad_W) ./ max(1e-6, abs(grad_W) + abs(ngrad_W));  
%relativeMaxW = max(absDifW(:,1))
%relativeMaxb = max(absDifb)
%averageRelativeW = sum(absDifW(:,1)) / size(absDifW, 2)
%averageRelativeb = sum(absDifb) / size(absDifb, 1)


[Wstar, bstar, trainingLoss, validationLoss, testAccuracy] = MiniBatch(trainX, trainY, testX, testY, testLabels, n_batch, eta, n_epoch, W, b, lambda);
accuracy = ComputeAccuracy(testX, testLabels, Wstar, bstar);
%for i=1:length(min(trainLabels):max(trainLabels));
%  im = reshape(Wstar(i, :), 32, 32, 3);
%  s_im{i} = (im-min(im(:)))/(max(im(:))-min(im(:)));
%  s_im{i} = permute(s_im{i}, [2, 1, 3]);
%end
%montage(s_im, 'Size', [1,10]);

function [Wstar, bstar, trainingLoss, validationLoss, testAccuracy] = MiniBatch(X, Y, testX, testY, testLabels, n_batch, eta, n_epoch, W, b, lambda)
    
    for i=1:n_epoch
        trainingLoss(i) = ComputeCost(X, Y, W, b, lambda);
        validationLoss(i) = ComputeCost(testX, testY, W, b, lambda);
        testAccuracy(i) = ComputeAccuracy(testX, testLabels, W, b) * 100;
        i
        for j=1:size(X, 2)/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            
            P = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
            W = W - eta*grad_W;
            b = b - eta*grad_b;
        end
    end
    
    Wstar = W;
    bstar = b;
    
end

function [X, Y, y] = LoadBatch(filename)
    A = load(filename);
    X = im2double(A.data');
    y = A.labels;
    Y = y == 0:max(y);
    Y = Y.';
end
 
function P = EvaluateClassifier(X, W, b)
    s = W * X + b;
    P = softmax(s);
end
 
function J = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    J = -sum(log(sum(Y .* P, 1))) / size(X, 2) + lambda * sumsqr(W);
end
 
function acc = ComputeAccuracy(X, Y, W, b)
    P = EvaluateClassifier(X, W, b);
    d = size(P, 1);
    s = size(P, 2);
    [~, index] = max(P);
    acc = sum((index - 1)' == Y) / size(X, 2);
end
 
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)

    grad_W = zeros(size(Y, 1), size(X, 1));
    grad_b = zeros(size(Y, 1), 1);

    for i=1:size(X,2)
      x = X(:, i);
      y = Y(:, i);
      p = P(:, i);
      g =  -(y-p).';
      grad_b = grad_b + g';
      grad_W = grad_W + g' .* x';
    end
 
    grad_W = grad_W/size(X, 2) + 2 * lambda*W;
    grad_b = grad_b/size(X, 2);
end

 
function [grad_W, grad_b] = ComputeGradsNum(X, Y, W, b, lambda, h)
 
    no = size(W, 1);
    d = size(X, 1);
 
    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);
 
    c = ComputeCost(X, Y, W, b, lambda);
 
    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c) / h;
    end
 
    for i=1:numel(W)  
 
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
 
        grad_W(i) = (c2-c) / h;
    end
end
 
 
function [grad_W, grad_b] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)
 
    no = size(W, 1);
    d = size(X, 1);
 
    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);
 
    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end
 
    for i=1:numel(W)
 
        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
 
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
 
        grad_W(i) = (c2-c1) / (2*h);
    end
end