
""" # Load Datasets # """;
addpath /home/alex/Courses/'Deep Learning in Data Science'/'Assignment One'/cifar-10-batches-mat/
filename_train1 = 'cifar-10-batches-mat/data_batch_1.mat';
filename_train2 = 'cifar-10-batches-mat/data_batch_2.mat';
filename_train3 = 'cifar-10-batches-mat/data_batch_3.mat';
filename_train4 = 'cifar-10-batches-mat/data_batch_4.mat';
filename_train5 = 'cifar-10-batches-mat/data_batch_5.mat';
filename_test = 'cifar-10-batches-mat/test_batch.mat';
[data1.X, data1.Y, data1.labels] = LoadBatch(filename_train1);
[data2.X, data2.Y, data2.labels] = LoadBatch(filename_train2);
[data3.X, data3.Y, data3.labels] = LoadBatch(filename_train3);
[data4.X, data4.Y, data4.labels] = LoadBatch(filename_train4);
[data5.X, data5.Y, data5.labels] = LoadBatch(filename_train5);
[testData.X, testData.Y, testData.labels] = LoadBatch(filename_test);

""" # Initialization # """;
%Dimensions
K = size(data1.Y, 1);
d = size(data1.X, 1);
N = size(data1.X, 2);
m = 50; 
n_batch = 100;
n_epoch = 50;
lambda = 0.01;
eta_min = 1e-5;
eta_max = 1e-1;
    
[W, b] = InitiateValues(m, d, K, N);
parameters = InitiateParameters(lambda, eta_min, eta_max, n_epoch, n_batch);

""" # Compare numerical and analytical gradients # """;
%[W_gradient_error, b_gradient_error] = CompareGradients(data1.X, data1.Y, W, b, parameters);

""" # Compare cost, small set of data 200 epochs # """;
%testExamples = 100;
%temp.X = data1.X(:,1:testExamples);
%temp.Y = data1.Y(:,1:testExamples);
%temp.labels = data1.labels(1:testExamples);
%[W_Test, b_Test, resultsCostTest] = MiniBatch(temp, testData, W, b, parameters);
%finalCostTest = ComputeCost(data1.X, data1.Y, W_Test, b_Test, parameters.lambda)

""" # Train Network to study CLR # """;
%[Wstar, bstar, results] = MiniBatch(data1, testData, W, b, parameters);
%accuracy = ComputeAccuracy(testData.X, testData.labels, Wstar, bstar)
%trainAccuracy = ComputeAccuracy(data1.X, data1.labels, Wstar, bstar)

%plot(results.trainAccuracy);
%hold on
%plot(results.testAccuracy);
%ylabel('Accuracy %');
%xlabel('Epochs');
%legend('Training data', 'Validation data');
%hold off

%plot(resluts.etaList);
%xlabel('Steps');
%ylabel('η_m_i_n                     η_m_a_x');

""" # Chunk data for Coarse to fine lambda search # """;

%completeDataX = [data1.X,data2.X,data3.X,data4.X,data5.X(:,1:5000)];
%completeDataY = [data1.Y,data2.Y,data3.Y,data4.Y,data5.Y(:,1:5000)];
%completeDataLabels = [data1.labels;data2.labels;data3.labels;data4.labels;data5.labels(1:5000)];
%completeData.X = completeDataX;
%completeData.Y = completeDataY;
%completeData.labels = completeDataLabels;
%tempTest.X = data5.X(:,5001:10000);
%tempTest.Y = data5.Y(:,5001:10000);
%tempTest.labels = data5.labels(5001:10000);

%res = CoarseToFine(completeData, tempTest, m, d, K, N);

""" # Final training with optimal lambda # """;

%completeDataX = [data1.X,data2.X,data3.X,data4.X,data5.X];
%completeDataY = [data1.Y,data2.Y,data3.Y,data4.Y,data5.Y];
%completeDataLabels = [data1.labels;data2.labels;data3.labels;data4.labels;data5.labels];
%completeData.X = completeDataX;
%completeData.Y = completeDataY;
%completeData.labels = completeDataLabels;
%tempTest.X = testData.X(:,1:1000);
%tempTest.Y = testData.Y(:,1:1000);
%tempTest.labels = testData.labels(1:1000);

%n_cycles = 5;
%parameters2 = InitiateParameters2(0.000519, 1e-5, 0.12, n_cycles, 100, size(completeData.X, 2));
%[~, ~, results] = MiniBatch(completeData, tempTest, W, b, parameters2);

plot(results.trainAccuracy);
hold on
plot(results.testAccuracy);
ylabel('Accuracy %');
xlabel('Epochs');
legend('Training data', 'Validation data');
hold off

plot(results.trainingCost);
hold on
plot(results.testCost);
ylabel('Cost');
xlabel('Epochs');
legend('Training data', 'Validation data');
hold off

plot(results.trainingLoss);
hold on
plot(results.testLoss);
ylabel('Loss');
xlabel('Epochs');
legend('Training data', 'Validation data');
hold off

function [Wstar, bstar, results] = MiniBatch(data, testData, W, b, parameters)
    iteration = 0;
    for i=1:parameters.n_epoch
        
            [trainCost, trainLoss] = ComputeCost(data.X, data.Y, W, b, parameters.lambda);
            results.trainingCost(i) = trainCost;
            results.trainingLoss(i) = trainLoss;
            [testCost, testLoss] = ComputeCost(testData.X, testData.Y, W, b, parameters.lambda);
            results.testCost(i) = testCost;
            results.testLoss(i) = testLoss;
            
            results.trainAccuracy(i) = ComputeAccuracy(data.X, data.labels, W, b) * 100; 
            results.testAccuracy(i) = ComputeAccuracy(testData.X, testData.labels, W, b) * 100;

        for j=1:size(data.X, 2)/parameters.n_batch
            iteration = iteration + 1;
            j_start = (j-1)*parameters.n_batch + 1;
            j_end = j*parameters.n_batch;
            Xbatch = data.X(:, j_start:j_end);
            Ybatch = data.Y(:, j_start:j_end);

            [P, h, s1] = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, h, s1, W, b, parameters.lambda);
            W{1} = W{1} - (grad_W{1} * parameters.eta);
            W{2} = W{2} - (grad_W{2} * parameters.eta);
            b{1} = b{1} - (grad_b{1} * parameters.eta);
            b{2} = b{2} - (grad_b{2} * parameters.eta);
            
            eta = CLR(parameters, iteration);
            parameters.eta = eta;
            results.etaList(iteration) = eta;
        end
    end
    
    Wstar = W;
    bstar = b;
    
end

function result = CoarseToFine(Data, Test, m, d, K, N)

    n_lambdas = 15;
    %l_min = -5;
    %l_max = -1;
    %l_min = -7;
    %l_max = -3;
    l_min = -4;
    l_max = -2;
    
    n_cycles = 2;
    [W, b] = InitiateValues(m, d, K, N);
    lambda_vals = GenerateLambda(n_lambdas, l_min, l_max);
    
    parameters = InitiateParameters2(lambda_vals(1), 1e-5, 0.1, n_cycles, 100, size(Data.X, 2));
    filename = 'CoarseToFine';
    file = fopen(filename, 'w');
    fprintf(file,'Results for Coars_to_fine with %0f different lambda values with l_max: %1f and l_min: %2f \n', n_lambdas, l_max, l_min);
    fprintf(file,'Stepsize: %1f, batchsize: %2f, epochs: %3f, cycles: %4f \n\n\n', parameters.n_s, parameters.n_batch, parameters.n_epoch, n_cycles);
    
    for i=1:n_lambdas
        
        parameters = InitiateParameters2(lambda_vals(i), 1e-5, 0.1, n_cycles, 100, size(Data.X, 2));
        [Wstar, bstar, result] = MiniBatch(Data, Test, W, b, parameters);
        
        testAccuracy = max(result.testAccuracy)
        trainAccuracy = max(result.trainAccuracy)
        
        fprintf(file,'lambda: %0f \n', lambda_vals(i));
        fprintf(file,'%2f percent Test Accuracy %0f \n',testAccuracy, lambda_vals(i));
        fprintf(file,'%2f percent Train Accuracy \n\n',trainAccuracy, lambda_vals(i));
        
    end
    
    fclose(file);
end

function lambda = GenerateLambda(n_lambdas, l_min, l_max)
    
    for i=1:n_lambdas
        l = l_min + (l_max - l_min) * rand(1, 1);
        lambda(i) = 10^l;
    end

end

function [W_diff, b_diff] = CompareGradients(X, Y, W, b, parameters)

%   Compare the values of our analytical gradie nt computations
%   with a more precise, numerical calculations of the gradients
%   by calculating the relative error between the two

    [P, h, s1] = EvaluateClassifier(X(:,1:parameters.n_batch), W, b);
    [ngrad_W, ngrad_b] = ComputeGradsNumSlow(X(:,1:parameters.n_batch), Y(:,1:parameters.n_batch), W, b, parameters.lambda, 1e-5);
    [grad_W, grad_b] = ComputeGradients(X(:,1:parameters.n_batch), Y(:,1:parameters.n_batch), P, h, s1, W, b, parameters.lambda);
    
    gradientDiff_W1 = sum(sum(abs(grad_W{1} - ngrad_W{1}) ./ max(1e-6, sum(sum(abs(grad_W{1}) + abs(ngrad_W{1}))))))
    gradientDiff_b1 = sum(abs(grad_b{1} - ngrad_b{1}) ./ max(1e-6, sum(abs(grad_b{1}) + abs(ngrad_b{1}))))
    gradientDiff_W2 = sum(sum(abs(grad_W{2} - ngrad_W{2}) ./ max(1e-6, sum(sum(abs(grad_W{2}) + abs(ngrad_W{2}))))))
    gradientDiff_b2 = sum(abs(grad_b{2} - ngrad_b{2} ) ./ max(1e-6, abs(grad_b{2}) + abs(ngrad_b{2})))
    
    W_diff = {gradientDiff_W1, gradientDiff_W2};
    b_diff = {gradientDiff_b1, gradientDiff_b2};

end

function[W, b] = InitiateValues(m, d, K, N)
    deviation1 = 1/sqrt(d);
    deviation2 = 1/sqrt(m);
    W1 = deviation1.*randn(m,d);   
    W2 = deviation2.*randn(K,m);
    b1 = zeros(m,1);
    b2 = zeros(K,1);
    W = {W1, W2};
    b = {b1, b2};
end

function eta = CLR(parameters, iteration)
    
    cycle = floor(1 + iteration/(parameters.n_s * 2));
    x = abs((iteration/parameters.n_s) - (2 * cycle) + 1);
    eta = parameters.eta_min + (parameters.eta_max - parameters.eta_min) * max(0, (1-x));
    
end

function parameters = InitiateParameters(lambda, eta_min, eta_max, n_epoch, n_batch)

    parameters.lambda = lambda;
    parameters.eta_min = eta_min;
    parameters.eta_max = eta_max;
    parameters.eta = eta_min;
    parameters.n_epoch = n_epoch;
    parameters.n_batch = n_batch;
    parameters.n_s = 800;
    
end

function parameters = InitiateParameters2(lambda, eta_min, eta_max, n_cycles, n_batch, n)

    parameters.lambda = lambda;
    parameters.eta_min = eta_min;
    parameters.eta_max = eta_max;
    parameters.eta = eta_min;
    parameters.n_batch = n_batch;
    parameters.n_s = 2 * floor(n/n_batch);
    parameters.n_epoch = floor((n_cycles * n_batch * parameters.n_s) / (n_batch * 100));
    
end

function [X, Y, y] = LoadBatch(filename)
    A = load(filename);
    X = im2double(A.data');
    mean_X = mean(X, 2);
    X = X - mean_X;
    y = A.labels;
    Y = y == 0:max(y);
    Y = Y.';
end
 
function [P, h, s1] = EvaluateClassifier(X, W, b)
    s1 = W{1}*X + b{1};
    h = max(0, s1);
    s = W{2} * h + b{2};
    P = softmax(s);
end
 
function [cost, loss] = ComputeCost(X, Y, W, b, lambda)
    [P, h] = EvaluateClassifier(X, W, b);
    loss = sum(-log(sum(Y .* P, 1))) / size(X, 2);
    cost = loss + lambda * (sumsqr(W{1}) + sumsqr(W{2}));
end
 
function acc = ComputeAccuracy(X, Y, W, b)
    P = EvaluateClassifier(X, W, b);
    [~, index] = max(P);
    acc = sum((index - 1)' == Y) / size(X, 2);
end
 

function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, S1, W, b, lambda)

    grad_W1 = zeros(size(W{1}));
    grad_W2 = zeros(size(W{2}));
    grad_b1 = zeros(size(b{1}));
    grad_b2 = zeros(size(b{2}));
    
    for i=1:size(X,2)
        x = X(:, i);
        y = Y(:, i);
        p = P(:, i);
        hn = h(:, i);
        s1 = S1(:,i);
        g =  -(y-p).';
        grad_b2 = grad_b2 + g'; 
        grad_W2 = grad_W2 + g' .* hn';
        g = g * W{2};
        g = g * diag(s1 > 0);
        grad_b1 = grad_b1 + g';
        grad_W1 = grad_W1 + (g' .* x');

    end

    grad_W1 = grad_W1/size(X, 2);
    grad_W2 = grad_W2/size(X, 2);
    grad_b1 = grad_b1/size(X, 2);
    grad_b2 = grad_b2/size(X, 2);
    grad_W1 = grad_W1 + 2 * lambda*W{1};
    grad_W2 = grad_W2 + 2 * lambda*W{2};
    
    grad_W = {grad_W1, grad_W2};
    grad_b = {grad_b1, grad_b2};
    
end

function [grad_W, grad_b] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);
    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));
    
        for i=1:length(b{j})
        
            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b_try, lambda);
        
            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);
        
            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
        for i=1:numel(W{j})
        
           W_try = W;
           W_try{j}(i) = W_try{j}(i) - h;
                c1 = ComputeCost(X, Y, W_try, b, lambda);
    
           W_try = W;
           W_try{j}(i) = W_try{j}(i) + h;
           c2 = ComputeCost(X, Y, W_try, b, lambda);
    
                grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end