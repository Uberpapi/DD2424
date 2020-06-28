clear
"""  # Read in Data  # """;
book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

book_chars = unique(book_data);

""" # Map Functions # """;

char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

for i=1:length(book_chars)
    char_to_ind(book_chars(i)) = i;
    ind_to_char(i) = book_chars(i);
end

""" # Initialization # """;

RNN = InitializeParameters(100, 1e-1, 25, length(book_chars));
RNN_m = InitializeSumSqr(RNN);

tParams.m = 100;
tParams.eta = 1e-1;
tParams.K = length(book_chars);
tParams.seq_length = 25;
tParams.char_to_ind = char_to_ind;
tParams.ind_to_char = ind_to_char;
tParams.book_data = book_data;
tParams.n_epochs = 10;

""" # Execution # """;

%synthText = SynthesizeText(RNN, h0, x0, n);
%text = GenerateText(ind_to_char, synthText);
%X_chars = book_data(1:RNN.seq_length);
%Y_chars = book_data(2:RNN.seq_length+1);
%X = OneHot(X_chars, char_to_ind);
%Y = OneHot(Y_chars, char_to_ind);
%error = CompareGradients(X, Y, RNN, tParams);

[RNN, Loss, best_model] = SGDTraining(RNN, RNN_m, tParams);

syntText = SynthesizeText(best_model.RNN, best_model.h, best_model.X, 1000, tParams.K);
txt = GenerateText(tParams.ind_to_char, syntText);
disp('----------- FINAL TEXT -----------')
disp(['Loss at ', num2str(best_model.loss)]);
txt

function RNN = InitializeParameters(m, eta, seq_length, K)

    RNN.c = zeros(K, 1);
    RNN.b = zeros(m, 1);
    sig = 1e-2;
    RNN.U = randn(m, K)*sig;
    RNN.W = randn(m, m)*sig;
    RNN.V = randn(K, m)*sig;
    RNN.m = m;
    RNN.seq_length = seq_length;
    
end

function [RNN, Loss, best_model] =  SGDTraining(RNN, RNN_m, tParams)

    n = length(tParams.book_data) - tParams.seq_length;
    h_prevs = zeros(tParams.m, floor(n/tParams.seq_length));
    step = 1;
   % Loss = zeros(tParams.n_epochs, floor(n/tParams.seq_length));
    epoch_step = 0;
    
    for i=1:tParams.n_epochs
        for e = 1 : tParams.seq_length : n
            X = OneHot(tParams.book_data(e:e + tParams.seq_length-1), tParams.char_to_ind);
            Y = OneHot(tParams.book_data(e+1: e+ tParams.seq_length), tParams.char_to_ind);

            if e == 1
                h_prev = zeros(tParams.m, 1);
            else
                h_prev = h(:, end);
            end

            h_prevs(:,e) = h_prev;
            [loss, P, a, h] = ForwardPass(X, Y, h_prev, RNN, tParams);
            grads = BackwardPass(RNN, X, Y, P, a, h, tParams);
            grads = ClipGradients(grads);
            [RNN, RNN_m] = AdaGrad(RNN, RNN_m, grads, tParams.eta);

            if e == 1 && i == 1
                smooth_loss = loss;
                best_model.loss = loss;
            else
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss;
            end
            
            if best_model.loss >= smooth_loss
               best_model.loss = smooth_loss;
               best_model.RNN = RNN;
               best_model.h = h_prev;
               best_model.X = X;
            end

            %if mod(step, 1000) == 0
            %    disp(['At iteration: ,', num2str(step), ' - loss is Currently at: ', num2str(smooth_loss), ' progress at ', num2str((e+n*(i-1))/(n*tParams.n_epochs) *100), '%', ' Epoch: ', num2str(i)]);
            %end

            if mod(step, 10000) == 0 || step == 1
                synthText = SynthesizeText(best_model.RNN, best_model.h, best_model.X, 200, tParams.K);
                text = GenerateText(tParams.ind_to_char, synthText);
                disp(['At iteration: ', num2str(step), ' loss is Currently at: ', num2str(best_model.loss)]);
                disp('--------------')
                disp(text);
                disp('--------------')
            end
            Loss(step) = smooth_loss;
            step = step + 1;
        end
        epoch_step = epoch_step + 1;
    end
    'Done with SGD'
end

function Y = SynthesizeText(RNN, h0, x0, n, K)
    h = h0;
    x = x0;
    Y = zeros(K, n);
    
    for t=1:n
        a = RNN.W * h + RNN.U*x + RNN.b;
        h = tanh(a);
        o = RNN.V*h + RNN.c;
        p = softmax(o);
        
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a >0);
        ii = ixs(1);
        
        x = zeros(K, 1); 
        x(ii) = 1; 
        Y(:,t) = x;
        
    end
end

function oneHot = OneHot(chars, char_to_ind)

    oneHot = zeros(length(char_to_ind), length(chars));
    for i=1:length(chars)
        index = char_to_ind(chars(i));
        oneHot(index, i) = 1;
    end
    
end

function sequence = GenerateText(ind_to_char, text)
    sequence = [];
    [~, index] = max(text, [], 1);

    for i=1:length(index)
        sequence = [sequence ind_to_char(index(i))];
    end
end

function [loss, P, a, h] = ForwardPass(X, Y, h0, RNN, tParams)

    h = zeros(tParams.m, tParams.seq_length);
    h(:, 1) = h0;
    P = zeros(tParams.K, tParams.seq_length);
    a = zeros(tParams.m, tParams.seq_length);
    o = zeros(tParams.K, tParams.seq_length);
   

    for t=1:tParams.seq_length
       a(:,t) = RNN.W*h(:,t) + RNN.U*X(:,t) + RNN.b;
       h(:,t+1) = tanh(a(:,t));
       o(:,t) = RNN.V*h(:,t+1) + RNN.c;
       P(:,t) = softmax(o(:,t));
    end

    loss = sum(-log(sum(Y .* P)));
end

function grads = BackwardPass(RNN, X, Y, P, A, H, tParams)

    for f = fieldnames(RNN)'
        grads.(f{1}) = zeros(size(RNN.(f{1})));
    end

    G = -(Y-P)';
    h = H(:,tParams.seq_length+1); 
    h2 = H(:, tParams.seq_length);
    a = A(:,tParams.seq_length);
    x = X(:,tParams.seq_length);
    g = G(tParams.seq_length,:);
    grads.c = g';
    grads.V = g'*h';
    
    grad_h = g*RNN.V;
    grad_a = grad_h*diag(1-tanh(a).^2);
    grads.b = grad_a';
    grads.W = grad_a'*h2';
    grads.U = grad_a'*x';

    for i = flip(1:tParams.seq_length-1)
        g = G(i,:);
        h = H(:,i+1);
        h2 = H(:,i);
        a = A(:,i);
        x = X(:,i);
    
        grads.c = grads.c + g';    
        grads.V = grads.V + g'*h';
    
        grad_h = g*RNN.V + grad_a*RNN.W;
        grad_a = grad_h*diag(1-tanh(a).^2);
    
        grads.b = grads.b + grad_a';
        grads.W = grads.W + grad_a'*h2';
        grads.U = grads.U + grad_a'*x';
    end
end

function errors = CompareGradients(X, Y, RNN, tParams)
    
    h0 = zeros(RNN.m, 1);
    grads = ComputeGradients(X, Y, h0, RNN, tParams);
    num_grads = ComputeGradsNum(X, Y, 1e-4, RNN, tParams);

    eps = 1e-6;
    errors.gradb = max(abs(grads.b-num_grads.b)./max(eps,sum(abs(grads.b)+ abs(num_grads.b))));
    errors.gradc = max(abs(grads.c-num_grads.c)./max(eps,sum(abs(grads.c)+ abs(num_grads.c))));
    errors.gradU = max(max(abs(grads.U-num_grads.U)./max(eps,sum(abs(grads.U)+ abs(num_grads.U)))));
    errors.gradW = max(max(abs(grads.W-num_grads.W)./max(eps,sum(abs(grads.W)+ abs(num_grads.W)))));
    errors.gradV = max(max(abs(grads.V-num_grads.V)./max(eps,sum(abs(grads.V)+ abs(num_grads.V)))));
    
end

function grads = ComputeGradients(X, Y, h0, RNN, tParams)

    [~, P, a, h] = ForwardPass(X, Y, h0, RNN, tParams);
    grads = BackwardPass(RNN, X, Y, P, a, h, tParams);
    
end

function num_grads = ComputeGradsNum(X, Y, h, RNN, tParams)

    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h, tParams);
    end
    
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h, tParams)

    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(X, Y, RNN_try, hprev, tParams);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(X, Y, RNN_try, hprev, tParams);
        grad(i) = (l2-l1)/(2*h);
    end
    
end

function loss = ComputeLoss(X_batch, Y_batch, RNN, h, tParams)

    [loss, ~, ~, ~] = ForwardPass(X_batch, Y_batch, h, RNN, tParams);
    
end

function [RNN, RNN_m] = AdaGrad(RNN, RNN_m, grads, eta)

    RNN_m.W = RNN_m.W + grads.W.^2;
    RNN_m.U = RNN_m.U + grads.U.^2;
    RNN_m.V = RNN_m.V + grads.V.^2;
    RNN_m.b = RNN_m.b + grads.b.^2;
    RNN_m.c = RNN_m.c + grads.c.^2;
    RNN.W = RNN.W - eta*(grads.W./(RNN_m.W + eps).^(0.5));
    RNN.U = RNN.U - eta*(grads.U./(RNN_m.U + eps).^(0.5));
    RNN.V = RNN.V - eta*(grads.V./(RNN_m.V + eps).^(0.5));
    RNN.b = RNN.b - eta*(grads.b./(RNN_m.b + eps).^(0.5));
    RNN.c = RNN.c - eta*(grads.c./(RNN_m.c + eps).^(0.5));

    
end

function grads = ClipGradients(grads)

    for f = fieldnames(grads)
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end
    
end

function RNN_m = InitializeSumSqr(RNN)
    RNN_m.c = zeros(size(RNN.c));
    RNN_m.b = zeros(size(RNN.b));
    RNN_m.U = zeros(size(RNN.U)); 
    RNN_m.W = zeros(size(RNN.W)); 
    RNN_m.V = zeros(size(RNN.V));
end