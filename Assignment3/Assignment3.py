import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import random

pathFolder = "D:/cifar-10-batches-py/"

d = 3072  # (32x32x3) -
k = 10  # number of classes

layers = [50, 50, k]
n_batch = 100
n_cycles = 2

lam = 0.00888
#lam = 0.005
eta_min = 1e-5
eta_max = 0.1

b_n = False #Batch normalization, set to true if you wish to have active
plot = True #Do we want to plot?


def InitiateParameters():
    #sigma = 1/np.sqrt(d) #Xavier Initialization
	W = [np.random.normal(1/np.sqrt(d), 1e-4, (layers[0] , d))]
	b = [np.random.normal(1/np.sqrt(d), 1e-4,(layers[0] , 1))]
	G = [np.random.normal(1/np.sqrt(d), 1e-4, (layers[0] , 1))]
	B = [np.random.normal(1/np.sqrt(d), 1e-4, (layers[0] , 1))]
	
	for l in range(1, len(layers)):
		#sigma = 1/np.sqrt(Ws[l-1].shape[0])
		W_temp = np.random.normal(0, 1e-4, (layers[l], W[l-1].shape[0]))
		b_temp = np.zeros( (layers[l]  , 1))
		G_temp = np.random.normal(0, 1e-4, (layers[l], 1))
		B_temp = np.random.normal(0, 1e-4, (layers[l], 1))
		W.append(W_temp)
		b.append(b_temp)
		G.append(G_temp)
		B.append(B_temp)

	return W , b, G, B


def CLR(n_s, iter, cycle):

    minimum = 2*cycle*n_s
    maximum = 2*(cycle + 1)*n_s
    middle = (2*cycle + 1)*n_s

    if (minimum <= iter and iter <= middle):
        eta = eta_min + ((eta_max - eta_min)*((iter-minimum)/n_s))
    elif(middle <= iter and iter <= maximum):
        eta = eta_max - ((eta_max - eta_min)*((iter - middle)/n_s))
    return eta

def EvaluateClassifier(X, W, b, G, B):
    x = [X]
    s = list()
    means = list()
    vars = list()
    sH = list()
    sT = list()

    for l in range(len(layers) - 1):
        s.append(np.dot(W[l], x[l]) + b[l])

        if(b_n):
            s_H, mean, var = BatchNormalization(s[l])
            sH.append(s_H)
            means.append(mean)
            vars.append(var)
            sT.append(np.multiply(G[l], sH[l]) + B[l])
            x.append(np.maximum(0, sT[l]))
        else:
            x.append(np.maximum(0, s[l]))

    aver_means = means
    aver_vars = vars

    for l in range(len(means)):
        aver_means[l] = 0.9 * aver_means[l] + (0.1) * means[l]
        aver_vars[l] = 0.9 * aver_vars[l] + (0.1) * vars[l]

    numerator = np.exp(
        np.dot(W[len(layers) - 1], x[len(layers) - 1]) + b[len(layers) - 1])
    prob = numerator / np.sum(numerator, axis=0)
    pred = np.argmax(prob, axis=0)

    return x, prob, pred, s, sH, means, vars


def ComputeCost(prob, Y, W):
    py = np.multiply(Y, prob).sum(axis=0)
    py[py == 0] = np.finfo(float).eps

    WeightSquareSum = 0
    for i in range(len(layers)):
        WeightSquareSum += np.sum(np.square(W[i]))

    ridgeRegression = lam * WeightSquareSum
    loss = ((-np.log(py).sum() / prob.shape[1]))
    cost = loss + ridgeRegression

    return loss, cost


def BatchNormalization(s, mean=None, var=None):
    eps = 1e-12
    if (mean == None):
        mean = np.mean(s, axis=1, keepdims=True)

    if (var == None):
        var = np.var(s, axis=1, keepdims=False)

    part1 = np.diag(1 / np.sqrt(var + eps))
    part2 = s - mean
    sH = np.dot(part1, part2)

    return sH, mean, var

def BackwardsPass(g, S, means):
    S1 = 1 / np.sqrt(np.mean(np.power((S-means), 2), axis=1, keepdims=True))
    S2 = np.power(S1, 3)

    A = np.multiply(g, S1)
    B = np.multiply(g, S2)
    D = np.subtract(S, means)
    c = np.sum(np.multiply(B, D), axis=1, keepdims=True)

    foo = np.subtract(A, np.sum(A, axis=1, keepdims=True) / S.shape[1])
    bar = np.multiply(D, c) / S.shape[1]

    return np.subtract(foo, bar)

def ComputeAccuracy(pred, y):
    totalCorrect = 0

    for i in range(pred.shape[0]):
        if(pred[i] == y[i]):
            totalCorrect = totalCorrect + 1
    acc = (totalCorrect / pred.shape[0]) * 100

    return acc


def ComputeGradients(X, Y, W, b, G, B):
    grad_W = list()
    grad_b = list()
    grad_G = list()
    grad_B = list()
    onesVector = np.ones((X.shape[1], 1))
    l = len(layers) - 1

    x, prob, pred, S, sH, means, vars = EvaluateClassifier(X, W, b, G, B)
    g = -(Y-prob)
    grad_b.append(np.dot(g, onesVector) / X.shape[1])
    grad_W.append((np.dot(g, x[l].T) / X.shape[1]) + (2*lam*W[l]))

    if(b_n):
        g = np.dot(W[l].T, g)
        indicator = 1 * (x[l] > 0)
        g = np.multiply(g, indicator)
        l = l - 1

    while (l >= 0):

        if(b_n):
            grad_G.append(
                np.dot(np.multiply(g, sH[l]), onesVector) / X.shape[1])
            grad_B.append(np.dot(g, onesVector) / X.shape[1])
            g = np.multiply(g, np.dot(G[l], onesVector.T))
            g = BackwardsPass(g, S[l], means[l])

        g_b = np.dot(g, onesVector) / X.shape[1]
        g_W = np.dot(g, x[l].T) / X.shape[1]
        g_W = g_W + (2*lam*W[l])

        grad_b.append(g_b)
        grad_W.append(g_W)

        if (l > 0):
            indicator = 1 * (x[l] > 0)
            g = np.dot(W[l].T, g)
            g = np.multiply(g, indicator)

        l -= 1

    grad_b.reverse()
    grad_W.reverse()
    grad_G.reverse()
    grad_B.reverse()

    return grad_b, grad_W, grad_G, grad_B


def ComputeGradsNumSlow(X, Y, W, b, G, B):
    h = 1e-5
    grad_Ws = list()
    grad_bs = list()
    grad_gammas = list()
    grad_betas = list()

    for layer in range(len(layers)):
        grad_b = np.zeros_like(b[layer])
        grad_W = np.zeros_like(W[layer])
        grad_G = np.zeros_like(G[layer])
        grad_B = np.zeros_like(B[layer])
        for i in range(b[layer].shape[0]):
            sb = b[layer]

            temp = sb
            temp[i] += h
            b[layer] = temp
            _, prob, _, _, _, _, _ = EvaluateClassifier(X, W, b, G, B)
            _, cost_one = ComputeCost(prob, Y, W)

            temp = sb
            temp[i] -= h
            b[layer] = temp
            _, prob, _, _, _, _, _ = EvaluateClassifier(X, W, b, G, B)
            _, cost_two = ComputeCost(prob, Y, W)

            b[layer] = sb
            grad_b[i] = (cost_one - cost_two) / h
        for i in range(W[layer].shape[0]):
            for j in range(W[layer].shape[1]):
                sW = W[layer]

                temp = sW
                temp[i][j] += h
                W[layer] = temp
                _, prob, _, _, _, _, _ = EvaluateClassifier(X, W, b, G, B)
                _, cost_one = ComputeCost(prob, Y, W)

                temp = sW
                temp[i][j] -= h
                W[layer] = temp
                _, prob, _, _, _, _, _ = EvaluateClassifier(X, W, b, G, B)
                _, cost_two = ComputeCost(prob, Y, W)

                W[layer] = sW
                grad_W[i][j] = (cost_one - cost_two) / h
        for i in range(G[layer].shape[0]):
            sG = G[layer]

            temp = sG
            temp[i] += h
            G[layer] = temp
            _, prob, _, _, _, _, _ = EvaluateClassifier(X, W, b, G, B)
            _, cost_one = ComputeCost(prob, Y, W)

            temp = sG
            temp[i] -= h
            G[layer] = temp
            _, prob, _, _, _, _, _ = EvaluateClassifier(X, W, b, G, B)
            _, cost_two = ComputeCost(prob, Y, W)

            G[layer] = sG
            grad_G[i] = (cost_one - cost_two) / h

        for i in range(B[layer].shape[0]):
            sB = B[layer]

            temp = sB
            temp[i] += h
            B[layer] = temp
            _, prob, _, _, _, _, _ = EvaluateClassifier(X, W, b, G, B)
            _, cost_one = ComputeCost(prob, Y, W)

            temp = sB
            temp[i] -= h
            B[layer] = temp
            _, prob, _, _, _, _, _ = EvaluateClassifier(X, W, b, G, B)
            _, cost_two = ComputeCost(prob, Y, W)

            B[layer] = sB
            grad_B[i] = (cost_one - cost_two) / h

        grad_bs.append(grad_b)
        grad_Ws.append(grad_W)
        grad_gammas.append(grad_G)
        grad_betas.append(grad_B)
    return grad_bs, grad_Ws, grad_gammas, grad_betas


def CompareGradients():

    X, Y, y = ReadData("data_batch_1")
    Ws, b, G, B = InitiateParameters()
    temp_d = 20

    grad_W_mean = 0
    grad_B_mean = 0
    X = X[:temp_d, 0:1]
    Y = Y[:, 0:1]

    W = list()
    W.append(Ws[0][:, :temp_d])
    for layer in range(1, len(layers)):
        W.append(Ws[layer])

    grad_b, grad_W, _, _ = ComputeGradients(X, Y, W, b, G, B)
    ngrad_b, ngrad_W, _, _ = ComputeGradsNumSlow(X, Y, W, b, G, B)

    print('\n -- Difference between numerical and analytical computations --', '\n')
    for layer in range(len(layers)):
        temp = np.mean(np.abs(grad_W[layer] - ngrad_W[layer]))
        grad_W_mean += temp
        print(' grad_W difference in layer ', layer + 1, " = ", temp)
    print('\n  grad_W mean difference = ', grad_W_mean / layer, '\n')

    for layer in range(len(layers)):
        temp = np.mean(np.abs(grad_b[layer] - ngrad_b[layer]))
        grad_B_mean += temp
        print(' grad_b difference in layer ', layer + 1, " = ", temp)
    print('\n  grad_b mean difference = ', grad_B_mean / layer)


def MiniBatch():
    X, Y, y, XVal, YVal, yVal, XTest, YTest, yTest = GetData()
    W, b, G, B = InitiateParameters()
    print('lambda is : ', lam)
    trainCost = list()
    trainLoss = list()
    trainAccuracy = list()
    valCost = list()
    valLoss = list()
    valAcc = list()
    iters = list()

    n_s = 5 * 45000/n_batch
    numBatches = int(X.shape[1] / n_batch)
    n_epochs = int((2 * n_cycles * n_s) / numBatches)
    iter = 0
    cycleCounter = 0
    eta = eta_min

    for epoch    in range(n_epochs):
        for j in range(numBatches):
            j_start = j * n_batch
            j_end = j_start + n_batch
            X_batch = X[:, j_start:j_end]
            Y_batch = Y[:, j_start:j_end]
            grad_b, grad_W, grad_G, grad_B = ComputeGradients(
                X_batch, Y_batch, W, b, G, B)

            for layer in range(len(layers)):
                W[layer] = W[layer] - (eta * grad_W[layer])
                b[layer] = b[layer] - (eta * grad_b[layer])

                if(layer == len(layers) - 1):
                    break

                if(b_n):
                    G[layer] = G[layer] - (eta * grad_G[layer])
                    B[layer] = B[layer] - (eta * grad_B[layer])

            if (iter % (2 * n_s) == 0) and iter != 0:
                cycleCounter += 1

            iter += 1
            eta = CLR(n_s, iter, cycleCounter)

            if (iter % 100 == 0):

                _, trainProb, trainPred, _, _, _, _ = EvaluateClassifier(
                    X, W, b, G, B)
                tLoss, tCost = ComputeCost(trainProb, Y, W)
                tAcc = ComputeAccuracy(trainPred, y)
                trainLoss.append(tLoss)
                trainCost.append(tCost)
                trainAccuracy.append(tAcc)

                _, valProb, valPred, _, _, _, _ = EvaluateClassifier(
                    XVal, W, b, G, B)
                vLoss, vCost = ComputeCost(valProb, YVal, W)
                vAcc = ComputeAccuracy(valPred, yVal)
                valLoss.append(vLoss)
                valCost.append(vCost)
                valAcc.append(vAcc)

                iters.append(iter)

        print(int(epoch/n_epochs*100), '% complete')

    # Calculate accuracy on the test dataset
    _, _, predictionsTest, _, _, _, _ = EvaluateClassifier(XTest, W, b, G, B)
    testAccuracy = ComputeAccuracy(predictionsTest, yTest)
    print("\n")
    print("Test Accuracy: ", testAccuracy, "% \n")

    if(plot):

        # Plot Accuracy
        plt.figure(1)
        plt.plot(iters, trainAccuracy, 'r-')
        plt.plot(iters, valAcc, 'b-')
        plt.xlabel("# of Iterations")
        plt.ylabel("Accuracy")
        plt.show()

        # Plot Cost
        #plt.figure(1)
        #plt.plot(iters, trainCost, 'r-')
        #plt.plot(iters, valCost, 'b-')
        #plt.xlabel("# of Iterations")
        #plt.ylabel("Computed Cost")
        #plt.show()

        # Plot Loss
        plt.figure(1)
        plt.plot(iters, trainLoss, 'r-')
        plt.plot(iters, valLoss, 'b-')
        plt.xlabel("# of Iterations")
        plt.ylabel("Computed Loss")
        plt.show()

    return testAccuracy

def GetData():
    X, Y, y = ReadData("data_batch_1")
    for j in range(2, 6):
        filename = "data_batch_" + str(j)
        tempX, tempY, tempy = ReadData(filename)
        X = np.concatenate((X, tempX), axis=1)
        Y = np.concatenate((Y, tempY), axis=1)
        y = np.concatenate((y, tempy))
    trainX, validX = np.split(X, [45000], axis=1)
    trainY, validY = np.split(Y, [45000], axis=1)
    trainy, validy = np.split(y, [45000])
    testX, testY, testy = ReadData("test_batch")
    return trainX, trainY, trainy, validX, validY, validy, testX, testY, testy


def ReadData(fileName):
    temp_ = pathFolder + fileName
    with open(temp_, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    file.close()
    X = (np.array(data[b'data'])).T
    m_X = np.mean(X)
    s_X = np.std(X)
    X = (X - m_X) / s_X
    y = np.array(data[b'labels'])
    Y = np.zeros((k, X.shape[1]))
    for i in range(X.shape[1]):
        Y[y[i]][i] = 1
    return X, Y, y


def InitialLambdaSearch(amount):
    global lam
    for i in range(1, amount+1):
        lam = 1/math.pow(10, i)
        print('Accuracy with lambda: ', lam, ' is ', MiniBatch())

def CorseToFine():
    global lam

    for i in range(0, 10):
        lam = random.uniform(1e-2, 1e-3)
        print('Accuracy with lambda: ', lam, ' is ', MiniBatch())

# CompareGradients()
accuracy = MiniBatch()
#print(accuracy)
#InitialLambdaSearch(6)
#CorseToFine()
