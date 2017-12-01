#-*- coding: utf-8 -*- 

import numpy as np 
import scipy.spatial.distance as sciDist


# !!! redefine feedforward function for RBFNN 
def run_rbfnn(trainX, trainY, W, U, V):
    D = np.power(sciDist.cdist(trainX, U), 2)
    Z = np.exp(D/np.transpose((-2 * np.power(V, 2)))) # how to implement element-wise operation?
    A = np.dot(Z, W)
    assert np.shape(A) == np.shape(trainY)
    return A 


# outline of LGEM calculating
def calc_stsm(trainX, trainY, W, U, V, Q=0.1):
    deltaX = np.random.uniform(-Q, Q, (1000, np.shape(trainX)[1]))
    H = np.shape(deltaX)[0]

    trainOut = run_rbfnn(trainX, trainY, W, U, V)
    STSM = np.zeros(np.shape(trainY))
    for i in range(H):
        deltaOut = run_rbfnn(trainX + deltaX[i,:], trainY, W, U, V)
        STSM = STSM + np.power((deltaOut - trainOut), 2)

    STSM = STSM/H
    STSM = np.mean(STSM, axis = 0)

    return STSM

class RadialBaiscFunctionNeuralNetwork(object):
    def __init__(self, sizes=[]):
        pass
    def fit(self):
        pass
    def predict(self):
        pass
model = RadialBaiscFunctionNeuralNetwork()


def calc_stsm_vector(model, trainX, cost_func=None, num_delta=1000, Q=0.1):
    deltaX = np.random.uniform(-Q, Q, (num_delta, trainX.shape[1]))
    trainY = model.predict(trainX)

    STSM = np.zeros(trainY.shape)
    for i in range(num_delta):
        deltaY = model.predict(trainX + deltaX[i,:])
        STSM += cost_func(deltaY, trainY)
    STSM = STSM / num_delta
    stsm_vector = STSM.mean(axis=0)
    return stsm_vector