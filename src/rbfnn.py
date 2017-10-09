#-*- coding: utf-8 -*- 


import numpy as np
import scipy.spatial.distance as sciDist
import sklearn.cluster as sklCluster


def kMeans(dataSet=None, k=None):
    kms = sklCluster.KMeans(n_clusters=k, n_jobs=1, random_state=0)
    kms.fit(dataSet)

    return kms


def numCenterAssign(numCenter, numSamples):
    # ct = np.mat(np.sum(trainY, axis = 0))
    # cn = np.mat(np.ones(np.shape(ct)))

    assert numSamples.shape[0] == 1
    assert numCenter > numSamples.shape[1]
    assert numCenter <= numSamples.sum()

    cn = np.ones(numSamples.shape, dtype=int)
    numCenter = numCenter - cn.sum()

    for i in range(numCenter):
        # res = ct/cn
        # idx = np.mat(np.nonzero(res[0,:] == np.max(res[0,:], axis = 1))[1])[0,0]
        cnf = cn.astype(float)
        idx = np.argmax(numSamples/cnf)
        # cn[0,idx] = cn[0,idx] + 1 # is cn proportional to number of every catogory in TrainY ?!!
        cn[0,idx] = cn[0,idx] + 1

    return cn.astype(int)


def innerCluster(trainX, trainY, numCenter, alpha):
    '''Center Assignment'''
    numSample = trainY.sum(axis=0).reshape((-1,1)).T
    cn = numCenterAssign(numCenter, numSample)

    '''define U'''
    U = np.zeros((numCenter, trainX.shape[1]))
    '''define V'''
    V = np.zeros((numCenter, 1))

    '''for debug'''
    centerLabel = np.zeros((numCenter, 1),dtype=int)

    begin, end = 0, 0
    for i in range(cn.shape[1]):
        end = begin + cn[0, i]
        centerLabel[begin:end,0] = i
        subX = trainX[trainY[:, i] == 1, :]

        '''compute U'''
        # u = kMeans(dataSet=subX, k=int(cn[0, i]))
        u = sklCluster.KMeans(n_clusters=int(cn[0, i]), n_jobs=1, random_state=0).fit(subX).cluster_centers_
        U[begin:end, :] = u


        '''bug bug bug bug'''
        '''the definition of radius of RBFNN'''

        '''compute V'''
        '''version 3.0'''
        # v = sciDist.cdist(subX, U[begin:end, :])
        # v = v.max(axis=0).reshape((-1,1)) # max() can be substituted with mean() or min()
        # V[begin:end, :] = v * alpha

        '''compute V'''
        '''version 2.0'''
        # v = sciDist.pdist(subX).max() # max() can be substituted with mean() or min()
        # V[begin:end, :] = v * alpha

        '''compute V'''
        '''version 4.0'''
        # v = sciDist.pdist(U[begin:end, :]).max() # max() can be substituted with mean() or min()
        # V[begin:end,:] = v * alpha

        begin = end

    assert begin == numCenter

    '''compute V'''
    '''version 1.0'''
    '''It seems that this version achieves the best performance! '''
    V[:,0] = sciDist.pdist(U).mean() * alpha # max() can be substituted with mean() or min()

    return U, V


def discritize(dataset, numClass=None, num_each_class=None, mode='kMeans'): 
    '''for regression problem purpose'''
    m,n = dataset.shape 
    out = np.zeros((m,1))
    
    flag = False 
    while not flag: 
        numClass -= 1

        kms = sklCluster.KMeans(n_clusters=numClass, n_jobs=-1-4, random_state=None)
        out = kms.fit_predict(dataset)
        print(out)
        n_lbs = len(set(out))

        count_each_class = np.histogram(out, numClass, range=(0,numClass))[0]
        print(count_each_class)
        flag = (count_each_class >= num_each_class).all()

        print(out.shape, n_lbs, numClass)
    
    ''' how to optimize the codes below '''
    # res[:,out] = 1 ?!!
    res = np.zeros((m,numClass))
    for i in range(m): 
        res[i,out[i]] = 1

    return res


class RBFNN(object):
    '''to provide the same api as sklearn'''
    # initialize the archietecture for Neural Network
    def __init__(self, indim=None, numCenter=None, outdim=None, alpha=1.0):
        # components 1
        self.indim = indim # K
        self.numCenter = numCenter # M 
        self.outdim = outdim # category 

        # components 2
        self.U = np.random.standard_normal((self.numCenter, self.indim)) 
        self.V = np.random.standard_normal((self.numCenter, 1))
        self.W = np.random.standard_normal([self.numCenter, self.outdim])
        
        '''new'''
        # components 3
        self.alpha = alpha #???
        self.func = self.run_rbfnn
        '''new'''
    
    def run_rbfnn(trainX, trainY, W, U, V):
        D = np.power(sciDist.cdist(trainX, U), 2)
        Z = np.exp(D/np.transpose((-2 * np.power(V, 2)))) # how to implement element-wise operation?
        A = np.dot(Z, W)
        assert np.shape(A) == np.shape(trainY)
        return A 

    # calculate activations
    def activCalc(self, trainX):
        Z = np.power(sciDist.cdist(trainX, self.U), 2)
        # A = np.exp(Z/np.transpose((-2 * np.power(self.V, 2)))) # how to implement element-wise operation?
        '''for debug'''
        # radius = -2.0 * np.power(self.V, 2).T
        # z = Z/radius
        # a = radius.sum()
        # b = z.sum()

        A = np.exp(Z/(-2.0 * np.power(self.V, 2).T))

        assert np.shape(A) == (np.shape(trainX)[0], self.numCenter)
        return A

    # calculate the parameters of hidden neurons in Network
    def centersCalc(self, trainX, trainY):
        self.U, self.V = innerCluster(trainX, trainY, self.numCenter, self.alpha)
        '''bug bug bug bug'''
        '''alpha'''
        '''bug bug bug bug'''
        assert np.shape(self.U) == (self.numCenter, np.shape(trainX)[1])
        assert np.shape(self.V) == (self.numCenter, 1)

    # classifier training
    def fit(self, trainX, trainY):
        self.centersCalc(trainX, trainY)
        H = self.activCalc(trainX)
        self.W = np.dot(np.linalg.pinv(H), trainY)
        '''for debug'''
        # a = self.W.sum()

        assert np.shape(self.W) == (self.numCenter, np.shape(trainY)[1])

    # classifier testing
    def predict(self, trainX):
        H = self.activCalc(trainX)
        output = np.dot(H, self.W)
        assert np.shape(output) == (np.shape(trainX)[0], self.outdim)
        return output