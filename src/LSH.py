import numpy as np

from scipy import sparse

N = 10000

class LSH:
    def __init__(self, dataset, d, k):
        self.dataset = dataset
        self.hash = self.makeHash(d, k)

    def makeHash(self, dimensionOfInput, dimensionOfOutput):
        weight = np.random.randn(dimensionOfOutput, dimensionOfInput)
        hash = np.floor(self.dataset @ weight.T)

        return hash

    def query(self, queryIndex):
        distanceOfVectors = np.sum((self.hash - self.hash[queryIndex]) ** 2, axis = 1)

        return np.argsort(distanceOfVectors)[1:201]

    def findAP(self, prediction, truth):
        precision = []

        for index in range(0, 200):
            intersect = len(np.intersect1d(prediction[:index], truth))
            precision.append(intersect / (index + 1))

        return np.mean(precision)

    def findmAP(self):
        randomIndex=np.random.choice(N, 100)
        mAP = 0
        for index in randomIndex:
            distanceOfVectors = np.sum((self.dataset - self.dataset[index]) ** 2, axis = 1)
            mAP += self.findAP(self.query(index), np.argsort(distanceOfVectors)[1:201])

        return mAP

class FlyLSH(LSH):
    def __init__(self, dataset, d, k):
        self.dataset = dataset
        self.k = k
        self.hash = self.makeHash(d, 10 * d)

    def makeHash(self, dimensionOfInput, dimensionOfOutput):
        weight = np.zeros((dimensionOfOutput,dimensionOfInput))

        for i in range(0, dimensionOfOutput):
            weight[i, :] = sparse.random(1, dimensionOfInput, density = 0.1, data_rvs = np.ones).toarray()

        hash = self.dataset @ weight.T

        H = np.zeros((N, dimensionOfOutput))
 
        for i in range(N):
            threshold = np.sort(hash[i, :])[-self.k] 
            H[i, :] = np.floor(hash[i, :] * (hash[i, :] >= threshold))

        return H