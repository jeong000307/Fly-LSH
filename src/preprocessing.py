import os
import sys
import random
import pandas as pd
import numpy as np

from struct import *
from scipy import spatial

def readUbyte(filePath):
    images = []

    imageFile = open(filePath, 'rb')
    imageFile.read(16)

    while True:
        imageBuffer = imageFile.read(28 * 28)

        if not imageBuffer:
            break;

        imageBuffer = np.reshape(unpack(len(imageBuffer) * 'B', imageBuffer), (28 * 28))
        images.append(imageBuffer)

    return np.array(images)

def readFvecs(filePath):
    vectors = np.fromfile(filePath, dtype = np.float32)

    dimensionOfVector = vectors.view(np.int32)[0]
    vectors = vectors.reshape(-1, 1 + dimensionOfVector)[:, 1:]

    randomIndex = np.random.randint(vectors.shape[0], size = 10000)
    vectors = vectors[randomIndex, :]
    
    return np.array(vectors)

def readTxt(filePath):
    words = []

    wordFile = open(filePath, 'rb')

    while True:
        wordBuffer = wordFile.readline()

        if not wordBuffer:
            break;

        words.append(np.array(wordBuffer.split()[1:], dtype = np.float32))

    words = np.array(words)
    randomIndex = np.random.randint(words.shape[0], size = 10000)
    words = words[randomIndex, :]
    
    return np.array(words)

def preprocessVector(vectors):

    for i in range(vectors.shape[0]):
        vectors[i, :] += abs(min(vectors[i, :]))
        vectors[i, :] = vectors[i, :] / np.mean(vectors[i, :])

    return vectors

if __name__ == '__main__':

    sift = readFvecs("../raw/sift_base.fvecs")
    glove = readTxt("../raw/glove.42B.300d.txt")
    mnist = readUbyte("../raw/t10k-images.idx3-ubyte")
    gist = readFvecs("../raw/gist_base.fvecs")

    print("reading complete")
    
    pd.DataFrame(preprocessVector(sift)).to_csv("../data/sift.csv", index = False, header = False)
    pd.DataFrame(preprocessVector(glove)).to_csv("../data/glove.csv", index = False, header = False)
    pd.DataFrame(preprocessVector(mnist)).to_csv("../data/mnist.csv", index = False, header = False)
    pd.DataFrame(preprocessVector(gist)).to_csv("../data/gist.csv", index = False, header = False)

    print("writing complete")