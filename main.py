import math

from PIL import Image
import numpy as np
import scipy.io
import glob
import matplotlib.pyplot as plt
import random


# import plotly.express as px


def LoadPictures(directory):
    images = []
    for path in sorted(glob.glob(directory)):
        image = Image.open(path)
        image = np.asarray(image)
        # image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
        images.append(image)
    return np.array(images, dtype=object)


def LoadGroundTruth(directory):
    # Needs Enhancing
    images = []
    for path in sorted(glob.glob(directory)):
        image = scipy.io.loadmat(path)
        images.append(image)
    return np.array(images)


def KMeans(image, k):
    shape = image.shape
    image = image.reshape(shape[0] * shape[1], shape[2])
    size = len(image)
    centroid = []
    for i in range(k):
        centroid.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    Distance = np.zeros((size, k))
    label = np.zeros(size, dtype=int)
    while True:
        for i in range(size):
            for j in range(k):
                Distance[i][j] = math.dist(image[i], centroid[j])
        for i in range(size):
            label[i] = np.argmin(Distance[i])
        cluster = []
        for i in range(k):
            cluster.append([])
        for i in range(size):
            cluster[label[i]].append(image[i])
        oldCentroid = centroid.copy()
        for i in range(k):
            if not len(cluster[i]):
                centroid[i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            else:
                centroid[i] = np.mean(cluster[i], axis=0)
        Difference = np.zeros(k)
        for i in range(k):
            Difference[i] = math.dist(oldCentroid[i], centroid[i])
        if np.mean(Difference) <= 0.05:
            break
    segmentation = label.reshape(shape[0], shape[1])
    plt.figure()
    plt.imshow(segmentation)
    plt.show()


# def Visulaize(matfile):


if __name__ == '__main__':
    Data = LoadPictures("Dataset/BSR/BSDS500/data/images/train/*.jpg")
    print(Data[0].shape)
    # Data2 = LoadGroundTruth("Dataset/BSR/BSDS500/data/groundTruth/train/*.mat")
    # x = Data2[0]['groundTruth'][0][1][0][0][0]
    # print(np.array(x).shape)
    # print(x)
    # for i in range(len(Data2[0]['groundTruth'][0])):
    #     z = Data2[5]['groundTruth'][0][i][0][0][0]
    #     z = np.array(z)
    #     print(z.shape)
    #     print(np.unique(z))
    #     plt.figure()
    #     plt.title(i)
    #     plt.imshow(z)
    # plt.imshow(Data[0])
    # fig = plt.figure()
    # size = fig.get_size_inches() * fig.dpi  # size in pixels
    # plt.show()
    # print(size)
    KMeans(Data[0], 7)
