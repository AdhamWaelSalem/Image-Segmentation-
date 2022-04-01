import math

from PIL import Image
import numpy as np
import scipy.io
import glob
import matplotlib.pyplot as plt
import matplotlib.image as im
import random
import math
from sklearn.metrics import f1_score
from scipy.stats import entropy
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import contingency_matrix
import pandas as pd


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


def KMeans(image, K):
    size = len(image)
    for y in range(size):
        for k in K:
            shape = image[y].shape
            image_temp = image[y].reshape(shape[0] * shape[1], shape[2])
            size = len(image_temp)
            centroid = []
            for i in range(k):
                centroid.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            Distance = np.zeros((size, k))
            label = np.zeros(size, dtype=int)
            while True:
                for i in range(size):
                    for j in range(k):
                        Distance[i][j] = np.linalg.norm(image_temp[i] - centroid[j])
                for i in range(size):
                    label[i] = np.argmin(Distance[i])
                cluster = []
                for i in range(k):
                    cluster.append([])
                for i in range(size):
                    cluster[label[i]].append(image_temp[i])
                oldCentroid = centroid.copy()
                for i in range(k):
                    if not len(cluster[i]):
                        centroid[i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                    else:
                        centroid[i] = np.mean(cluster[i], axis=0)
                Difference = np.zeros(k)
                for i in range(k):
                    Difference[i] = math.dist(oldCentroid[i], centroid[i])
                if np.mean(Difference) <= 0.5:
                    break
            # segmentation = []
            # for i in range(size):
            #     segmentation.append(np.array(centroid[label[i]], dtype=int))
            segmentation = label.reshape(shape[0], shape[1])
            # plt.figure()
            # plt.title(f'{k}-Means')
            # plt.imshow(segmentation)
            im.imsave(f'BigPicture/{y}_{k}_Means.png', segmentation)
            print(f"Done Picture Number {y} ---> {k}-Means")
            # plt.show()


def normalizedCut(images, s):
    size = len(images)
    dimension = images[0].shape[0] * images[0].shape[1]
    distances = np.zeros((dimension, dimension))
    similarity = np.zeros((dimension, dimension), dtype=np.int8)
    degree = np.zeros((dimension, dimension), dtype=np.int8)
    for k in range(size):
        shape = images[k].shape
        images[k] = images[k].reshape(shape[0] * shape[1], shape[2])
        for i in range(dimension):
            for j in range(dimension):
                diff = images[k][i] - images[k][j]
                diff = np.array(diff)
                if np.all((diff == 0)):
                    distances[i][j] = 0
                else:
                    # distances[i][j] = np.linalg.norm(images[k][i] - images[k][j])
                    distances[i][j] = math.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
        for i in range(dimension):
            furthest = np.amax(distances[i])
            j = 0
            while j < s:
                index = np.argmin(distances[i])
                distances[i][index] = furthest + 1
                if index != i:
                    similarity[i][index] = 1
                    j += 1
        for i in range(dimension):
            degree[i][i] = np.sum(similarity[i])
        L = degree - similarity
        Dinv = np.linalg.inv(degree)
        La = np.dot(Dinv, L)
        eigenValues, eigenVectors = np.linalg.eigh(La)
        start = 0
        while eigenValues[start] <= 0:
            start += 1
        K_EigenVectors = eigenVectors[:, range(start, s + start)]
        for i in range(dimension):
            K_EigenVectors[i] = K_EigenVectors[i] / np.linalg.norm(K_EigenVectors[i])
        return K_EigenVectors
        # kmeans = KMeans(n_clusters=s, random_state=0).fit(K_EigenVectors)
        # # label = KMean_NormalizedCut(K_EigenVectors, s)
        # label = kmeans.labels_
        # label = label.reshape(shape[0], shape[1])
        # im.imsave(f'/content/gdrive/MyDrive/Assignment2/BigPicture/NormalizedCut/{k}_{s}-NN.png', label)
        # return K_EigenVectors, label


def KMean_NormalizedCut(K_EigenVectors, k):
    size = len(K_EigenVectors)
    centroid = []
    for i in range(k):
        centroid.append(K_EigenVectors[random.randint(0, size - 1)].copy())
    Distance = np.zeros((size, k))
    label = np.zeros(size, dtype=int)
    while True:
        for i in range(size):
            for j in range(k):
                Distance[i][j] = np.linalg.norm(K_EigenVectors[i] - centroid[j])
        for i in range(size):
            label[i] = np.argmin(Distance[i])
        cluster = []
        for i in range(k):
            cluster.append([])
        for i in range(size):
            cluster[label[i]].append(K_EigenVectors[i])
        oldCentroid = centroid.copy()
        for i in range(k):
            if not len(cluster[i]):
                centroid[i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            else:
                centroid[i] = np.mean(cluster[i], axis=0)
        Difference = np.zeros(k)
        for i in range(k):
            Difference[i] = np.linalg.norm(oldCentroid[i] - centroid[i])
        if np.mean(Difference) <= 0.5:
            break
    return label


def reduceClusters(gtlabels, names, k):
    gtlabels = gtlabels.copy()
    if len(names) >= k:
        clusters = []
        for i in range(k - 1):
            clusters.append(names[i])
        for i, n in enumerate(gtlabels):
            if n not in clusters:
                gtlabels[i] = names[k - 1]
    return gtlabels


def getLabels(image):
    label = []
    test = [image[0][0].tolist()]
    h = 2
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(len(test)):
                if image[i][j].tolist() == test[k]:
                    label.append(k + 1)
            if not image[i][j].tolist() in test:
                test.append(image[i][j].tolist())
                label.append(h)
                h += 1
    return label


def F_Measure(Truth, Predicted):
    Contingency = contingency_matrix(Truth, Predicted)
    rows = len(Contingency)
    measures = []
    for i in range(rows):
        index = np.argmax(Contingency[i])
        Precision = Contingency[i][index] / np.sum(Contingency[i])
        Recall = Contingency[i][index] / np.sum(Contingency.transpose()[i])
        measures.append((2 * Precision * Recall) / (Precision + Recall))
    F = np.mean(measures)
    return F


def Conditional_Entropy(Truth, Predicted):
    Contingency = contingency_matrix(Truth, Predicted)
    rows = len(Contingency)
    conditional_entropy = []
    total = np.sum(Contingency)
    for i in range(rows):
        cluster_size = np.sum(Contingency[i])
        temp = 0
        for j in range(len(Contingency[i])):
            probability = Contingency[i][j] / cluster_size
            if probability == 0:
                continue
            temp -= probability * math.log2(probability)
        conditional_entropy.append(temp * (cluster_size / total))
    CE = np.sum(conditional_entropy)
    return CE


def GenerateMeasures():
    K = [3, 5, 7, 9, 11]
    FM = []
    CE = []
    for i in range(2):
        FM_PIC = []
        CE_PIC = []
        for j in range(len(matFiles[i]['groundTruth'][0])):
            FM_GT = []
            CE_GT = []
            gtLabels = matFiles[i]['groundTruth'][0][j][0][0][0].copy()
            values, names = zip(
                *sorted(
                    zip(
                        np.unique(gtLabels, return_counts=True)[1],
                        np.unique(gtLabels, return_counts=True)[0]),
                    reverse=True)
            )
            values = list(values)
            names = list(names)
            gtLabels = gtLabels.ravel()
            for k in range(5):
                newGT = reduceClusters(gtLabels, names, K[k])
                image = Image.open(f'K_Means/{i}_{K[k]}_Means.png')
                image = np.asarray(image)
                imageLabels = getLabels(image)
                FM_GT.append(F_Measure(newGT, imageLabels))
                CE_GT.append(Conditional_Entropy(newGT, imageLabels))
            FM_PIC.append(FM_GT)
            CE_PIC.append(CE_GT)
        FM.append(FM_PIC)
        CE.append(CE_PIC)
    FM = np.array(FM, dtype=object)
    CE = np.array(CE, dtype=object)
    return FM, CE


if __name__ == '__main__':
    Data = LoadPictures("Dataset/BSR/BSDS500/data/images/test/*.jpg")
    matFiles = LoadGroundTruth("Dataset/BSR/BSDS500/data/groundTruth/test/*.mat")
    # list = []
    # for i in range(len(matFiles)):
    #     list.append(len(matFiles[i]['groundTruth'][0]))
    # print(np.unique(list))

    FM, CE = GenerateMeasures()
    print(FM)
    # columns = ['GroundTruth 1',
    #            'GroundTruth 2',
    #            'GroundTruth 3',
    #            'GroundTruth 4',
    #            'GroundTruth 5',
    #            'GroundTruth 6',
    #            'GroundTruth 7',
    #            'GroundTruth 8']
    # dataset = pd.DataFrame(FM, columns=columns)
    # print(dataset)

    # dataset = pd.DataFrame({'GroundTruth 1': FM[:, 0],
    #                         'GroundTruth 2': FM[:, 1],
    #                         'GroundTruth 3': FM[:, 2],
    #                         'GroundTruth 4': FM[:, 3],
    #                         'GroundTruth 5': FM[:, 4],
    #                         'GroundTruth 6': FM[:, 5],
    #                         'GroundTruth 7': FM[:, 6]})
    # print(dataset)
    # A = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    # C = [1, 1, 1, 1, 1, 2, 1, 3, 3, 2, 2, 2, 2, 1, 3, 1, 3, 3]
    # F = F_Measure(A, C)
    # print(f"F_Measures: {F}")
    # print(f"Library: {f1_score(A, C, average='macro')}")
    #
    # CT = Conditional_Entropy(A, C)
    # print(f"Conditional Entropy: {CT}")
