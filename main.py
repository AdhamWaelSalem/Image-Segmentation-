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

def normalizedCut(images, s):
    size = len(images)
    for i in range(size):
        images[i] = images
    dimension = images[0].shape[0] * images[0].shape[1]
    distances = np.zeros((dimension, dimension))
    similarity = np.zeros((dimension, dimension))
    for k in range(size):
        for i in range(dimension):
            for j in range(dimension):
                distances[i][j] = math.dist(images[k][i], images[k][j])
        for i in range(dimension):
            furthest = np.amax(dimension[i])
            for j in range(s+1):
                index = np.argmin(distances[i])
                distances[i][index] = furthest
                if index != i:
                    similarity[i][index] = 1


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
                        Distance[i][j] = math.dist(image_temp[i], centroid[j])
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


# def Visualize(matfile):
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


if __name__ == '__main__':
    # Data = LoadPictures("Dataset/BSR/BSDS500/data/images/5images/*.jpg")
    # Data2 = LoadGroundTruth("Dataset/BSR/BSDS500/data/groundTruth/5images/*.mat")
    # # image = Image.open('K_Means/0_11_Means.png')
    # # image = np.asarray(image)
    #
    # for i in range(len(Data2)):
    #     for j in range(len(Data2[i]['groundTruth'][0])):
    #         x = Data2[i]['groundTruth'][0][j][0][0][0]
    #         im.imsave(f'BigPicture/GroundTruth/{i}_{j}_GT.png', x)

    # print(np.array(x).shape)
    # for i in range(x.shape[0])
    # print(x.tolist())
    # for i in range(len(Data2[0]['groundTruth'][0])):
    #     z = Data2[5]['groundTruth'][0][i][0][0][0]
    #     z = np.array(z)
    #     print(z.shape)
    #     print(np.unique(z))
    #     plt.figure()
    #     plt.title(i)
    #     plt.imshow(z)
    # plt.imshow(x)
    # fig = plt.figure()
    # size = fig.get_size_inches() * fig.dpi  # size in pixels
    # plt.show()
    # print(size)
    # k = [3, 5, 7, 9, 11]

    # label = []
    # test = [image[0][0].tolist()]
    # h = 2
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         for k in range(len(test)):
    #             if image[i][j].tolist() == test[k]:
    #                 label.append(k+1)
    #         if not image[i][j].tolist() in test:
    #             test.append(image[i][j].tolist())
    #             label.append(h)
    #             h += 1

    # k = 1
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         for k in range(len(test)):
    #             if image[i][j].tolist() == test[k]:
    #                 label.append(k)
    #         if not image[i][j].tolist() in test:
    #             test.append(image[i][j].tolist())
    #             label.append(k)
    #             k += 1

    # fScore = f1_score(x, label, average='weighted')
    # label += 1
    # print(label)
    # print(fScore)
    # KMeans(Data, [5])
    # normalizedCut([Data[0]], 5)
    array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    array = np.array(array)




