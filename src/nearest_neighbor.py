import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter


def distance(u, v):
    """ Returns the L2 distance between two vectors """
    d = 0
    for i in range(len(u)):
        d += (u[i] - v[i]) ** 2
    return d ** (1 / 2)


def get_min(X, neighbors=[]):
    """ Returns the position (index) of the smallest element, checking if it is
    not in the neighbors list (already calcualted)
    """

    minimum = float("inf")
    pos = 0
    for i in range(0, len(X)):
        if X[i] < minimum and i not in neighbors:
            minimum = X[i]
            pos = i

    return pos


def KNN(x, x_train, y_train, K=1):
    """Returns the position of KNN guess for the handwritten digit"""

    print("Running. . .")

    neighbors = []
    if K == 1:
        distances = [distance(x, x_train[i]) for i in range(0, len(x_train))]
        smallest_index = get_min(distances)

        return smallest_index
    else:
        result = []
        for k in range(0, K):
            distances = [distance(x, x_train[i]) for i in range(0, len(x_train))]
            smallest_index = get_min(distances, neighbors)
            neighbors.append(smallest_index)
            result.append(y_train[smallest_index])
        c = Counter(result)
        # print(c)
        # print(neighbors)
        # print(result)
        # print(max(c, key=c.get))
        return max(c, key=c.get)


def visualize_img(X):
    """Given a vector (1x728) it reshapes to a 28x28 image and plots it"""

    x = np.reshape(X, (28, 28))
    plt.imshow(x, cmap=plt.cm.gray)
    plt.show()


if __name__ == "__main__":

    # load data

    X_train = np.load("../MNIST/train_data.npy")
    y_train = np.load("../MNIST/train_labels.npy")
    X_test = np.load("../MNIST/test_data.npy")
    y_test = np.load("../MNIST/test_labels.npy")

    # end load data

    # generates a random index to test the KNN
    index = random.randint(0, len(X_test) - 1)
    visualize_img(X_test[index])
    print(f"Random index: {index}")

    # runs KNN for the test_example using the index generated above
    i = KNN(X_test[index], X_train, y_train, 4)

    print(f"KNN result: {y_train[i]} Expected result: {y_test[index]}")

