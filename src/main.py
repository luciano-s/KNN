import numpy as np
from random import randint
from knn import KNN
import time

def get_classes(Y):
    result = []
    for y in Y:
        if y not in result:result.append(y)

    return result


def test_accuracy(X_train, y_train, X_test, y_test, size, K):
    if size > 1000:
        size = 1000
    k_nn = KNN(X_train, y_train, get_classes(y_train), K=k)
    t1 = time.process_time()
    result = k_nn.evaluate_accuracy(X_test, y_test, size)
    t2 = time.process_time()
    print(f'Accuracy: {result}')
    print(f'tempo de execução: {t2-t1}')

def run_knn(X_train, y_train, X_test, y_test, k):
    k_nn = KNN(X_train, y_train, get_classes(y_train), K=k)
    index = randint(0, len(X_test))
    value = X_test[index]
    k_nn.visualize_img(value)
    predicted = k_nn.predict(value)
    
    print(f'Predicted class: {predicted}')
    print(f'Correct class: {y_test[index]}')

if __name__ == "__main__":
    # load data

    X_train = np.load("../MNIST/train_data.npy")
    y_train = np.load("../MNIST/train_labels.npy")
    X_test = np.load("../MNIST/test_data.npy")
    y_test = np.load("../MNIST/test_labels.npy")


    print('Number of neighbors: ')
    k = int(input())

    print('1 - Test accuracy')
    print('2 - Run KNN for a specific value')
    print('E - Exit')
    op = input()
    
    if op == '1':
        print('Running. . .')
        print(f'Test set size (MAX_SIZE = {len(y_test)}): ')
        size = int(input())
        test_accuracy(X_train, y_train, X_test, y_test, size, k)
    
    elif op =='2':
        print('Running. . .')
        run_knn(X_train, y_train, X_test, y_test, k)

    else:
        exit()



