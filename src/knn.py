import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

class KNN:
    

    def __init__(self, x_train, y_train, classes, K=1):
        self.K = K
        self.neighbors = []
        self.memory = dict(zip([i for i in range(len(x_train))], x_train) )
        self.X = x_train[:]
        self.Y = y_train[:]
        self.classes = y_train[:]
        self.function = lambda x,y: np.sum( [(x[i]-y[i])**2 
            for i in range(0, len(x))] )**(1/2)
        

    def set_distance_fuction(self, function):
        self.function = function


    def get_distance(self, vector_1, vector_2):
        
        # lambda implements norma L2

        if len(vector_1) != len(vector_2):
            return None
        else:
            return self.function(vector_1, vector_2)

    def predict(self, x):
        keys = self.memory.keys()
        
        if self.K == 1:
            values = [(self.get_distance(x, self.X[key]), key, self.X[key])
                for key in keys]
            values.sort()
            self.neighbors.append((values[0][1], values[0][2]))
        else:
            for k in range(self.K):
                values = [(self.get_distance(x, self.X[key]), key, self.X[key])
                    for key in keys]
                values.sort()
                
                self.neighbors.append( (values[0][1], values[0][2]) )

  
        
        
        c = Counter([items[0] for items in self.neighbors])
        
        
        
        index = max(c, key=c.get)
        self.neighbors = []
        return self.Y[index]

        


    
    def visualize_img(self, X):
        """Given a vector (1x728) it reshapes to a 28x28 image and plots it"""

        x = np.reshape(X, (28, 28))
        plt.imshow(x, cmap=plt.cm.gray)
        plt.show()

    def evaluate_accuracy(self, test_data, test_label, size):
        acc = 0
        inf = random.randint(0, len(test_data)-size)
        for i in range(inf, inf+size):
            pred = self.predict(test_data[i])

            print(f'Predicted: {pred}')
            print(f'Expected: {test_label[i]}')
            if pred == test_label[i]:
                print('Igual') 
                acc +=1
        return acc/size
