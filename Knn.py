import math
import numpy as np

class KNearestNeighbors:

    def __init__(self, n_neighbors = 3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.target_values = len(np.unique(y))


    # tinh khoang cach euclid
    def euclid_distance (self, X1, X2):
        dist = 0
        for x1,x2 in zip(X1,X2):
            d = x1 - x2
            dist += d*d
        dist = pow(dist, 0.5)
        return dist

    # tim K neighbors gan voi diem can test nhat
    def neighbors (self, test):
        dist = list()
        index = list()
        # tinh khoang cach cua moi diem du lieu training
        for x in self.X_train:
            d = self.euclid_distance(test, x)
            dist.append(d)
        temp = list(dist)
        # sap xep khoang cach theo chieu tang dan
        temp.sort()
        # tim chi so cua K diem co khoang cach nho nhat trong tap train
        for i in range(self.n_neighbors):
            value = temp[i]
            ind = dist.index(value)
            index.append(ind)
        return index


    # dua tren K diem gan nhat va label cua chung, du doan label cua du lieu can test
    def predict(self, neighbors):
        count = list()
        # so luong cac label
        # count[i] tuong ung voi so lan xuat hien cua label i trong K diem du lieu gan nhat
        for i in range(self.target_values):
            count.append(0)
        for neighbor in neighbors:
            target = self.y_train[neighbor]
            count[target] += 1
        # chon ra label xuat hien nhieu nhat
        prediction = count.index(max(count))
        return prediction


    def score(self, X, y):
        predictions = list()  # predictions cua moi du lieu trong tap test
        count = 0  # so lan du doan dung
        for i in range(len(X)):
            neighbors = self.neighbors(X[i])
            predict = self.predict(neighbors)
            predictions.append(predict)
            # neu du doan dung, count+=1
            if (predict == y[i]):
                count += 1
        # tinh xac suat du doan dung
        _score = (count / (len(y) * 1.0)) * 100
        return _score













