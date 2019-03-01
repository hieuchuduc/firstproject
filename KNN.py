from sklearn import datasets
import math
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
iris_x = iris.data   #tap du lieu
iris_y = iris.target   #tap label
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=50)  # train:test = 100:50
print ("Training size: %d" %len(y_train))
print ("Test size    : %d" %len(y_test))

# tinh khoang cach euclid
def Euclid_distance (x1, x2):
    dist = 0
    for i in range(len(x1)):
        d = x1[i] - x2[i]
        dist += d*d
    dist = pow(dist, 0.5)
    return dist

# tim K neighbors gan voi diem can test nhat
def Neighbors (x_train, test, K):
    dist = list()
    neighbors = list()
    # tinh khoang cach cua moi diem du lieu training
    for i in range(len(x_train)):
        d = Euclid_distance(test, x_train[i])
        dist.append(d)
    temp = list(dist)
    # sap xep khoang cach theo chieu tang dan
    temp.sort()
    # tim chi so cua K diem co khoang cach nho nhat trong tap train
    for i in range(K):
        value = temp[i]
        index = dist.index(value)
        neighbors.append(index)
    return neighbors


# dua tren K diem gan nhat va label cua chung, du doan label cua du lieu can test
def Predict(y_train, neighbors, target_number):
    count = list()
    # so luong cac label, o day la 3 loai hoa
    # count[i] tuong ung voi so lan xuat hien cua label i trong K diem du lieu gan nhat
    for i in range(target_number):
        count.append(0)
    for i in range(len(neighbors)):
        target = y_train[neighbors[i]]
        count[target] += 1
    # chon ra label xuat hien nhieu nhat
    predict = count.index(max(count))
    return predict

# KNN
K = 3   # chon K=3
predict = list()    # prediction cua moi du lieu trong tap test
count = 0    # so lan du doan dung
for i in range(len(x_test)):
    test = x_test[i]
    neighbors = Neighbors(x_train, test, K)
    result = Predict(y_train, neighbors, len(iris.target_names))
    predict.append(result)
    # neu du doan dung, count+=1
    if (result == y_test[i]):
        count += 1
print(predict)
print (y_test)
# tinh xac suat du doan dung
accuracy = (count/(len(y_test)*1.0))*100
print ("Accuracy: ", accuracy, '%')

#my_name_is_hieu






