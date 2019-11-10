# 使用sklearn中的kNN分类器
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def get_data():
    raw_data_X = [[3.3935, 2.3312],
                  [3.1101, 1.7815],
                  [1.3438, 3.3684],
                  [3.5823, 4.6792],
                  [2.2804, 2.8670],
                  [7.4234, 4.6965],
                  [5.7451, 3.5340],
                  [9.1722, 2.5111],
                  [7.7928, 3.4241],
                  [7.9398, 0.7916]]
    raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    X_train = np.array(raw_data_X)
    y_train = np.array(raw_data_y)
    return X_train, y_train


def sk_classifier(k, X_train, y_train, X):
    kNN_classifier = KNeighborsClassifier(n_neighbors=k)
    kNN_classifier.fit(X_train, y_train)
    print(kNN_classifier.predict(X))


if __name__ == "__main__":
    X_train, y_train = get_data()

    print(X_train, y_train)

    x = np.array([[8.0936, 3.3657]])
    sk_classifier(6, X_train, y_train, x)
