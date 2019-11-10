# 实现一个自己的kNN分类器
import numpy as np
from collections import Counter
from matrics import get_accuracy


class KNNClassifier:

    # 初始化KNN分类器
    def __init__(self, k):
        assert k >= 1, "k必须为合法值"
        self.k = k
        # 以_开头代表私有变量，外界不能访问
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """
        根据训练集训练分类器
        :param X_train: 用户传入的训练集特征值
        :param y_train: 用户传入的训练集目标值
        :return: self自身对象
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "训练集X必须和y的大小一致"
        assert self.k <= X_train.shape[0], \
            "训练集X必须至少k个样本"

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """
        预测函数
        :param X_predict: 待预测数据集
        :return: 对单个向量预测结果的数组
        """
        assert self._X_train is not None and self._y_train is not None, \
            "在预测前必须先训练"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "预测数据的特征数必须和训练集X的一致"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """
        对一个向量进行预测
        :param x: 需要预测的单个向量
        :return:
        """
        assert x.shape[0] == self._X_train.shape[1], \
            "预测数据的特征数必须和训练集X的一致"
        distances = [((np.sum((x_train - x) ** 2)) ** 0.5) for x_train in self._X_train]
        nearest = np.argsort(distances)

        top_K = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(top_K)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return get_accuracy(y_test, y_predict)


def loadData():
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


if __name__ == "__main__":
    # 获取数据集
    X_train, y_train = loadData()

    # 待预测数据
    x = [8.0936, 3.3657]
    # 将待预测数据转换成数组
    X_predict = np.array([x])

    knn_clf = KNNClassifier(k=6)
    knn_clf.fit(X_train, y_train)
    y_predict = knn_clf.predict(X_predict)
    print(y_predict[0])
