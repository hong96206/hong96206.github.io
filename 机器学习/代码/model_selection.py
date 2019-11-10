# 自定义的数据集划分
import numpy as np
from sklearn import datasets
from knn import KNNClassifier  # 使用自己的分类器


def shuffle_index(len):
    """
    对len个索引进行乱序化
    :param len: 要乱序化的索引个数
    :return: 乱序完的索引数组
    """
    new_index = np.random.permutation(len)
    return new_index


def train_test_split(X, y, test_radio, seed=None):
    """
    划分数据集
    :param X: 传入的数据集的特征值
    :param y: 传入的数据集的目标值
    :param test_radio: 测试样本的比例
    :param seed: 随机种子，默认为None
    :return: X_train, X_test, y_train, y_test
    """
    assert X.shape[0] == y.shape[0], "X的大小必须和y的一致"
    assert 0.0 <= test_radio <= 1.0, "测试比例必须合法(0-1之间)"

    if seed:
        np.random.seed(seed)

    new_index = shuffle_index(len(X))

    # 测试数据集个数
    test_size = int(test_radio * len(X))
    test_index = new_index[:test_size]
    train_index = new_index[test_size:]

    X_test = X[test_index]
    y_test = y[test_index]
    X_train = X[train_index]
    y_train = y[train_index]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # print(X.shape)  # (150, 4)
    # print(y.shape)  # (150,)
    test_radio = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_radio, None)
    my_knn_clf = KNNClassifier(k=3)
    my_knn_clf.fit(X_train, y_train)
    y_predict = my_knn_clf.predict(X_test)
    print('预测结果集：', y_predict)
    print('实际结果集：', y_test)
    accuracy_rate = sum(y_predict == y_test) / len(y_test)
    print('预测准确率：', accuracy_rate)
