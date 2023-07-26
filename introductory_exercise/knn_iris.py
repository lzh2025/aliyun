from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

# 数据集导入
iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

def euc_dis(x1, x2):
    '''
    :param x1: 样本1 array型
    :param x2: 样本2 array型
    :return: 样本1和2的欧式距离
    '''
    dist = np.sqrt(np.sum((x1 - x2)**2))
    return dist

def knn_classifier(X, Y, test_data, k):
    '''
    :param x: 训练集样本
    :param y: 训练集标签
    :param test_data: 测试集数据
    :param k: 根据k个邻居进行预测
    :return: 测试集预测标签
    '''
    # 计算test_data与训练集样本x的距离
    knn_dist = [euc_dis(x, test_data) for x in X]
    # 找出最近的k个元素
    # 根据dist排序，取前k个（0 → k-1）
    neighbors_k = np.argsort(knn_dist)[:k]
    # k个邻居的标签
    knn_target = Y[neighbors_k];
    # k邻居标签出现次数最多的即为预测标签
    # most_common返回的数据类型为[(标签，出现次数)]，取[0][0]即为所求
    return Counter(knn_target).most_common(1)[0][0]

predicts = [knn_classifier(X_train, Y_train, test_data, 3) for test_data in X_test]
correct = np.count_nonzero((predicts == Y_test) == True)
print("Accuracy is: %.3f" %(correct/len(X_test)))
