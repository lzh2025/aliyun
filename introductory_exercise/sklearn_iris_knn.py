from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target
# 调用KNN模块
knn = KNeighborsClassifier(n_neighbors=1)
# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y)
# 利用训练集训练模型
knn.fit(X_train, y_train)
# 利用测试集做预测且计算准确率
correct = np.count_nonzero((knn.predict(X_test) == y_test) == True)
print ("Accuracy is: %.3f" %(correct/len(X_test)))