from sklearn.datasets import load_iris
iris = load_iris()
print(type(iris))
print(iris.feature_names)
print(iris.target_names)
# 在scikit-learn中对数据有如下要求：
# 特征和标签要用分开的对象存储
# 特征和标签要是数字
# 特征和标签都要使用numpy的array来存储
print(type(iris.data))
print(type(iris.target))
# store features matrix in "X"
X = iris.data
# store response vector in "y"
y = iris.target

import matplotlib.pyplot as plt
X_sepal = X[:, :2]
plt.scatter(X_sepal[:, 0], X_sepal[:, 1], c=y, cmap=plt.cm.gnuplot)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()
X_petal = X[:, 2:4]
plt.scatter(X_petal[:, 0], X_petal[:, 1], c=y, cmap=plt.cm.gnuplot)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()