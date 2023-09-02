## Feature Engineering
## 1.feature processing
#  1.1 Standardization
#  (x-mean)/std
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
iris = load_iris()
iris_norm = StandardScaler().fit_transform(iris.data)

#  1.2 Rescaling
#  (x-min)/(max-min)
from sklearn.preprocessing import MinMaxScaler
iris_rescale = MinMaxScaler().fit_transform(iris.data)

#  1.3 Normalization
#  x/||x|| l2-norm
from sklearn.preprocessing import Normalizer
iris_norm = Normalizer().fit_transform(iris.data)

#  1.4 Binarization
#  x > threshold: x=1; otherwise: x=0
from sklearn.preprocessing import Binarizer
iris_binary = Binarizer(threshold=3).fit_transform(iris.data)

#  1.5 Dummy Variable (one-hot vector)
from sklearn.preprocessing import OneHotEncoder
iris_one_hot = OneHotEncoder().fit_transform(iris.data)

#  1.6 Missing Data Imputation
from numpy import vstack, array, nan
from sklearn.impute import SimpleImputer

#  strategy为缺失值填充方式，默认为均值
#  vstack是按行堆叠数组（垂直），先创建临时对象array([nan, nan, nan, nan]人造了缺失值
#  再用SimpleImputer默认方式进行均值填充，最后结果第一行就是各列的均值
iris_impute = SimpleImputer().fit_transform(vstack((array([nan, nan, nan, nan]),
                                            iris.data)))

#  1.7 Data Transform
#  1.7.1 Polynomial Transformation
#  e.g. feature:(a, b) polynomial degree=2: (1, a, b, ab, a^2, b^2)
from sklearn.preprocessing import PolynomialFeatures
iris_poly = PolynomialFeatures().fit_transform(iris.data)

#  1.7.2 Log Transformation
#  log1p = log(x+1)
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
iris_log = FunctionTransformer(log1p).fit_transform(iris.data)

## 2.Dimensionality Reduction
#  2.1 Feature Selection
#  2.1.1 Variance Threshold
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
iris_con = np.concatenate([iris.data, iris.target.reshape(-1, 1)], axis=1)
iris_pd = pd.DataFrame(iris_con, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
iris_var_select = VarianceThreshold(threshold=3)
iris_var_select.fit_transform(iris_pd)
print(iris_var_select.feature_names_in_)
print(iris_var_select.get_feature_names_out())
#  2.1.2 SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
#  2.1.2.1 f_regression
#  选择K个最好的特征,返回选择特征后的数据
#  第一个参数为计算评估特征的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组,
#  数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#  参数k为选择的特征个数
iris_k_best_fg = SelectKBest(f_regression, k = 2).fit_transform(iris.data, iris.target)
print(iris_k_best_fg)
#  2.1.2.2 Chi-Square
from sklearn.feature_selection import chi2
iris_k_best_chi2 = SelectKBest(chi2, k = 2).fit_transform(iris.data, iris.target)
print(iris_k_best_chi2)
#  2.1.3 RFE 递归消除特征法
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#  递归特征消除法，返回特征选择后的数据
#  参数estimator为基模型
#  参数n_feature_to_select为选择的特征个数
iris_RFE = RFE(estimator=LogisticRegression(multi_class='auto',
                                            solver='lbfgs',
                                            max_iter=500),
               n_features_to_select=2).fit_transform(iris.data, iris.target)
print(iris_RFE)
#  2.1.4 SelectFromModel
#  主要采用基于模型的特征选择法，常见的有基于惩罚项的特征选择法和基于树模型的特征选择法
#  2.1.4.1 基于惩罚项的特征选择法
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
iris_l2 = SelectFromModel(LogisticRegression(penalty='l2', C=0.1, solver='lbfgs',
                                             multi_class='auto')).fit_transform(iris.data, iris.target)
print(iris_l2)
#  2.1.4.2 基于树模型的特征选择法
#  梯度提升迭代决策树 Gradient Boosting Decision Tree
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
iris_gbdt = SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)
print(iris_gbdt)

#  2.2 Linear Dimensionality Reduction
#  2.2.1 PCA主成分分析法
#  主成分分析法返回降维后的数据
#  参数n_components为主成分的数目
from sklearn.decomposition import PCA
iris_pca = PCA(n_components=2).fit_transform(iris.data)

#  2.2.1 LDA主成分分析法
# 1、相同点
# （1）两者的作用是用来降维的
# （2）两者都假设符合高斯分布
# 2、不同点
# （1）LDA是有监督的降维方法，PCA是无监督的
# （2）LDA降维最多降到类别数K-1的维数，PCA没有这个限制
# （3）LDA更依赖均值，如果样本信息更依赖方差的话，效果将没有PCA好
# （4）LDA可能会过拟合数据
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
iris_lda = LDA(n_components=2).fit_transform(iris.data, iris.target)

