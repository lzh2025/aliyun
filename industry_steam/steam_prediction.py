import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# 忽略warining
import warnings
warnings.filterwarnings("ignore")
train_data_file = "./zhengqi_train.txt"
test_data_file = "./zhengqi_test.txt"
train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

## pandas输出数据集信息
# 基本信息
# train_data.info()
# test_data.info()
# 统计信息
# print(train_data.describe())
# print(test_data.describe())
# 字段信息 默认输出前五行
# print(train_data.head())
# print(test_data.head())

## 数据预处理
# 绘制箱型图
# 目的：寻找异常值
column = train_data.columns.tolist()[:39]
fig = plt.figure(figsize=(80, 60), dpi=75)
for i in range (38):
    plt.subplot(5, 8, i + 1)
    sns.boxplot(train_data[column[i]], orient="v", width=0.5)
    plt.ylabel(column[i])
plt.show()