import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools.detect_outliers import detect_outliers
from scipy import stats
# 忽略warining
import warnings
warnings.filterwarnings("ignore")
train_data_file = "data/zhengqi_train.txt"
test_data_file = "data/zhengqi_test.txt"
train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

## pandas输出数据集信息
# 基本信息
train_data.info()
test_data.info()
# 统计信息
print(train_data.describe())
print(test_data.describe())
# 字段信息 默认输出前五行
print(train_data.head())
print(test_data.head())

### 数据预处理
##  绘制箱型图
#   目的：检测数据分布中是否存在较多异常值
column = train_data.columns.tolist()[:39]
fig = plt.figure(figsize=(80, 60), dpi=150)
for i in range (38):
    plt.subplot(5, 8, i + 1)
    sns.boxplot(train_data[column[i]], orient="v", width=0.5)
    plt.ylabel(column[i])
plt.tight_layout()
plt.savefig("fig/boxplot.png")

## 利用模型预测找出异常值
#  岭回归模型找异常值
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
X_train = train_data.iloc[:, 0:-1]
y_train = train_data.iloc[:, -1]
outliers = detect_outliers(Ridge(), X_train, y_train)

## 查看数据分布，观察是否符合正态分布
#  通过直方图查看统计分布，通过Q-Q图查看分布是否近似正态分布
#  一共有38个特征，每个特征两个图，拆分为两张图保存
train_col = 6
train_rows = 7
plt.figure(figsize=(8*train_rows, 8*train_col), dpi=100)
i = 0
for col in train_data.columns[:len(train_data.columns)//2]:
    i += 1
    ax = plt.subplot(train_rows, train_col, i)
    sns.distplot(train_data[col], fit=stats.norm)
    i += 1
    ax = plt.subplot(train_rows, train_col, i)
    stats.probplot(train_data[col], plot=plt)
plt.tight_layout()
plt.savefig('fig/Q-Q_fig_(V0-V18).png')

plt.figure(figsize=(8*train_rows, 8*train_col), dpi=100)
i = 0
for col in train_data.columns[len(train_data.columns)//2:-1]:
    i += 1
    ax = plt.subplot(train_rows, train_col, i)
    sns.distplot(train_data[col], fit=stats.norm)
    i += 1
    ax = plt.subplot(train_rows, train_col, i)
    stats.probplot(train_data[col], plot=plt)
plt.tight_layout()
plt.savefig('fig/Q-Q_fig_(V19-V37&target).png')

## 绘制KDE分布图
#  对比训练集和测试集特征变量分布情况
#  V5 V9 V11 V17 V21 V22训练集与测试集差异太大，考虑删除
dist_rows = 6
dist_cols = 7
plt.figure(figsize=(8*dist_rows, 8*dist_cols), dpi=100)
i = 0
for col in test_data.columns:
    i += 1
    ax = plt.subplot(dist_rows, dist_cols, i)
    ax = sns.kdeplot(train_data[col], color='Red', shade=True)
    ax = sns.kdeplot(test_data[col], color='Blue', shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])
plt.tight_layout()
plt.savefig('fig/KDE.png')

## 线性回归关系
#  分析变量之间的线性回归关系，即特征Vn与target的关系
f_col = 6
f_rows = 7
plt.figure(figsize=(8*f_rows, 8*f_col), dpi=100)
i = 1
for col in test_data.columns:
    ax=plt.subplot(f_rows, f_col, i)
    sns.regplot(x=col, y='target', data=train_data, ax=ax,
                scatter_kws={'marker':'.','s':3,'alpha':0.3},
                line_kws={'color':'k'})
    plt.xlabel(col)
    plt.ylabel('target')
    i += 1
plt.tight_layout()
plt.savefig('fig/reg.png')

## 计算相关性系数
#  分析特征变量和目标变量、特征变量之间的关系
#  以表格形式展示，通过pandas
#  V5 V9 V11 V17 V21 V22训练集与测试集差异太大，考虑删除
drop_cols = ['V5','V9','V11','V17','V21','V22']
data_train = train_data.drop(drop_cols, axis=1)
data_test = test_data.drop(drop_cols, axis=1)
train_corr = data_train.corr()
print(train_corr)
ax=plt.subplots(figsize=(20, 16))
ax=sns.heatmap(train_corr, vmax=0.8, square=True, annot=True)
plt.savefig('fig/heatmap.png')
# 根据相关系数筛选特征变量
# 首先寻找k个与target变量最相关的特征变量
k = 10
cols = train_corr.nlargest(k, 'target')['target'].index
# 找到最大的k列后，不能直接用train_corr[cols]，因为里面包含k个特征对所有0-37的特征的相关系数
# 必须先去原数据将这k列挑出来，再单独求这k列对target的相关系数
train_k_corr = train_data[cols].corr()
plt.figure(figsize=(10,10))
sns.heatmap(train_k_corr, annot=True, square=True)
plt.savefig('fig/k_largest_corr.png')
#  筛选相关系数大于0.5的，相关系数越大，认为这些特征变量对target的线性影响越大
#  如果存在更复杂的影响，可以考虑使用树模型的特征重要性去选择
threshold = 0.5
top_corr_features = train_k_corr.index[abs(train_k_corr['target']>threshold)]
plt.figure(figsize=(10,10))
sns.heatmap(train_k_corr[top_corr_features],annot=True,cmap='RdYlGn')
plt.savefig('fig/larger_than_threshold_corr.png')
#  也可以根据threshold去移除相关系数过小的特征（这里不删除，后续分析能用到）
drop_cols = train_corr[abs(train_corr['target'])<threshold].index
#  inplace默认为false，不对原数组进行修改，所以这里需要一个数组保存修改结果
data_larger_threshold = train_data.drop(drop_cols, axis=1)

## Box-Cox：将数据转变为正态分布
#  在变换之前需要对数据进行归一化
#  线下分析时可以对数据合并操作，能使训练数据和测试数据一致
#  线上部署只需要归一化训练数据
train_x = train_data.drop(['target'], axis=1)
data_all = pd.concat([train_x, test_data])
data_all.drop(drop_cols, axis=1, inplace=True)
#  训练集与测试集统一归一化
def norm(df):
    return (df-df.min())/(df.max()-df.min())
norm_all_data = norm(data_all)
#  分开归一化
train_data_norm = norm(data_train)
test_data_norm = norm(data_test)
#  进行Box-Cox变换，计算分位数并画图展示，使用分开归一化的结果
fcols = 6
frows = len(train_data_norm.columns)//2
cols_left = train_data_norm.columns[:frows]
plt.figure(figsize=(4*frows, 10*fcols))
i=0

for c in cols_left:
    i+=1
    # 利用index索引，丢弃存在空值的行
    data_c = train_data_norm[[c, 'target']].dropna()
    plt.subplot(frows, fcols, i)
    # hist & kde
    sns.distplot(data_c[c],fit=stats.norm)
    plt.title(c+' original')
    plt.xlabel('')

    i+=1
    plt.subplot(frows, fcols, i)
    stats.probplot(data_c[c], plot=plt)
    plt.title('skew='+'{:.4f}'.format(stats.skew(data_c[c])))
    plt.xlabel('')
    plt.ylabel('')

    i+=1
    plt.subplot(frows, fcols, i)
    # alpha：线条透明度
    plt.plot(data_c[c], data_c['target'], '.', alpha=0.5)
    plt.title('corr=' + ':.4f'.format(np.corrcoef(data_c[c], data_c['target'])[0][1]))

    i+=1
    plt.subplot(frows, fcols, i)
    #  boxcox要求输入值大于0，而归一化的结果最小值为0，所以加一保证正数
    #  函数返回转换后的x与最优lambda
    trans_data_c, lambda_data_c = stats.boxcox(data_c[c].dropna() + 1)
    #  加了1之后重新归一化
    trans_data_c = norm(trans_data_c)
    sns.distplot(trans_data_c,fit=stats.norm)
    plt.title(c+' transformed')
    plt.xlabel('')

    i+=1
    plt.subplot(frows, fcols, i)
    stats.probplot(trans_data_c, plot=plt)
    plt.title('skew='+'{:.4f}'.format(stats.skew(trans_data_c)))
    plt.xlabel('')
    plt.ylabel('')

    i+=1
    plt.subplot(frows, fcols, i)
    # alpha：线条透明度
    plt.plot(trans_data_c, data_c['target'], '.', alpha=0.5)
    plt.title('corr=' + ':.4f'.format(np.corrcoef(trans_data_c, data_c['target'])[0][1]))
plt.tight_layout()
plt.savefig('fig/Box_Cox_transformed_left.png')

cols_right = train_data_norm.columns[frows:-1]
plt.figure(figsize=(4*frows, 10*fcols))
i=0
for c in cols_right:
    i+=1
    # 利用index索引，丢弃存在空值的行
    data_c = train_data_norm[[c, 'target']].dropna()
    plt.subplot(frows, fcols, i)
    # hist & kde
    sns.distplot(data_c[c],fit=stats.norm)
    plt.title(c+' original')
    plt.xlabel('')

    i+=1
    plt.subplot(frows, fcols, i)
    stats.probplot(data_c[c], plot=plt)
    plt.title('skew='+'{:.4f}'.format(stats.skew(data_c[c])))
    plt.xlabel('')
    plt.ylabel('')

    i+=1
    plt.subplot(frows, fcols, i)
    # alpha：线条透明度
    plt.plot(data_c[c], data_c['target'], '.', alpha=0.5)
    plt.title('corr=' + ':.4f'.format(np.corrcoef(data_c[c], data_c['target'])[0][1]))

    i+=1
    plt.subplot(frows, fcols, i)
    #  boxcox要求输入值大于0，而归一化的结果最小值为0，所以加一保证正数
    #  函数返回转换后的x与最优lambda
    trans_data_c, lambda_data_c = stats.boxcox(data_c[c].dropna() + 1)
    #  加了1之后重新归一化
    trans_data_c = norm(trans_data_c)
    sns.distplot(trans_data_c,fit=stats.norm)
    plt.title(c+' transformed')
    plt.xlabel('')

    i+=1
    plt.subplot(frows, fcols, i)
    stats.probplot(trans_data_c, plot=plt)
    plt.title('skew='+'{:.4f}'.format(stats.skew(trans_data_c)))
    plt.xlabel('')
    plt.ylabel('')

    i+=1
    plt.subplot(frows, fcols, i)
    # alpha：线条透明度
    plt.plot(trans_data_c, data_c['target'], '.', alpha=0.5)
    plt.title('corr=' + ':.4f'.format(np.corrcoef(trans_data_c, data_c['target'])[0][1]))
plt.tight_layout()
plt.savefig('fig/Box_Cox_transformed_right.png')