import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
def detect_outliers(model, X, y, sigma=3) :
    # 利用模型预测标签
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # 如果预测失败，先尝试调整模型
    except:
        model.fit(X,y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    # 计算预测值和真值的残差
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # 定义outliers：|(resid - mean_resid)/std_resid|>sigma
    # pd.Series一维数组已设定索引值为真值y，先通过不等式筛选出异常值，再取异常值对应的真值标签
    z = (resid - mean_resid)/std_resid
    outliers = z[abs(z)>sigma].index

    # print % plot
    print('R2=', model.score(X,y))
    print('mse=', mean_squared_error(y, y_pred))
    print('-----------------------------------------')

    print('mean of residuals', mean_resid)
    print('std of residuals', std_resid)
    print('-----------------------------------------')

    print(len(outliers), ' ouliers:')
    print(outliers.tolist())
    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    # 检索异常值
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y-y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers]-y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred')

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('z')

    plt.savefig('fig/outliers.png')
    return outliers