import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm

# 导入数据集
income = pd.read_csv("./line-ext.csv")
# 绘制散点图
sns.lmplot(x='YearsExperience', y='Salary', data=income, ci=None)
# 显示图形
plt.show()
# 样本量
n = income.shape[0]

# 计算自变量、因变量、自变量平方、自变量与因变量乘积的和
sum_x = income.YearsExperience.sum()
sum_y = income.Salary.sum()
sum_x2 = income.YearsExperience.pow(2).sum()
xy = income.YearsExperience * income.Salary
sum_xy = xy.sum()

# 根据公式计算回归模型的参数
w = (sum_xy - sum_x * sum_y / n) / (sum_x2 - sum_x ** 2 / n)
b = income.Salary.mean() - w * income.YearsExperience.mean()  # 打印出计算结果
print('回归参数 w 的值：', w)
print('回归参数 b 的值：', b)
print('模型表达式：f(x)=', w, 'x+', b)
# 打印出均方误差
predictY = 0.8542 * w + b
print("预测值", predictY)
# 打印出变量 x =0.8452 时，变量 y 的预测值

# 利用收入数据集，构建回归模型
fit = sm.formula.ols('Salary ~ YearsExperience', data=income).fit()
# 返回模型的参数值
fit.params
print("参数", fit.params)

linearRegression = linear_model.LinearRegression()
arrayX = np.array(income.YearsExperience)
arrayY = np.array(income.Salary)
xTest = (arrayX.reshape((len(arrayX)), -1))
yTest = (arrayY.reshape((len(arrayY)), -1))
linearRegression.fit(xTest, yTest)
print("系数：", linearRegression.coef_)
print("截距：", linearRegression.intercept_)
