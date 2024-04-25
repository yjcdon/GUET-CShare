import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model

# 导入数据
Profit = pd.read_excel('Predict to Profit.xlsx')
X = Profit.loc[:, ['RD_Spend', 'Administration', 'Marketing_Spend']]
Y = Profit.loc[:, ['Profit']]
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=1234)
# 根据train数据集建模
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
print('模型的偏回归系数分别为：\n', model.coef_)
print('模型的截距为：\n', model.intercept_)
print('模型为:y={}x1 {}x2 {}x3 {}'.format(model.coef_[0][0],
                                          model.coef_[0][1] if model.coef_[0][1] < 0 else '+' + str(model.coef_[1][1]),
                                          model.coef_[0][2] if model.coef_[0][2] < 0 else '+' + str(model.coef_[0][2]),
                                          model.intercept_[0] if model.intercept_[0] < 0 else '+' + str(
                                              model.intercept_[0])))
# 进行预测
y_predict = model.predict(X_test)
print("预测值{}".format(y_predict))
print("原值{}".format(y_test))
print("得分：{}".format(model.score(X_test, y_test)))
