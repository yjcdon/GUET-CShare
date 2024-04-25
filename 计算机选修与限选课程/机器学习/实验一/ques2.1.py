# 导入第三方包
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
from sklearn import tree

# 读入数据
fr = open('glass-lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate', 'type']
lens = pd.DataFrame.from_records(lenses, columns=lensesLabels)

# 数据预处理
# 哑变量处理
dummy = pd.get_dummies(lens[['age', 'prescript', 'astigmatic', 'tearRate']])
# 水平合并数据集和哑变量的数据集
lens = pd.concat([lens, dummy], axis=1)
# 删除原始的 age, prescript, astigmatic 和 tearRate 变量
lens.drop(['age', 'prescript', 'astigmatic', 'tearRate'], inplace=True, axis=1)
lens.head()

# 将数据集拆分为训练集和测试集，且测试集的比例为 25%
X_train, X_test, y_train, y_test = model_selection.train_test_split(lens.loc[:, 'age_pre':'tearRate_reduced'],
                                                                    lens.type, test_size=0.25, random_state=1234)

# 构建分类决策树
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = clf.predict(X_test)

# 计算预测准确率
accuracy = metrics.accuracy_score(y_test, y_pred)
print("预测准确率为: {:.2f}%".format(accuracy * 100))
