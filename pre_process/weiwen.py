import pandas as pd
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
import warnings
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


warnings.filterwarnings("ignore", category=Warning)
file_path = '../data/people_data_weiwen.csv'
df = pd.read_csv(file_path, encoding='gbk')
df = pd.get_dummies(df)
Y = df['是否需要维稳']
X = df.drop('是否需要维稳', axis=1)
print("初始样本中各类数目为:{}".format(Counter(Y)))
#SMOTE过采样
smo = SMOTE(random_state=42)
x_smo, y_smo = smo.fit_resample(X, Y)
print("过采样后各类数目为:{}".format(Counter(y_smo)))
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_smo, y_smo, random_state=2, test_size=.20)
scaler = MinMaxScaler()
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = pd.DataFrame(x_train, columns=X.columns)
x_test = pd.DataFrame(x_test, columns=X.columns)
xg_model = XGBClassifier()
xg_model.fit(x_train, y_train)
y_pred = xg_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("xgboost_accuracy:{:.2f}".format(accuracy * 100.0))
plot_importance(xg_model)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.show()
svm_model = svm.SVC(kernel='linear')
svm_model.fit(x_train, y_train)
y_pred = svm_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("svm_accuracy:{:.2f}".format(accuracy * 100.0))
tree_model = DecisionTreeClassifier().fit(x_train, y_train)
y_pred = tree_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("decision_tree_accuracy:{:.2f}".format(accuracy * 100.0))
rf_model = RandomForestClassifier().fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("random_forest_accuracy:{:.2f}".format(accuracy * 100.0))
ada_model = AdaBoostClassifier().fit(x_train, y_train)
y_pred = ada_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("adaboost_accuracy:{:.2f}".format(accuracy * 100.0))