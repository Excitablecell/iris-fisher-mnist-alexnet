# 导入相关包
import numpy as np
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import seaborn as sns
sns.set_style("whitegrid")

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier

# 导入数据集
iris = pd.read_csv('iris.csv', usecols=[1, 2, 3, 4, 5])
iris.info()
iris.describe()
# 设置颜色主题
antV = ['pink', 'blue', 'red', 'green', '#8543E0', '#13C2C2', '#3436c7', '#F04864'] 
# 绘制  Violinplot
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True)

sns.violinplot(x='Species', y='SepalLengthCm', data=iris, palette=antV, ax=axes[0, 0])
sns.violinplot(x='Species', y='SepalWidthCm', data=iris, palette=antV, ax=axes[0, 1])
sns.violinplot(x='Species', y='PetalLengthCm', data=iris, palette=antV, ax=axes[1, 0])
sns.violinplot(x='Species', y='PetalWidthCm', data=iris, palette=antV, ax=axes[1, 1])
plt.show()

heatfig=plt.gcf()
heatfig.set_size_inches(12, 8)
heatfig=sns.heatmap(iris.corr(), annot=True, linewidths=1, linecolor='k', square=True, mask=False, vmin=-1, vmax=1, cbar_kws={"orientation": "vertical"}, cbar=True)
plt.show()


