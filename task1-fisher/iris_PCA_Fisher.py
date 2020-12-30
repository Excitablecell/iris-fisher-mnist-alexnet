import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
def PCA_convert(dataMat,n):
    #零均值化
    meandata=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData= dataMat - meandata

    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    lowDDataMat=newData*n_eigVect               #低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meandata  #重构数据
    return lowDDataMat,reconMat

#绘制PCA重构后的矩阵
def plot_scatter(pcadata):
    x1 = pcadata[:,0:1]
    y1 = pcadata[:,1:]
    setosa = []
    y_data = []
    versicolor = []
    y_data2 = []
    virginica = []
    y_data3 = []
    for i in range(len(x1)):
        if y[i] == 0:
            setosa.append(x1[i])
        elif y[i] == 1:
            versicolor.append(x1[i])
        elif y[i] == 2:
            virginica.append(x1[i])
    for i in range(len(y1)):
        if y[i] == 0:
            y_data.append(y1[i])
        elif y[i] == 1:
            y_data2.append(y1[i])
        elif y[i] == 2:
            y_data3.append(y1[i])

    plt.scatter(setosa,y_data,c='c',label = 'setosa')
    plt.scatter(versicolor,y_data2,c='r',label = 'versicolor')
    plt.scatter(virginica,y_data3,c='b',label = 'virginica')
    plt.legend()
    plt.show()

data=datasets.load_iris()
X=data['data']
y=data['target']
data_pca,data_avr = PCA_convert(X,2)

Iris1=data_pca[0:50,0:2]
Iris2=data_pca[50:100,0:2]
Iris3=data_pca[100:150,0:2]

#定义类均值向量
m1=np.mean(Iris1,axis=0)
m2=np.mean(Iris2,axis=0)
m3=np.mean(Iris3,axis=0)

#定义类内离散度矩阵
s1=np.zeros((2,2))#生成4*4的0矩阵
s2=np.zeros((2,2))
s3=np.zeros((2,2))

for i in range(0,50,1): #公式4-11，求得类内离散度矩阵
    a=Iris1[i,:]-m1
    a=np.array(a)
    b=a.T
    s1=s1+np.dot(b,a)

for i in range(0,50,1):
    c=Iris2[i,:]-m2
    c=np.array(c)
    d=c.T
    s2=s2+np.dot(d,c) 
for i in range(0,50,1):
    a=Iris3[i,:]-m3
    a=np.array(a)
    b=a.T
    s3=s3+np.dot(b,a)
    
#定义总类内离散矩阵
sw12 = s1 + s2  #公式4-12，求得总类内离散矩阵
sw13 = s1 + s3
sw23 = s2 + s3

#定义投影方向
a=np.array(m1-m2)
sw12=np.array(sw12,dtype='float')
sw13=np.array(sw13,dtype='float')
sw23=np.array(sw23,dtype='float')

#判别函数以及阈值T（即w0）
a=np.array(m1-m2).T
b=np.array(m1-m3).T
c=np.array(m2-m3).T
#Sw12^-1 *(m1-m2) = w*
w12=(np.dot(np.linalg.inv(sw12),a)).T 
w13=(np.dot(np.linalg.inv(sw13),b)).T
w23=(np.dot(np.linalg.inv(sw23),c)).T
#w0 = -1/2*(m1+m2)
T12=-0.5*(np.dot(np.dot((m1+m2),np.linalg.inv(sw12)),a))
T13=-0.5*(np.dot(np.dot((m1+m3),np.linalg.inv(sw13)),b))
T23=-0.5*(np.dot(np.dot((m2+m3),np.linalg.inv(sw23)),c))
#计算正确率
kind1=0
kind2=0
kind3=0
#创建分类后各类存放样本的矩阵
newiris1=[]
newiris2=[]
newiris3=[]

#剩下的样本用来分类并测试正确率
errorflag = 0
for i in range(0,49):
    x=Iris1[i,:]
    x=np.array(x)
    g12=np.dot(w12,x.T)+T12
    g13=np.dot(w13,x.T)+T13
    g23=np.dot(w23,x.T)+T23
    if g12>0 and g13>0:#两判别函数均大于0，属于第一类
        newiris1.extend(x)
        kind1=kind1+1
    elif g12<0 and g23>0:   #属于第二类
        newiris2.extend(x)
    elif g13<0 and g23<0 :  #属于第三类
        newiris3.extend(x)
    else:                   #出现不可分的情况
        errorflag +=1

for i in range(0,49):
    x=Iris2[i,:]
    x=np.array(x)
    g12=np.dot(w12,x.T)+T12
    g13=np.dot(w13,x.T)+T13
    g23=np.dot(w23,x.T)+T23
    if g12>0 and g13>0:
        newiris1.extend(x)
    elif g12<0 and g23>0:
       
        newiris2.extend(x)
        kind2=kind2+1
    elif g13<0 and g23<0 :
        newiris3.extend(x)
    else:
        errorflag +=1

for i in range(0,49):
    x=Iris3[i,:]
    x=np.array(x)
    g12=np.dot(w12,x.T)+T12
    g13=np.dot(w13,x.T)+T13
    g23=np.dot(w23,x.T)+T23
    if g12>0 and g13>0:
        newiris1.extend(x)
    elif g12<0 and g23>0:     
        newiris2.extend(x)
    elif g13<0 and g23<0 :
        newiris3.extend(x)
        kind3=kind3+1
    else:
        errorflag +=1
correct=(kind1+kind2+kind3)/150
print(errorflag)
print('判断出来的综合正确率：',correct*100,'%')
plot_scatter(data_pca)