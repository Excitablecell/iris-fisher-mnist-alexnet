import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
data=datasets.load_iris()
X=data['data']
y=data['target']

Iris1=X[0:50,0:4]
Iris2=X[50:100,0:4]
Iris3=X[100:150,0:4]

#定义类均值向量
m1=np.mean(Iris1,axis=0)
m2=np.mean(Iris2,axis=0)
m3=np.mean(Iris3,axis=0)

#定义类内离散度矩阵
s1=np.zeros((4,4))#生成4*4的0矩阵
s2=np.zeros((4,4))
s3=np.zeros((4,4))
for i in range(0,50,1): #公式4-11，求得类内离散度矩阵
    a=Iris1[i,:]-m1
    a=np.array([a])
    b=a.T
    s1=s1+np.dot(b,a)
for i in range(0,50,1):
    c=Iris2[i,:]-m2
    c=np.array([c])
    d=c.T
    s2=s2+np.dot(d,c) 
for i in range(0,50,1):
    a=Iris3[i,:]-m3
    a=np.array([a])
    b=a.T
    s3=s3+np.dot(b,a)
    
#定义总类内离散矩阵
sw12 = s1 + s2  #公式4-12，求得总类内离散矩阵
sw13 = s1 + s3
sw23 = s2 + s3
#定义投影方向
a=np.array([m1-m2])
sw12=np.array(sw12,dtype='float')
sw13=np.array(sw13,dtype='float')
sw23=np.array(sw23,dtype='float')
#判别函数以及阈值T（即w0）
a=np.array([m1-m2]).T
b=np.array([m1-m3]).T
c=np.array([m2-m3]).T
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
    x=np.array([x])
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
    x=np.array([x])
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
    x=np.array([x])
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