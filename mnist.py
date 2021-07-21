import pandas as pd
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

train=pd.read_csv('mnist_train.csv')
test=pd.read_csv('mnist_test.csv')

X=train.drop('label',axis=1) 

x_nt=X+0.00001*np.random.randn(60000,784)#to remove linearly dependent variables we should add small values

Y=train.label

x_test=test.drop('label',axis=1)

#print(x_test)
y_test=test.label

#analytical solution
W=inv(x_nt.T.dot(x_nt)).dot(X.T).dot(Y) #X.T gives transpose of matrix x, dot returns dot product of 2 arrays
print(W.shape)
yhat=x_test.dot(W)
for i in range(len(yhat)):
    yhat[i]=int(np.round(yhat[i]))


from sklearn.metrics import mean_squared_error
print('MSE for Analytical:',mean_squared_error(yhat,y_test))
print('Accuracy for Analytical:',accuracy_score(y_test,yhat))
#as this is not a classifaction algorith
#if we get value as 0.1 instead of 0 it will give accuracy so I used for loop to convert all numbers to their nearest int 

#LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
model=LinearRegression()
model.fit(X,Y)
pred=model.predict(x_test)
#cheching performance of model
for i in range(len(pred)):
    pred[i]=int(np.round(pred[i]))

print('MSE for LinearRegression:',mean_squared_error(y_test,pred))
print('Accuracy for LinearRegression:',accuracy_score(y_test,pred))



#gradientdescent
A=np.random.normal(size=[784,1])#x*A-y->(120*5)(A)=(120*1) #y use normal instead of size
A=A.reshape(784,1)
print(X.shape)
Y=Y.values.reshape(60000,1)

loss=X.dot(A)-Y
final_loss=(np.linalg.norm(loss))**2 #to find value of matrix
#now to minimize this loss we use gradient..so partial differentiation with A
gradient=X.T.dot((X.dot(A)-Y)) #so that gradient will have same dimensions as A, to match dimenions and do computations we transpose matix           
print("gradient shape",gradient.shape)

n=5
while n:
    A=A-0.000000001*gradient
    loss=np.linalg.norm(X.dot(A)-Y)**2
    if(loss>final_loss):
        break
    else:
        final_loss=loss
        n=n-1
pred=x_test.dot(A)
print(pred.shape)
pred.reshape(10000)
print(pred.shape)
print(y_test.shape)
def accuracy(pred,y_test):
    c=0
    for i in range(len(pred)):
        pred[i]=int(np.round(pred[i]))
        if(pred[i]==y_test[i]):
            c+=1
    return c/len(pred)*100
    

print("MSE for gradient descent",mean_squared_error(y_test,pred))
print("Accuracy for gradient descent",accuracy(pred,y_test))
#we can find accuracy by graident descent method by same as above by converting y pred
#values to int and then use accuracu_score
