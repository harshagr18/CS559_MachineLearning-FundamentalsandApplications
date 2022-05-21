import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

names = ["val1","val2", "val3", "val4", "val5", "val6", "val7", "val8", "val9", "Target"]
df = pd.read_csv("breast-cancer-wisconsin.csv", header = None, names = names)
#df.drop('Unnamed',axis=1,inplace=True)
df.fillna(0)
df['Target'] = df['Target'].map({'M':1,'B':0})

X = df[df.columns[2:32]]
Y = df['Target']
Y = Y.values.reshape(Y.shape[0],1)

#train data is (80%):
X_train = X.loc[0:454,X.columns[0:]] 
y_train = Y[0:455]

#testing data (20%):
test_X = X.loc[0:143,X.columns[0:]] 
test_Y = Y[0:144]

#training data:
mean = X_train.mean()
std_error = X_train.std()
X_train = (X_train - mean)/std_error

#test set:
mean = test_X.mean()
std_error = test_X.std()
test_X = (test_X - mean)/std_error

print("Shape of X_train data",X_train.shape) 
print("Shape of test_X data ",test_X.shape) 
print("Shape of y_train data ",y_train.shape) 
print("Shape of test_Y data",test_Y.shape)


def sigmoid(z):
  return 1/(1+np.exp(-z))

### Random initialization of w and b
def random_init(dim): 
  w = np.zeros((dim,1))
  b=0
  return w,b

(random_init(X_train.shape[1]))

### Forward and backward propogation
#hypothesis in logistic regression is y = a = sigmoid(z) = sigmoid(w^TX + b)
def propo(w,b,X,Y): 
  m = X.shape[0]
#forward propogation
  z = np.dot(X,w) + b
  a = sigmoid(z)
  cost = -np.sum(Y*np.log(a) - (1-Y)*np.log(1-a))/m
#backpropogation:
  dz = a-Y
  dw = np.dot(np.transpose(X),dz)/m 
  db = np.sum(dz)/m
  grad = { "dw":dw,"db":db }
  return grad,cost

### Gradient descent over number of iteration
def optim(w,b,X,Y,learning_rate,num_iteration):
   costs = []
   for i in range(num_iteration): 
    grads, cost=propo(w,b,X,Y)
    dw = grads["dw"] 
    db = grads["db"]
    #updating w and b
    w = w - learning_rate*dw 
    b = b - learning_rate*db
    if(i%100==0): 
      costs.append(cost)
   params= { "w":w,"b":b }
   grads = { "dw":dw,"db":db }
   return params,grads,costs

#random init of w,b
w,b = random_init(X_train.shape[1])
#forward, backward & grad. descent:
params,grads,costs = optim(w,b,X_train,y_train,0.01,2000)
print(params) 
print(grads) 
print(costs)

### Cost vs iteration graph For checking learning rate

#plt.plot(cost_all,range(len(cost_all))) 
costs = np.squeeze(costs) 
plt.plot(costs)
plt.xlabel('No. of iteration') 
plt.ylabel('Cost')
plt.show()

def predict(w,b,X):
  a = sigmoid(np.dot(X,w) + b) 
  return a
def oneORzero(x): 
  if(x>=0.5):
   return 1 
  elif(x<0.5): 
    return 0

### Prediction accuracy for Train and test set
# Accuracy for training set:
temp = predict(params["w"],params["b"],X_train) 
train_pred = np.array(list(map(oneORzero,temp)))
train_pred = train_pred.reshape((train_pred.shape[0],1))

# Accuracy for test set:
temp = predict(params["w"],params["b"],test_X)
test_pred = np.array(list(map(oneORzero,temp)))
test_pred = test_pred.reshape((test_pred.shape[0],1))

print("Training set accuracy = ",(100 - np.mean(np.abs(train_pred - y_train))*100))
print("Test set accuracy = ",(100 - np.mean(np.abs(test_pred - test_Y))*100))