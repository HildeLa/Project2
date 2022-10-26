from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random
import numpy as np

seed = np.random.seed(12345)

def create_data(x):
    return 1 + 3*x + 5*x**2 #+ np.random.randn(n, 1)

def create_X(x):
    return np.c_[np.ones((n, 1)), x, x**2]

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

n = 5000
x = np.random.randint(1,10,n)#.reshape(1,-1)

y = create_data(x)#.ravel()

X = create_X(x)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

XT_X = X.T @ X #Identity matrix?

model = SGDRegressor(max_iter=1000)

def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

mse_train= []
mse_test= []
epochs = np.arange(1, 100)
for epoch in range(len(epochs)):
    for batch in batches(range(len(X_train)), 100): #
        model.partial_fit(X_train[batch[0]:batch[-1]+1], y_train[batch[0]:batch[-1]+1])

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    mse_train.append(MSE(y_train, y_pred_train))
    mse_test.append(MSE(y_test, y_pred_test))

    print(model.coef_)
    print(model.intercept_)

fig, ax = plt.subplots()
ax.plot(epochs, mse_test, label = 'mse test')
ax.plot(epochs, mse_train, label = 'mse train')
plt.xlabel('epoch')
plt.ylabel('MSE scores')
plt.legend()
plt.show()
