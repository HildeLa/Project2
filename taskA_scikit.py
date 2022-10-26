'''
Solving task a using scikit learn
'''
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
#print(np.max(y))
#plt.plot(x,y,'o')
#plt.show()
X = create_X(x)
#scaler = StandardScaler()
#scaler.fit(X)
#X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

XT_X = X.T @ X #Identity matrix?
H = (2.0/n)* XT_X
EigValues, EigVectors = np.linalg.eig(H)
eta = 1.0/np.max(EigValues)

learn_rates =  np.geomspace(0.0001, 0.1, 10, endpoint = True)
epoch_range = np.geomspace(10, 1000, 10, endpoint = True)


def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

for rate in learn_rates:
    for range in epoch_range:
        model = SGDRegressor(max_iter=10000,eta0=rate,fit_intercept=False)
        mse_train= []
        mse_test= []
        epochs = np.arange(1, range+1)
        n_batch = 10
        batches = np.split(X_train, n_batch, axis=0)
        for epoch in epochs:
            batch_i= 0
            for b, batch in enumerate(batches):
                y_batch = y_train[batch_i:batch_i+int(len(y_train)/n_batch)]
                try:
                    model.partial_fit(batch, y_batch)
                    batch_i += n_batch
                except ValueError:
                    print('for some reason y_batch is now ', len(y_batch))
                    print('while x-batch ', len(x_batch))
                    pass

            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            mse_train.append(MSE(y_train, y_pred_train))
            mse_test.append(MSE(y_test, y_pred_test))

        print('Coefficients: ', model.coef_)

fig, ax = plt.subplots()
ax.plot(epochs, mse_test, label = 'mse test')
ax.plot(epochs, mse_train, label = 'mse train')
plt.xlabel('epoch')
plt.ylabel('MSE scores')
plt.legend()
plt.show()
