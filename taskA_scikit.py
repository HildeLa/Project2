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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) #Splitter dataene

XT_X = X.T @ X #Identity matrix?
H = (2.0/n)* XT_X
EigValues, EigVectors = np.linalg.eig(H) #To get eigenvalues
eta = 1.0/np.max(EigValues)

learn_rates =  np.geomspace(0.0001, 0.1, 10, endpoint = True) # this is ineffective,
epoch_range = np.geomspace(10, 1000, 10, endpoint = True)#but the thought is to be able to plot this


for rate in learn_rates: # looping through the learning rates I want to test
    print(f'Results for learning rate {rate}:\n')
    for range in epoch_range: # for each learning rate, I test each epoch lenght
        model = SGDRegressor(max_iter=10000,eta0=rate,fit_intercept=False) #Scikit learn Stocastic gradient descent regression
        #fit intercept is false because of the 1 in the feature matrix
        mse_train= [] # to be filled with train and test mses
        mse_test= [] #Planning to exchange with a matrix defined outside of loop for plotting all MSEs for each rate and epoch size
        epochs = np.arange(1, range+1) #making a list for looping through
        n_batch = 10 #I use 10 batches in each epoch
        batches = np.split(X_train, n_batch, axis=0) #Splitting the training data into the batches
        ybatches = np.split(y_train, n_batch, axis=0)
        batch_i= 0
        for b, batch in enumerate(batches): #getting index to get the right y-batch
            y_batch = ybatches[b]
            try:
                model.partial_fit(batch, y_batch)
                batch_i += n_batch
            except ValueError: #An error that occured, perhaps not anymore
                print('for some reason y_batch is now ', len(y_batch))
                print('while x-batch ', len(x_batch))
                pass

            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            mse_train.append(MSE(y_train, y_pred_train))
            mse_test.append(MSE(y_test, y_pred_test))

        print('Coefficients: ', model.coef_) #printing coefficients for each epoch epoch_range
        #Now, there is something weird
    #fig, ax = plt.subplots()
    #ax.plot(epochs, mse_test, label = 'mse test')
    #ax.plot(epochs, mse_train, label = 'mse train')
    #plt.xlabel('epoch')
    #plt.ylabel('MSE scores')
    #plt.legend()
    #plt.show()
