import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from RandomForestKernel import get_kernel
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from help_functions import *
from sklearn.preprocessing import normalize

dataset = "wine"  # square, wine ,sine

if dataset == "wine":
    data = pd.read_csv("datasets/winequality-red.csv", delimiter=";")
    truth = data["alcohol"].as_matrix()
    cols = ['fixed acidity', 'pH', 'density', 'chlorides']
    
    features = data[cols].as_matrix()
    
    # Normalize histogramms
    features = normalize_features(features)

    train_features = features[:750, :]
    train_measures = truth[:750]

    test_features = features[750:1000, :]
    test_measures = truth[750:1000]

elif dataset == "square":

    train_features = np.linspace(-10, 10, 200)
    train_measures = train_features**2


    train_features = train_features[:, None]

    # Add noise to data
    train_measures = train_measures + np.random.normal(0, 10, len(truth))

    test_features = np.linspace(-10, 10, 50)
    test_features = test_features[:, None]

elif dataset == "sine":
    train_features = np.linspace(-np.pi, np.pi, 100)
    
    truth = np.sin(train_features)
    train_features = train_features[:, None]

    # truth = train_features*np.sin(train_features*0.1)
    train_measures = truth + np.random.normal(0, 0.2, len(truth))

    test_features = np.linspace(-np.pi, np.pi, 50)
    test_features = test_features[:, None]
    # test_points_y = [0.25, 26]


kernel = get_kernel(train_features, test_features, train_measures)

test_size = test_features.shape[0]
train_size = train_features.shape[0]

test_kernel = kernel[train_size:, :train_size]
train_kernel = kernel[:train_size, :train_size]

def my_kernel(X, Y):
    return kernel

plt.imshow(kernel)



# Random Forest Kernel
svr_rf = SVR(kernel="precomputed")
svr_rf.fit(train_kernel, train_measures)



# RBF Kernel
# Selecting hyper-parameters
C = np.logspace(-2, 10, 10)
gamma = np.logspace(1, 10, 10)

print('Performing cross-validation for SVR parameters...')
Copt, gammaOpt = select_parameters(C, gamma, train_features, train_measures)

print Copt, gammaOpt

svr_rbf = SVR(kernel='rbf', C=Copt, gamma=gammaOpt)
svr_rbf.fit(train_features, train_measures)


# Plotting the data
if dataset in ['square', 'sine']:
    fig, ax = plt.subplots()
    ax.plot(train_features, truth, label='Ground truth')
    ax.scatter(train_features, train_measures, label='Points')

    result = svr_rf.predict(test_kernel)
    print result

    ax.plot(test_features, result, label='Random Forest kernel')
    ax.plot(test_features, svr_rbf.predict(test_features), label='RBF')
    legend = ax.legend()
    plt.show()
    fig.savefig("plot.png")

elif dataset == 'wine':
    RF_predict = svr_rf.predict(test_kernel)
    RBF_predict = svr_rbf.predict(test_features)
    r2_rf = r2_score(test_measures, RF_predict)
    mse_rf = np.mean((test_measures - RF_predict) ** 2)
    r2_rbf = r2_score(test_measures, RBF_predict)
    mse_rbf = np.mean((test_measures - RBF_predict) ** 2)

    print("MSE Random forest:", mse_rf)
    print("MSE RBF: ", mse_rbf)

    fig, ax = plt.subplots()
    ax.scatter(test_measures, RF_predict)
    plt.plot(np.arange(8, 15), np.arange(8, 15), c="r")
    plt.legend(loc="lower right")
    plt.title('RF kernel')
    ax.set_xlabel('Correct value')
    ax.set_ylabel('Predicted value')
    plt.show
    fig, ax2 = plt.subplots()
    ax2.scatter(test_measures, RBF_predict)
    plt.plot(np.arange(8, 15), np.arange(8, 15), c="r")
    plt.legend(loc="lower right")
    plt.title('RBF kernel')
    ax2.set_xlabel('Correct value')
    ax2.set_ylabel('Predicted value')
    plt.xlabel("")
    plt.show()
