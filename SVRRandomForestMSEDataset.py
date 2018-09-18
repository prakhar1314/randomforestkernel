import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from RandomForestKernel import get_kernel
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from help_functions import *
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split


total_data = ['auto' ,'wine', 'housing', 'Concrete' , 'servo']

rf_mse = []
rbf_mse = []

for dataset in total_data:
    
    if dataset == "auto":
        data = pd.read_csv("datasets/auto-mpg.csv",header= None)
        data = data.as_matrix()
        truth = data[:,1]
        features = data[:,:1]
       # features = normalize_features(features)
        train_features, test_features, train_measures, test_measures = train_test_split(features, truth, test_size=0.33, random_state=42)
    
    elif dataset == "wine":
        data = pd.read_csv("datasets/winequality-red.csv", delimiter=";")
        truth = data["alcohol"].as_matrix()
        cols = ['fixed acidity', 'pH', 'density', 'chlorides','volatile acidity', 'citric acid','residual sugar','free sulfur dioxide','total sulfur dioxide']
        # cols = ['fixed acidity']
        features = data[cols].as_matrix()
        
        features = normalize_features(features)
    
        train_features, test_features, train_measures, test_measures = train_test_split(features, truth, test_size=0.33, random_state=42)

    
    elif dataset == "linearReg":
        features = np.linspace(-10,10, 100)
        train_features = features[:, None]
        truth = (train_features)**2
        truth = normalize(truth, axis = 0)
        # truth = features*np.sin(features*0.1)
        train_measures = truth + np.random.normal(0, 0.1, [len(truth), 1])
    
        test_features = np.linspace(-10 , 10 , 200)
        test_features = test_features[:, None]
        # test_points_y = [0.25, 26]
        
        
    elif dataset == "housing":
        data = pd.read_csv("datasets/housing.csv", header = None)
        data = data.as_matrix()
        truth = data[:,-1]
        for i in range(len(data)):
            features = data[:,1:i+1]
            features = normalize_features(features)
            train_features, test_features, train_measures, test_measures = train_test_split(features, truth, test_size=0.33, random_state=42)
        
    elif dataset == "Concrete":
        data = pd.read_csv("datasets/Concrete.csv",header= 1)
        data = data.as_matrix()
        truth = data[:,-1]
        features = data[:,:-1]
        features = normalize_features(features)
        train_features, test_features, train_measures, test_measures = train_test_split(features, truth, test_size=0.33, random_state=42)
    
    elif dataset == "servo":
        data = pd.read_csv("datasets/servo.csv", header = None)
        data = data.replace(to_replace=('A','B','C','D','E'), value=(1,2,3,4,5))
        data = data.as_matrix()
        truth = data[:,-1]
        features = data[:,:-1]   
        features = normalize_features(features)
        train_features, test_features, train_measures, test_measures = train_test_split(features, truth, test_size=0.33, random_state=42)
    
    elif dataset == "communities":
        data = pd.read_csv("datasets/communities.csv")
        print data.replace('?', np.nan) 
        print(data)
        data = data.as_matrix()
        data = data[:,:4]
       
        print(data.mean())
        truth = data[:,-1]
        features = data[:,:-1]
        features = normalize_features(features)
        print(features)
        train_features, test_features, train_measures, test_measures = train_test_split(features, truth, test_size=0.33, random_state=42)
    
    
    
    kernel = get_kernel(train_features, test_features, train_measures)
    
    test_size = test_features.shape[0]
    train_size = train_features.shape[0]
    
    print kernel.shape
    
    test_kernel = kernel[train_size:, :train_size]
    train_kernel = kernel[:train_size, :train_size]
    
    plt.imsave("kernel.png", kernel)
    
    def my_kernel(X, Y):
        return kernel
    
    plt.imshow(kernel)
    
    
    # Random Forest Kernel
    svr_rf = SVR(kernel="precomputed")
    svr_rf.fit(train_kernel, train_measures)
    print(svr_rf.intercept_)
    
    
    
    # RBF Kernel
    C = np.logspace(-2, 10, 10)
    gamma = np.logspace(1, 3, 10)
    print('Performing cross-validation for SVR parameters...')
    Copt, gammaOpt = select_parameters(C, gamma, train_features, train_measures)
    
    print Copt, gammaOpt
    
    svr_rbf = SVR(kernel='rbf', C=Copt, gamma=gammaOpt)
    svr_rbf.fit(train_features, train_measures)
    


    if dataset == 'linearReg':
        fig, ax = plt.subplots()
        ax.plot(train_features, truth, label='Ground truth')
        ax.scatter(train_features, train_measures, label='Points')
    
        data = np.concatenate((train_features, test_features), axis=0)
    
        result = svr_rf.predict(test_kernel)
        print result
    
        ax.plot(test_features, result, label='Random Forest kernel')
        ax.plot(test_features, svr_rbf.predict(test_features), label='RBF')
        legend = ax.legend()
        plt.show()
        fig.savefig("plot.png")
    
    elif dataset in ['wine', 'housing', 'auto', 'servo', 'Concrete']:
        RF_predict = svr_rf.predict(test_kernel)
        RBF_predict = svr_rbf.predict(test_features)
        r2_rf = r2_score(test_measures, RF_predict)
        mse_rf = np.mean((test_measures - RF_predict) ** 2)
        r2_rbf = r2_score(test_measures, RBF_predict)
        mse_rbf = np.mean((test_measures - RBF_predict) ** 2)
    
        rf_mse.append(mse_rf/np.std(test_measures))
        rbf_mse.append(mse_rbf/np.std(test_measures))
    
        # fig, ax = plt.subplots()
        # ax.scatter(test_measures, RF_predict)
        # plt.plot(np.arange(8, 15), np.arange(8, 15), label="r^2=" + str(r2_rf), c="r")
        # plt.legend(loc="lower right")
        # plt.title('RF kernel')
        # plt.show
        # fig, ax2 = plt.subplots()
        # ax2.scatter(test_measures, RBF_predict)
        # plt.plot(np.arange(8, 15), np.arange(8, 15), label="r^2=" + str(r2_rbf), c="r")
        # plt.legend(loc="lower right")
        # plt.title('RBF kernel')
        # plt.show()


# Barchart
ind = np.arange(len(total_data))  # the x locations for the groups
width = 0.35  

print rf_mse
print rbf_mse

fig, ax = plt.subplots()
rects1 = ax.bar(ind, rf_mse, width, color='r')

rects2 = ax.bar(ind + width, rbf_mse, width, color='b')

ax.set_ylabel('nMSE')
ax.set_xlabel('Datasets')
ax.set_title('Normalized MSE for different datasets')
ax.set_xticks(ind + width)
ax.set_xticklabels(total_data)

ax.legend((rects1[0], rects2[0]), ('Random forest', 'RBF'), loc=2)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % height,
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()
