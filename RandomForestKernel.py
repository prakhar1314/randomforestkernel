import pandas as pd
import numpy as np
from random import randint
from sklearn.ensemble import RandomForestRegressor
from help_functions import get_leaves, get_lineage

def get_partial_kernel(forest, dataset):
    SAMPLE_SIZE = len(dataset)

    #Select random tree
    treeIndex = randint(0,forest.n_estimators-1)
    Estimator = forest.estimators_[treeIndex]

    #Define tree parameters
    n_nodes = Estimator.tree_.node_count
    children_left = Estimator.tree_.children_left
    children_right = Estimator.tree_.children_right


    #Node selection
    leaf_index = Estimator.apply(dataset)
    leaf_nodes = np.array(list(set(leaf_index)))  # Remove duplicates

    is_leaves, node_depth = get_leaves(children_left, children_right, n_nodes)
    myparents = get_lineage(Estimator, leaf_index, node_depth)

    myparents = np.array(myparents)

    h = max(node_depth[is_leaves])
    d = randint(0, h)

    # Build heritage tree for each leaf node.
    nth_parent_for_sample = []

    for lineage in myparents:
        ancest = None
        if len(lineage) <= d:
            ancest = lineage[-1]
        else:
            ancest = lineage[d]
        nth_parent_for_sample.append(ancest)

    partial_kernel = np.zeros([SAMPLE_SIZE, SAMPLE_SIZE])

    #Loop over each datapoint : two data point are assigned to the same cluster if they have the same ancestor
    for i in range(SAMPLE_SIZE):
        for j in range(i, SAMPLE_SIZE):
            if nth_parent_for_sample[i] == nth_parent_for_sample[j]:
                partial_kernel[i][j] = 1
                partial_kernel[j][i] = 1

    return partial_kernel


def get_kernel(train_data, test_data, label):

    #Define forest (n_estimators = number of trees)
    forest = RandomForestRegressor(n_estimators=500, bootstrap=True)
    forest = forest.fit(train_data, label)

    # After fitting, we concatinate train and test data to feed all through the selected
    # estimators. This is to create a n+m * n+m kernel matrix where n is number of training
    # points and m in number of testing points
    dataset = np.concatenate((train_data, test_data), axis=0)

    SAMPLE_SIZE = len(dataset)

    # The number of selected estimators used for building the partial kernels.
    M = 100

    #Loop that generates samples of the PDF

    kernel_list = np.empty([M, SAMPLE_SIZE, SAMPLE_SIZE])

    for m in range(M):
        print("Building partial kernel: {}".format(m))
        kernel_list[m,:,:] = get_partial_kernel(forest, dataset)

    #Average the samples to compute the kernel
    kernel = np.mean(kernel_list, axis=0)

    return kernel

