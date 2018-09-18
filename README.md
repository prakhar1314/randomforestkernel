# The Random Forest Kernel
The commonly used kernels are usually unsupervised.
Though having proven their worth, they usually don’t adapt to
the underlying statistics of the data. We have a wide range of
known kernel methods, as the Linear kernel, Periodic kernel,
Radial Basis function (RBF) and Polynomial to mention some
of them. The most popular kernel is the RBF kernel, and it is
often presented as the default kernel.

In this project, we describe a supervised kernel method called
the Random Forest Kernel. We showed how we can sample
random partitions from the dataset using an implementation
of the Random Forest.

Details of the python files
# help_functions.py
contains function to access every information of the trees built by random forest like getting parent/child of nodes, leave nodes, height of the trees etc.

# RandomForestKernel.py
gives the supervised Random Forest Kernel which can be given input to Support Vector Regression, Gaussian Process Regression as custom kernel.

# SVRRandomForest.py
gives the comparison of Random Forest kernel with RBF kernel on different regression dataset. Any dataset can be choosen to obtain the comparison with different plots.

# SVRRandomForestMSEDataset.py
gives the comparison of overall statistics(bargraph) of applying SVR with RDF kernel and Random Forest Kernel.
