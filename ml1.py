from sklearn import datasets
iris = datasets.load_iris()
x_iris, y_iris = iris.data, iris.target
print x_iris.shape, y_iris.shape
print x_iris[0], y_iris[0]