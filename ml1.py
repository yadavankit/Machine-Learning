from sklearn import datasets

#Load Iris 
iris = datasets.load_iris()
x_iris, y_iris = iris.data, iris.target

#Print Shape of Data and Target x= 150 instances * 4 features (sepal len, width ; petal len, width)
print x_iris.shape, y_iris.shape

#Print an example Instance from x_iris, & Species it belongs to (y_iris)
print x_iris[0], y_iris[0]

#Print all names of Species 0: Setosa, 1: Versicolor, 2: Virginica
print iris.target_names


#Try predicting species using just sepal length & sepal width

#Building a Training Dataset
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

#Get dataset with only Frist Two Attributes (Sepal length, width)
x, y = x_iris[:, :2], y_iris

#Split Data into Training and Testing Set
#Test Set will be the 25% Taken Randomly using test_train_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 33)

#Print shapes of training sets (x_train has 112 elements, 25% of 150 taken by x_test)
print x_train.shape, y_train.shape

#Standardize the features (***Feature Scaling***)
#For each feature, calculate the average, subtract the mean value from feature_value,
#and divide the result by their Standard deviation. After Scaling, each feature
#will have a Zero Average, with Standard Deviation of 1
scalar = preprocessing.data.StandardScalar().fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

#Print new x_train
print x_train

import matplotlib.pyplot as plt
colors = ['red', 'greenyellow', 'blue']
for i in xrange(len(colors)):
	xs = x_train[:, 0][y_train == i]
	ys = x_train[:, 1][y_train == i]
	plt.scatter(xs,ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
