from sklearn import datasets
import numpy as np

# Load Iris Dataset
iris = datasets.load_iris()
x_iris, y_iris = iris.data, iris.target

# Print Shape of Data and Target x= 150 instances * 4 features (sepal len, width ; petal len, width)
print x_iris.shape, y_iris.shape

# Print an example Instance from x_iris, & Species it belongs to (y_iris)
print x_iris[0], y_iris[0]

# Print all names of Species 0: Setosa, 1: Versicolor, 2: Virginica
print iris.target_names


# Try predicting species using just sepal length & sepal width

# Building a Training Dataset
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

# Get dataset with only Frist Two Attributes (Sepal length, width)
x, y = x_iris[:, :2], y_iris

# Split Data into Training and Testing Set
# Test Set will be the 25% Taken Randomly using test_train_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 33)

# Print shapes of training sets (x_train has 112 elements, 25% of 150 taken by x_test)
print x_train.shape, y_train.shape

# Standardize the features (***Feature Scaling***)
# For each feature, calculate the average, subtract the mean value from feature_value,
# and divide the result by their Standard deviation. After Scaling, each feature
# will have a Zero Average, with Standard Deviation of 1
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Print new x_train
print x_train

# Plot Graphs and Curves
import matplotlib.pyplot as plt
from matplotlib import pylab

# Colours of Species (Identifiers)
colors = ['red', 'greenyellow', 'blue']

# All identifiers
for i in xrange(len(colors)):
	xs = x_train[:, 0][y_train == i]
	ys = x_train[:, 1][y_train == i]
	plt.scatter(xs,ys, c=colors[i])

# Plot Specifications
plt.legend(iris.target_names)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal width')

# Show Plot
plt.show()

# Import Stochastic Gradient Descent Classifier (finds local minimum of a function)
from sklearn.linear_model import SGDClassifier

# Create Classifier Object
clf = SGDClassifier()
# Fit function (receives the training data and training classes, and builds Classifier)
clf.fit(x_train, y_train)

# State Min and Max Values
x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5

xs = np.arange(x_min, x_max, 0.5)
# Three Subplots for 3 Species Classification
fig, axes = plt.subplots(1,3)
# Set Size (distance from Left and Bottom Border) of SubGraphs
fig.set_size_inches(8,6)

for i in [0,1,2]:
	axes[i].set_aspect('equal')
	axes[i].set_title('Class ' + str(i) + ' versus the rest')
	axes[i].set_xlabel('Sepal Length')
	axes[i].set_ylabel('Sepal Width')
	axes[i].set_xlim(x_min, x_max)
	axes[i].set_ylim(y_min, y_max)
	pylab.sca(axes[i])
	plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap= plt.cm.prism)
	ys = (-clf.intercept_[i] - xs * clf.coef_[i,0])/ clf.coef_[i,1]
	plt.plot(xs, ys, hold=True)

# Show Triple Binary Classifier
plt.show()

# Predicts the Species of Flower with Sepal Width 4.7 and Sepal Length 3.1
# Selects the Class in which it is more confident (Boundary line whose distance
# to instance is longer)
print clf.predict(scaler.transform([[4.7, 3.1]]))

# Prints distance of all three boundary lines from the Point(4.7, 3.1)
print clf.decision_function(scaler.transform([[4.7, 3.1]]))
