# Data set: https://en.wikipedia.org/wiki/Iris_flower_data_set
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

iris = load_iris()
test_idx = [0, 10, 50, 100, 140]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# train
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# predict
predicted = iris.target_names[clf.predict(test_data)]
answer = iris.target_names[iris.target[test_idx]]

print 'Predicted : ' + \
      str(predicted)

print 'Answer : ' + \
      str(answer)

if np.array_equal(predicted, answer):
    print 'Prediction was correct!'
else:
    print 'Prediction incorrect!'

