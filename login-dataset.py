print(__doc__)

from random import randint

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import logParser

log_train = logParser.get_log("generated-log-train.csv")
log_test = logParser.get_log("generated-log-test.csv")
train_data = []
train_target = []
test_data = []
test_target = []

positive_count = 0
negative_count = 0

# load
for entry in log_train:
    if entry[2] == '1' and randint(0, 1250) == randint(0, 1250):
        train_data.append([float(entry[0]), float(entry[1])])
        train_target.append(float(entry[2]))

        positive_count += 1
    elif entry[2] == '0' and randint(0, 2) == randint(0, 2):
        train_data.append([float(entry[0]), float(entry[1])])
        train_target.append(float(entry[2]))

        negative_count += 1

X_train = np.asarray(train_data)
y_train = np.asarray(train_target)

print (positive_count)
print (negative_count)

positive_count = 0
negative_count = 0

for entry in log_test:
    if entry[2] == '1' and randint(0, 1250) == randint(0, 1250):
        test_data.append([float(entry[0]), float(entry[1])])
        test_target.append(float(entry[2]))

        positive_count += 1
    elif entry[2] == '0' and randint(0, 2) == randint(0, 2):
        test_data.append([float(entry[0]), float(entry[1])])
        test_target.append(float(entry[2]))

        negative_count += 1

X_test = np.asarray(test_data)
y_test = np.asarray(test_target)

x_list = []
x_list.extend(X_train)
x_list.extend(X_test)
X = np.asarray(x_list)

y_list = []
y_list.extend(y_train)
y_list.extend(y_test)
Y = np.asarray(y_list)

print (positive_count)
print (negative_count)

h = 3  # step size in the mesh
datasets = 3  # amount of datasets

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

figure = plt.figure(figsize=(27, 9))
i = 1

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(3, len(classifiers) + 1, i)

ax.set_title("Input data")
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(datasets, len(classifiers) + 1, i)
    clf.fit(X_train, y_train)

    result = clf.predict(X_test)

    from sklearn.metrics import accuracy_score

    score = round(accuracy_score(result, y_test), 4)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    N_split = 10

    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, score,
            size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()
