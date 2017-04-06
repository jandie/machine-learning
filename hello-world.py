from sklearn import tree

# Training data:
# smooth = 1
# bumpy = 0
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
texture_names = ['bumpy', 'smooth']

# Labels:
# apple = 0
# orange = 1
labels = [0, 0, 1, 1]
fruit_names = ['apple', 'orange']

# train
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)

# predict
print fruit_names[clf.predict([[200, 0]])[0]]

