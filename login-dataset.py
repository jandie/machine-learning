from sklearn import neighbors
from sklearn import tree
import logParser
from random import randint

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
    if entry[2] == '1' and randint(0, 1200) == randint(0, 1200):
        train_data.append([entry[0], entry[1]])
        train_target.append(entry[2])

        positive_count += 1
    elif entry[2] == '0':
        train_data.append([entry[0], entry[1]])
        train_target.append(entry[2])

        negative_count += 1

print positive_count
print negative_count

positive_count = 0
negative_count = 0

for entry in log_test:
    if entry[2] == '1' and randint(0, 1200) == randint(0, 1200):
        test_data.append([entry[0], entry[1]])
        test_target.append(entry[2])

        positive_count += 1
    elif entry[2] == '0':
        test_data.append([entry[0], entry[1]])
        test_target.append(entry[2])

        negative_count += 1

print positive_count
print negative_count

# train
clf = neighbors.KNeighborsClassifier()
# clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

result = clf.predict(test_data)

from sklearn.metrics import accuracy_score

score = accuracy_score(result, test_target)

print(score)


