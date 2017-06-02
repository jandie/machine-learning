from sklearn import neighbors
import logParser

log_train = logParser.get_log("generated-log-train.csv")
log_test = logParser.get_log("generated-log-test.csv")
train_data = []
train_target = []
test_data = []
test_target = []

# load
for entry in log_train:
    train_data.append([entry[0], entry[1]])

    train_target.append(entry[2])

for entry in log_test:
    test_data.append([entry[0], entry[1]])

    test_target.append(entry[2])

# train
clf = neighbors.KNeighborsClassifier()
clf.fit(train_data, train_target)

result = clf.predict(test_data)


from sklearn.metrics import accuracy_score

score = accuracy_score(result, test_target)

print(score)


