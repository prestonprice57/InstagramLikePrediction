import json
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
import numpy as np

json_file = open('data.json')
json_str = json_file.read()
json_dict = json.loads(json_str)
print type(json_dict)


#
#
# Save all of the like ratios as targets
target = []
for d in json_dict['data']['posts']:
    print d['likeRatio']
    target.append(d['likeRatio'])


#
#
# Now remove the like ratios AND the likes (so that we can predict)
for d in json_dict['data']['posts']:
    hour, minute = d['hour'].split(':')
    d['hour'] = float(hour + minute)
    del d['likeRatio']
    del d['likes']
data = json_dict['data']['posts']


#
#
# Copy the data
x = []
for d in data:
    temp = []
    for key, val in d.iteritems():
        temp.append(val)
    x.append(temp)


#
#
# Prepare the training and testing sets
x_train = x[0:int(len(x) * 0.9)]
x_test = x[int(len(x) * 0.9):]
target_train = target[0:int(len(target) * 0.9)]
target_test = target[int(len(target) * 0.9):]


#
#
# Decision Tree
# clf = DecisionTreeRegressor(max_depth=20)
# clf = clf.fit(x_train, target_train)
# clf.predict(x_test)    # TODO: This classifier has no .predict method


# SVM
# sv = svm.SVR(kernel='poly')
# sv = sv.fit(x_train, target_train)   # TODO: ONE OF THESE LINES CAUSES ISSUES
# predict = sv.predict(x_test)         # TODO: ONE OF THESE LINES CAUSES ISSUES


#
#
# Naive Bayes
# The predicted like ratios, on average, are usually a lot higher than the actual ratios
# Example: predicted: 12.3, actual: 1.6
bayes = linear_model.BayesianRidge()
bayes = bayes.fit(x_train, target_train)


#
#
# K Nearest Neighbors
# More accurate than Naive Bayes
# There are more predicted ratios that fall into the correct range, but there are much crazier outliers!
# Example: predicted: 51.3, actual: 5.3
knn = neighbors.KNeighborsRegressor()
knn = knn.fit(x_train, target_train)


#
#
# Linear Model
lr = linear_model.LinearRegression(normalize=True)
lr = lr.fit(x_train, target_train)


#
#
# Which model do we want to test?
kNN = False
naive_bayes = True
linear = False
decisionTree = False

if kNN:
    predict = knn.predict(x_test)
elif naive_bayes:
    predict = bayes.predict(x_test)
elif linear:
    lr.predict(x_test)


#
#
# Get the accuracy
correct = 0
wrong = 0
for i, val in enumerate(predict):
    print i, "predicted:", val, "actual:", target_test[i]
    accuracy = target_test[i] * 1
    if target_test[i] - accuracy <= val <= target_test[i] + accuracy:
        correct += 1
    else:
        wrong += 1


#
#
# Print results
print '\n'
print 'CORRECT:', correct
print 'WRONG:', wrong
print 'TOTAL ACCURACY:', float(correct) / len(predict)
