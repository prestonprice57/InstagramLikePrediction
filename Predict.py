import json
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
import numpy as np
from sklearn import ensemble
import pandas as pd
from pandas.io.json import json_normalize

json_file = open('data.json')
json_str = json_file.read()
json_dict = json.loads(json_str)
print type(json_dict)


#
#
# Save all of the like ratios as targets
target = []
for d in json_dict['data']['posts']:
    target.append(d['likeRatio'])

#
#
# Get summary of data for observation
normalized = json_normalize(json_dict['data']['posts'])
df = pd.DataFrame(normalized)
print df.describe()


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

np_x_train = np.array(x_train)
np_x_test = np.array(x_test)
np_target_train = np.array(target_train)
np_target_test = np.array(target_test)    


#
#
# Which model do we want to test?
support_vector_machine = False
kNN = False
naive_bayes = False
linear = False
decisionTree = False
bagging = True
random_forest = False
extra_random_forest = False
ada_boost = False

predict = []
if kNN:
	# K Nearest Neighbors
	# More accurate than Naive Bayes
	# There are more predicted ratios that fall into the correct range, but there are much crazier outliers!
	# Example: predicted: 51.3, actual: 5.3
	knn = neighbors.KNeighborsRegressor()
	knn = knn.fit(x_train, target_train)
	predict = knn.predict(x_test)
elif naive_bayes:
	# Naive Bayes
	# The predicted like ratios, on average, are usually a lot higher than the actual ratios
	# Example: predicted: 12.3, actual: 1.6
	bayes = linear_model.BayesianRidge()
	bayes = bayes.fit(x_train, target_train)
	predict = bayes.predict(x_test)
elif linear:
	# Linear Model
	lr = linear_model.LinearRegression(normalize=True)
	lr = lr.fit(x_train, target_train)
	predict = lr.predict(x_test)
elif decisionTree:
	# Decision Tree
	clf = DecisionTreeRegressor(max_depth=20)
	clf = clf.fit(x_train, target_train)
	predict = clf.predict(x_test)
elif support_vector_machine:
	# SVM
	sv = svm.SVR(kernel='rbf')
	sv = sv.fit(x_train, target_train) 
	predict = sv.predict(x_test)  
elif bagging:
	bag = ensemble.BaggingRegressor(neighbors.KNeighborsRegressor(n_neighbors=9, weights='uniform'), max_samples=1.0, max_features=0.7)
	bag = bag.fit(x_train, target_train)
	predict = bag.predict(x_test)
elif random_forest:
	forest = ensemble.RandomForestRegressor(n_estimators=20)
	forest = forest.fit(x_train, target_train)
	predict = forest.predict(x_test)
elif extra_random_forest:
	e_forest = ensemble.ExtraTreesRegressor(n_estimators=20)
	e_forest = e_forest.fit(x_train, target_train)
	predict = e_forest.predict(x_test)
elif ada_boost:
	ada = ensemble.AdaBoostRegressor(n_estimators=len(np_x_train)/4)
	ada = ada.fit(np_x_train, np_target_train)
	predict = ada.predict(np_x_test)

#
#
# Get the accuracy
correct = 0
wrong = 0
for i, val in enumerate(predict):
    print i, "predicted:", val, "actual:", target_test[i]
    accuracyScale = 0.5
    accuracy = target_test[i] * accuracyScale
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
