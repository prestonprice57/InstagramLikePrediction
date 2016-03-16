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

target = []
for d in json_dict['data']['posts']:
	target.append(d['likeRatio'])


for d in json_dict['data']['posts']:
	hour, minute = d['hour'].split(':')
	d['hour'] = float(hour+minute)
	del d['likeRatio']
	del d['likes']
data = json_dict['data']['posts']


x = []
for d in data:
	temp = []
	for key,val in d.iteritems():
		temp.append(val)
	x.append(temp)

x_train = x[0:int(len(x)*0.9)]
x_test = x[int(len(x)*0.9):]
target_train = target[0:int(len(target)*0.9)]
target_test = target[int(len(target)*0.9):]

#for val in x_test:
#	val[8] = 0
'''
clf = DecisionTreeRegressor(max_depth=20)
clf = clf.fit(x_train, target_train)

predict = clf.predict(x_test)
'''
sv = svm.SVR(kernel='poly')
sv = sv.fit(x_train, target_train)
predict = sv.predict(x_test)
'''

bayes = linear_model.BayesianRidge()
bayes = bayes.fit(x_train, target_train)
predict = bayes.predict(x_test)

knn = neighbors.KNeighborsRegressor()
knn = knn.fit(x_train, target_train)
predict = knn.predict(x_test)


lr = linear_model.LinearRegression(normalize=True)
lr = lr.fit(x_train, target_train)
predict = lr.predict(x_test)
'''

correct = 0
wrong = 0
for i, val in enumerate(predict):
	print i, val, target_test[i]
	accuracy = target_test[i]*1
	if target_test[i]-accuracy <= val <= target_test[i]+accuracy:
		correct+=1
	else:
		wrong+=1

print 'CORRECT:', correct
print 'WRONG:', wrong
print 'TOTAL ACCURACY:', float(correct)/len(predict)


