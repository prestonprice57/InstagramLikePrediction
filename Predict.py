import json
import requests
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
import numpy as np
from sklearn import ensemble
import pandas as pd
from pandas.io.json import json_normalize
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities           import percentError

import indicoio
from instagram.client import InstagramAPI

client_id = "bec82b4b69cc435998eb2c9f82212fb4"
client_secret = "6f7cd017a78945afaffcd992840a8fe5"
access_token = "1147536024.bec82b4.fb48b565d9ad4fe09f64f63d64d4f664"
INDICO_API_KEY = '61cdd30af4bbdfe5a21b92689a872234'


api = InstagramAPI(access_token=access_token, client_secret=client_secret)
indicoio.config.api_key = INDICO_API_KEY

def get_user_prediction(fit):
	url = 'https://api.instagram.com/v1/users/self/?access_token=%s' % access_token
	resp = requests.get(url=url)
	data = resp.json()
	followers = data['data']['counts']['followed_by']
	follows = data['data']['counts']['follows']
	day = 0
	hour_float = 0.0
	image_url = ''
	new_post = []
	count = 4
	recent_media, next = api.user_recent_media(user_id='self', count=count)
	for i, media in enumerate(recent_media):
		new_post = []
		image_url = media.get_standard_resolution_url()

		day = media.created_time.weekday()
		hour = str(media.created_time.hour) + ':' + str(media.created_time.minute)
		likes = media.like_count
		hashtags = len(media.tags)

		if i == count-1:
			captionSentiment = 0.5
			if media.caption != None:
				caption = media.caption.text.replace('\n', ' ').replace('\r', ' ').encode('utf-8')
				captionSentiment = indicoio.sentiment(caption)
			fer = indicoio.fer(image_url)

			new_hour, minute = hour.split(':')
			hour_float = new_hour
			
			new_post.append(captionSentiment)
			new_post.append(hour_float)
			new_post.append(follows)
			new_post.append(fer['Angry'])
			new_post.append(hashtags)
			new_post.append(day)
			new_post.append(fer['Neutral'])
			new_post.append(followers)
			new_post.append(fer['Surprise'])
			if follows > 0:
				new_post.append(float(followers)/follows)
			else:
				new_post.append(float(followers))
			new_post.append(fer['Sad'])
			new_post.append(fer['Fear'])
			new_post.append(fer['Happy'])

			if followers > 0:
				target = float(likes) / followers
			else:
				target = float(likes)

	print image_url
	print day
	print hour
	for i in xrange(0,24):
		new_post[1] = i
		predict = fit.predict(new_post)
		prediction = predict[0]*followers
		print "Predicted:", int(prediction)
		print "Actual:", likes
		print
		print

def stack(x_train, target_train, x_test):

	knn = neighbors.KNeighborsRegressor()
	knn = knn.fit(x_train, target_train)
	predict3 = knn.predict(x_test)

	bag = ensemble.BaggingRegressor(neighbors.KNeighborsRegressor(n_neighbors=9, weights='uniform'), max_samples=1.0, max_features=0.7)
	bag = bag.fit(x_train, target_train)
	predict5 = bag.predict(x_test)

	forest = ensemble.RandomForestRegressor(n_estimators=20)
	forest = forest.fit(x_train, target_train)
	predict7 = forest.predict(x_test)

	e_forest = ensemble.ExtraTreesRegressor(n_estimators=20)
	e_forest = e_forest.fit(x_train, target_train)
	predict8 = e_forest.predict(x_test)

	for i, arr in enumerate(x_test):
		#arr.append(predict1[i])
		#arr.append(predict2[i])
		arr.append(predict3[i])
		#arr.append(predict4[i])
		arr.append(predict5[i])
		#arr.append(predict6[i])
		arr.append(predict7[i])
		arr.append(predict8[i])

	return x_test


def average(x_train, target_train, x_test):
	clf = DecisionTreeRegressor(max_depth=20)
	clf = clf.fit(x_train, target_train)
	predict1 = clf.predict(x_test)

	sv = svm.SVR(kernel='rbf')
	sv = sv.fit(x_train, target_train) 
	predict2 = sv.predict(x_test)

	knn = neighbors.KNeighborsRegressor()
	knn = knn.fit(x_train, target_train)
	predict3 = knn.predict(x_test)

	bayes = linear_model.BayesianRidge()
	bayes = bayes.fit(x_train, target_train)
	predict4 = bayes.predict(x_test)

	bag = ensemble.BaggingRegressor(neighbors.KNeighborsRegressor(n_neighbors=9, weights='uniform'), max_samples=1.0, max_features=0.7)
	bag = bag.fit(x_train, target_train)
	predict5 = bag.predict(x_test)

	sv = svm.SVR(kernel='rbf')
	sv = sv.fit(x_train, target_train) 
	predict6 = sv.predict(x_test)

	forest = ensemble.RandomForestRegressor(n_estimators=50)
	forest = forest.fit(x_train, target_train)
	predict7 = forest.predict(x_test)

	e_forest = ensemble.ExtraTreesRegressor(n_estimators=50)
	e_forest = e_forest.fit(x_train, target_train)
	predict8 = e_forest.predict(x_test)

	prediction = []
	for i, num in enumerate(predict1):
		temp = [predict1[i], predict2[i], predict3[i], predict4[i], predict5[i], predict6[i], predict7[i], predict8[i]]
		avg = float(sum(temp))/len(temp)
		prediction.append(avg)

	return prediction

	#
	#
	# Get the accuracy
def getAccuracy(predict, target_test):
	correct = 0
	wrong = 0
	for i, val in enumerate(predict):
	    #print i, "predicted:", val, "actual:", target_test[i]
	    accuracyScale = 0.5
	    accuracy = target_test[i] * accuracyScale
	    if target_test[i] - accuracy <= val <= target_test[i] + accuracy:
	        correct += 1
	    else:
	        wrong += 1


	#
	#
	# Print results
	print 'CORRECT:', correct
	print 'WRONG:', wrong
	print 'TOTAL ACCURACY:', float(correct) / len(predict)
	print 


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
    d['hour'] = hour
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
x_train = x[0:10000]
x_train2 = x[10000:20000]
x_test = x[20000:]
target_train = target[0:10000]
target_train2 = target[10000:20000]  
target_test = target[20000:]

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
bagging = False
random_forest = True
extra_random_forest = False

stacking = False
averaging = False

if stacking:
	x_train2 = stack(x_train,target_train, x_train2)
	x_test = stack(x_train, target_train, x_test)

predict = []
if kNN:
	# K Nearest Neighbors
	# More accurate than Naive Bayes
	# There are more predicted ratios that fall into the correct range, but there are much crazier outliers!
	# Example: predicted: 51.3, actual: 5.3
	knn = neighbors.KNeighborsRegressor()
	knn = knn.fit(x_train2, target_train2)
	predict = knn.predict(x_test)

	print "kNN"
	getAccuracy(predict, target_test)
	#get_user_prediction(knn)
if naive_bayes:
	# Naive Bayes
	# The predicted like ratios, on average, are usually a lot higher than the actual ratios
	# Example: predicted: 12.3, actual: 1.6
	bayes = linear_model.BayesianRidge()
	bayes = bayes.fit(x_train2, target_train2)
	predict = bayes.predict(x_test)

	print "Naive Bayes"
	getAccuracy(predict, target_test)
	#get_user_prediction(bayes)
if linear:
	# Linear Model
	lr = linear_model.LinearRegression(normalize=True)
	lr = lr.fit(x_train2, target_train2)
	predict = lr.predict(x_test)

	print "Linear Regression"
	getAccuracy(predict, target_test)
	#get_user_prediction(linear)
if decisionTree:
	# Decision Tree
	clf = DecisionTreeRegressor(max_depth=20)
	clf = clf.fit(x_train2, target_train2)
	predict = clf.predict(x_test)

	print "Decision Tree"
	getAccuracy(predict, target_test)
	#get_user_prediction(clf)
if support_vector_machine:
	# SVM
	sv = svm.SVR(kernel='sigmoid')
	sv = sv.fit(x_train2, target_train2) 
	predict = sv.predict(x_test) 

	print "Support Vector Machine"
	getAccuracy(predict, target_test)
	#get_user_prediction(sv) 
if bagging:
	bag = ensemble.BaggingRegressor(neighbors.KNeighborsRegressor(n_neighbors=9, weights='uniform'), max_samples=1.0, max_features=0.7)
	bag = bag.fit(x_train2, target_train2)
	predict = bag.predict(x_test)

	print "Bagging"
	getAccuracy(predict, target_test)
	#get_user_prediction(bag)
if random_forest:
	forest = ensemble.RandomForestRegressor(n_estimators=200)
	forest = forest.fit(x_train2, target_train2)
	predict = forest.predict(x_test)

	print "Random Forest"
	getAccuracy(predict, target_test)
	get_user_prediction(forest)
if extra_random_forest:
	e_forest = ensemble.ExtraTreesRegressor(n_estimators=200)
	e_forest = e_forest.fit(x_train2, target_train2)
	predict = e_forest.predict(x_test)

	print "Extra Random Forest"
	getAccuracy(predict, target_test)
	#get_user_prediction(e_forest)
if averaging:
	predict = average(x_train2, target_train2, x_test)

	print "Averaging"
	getAccuracy(predict, target_test)


'''

net = buildNetwork(13, 10, 1)
ds = SupervisedDataSet(13,1)
ds_test = SupervisedDataSet(13,1)

for i, data in enumerate(x_train):
	ds.addSample(data, target_train[i])

for i, data in enumerate(x_test):
	ds.addSample(data, target_test[i])

trainer = BackpropTrainer(net, dataset=ds, momentum=0.1, weightdecay=0.01)
trainer.trainEpochs(5)

trnresult = percentError(trainer.testOnClassData(),target_train)
tstresult = percentError(trainer.testOnClassData(dataset=ds_test), target_test)

print error

'''