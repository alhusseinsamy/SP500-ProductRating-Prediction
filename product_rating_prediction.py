import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer



reviews = []
ratings = []
count1 = 2500
count2 = 2500
count3 = 2500
count4 = 2500
count5 = 2500

linesct = 1

with open('Digital_Music_5.json') as f:
	line = f.readline()
	x1 = json.loads(line)
	overall1 = x1['overall']
	ratings.append(overall1)
	reviews.append(x1['reviewText'])

	if (overall1 == 1.0):
		count1-=1
	

	if (overall1 == 2.0):
		count2-=1
	

	if (overall1 == 3.0):
		count3-=1
	

	if (overall1 == 4.0):
		count4-=1
	

	if (overall1 == 5.0):
		count5-=1
	

	line = f.readline()	

	while line:
		linesct+=1
		if((count1 == 0) and (count2==0) and (count3==0) and (count4==4) and (count5==0)):
			break

		x = json.loads(line)
		overall = x['overall']
		# print(overall)
		

		

		if (overall == 1.0):
			if(count1 != 0):
				count1-=1
				reviews.append(x['reviewText'])
				ratings.append(overall)	
		if (overall == 2.0):
			if count2 != 0:
				count2-=1
				reviews.append(x['reviewText'])
				ratings.append(overall)
		if (overall == 3.0):
			if count3 != 0:
				count3-=1
				reviews.append(x['reviewText'])
				ratings.append(overall)
		if (overall == 4.0):
			if count4 != 4:
				count4-=1
				reviews.append(x['reviewText'])
				ratings.append(overall)
		if (overall == 5.0):
			if count5 != 0:
				count5-=1
				reviews.append(x['reviewText'])
				ratings.append(overall)

		line = f.readline()	

		# print(count1)

f.close()			

# print(reviews[0])

X = np.array(reviews)
# X = X.reshape(len(X), 1)

Y = np.array(ratings)

le = preprocessing.LabelEncoder()
le.fit(Y)
Y = le.transform(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

# print(X_train[:2])

X_train = X_train.tolist()
X_test = X_test.tolist()

tfidf_vectorizer = TfidfVectorizer(max_features=40000)
X_train_transformed = tfidf_vectorizer.fit_transform(X_train)
X_test_transformed = tfidf_vectorizer.transform(X_test)

# print(X_train_transformed[0].shape)
# print(len(Y_train))

clf = MLPClassifier(hidden_layer_sizes=(64, ), verbose=True, batch_size = 32, early_stopping = True, shuffle=True)

# param_grid = {'hidden_layer_sizes': [(100,), (500,), (1000,), (2000,), (5000,)]}
# gs = GridSearchCV(clf, param_grid=param_grid)
# gs.fit(X_train_transformed, Y_train)
# gs.best_params_
# print(gs.best_params_)

clf.fit(X_train_transformed,Y_train)

predicted_targets = clf.predict(X_test_transformed)
#Check for minimal error
# print(metrics.mean_absolute_error(predicted_targets,Y_test))
print(metrics.accuracy_score(predicted_targets, Y_test))