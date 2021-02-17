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



df = pd.read_csv('sp500_27270.csv')


corr = df.iloc[:, 1:].corr().iloc[0, :]
# values = []
# for val in corr:
# 	if (val >= 0.95 or val <= -0.95) and val:
# 		values.append(val)


# print(values)	

# v = df.ix[:, 1:].corr()
# print(v)


# v = v.T
corr = corr.to_frame().iloc[1:, :]
res = corr[(corr.SP500>=0.95) | (corr.SP500 <=-0.95)]

res1 = res[(res.SP500 == res.max()[0])]

# print(res1)


my_df = df[['APH']]
# print(my_df)
X = np.array(my_df.iloc[0:9, :].T)[0][0:5].reshape(1,5)
Y = np.array(my_df.iloc[0:9, :].T)[0][5:10].reshape(1,4)
# print(Y)

for i in range(1,1252):
	values = np.array(my_df.iloc[i:i+9, :].T)[0]
	X = np.append(X, values[0:5].reshape(1, 5), axis=0)
	Y = np.append(Y, values[5:10].reshape(1,4), axis=0)
	# print(values[0:6].reshape(1, 6))

	
# X_scaled = preprocessing.scale(X)
# Y_scaled = preprocessing.scale(Y)

scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

scaler1 = preprocessing.StandardScaler()
scaler1.fit(Y)
Y_scaled = scaler1.transform(Y)




# print(x)
# X_scaled = X
# Y_scaled = Y



X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,Y_scaled,test_size=0.2)

reg = MLPRegressor(solver='adam', learning_rate_init=0.001, early_stopping = True, shuffle=True)

reg.fit(X_train, Y_train)
predictions = reg.predict(X_test)

preds = predictions
preds = scaler1.inverse_transform(preds)

# print(preds)

# print(preds)
trues = Y_test
trues = scaler1.inverse_transform(trues)


# score = metrics.r2_score(predictions,Y_test)
mse = metrics.mean_squared_error(predictions, Y_test)
print(mse)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


preds_flat = preds[:3].flatten()

trues_flat = trues[:3].flatten()

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])

ax.scatter(preds_flat, trues_flat, c='b')
ax.grid()
ax.legend(loc='best')
ax.set_xlabel('Predictions')
ax.set_ylabel('True Values')
ax.set_title('Predicted VS True')

plt.show()




	



