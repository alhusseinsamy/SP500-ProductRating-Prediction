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
corr = corr.to_frame().iloc[:, :]
res = corr[(corr.SP500>=0.95) | (corr.SP500 <=-0.95)] 

columns = (res.iloc[:, 0]).to_frame().index
cols = []

for val in columns:
	cols.append(val)


my_df = df[cols]
# print(my_df)

x = np.array(my_df.iloc[:, 1:])

y = np.array(my_df.iloc[:, 0])

X_scaled = preprocessing.scale(x)
Y_scaled = preprocessing.scale(y)

scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled = scaler.transform(x)

scaler1 = preprocessing.StandardScaler()
scaler1.fit(y.reshape(len(y), 1))
Y_scaled = scaler1.transform(y.reshape(len(y), 1))

Y_scaled = Y_scaled.reshape(len(Y_scaled))




# print(x)
# X_scaled = x
# Y_scaled = y



X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,Y_scaled,test_size=0.2)

# clf = MLPRegressor()

# param_grid = {'solver' : ['adam', 'sgd'], 'learning_rate_init': [0.0001, 0.001, 0.01]}
# gs = GridSearchCV(clf, param_grid=param_grid)
# gs.fit(X_train, Y_train)
# gs.best_params_
# print(gs.best_params_)

reg = MLPRegressor(solver='adam', learning_rate_init=0.001, early_stopping = True, shuffle=True)
# seed = 7
# np.random.seed(seed)
# kfold = KFold(n_splits=5, random_state=seed)
# results = cross_val_score(
#     estimator = reg,
#     X = X_scaled,
#     y = Y_scaled,
#     cv=10,


# )
reg.fit(X_train, Y_train)
predictions = reg.predict(X_test)


preds = predictions.reshape(len(predictions), 1)
preds = scaler1.inverse_transform(preds)
preds = preds.reshape(len(preds))
# print(preds)
trues = Y_test.reshape(len(Y_test), 1)
trues = scaler1.inverse_transform(trues)
trues = trues.reshape(len(trues))
# print(trues)


score = metrics.r2_score(predictions,Y_test)
mse = metrics.mean_squared_error(predictions, Y_test)
print(mse)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))




fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])

ax.scatter(preds, trues, c='b')
ax.grid()
ax.legend(loc='best')
ax.set_xlabel('Predictions')
ax.set_ylabel('True Values')
ax.set_title('Predicted VS True')

plt.show()

