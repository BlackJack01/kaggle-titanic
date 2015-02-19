
import pandas as pd
import statsmodels as sm
import numpy as np

import sklearn

from sklearn import cross_validation
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

df = pd.read_csv("data/train.csv")
#print "Columns: ", ",".join(list(df.columns))
#print "Length: ", len(df)
print df.head()
df = df[np.isfinite(df['Age'])]
df = df[np.isfinite(df['Fare'])]

#label_encoder = preprocessing.LabelEncoder()
#df['Survived'] = label_encoder.fit_transform(df['Survived'])
#df['Survived'] = df['Survived?'] == 'True.'

y = df['Survived'].as_matrix().astype(np.int)
X = df[['Age','Fare']].as_matrix().astype(np.float)

print('There are {} instances for survived class and {} instances for not-survived classes.'.format(y.sum(), y.shape[0] - y.sum()))
print('Ratio of survived class over all instances: {:.2f}'.format(float(y.sum()) / y.shape[0]))

#df.drop(['Age'], axis=1, inplace=True)


def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
	stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
	y_pred = y.copy()
	for ii, jj in stratified_k_fold:
		X_train, X_test = X[ii], X[jj]
		y_train = y[ii]
		clf = clf_class(**kwargs)
		clf.fit(X_train,y_train)
		y_pred[jj] = clf.predict(X_test)
	return y_pred
	

print("#####################################")	
print('Passive Aggressive Classifier: {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, linear_model.PassiveAggressiveClassifier))))
print('Gradient Boosting Classifier:  {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))
print('Support vector machine(SVM):	  {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, svm.SVC))))
print('Random Forest Classifier:	  {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
print('K Nearest Neighbor Classifier: {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))))
print('Logistic Regression:			  {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, linear_model.LogisticRegression))))
print("#####################################")	
