import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

# Read in the training data
train = pd.read_csv('train.csv')

# Select out the columns with numerical data
train_num = train[['survived', 'age','pclass','sibsp','parch', 'fare']].dropna(axis=0)

# Convert the sex into a numerical value and concatenate it
# on the dataeframe of numerical data
sex = train.ix[:,['sex']]
sex.columns = ['sexnumeric']
sex[sex == 'male'] = 1
sex[sex == 'female'] = 0
sex = sex.astype(np.uint8)
train_num = pd.concat([train_num, sex], axis=1, join_axes=[train_num.index])

# Try various different model parameters to find the best
param_grid = [
  {'C': [1e-2, 1e-1, 1, 1e1, 1e2,], 
      'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 
      'kernel': ['rbf']},
 ]

# Split the dataset in two parts
X_train, X_test, y_train, y_test = train_test_split(
        train_num.values[:,1:], train_num.values[:,0], test_size=0.3, random_state=0)

clf = GridSearchCV(svm.SVC(C=1), param_grid, 
        score_func=metrics.accuracy_score)
clf.fit(X_train, y_train, cv=3)

print clf.best_estimator_
for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params)
print 'Testing best model on holdout data'
y_true, y_pred = y_test, clf.predict(X_test)
print metrics.classification_report(y_true, y_pred)
print metrics.accuracy_score(y_true, y_pred)
