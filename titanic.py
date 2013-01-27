import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.cross_validation import KFold

# Read in the training data and take a quick look
train = pd.read_csv('train.csv')
pd.scatter_matrix(train)
plt.show()

# Select out the columns with numerical data
train_num = train[['survived', 'age','pclass','sibsp','parch', 'fare']].dropna(axis=0)
train_num_a = np.array(train_num)

# Create a support vector machine classifier
# using KFold for cross-validation
kf = KFold(len(train_num), 5, indices=False)
accuracy = []
rand_accuracy = [] # random classifier - want to do better
for train, test in kf:
    train_data = train_num_a[train,1:]
    train_y = train_num_a[train,0]
    test_data = train_num_a[test,1:]
    test_y = train_num_a[test,0]
    svc = svm.SVC(kernel='rbf', C=1.).fit(train_data, train_y)
    predictions = svc.predict(test_data)
    accuracy.append(metrics.accuracy_score(test_y, predictions))
    rand_pred = np.random.random(test_y.shape[0]) # 549 out of 891 died
    rand_pred[rand_pred > 549./891] = 1 # 549 out of 891 died
    rand_pred[rand_pred <= 549./891] = 0
    rand_accuracy.append(metrics.accuracy_score(test_y, rand_pred))
print accuracy, 'mean: ', np.array(accuracy).mean()
print rand_accuracy, 'mean: ', np.array(rand_accuracy).mean()


