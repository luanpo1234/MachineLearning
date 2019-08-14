import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from scipy.stats import mode

X, y = make_moons(n_samples=10000, noise=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y)

tree_clf = DecisionTreeClassifier(max_leaf_nodes=4, min_samples_leaf=2, min_samples_split=2)

def get_best_par():
    parameters = {"max_leaf_nodes": [2, 3, 4, 5, 6, 7, 8],
                  "min_samples_leaf": [2, 3, 4, 5, 6, 7, 8],
                  "min_samples_split": [2, 3, 4, 5, 6, 7, 8],}
    
    clf = GridSearchCV(tree_clf, parameters, cv=3)
    clf.fit(X_train, y_train)
    
    return clf.best_params_

tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

rs = ShuffleSplit(n_splits=1000, train_size=100, test_size=0)
index_list = rs.split(X_train)
forest = []
for l in index_list:
    X_, y_ = [], []
    for i in l:
        X_.append(X_train[i])
        y_.append(y_train[i])
    X_ = np.asarray(X_[0])
    y_ = np.asarray(y_[0])
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=4, min_samples_leaf=2, min_samples_split=2)
    tree_clf.fit(X_, y_)
    forest.append(tree_clf)

y_pred_maj_vote = []
for i, el in enumerate(X_test):
    print(i, "/", len(X_test))
    most_freq = []
    for tree in forest:
        most_freq.append(tree.predict(el.reshape(1, -1))) #reshaping to deal with single feature
    y_pred_maj_vote.append(mode(most_freq)[0][0])
    
print(accuracy_score(y_test, y_pred_maj_vote))