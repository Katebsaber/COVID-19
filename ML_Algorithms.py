from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
import numpy as np

df = pd.read_csv("data_20.csv")
X = df.drop(labels=["Unnamed: 0", "fileName", "label"], axis=1)
y = df["label"]

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:100]

idx = [str(x) for x in indices]

X = X[idx]

s = svm.SVC(kernel='sigmoid')
score_s = cross_val_score(s, X, y, cv=10, scoring='accuracy')

print(f"SVM score: {score_s}")
print(f"SVM mean score: {sum(score_s)/len(score_s)}")

r = RandomForestClassifier()
score_r = cross_val_score(r, X, y, cv=10, scoring='accuracy')

print(f"Random Forrest score: {score_r}")
print(f"Random Forrest mean score: {sum(score_r)/len(score_r)}")

b = BaggingClassifier(base_estimator=r, n_estimators=10)
score_b = cross_val_score(b, X, y, cv=10, scoring='accuracy')

print(f"Bagging score: {score_b}")
print(f"Bagging mean score: {sum(score_b)/len(score_b)}")
