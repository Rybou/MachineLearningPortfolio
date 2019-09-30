import pandas as pd 
import numpy as np
import os

import mglearn
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer

pos= [x for x in os.listdir("train/pos/") if x.endswith(".txt")]
neg= [x for x in os.listdir("train/neg/") if x.endswith(".txt")]
test= [x for x in os.listdir("test/") if x.endswith(".txt")]

posReviews=[]
for txt in pos:
    with open("train/pos/"+txt, encoding="ISO-8859-1") as f:
        posReviews.append(f.read())
negReviews=[]        
for txt in neg:
    with open("train/neg/"+txt, encoding="ISO-8859-1") as f:
        negReviews.append(f.read())
testReviews=[]        
for txt in test:
    with open("test/"+txt, encoding="ISO-8859-1") as f:
        testReviews.append(f.read())

reviews = pd.concat([
    pd.DataFrame({"review":posReviews, "label":1}),
    pd.DataFrame({"review":negReviews, "label":0}),
    pd.DataFrame({"review":testReviews, "label":-1})
], ignore_index=True).sample(frac=1, random_state=1)

# Examine firts 10 rows
reviews.head(10)

# Examine the class ditribution 
reviews.label.value_counts()

# Define X and y from the review dataset  for use with Countvectorizer
X= reviews[reviews.label!=-1].review
y= reviews[reviews.label!=-1].label
print (X.shape)
print (y.shape)

# Split X and y into training and testing/validation sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


vect = TfidfVectorizer(min_df=4, ngram_range=(1, 2))
text_train = vect.fit_transform(X_train)
text_test = vect.transform(X_test)

print("Vocabulary size: {}".format(len(vect.vocabulary_)))
print("text_train:\n{}".format(repr(text_train)))
print("text_test: \n{}".format(repr(text_test)))

feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(text_train, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)

mglearn.tools.visualize_coefficients(grid.best_estimator_.coef_, feature_names, n_top_features=30)
plt.show()

lr = grid.best_estimator_
lr.fit(text_train, y_train)
lr.predict(text_test)
print("Score: {:.2f}".format(lr.score(text_test, y_test)))
