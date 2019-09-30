# Test Bench to get the best performing model
# COMP 551
# Boury Mbodj
# Humayun Khan
# Michael Segev
# Feb 15 2019

# # Install spaCy (run in terminal/prompt)
# import sys
# !{sys.executable} -m pip install spacy
#
# # Download spaCy's  'en' Model
# !{sys.executable} -m spacy download en
from sklearn.linear_model import LogisticRegression

import ClassifierDataPrepper
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics, preprocessing
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
import pandas as pd

dataPath = "./"
trainingDataPath = dataPath + "train/"
positiveTrainingDataPath = trainingDataPath + "pos/"
negativeTrainingDataPath = trainingDataPath + "neg/"
# testDataPath = dataPath + "test/"
# positiveTrainingDataPath = trainingDataPath + "pos_small/"
# negativeTrainingDataPath = trainingDataPath + "neg_small/"
testDataPath = None

print("Opening training and test files...")
cdp = ClassifierDataPrepper.ClassifierDataPrepper(positiveTrainingDataPath, negativeTrainingDataPath, testDataPath)

print("Preparing data frames...")
X, Y = cdp.getXYlabeled()

print("Extracting features from data frames...")
# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp_spacy = spacy.load('en', disable=['parser', 'ner'])
X_lem = []
idx = 0
for comment in X:
    comment = cdp.cleanhtml(comment)
    sentences = nltk.sent_tokenize(comment)  # this gives us a list of sentences
    # Parse the sentence using the loaded 'en' model object `nlp`
    comment_lem = []
    for sentence in sentences:
        sentence_lem_tokens = nlp_spacy(sentence)
        sentence_lem = " ".join([token.lemma_ for token in sentence_lem_tokens])
        comment_lem.append(sentence_lem)

    comment_lem = " ".join(comment_lem)
    # print("Comment:")
    # print(comment)
    # print("Lemmatized Comment:")
    # print(comment_lem)
    X_lem.append(comment_lem)
    idx += 1
    if not idx % 2000:
        print("processed {} comments".format(idx))


vect = CountVectorizer(ngram_range=(1, 2), stop_words='english')

# vect = TfidfVectorizer(min_df=1, ngram_range=(1, 2))

# learn the vocabularies from training data for each vector type
vect.fit(X_lem)

# transform training data
X_lem = vect.transform(X_lem)

# normalize the data (scale it down to 0 -> 1)
X_lem = preprocessing.normalize(X_lem, norm='l2')

# model = MLPClassifier(solver='sgd', alpha=1e-5,
#                       hidden_layer_sizes=(5, 2), random_state=1)

print(X_lem.shape)
# model = MLPClassifier(hidden_layer_sizes=(X.shape[1], X.shape[1], X.shape[1]))
model = LogisticRegression()
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
# grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
# grid.fit(text_train, y_train)
#
# print("Best cross-validation score: {:.2f}".format(grid.best_score_))

# print("Performing k-fold cross validation for our model...")
# kfoldscores = cross_val_score(logRegModel, X, Y, cv=5)
# # print(scores)
# print("Mean model accuracy = {}".format(kfoldscores.mean()))


# Split labelled data into train and validation sets
print("Splitting the data only once and measuring performance for our trained model...")
X_train, X_validate, Y_train, Y_validate = train_test_split(X_lem, Y, test_size=0.2, random_state=0)

model.fit(X_train, Y_train)
predictions = model.predict(X_validate)
accuracy = metrics.accuracy_score(Y_validate, predictions)
print("Model accuracy = {}".format(accuracy))