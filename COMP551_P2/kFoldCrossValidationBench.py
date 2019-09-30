# Test Bench to show benefits of k-fold cross validation
# COMP 551
# Boury Mbodj
# Humayun Khan
# Michael Segev
# Feb 14 2019
import ClassifierDataPrepper
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

dataPath = "./"
trainingDataPath = dataPath + "train/"
positiveTrainingDataPath = trainingDataPath + "pos/"
negativeTrainingDataPath = trainingDataPath + "neg/"
# testDataPath = dataPath + "test/"
testDataPath = None

print("Opening training and test files...")
cdp = ClassifierDataPrepper.ClassifierDataPrepper(positiveTrainingDataPath, negativeTrainingDataPath, testDataPath)


print("Preparing data frames...")
X, Y = cdp.getXYlabeled()

# Building different vectorizers used to parse the text into features
print("Extracting features from data frames...")
vectCount = CountVectorizer(min_df=1, binary=False)

# learn the vocabularies from training data for each vector type
vectCount.fit(X)

# transform training data
X_count_total = vectCount.transform(X)

logRegModel = LogisticRegression()

print("Performing k-fold cross validation for our model...")
kfoldscores = cross_val_score(logRegModel, X_count_total, Y, cv=5)
# print(scores)
print("Mean model accuracy = {}".format(kfoldscores.mean()))


# Split labelled data into train and validation sets
print("Splitting the data only once and measuring performance for our trained model...")
X_train, X_validate, Y_train, Y_validate = train_test_split(X_count_total, Y, test_size=0.2, random_state=0)

logRegModel.fit(X_train, Y_train)
predictions = logRegModel.predict(X_validate)
accuracy = metrics.accuracy_score(Y_validate, predictions)
print("Model accuracy = {}".format(accuracy))




