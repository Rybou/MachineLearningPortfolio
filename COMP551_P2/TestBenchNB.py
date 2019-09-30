# Test bench for Bernouli Naive Bayes Model
# COMP 551
# Boury Mbodj
# Humayun Khan
# Michael Segev
# Feb 10 2019

import NaiveBayes
import DataGrabber
import numpy as np
import pickle
import time

# PICKLE_MODEL_FILENAME = None
PICKLE_MODEL_FILENAME = "./pickles/nbModel_submission1.pickle"

time = int(time.time())

print("Performing feature extraction...")
dataPath = "./comp-551-imbd-sentiment-classification/"
trainingDataPath = dataPath + "train/"

# positiveTrainingDataPath = trainingDataPath + "pos_small/"
# negativeTrainingDataPath = trainingDataPath + "neg_small/"
positiveTrainingDataPath = trainingDataPath + "pos/"
negativeTrainingDataPath = trainingDataPath + "neg/"

print("Getting positive training comments from " + positiveTrainingDataPath)
dg = DataGrabber.DataGrabber(positiveTrainingDataPath)
positiveComments = dg.readCommentFiles()

print("Getting negative training comments from " + negativeTrainingDataPath)
dg = DataGrabber.DataGrabber(negativeTrainingDataPath)
negativeComments = dg.readCommentFiles()

print("got " + str(len(positiveComments)) + " positive comments and " + str(len(negativeComments)) + " comments")

nbModel = NaiveBayes.NaiveBayes()
trainingDataComments = positiveComments[0:10000] + negativeComments[0:10000]
validationDataComments = positiveComments[10000:12500] + negativeComments[12500:25000]

trainingLabels = []
for comment in positiveComments[0:10000]:
    trainingLabels.append(1)
for comment in negativeComments[0:10000]:
    trainingLabels.append(0)

validationLabels = []
for comment in positiveComments[10000:12500]:
    validationLabels.append(1)
for comment in negativeComments[12500:25000]:
    validationLabels.append(0)

if PICKLE_MODEL_FILENAME is None:
    print("Training Model")
    nbModel.trainFromStrings(trainingDataComments, trainingLabels, 2500)
    # save model to pickle file
    pickleFile = open("nbModel_" + str(time) + ".pickle", "wb")
    pickle.dump(nbModel, pickleFile)
    pickleFile.close()

else:
    nbModel = pickle.load(open(PICKLE_MODEL_FILENAME, "rb"))

print("Running Model on new data")
predictions = nbModel.runFromStrings(validationDataComments)

f = open("predictions_" + str(time) + ".csv", "w")
f.write("Id,Category\n")
i = 0
for prediction in predictions:
    f.write(str(i) + "," + str(int(prediction)))
    f.write("\n")
    i += 1
f.close()

numCorrectPredictions = 0
for commentIdx in range(predictions.shape[0]):
    if predictions[commentIdx] == validationLabels[commentIdx]:
        numCorrectPredictions += 1
accuracy = numCorrectPredictions / predictions.shape[0]
print("accuracy is = " + str(accuracy * 100))
