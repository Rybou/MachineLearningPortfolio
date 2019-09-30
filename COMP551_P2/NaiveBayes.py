# Bernouli Naive Bayes Model
# COMP 551
# Boury Mbodj
# Humayun Khan
# Michael Segev
# Feb 10 2019
import numpy as np

# returns lower cased string split by whitespace
def split_words(string):
    return string.lower().split(" ")

class NaiveBayes:

    def __init__(self):
        print("Initializing Naive Bayes model")
        self.p_pos = 0
        self.p_neg = 0
        self.featureProbsGivenPos = None
        self.featureProbsGivenNeg = None
        self.condProbLogsPos = None
        self.condProbLogsNeg = None
        self.priorProbLog = None
        self.topWord2featureIdxHashMap = {}

    def trainFromStrings(self, comments, labels, numTopWords):
        # 1. Finding vocabulary using and only including most frequent words

        # join all comments
        joined = " ".join(comments)
        # split on blank spaces + remove duplicates
        uniqueWords = list(set(split_words(joined)))
        # create word count dictionary initialized to zero
        wordCount = {}
        for word in uniqueWords:
            wordCount[word] = 0

        # count how many times each word occurs in all comments
        for word in split_words(joined):
            wordCount[word] += 1

        # sort by word count
        def getDictValue(item):
            return item[1]
        sortedWordCount = sorted(wordCount.items(), key=getDictValue, reverse=True)

        # Make list of top words
        topWords = []
        for word, count in sortedWordCount[0:numTopWords]:
            topWords.append(word)

        # make hashmap of word to feature idx to speedup computation
        # and store it as vocabulary
        self.topWord2featureIdxHashMap = {}
        i = 0
        for word in topWords:
            self.topWord2featureIdxHashMap[word] = i
            i += 1

        X = self.convertCommentsToFeatures(comments)
        Y = np.array(labels).reshape(-1, 1)

        self.trainFromXY(X, Y)
        # return X, Y

    # takes list of strings as inputs and uses previously obtained vocabulary
    # to produce X which has Xij = 1 if comment i contains word j else zero
    def convertCommentsToFeatures(self, comments):
        # count how many times those words appear in comments
        commentIdx = 0
        X = np.zeros((len(comments), len(self.topWord2featureIdxHashMap)))
        for comment in comments:
            for word in split_words(comment):
                if word in self.topWord2featureIdxHashMap:
                    wordIdx = self.topWord2featureIdxHashMap[word]
                    X[commentIdx][wordIdx] = 1
            commentIdx += 1
            if not commentIdx % 5000:
                print("Processed " + str(commentIdx) + " comments...")
        return X

    # takes X nxm matrix and Y labels to compute the priors and conditionals
    def trainFromXY(self, X, Y):
        print("Training Naive Bayes model")
        numPositiveSamples = 0
        numNegativeSamples = 0
        # number of times given feature appeared in a sample of given class
        featureFrequencyGivenPos = np.zeros((X.shape[1], 1))
        featureFrequencyGivenNeg = np.zeros((X.shape[1], 1))
        # loop through the samples
        for sampleIdx in range(X.shape[0]):
            # positive label samples
            if Y[sampleIdx] == 1:
                # loop through sample features
                for featureIdx in range(X.shape[1]):
                    if X[sampleIdx][featureIdx] == 1:
                        featureFrequencyGivenPos[featureIdx] += 1
                numPositiveSamples += 1
            # negative label samples
            else:
                # loop through sample features
                for featureIdx in range(X.shape[1]):
                    if X[sampleIdx][featureIdx] == 1:
                        featureFrequencyGivenNeg[featureIdx] += 1
                numNegativeSamples += 1
        # compute conditionals with laplace smoothing (alpha = 1)
        self.featureProbsGivenPos = (featureFrequencyGivenPos + 1) / (numPositiveSamples + len(self.topWord2featureIdxHashMap))
        self.featureProbsGivenNeg = (featureFrequencyGivenNeg + 1) / (numNegativeSamples + len(self.topWord2featureIdxHashMap))
        # compute priors
        self.p_pos = numPositiveSamples / (numPositiveSamples + numNegativeSamples)
        self.p_neg = 1 - self.p_pos

        # pre-computing the log terms used in prediction
        self.priorProbLog = np.log10(self.p_pos / self.p_neg)
        self.condProbLogsPos = np.empty((len(self.topWord2featureIdxHashMap), 1))
        self.condProbLogsNeg = np.empty((len(self.topWord2featureIdxHashMap), 1))
        for featureIdx in range(len(self.topWord2featureIdxHashMap)):
            self.condProbLogsPos[featureIdx] = np.log10(self.featureProbsGivenPos[featureIdx] / self.featureProbsGivenNeg[featureIdx])
            self.condProbLogsNeg[featureIdx] = np.log10((1 - self.featureProbsGivenPos[featureIdx]) / (1 - self.featureProbsGivenNeg[featureIdx]))

    def runFromStrings(self, comments):
        print("Running Naive Bayes model")
        X = self.convertCommentsToFeatures(comments)
        return self.runFromX(X)

    def runFromX(self, X):
        delta = np.zeros((X.shape[0], 1))
        Ypredicted = np.zeros((X.shape[0], 1))
        # TODO: pre-compute the logs to increase speed
        for commentIdx in range(X.shape[0]):
            delta[commentIdx] = self.priorProbLog  # np.log10(self.p_pos / self.p_neg)
            for featureIdx in range(X.shape[1]):
                if X[commentIdx][featureIdx] == 1:
                    delta[commentIdx] += self.condProbLogsPos[featureIdx]   # np.log10(self.featureProbsGivenPos[featureIdx] / self.featureProbsGivenNeg[featureIdx])
                else:
                    delta[commentIdx] += self.condProbLogsNeg[featureIdx]   # np.log10((1 - self.featureProbsGivenPos[featureIdx]) / (1 - self.featureProbsGivenNeg[featureIdx]))
            if delta[commentIdx] > 0:
                Ypredicted[commentIdx] = 1
            else:
                Ypredicted[commentIdx] = 0
            if not commentIdx % 2500:
                print("predicted " + str(commentIdx) + " comments so far...")

        return Ypredicted
