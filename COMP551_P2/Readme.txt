P2BestPerformingModel.ipynb:
       Class that contains our best performing model

P2FeatureExtraction&ValidationModel.ipynb:
       Class that contains our different feature extraction(Binary, count, tf-idf) and validation pipelines(holdout cross validation and grid search k fold cross validation)


P2SupervisedLearningModels.ipynb:
       Class that contains our different supervised learning models (logistic regression, decision trees, support vector machines and neural networks)

TestBenchNB.py:
	Test Bench which opens training data set, performs feature extraction, runs the Naive Bayes model made from scratch, and performs leave one out validation to report accuracy. 
	Note that data must be placed in ./comp-551-imbd-sentiment-classification/ folder

NaiveBayes.py:
	Naive Bayes class made from scratch

DataGrabber.py:
	Class that opens review files in folders in order of their name.

ClassifierDataPrepper.py
	Class that contains 3 data grabbers and prepares labelled and unlabelled reviews in pandas data frame format. Also containes helper methods that cleans reviews from html tags and other non-letter characters.

spacy.ipynb
	Jupyter notebook showing advanced lemmatizing experiment using spacy PoS tagging library.

doc2vec.ipynb
	Jupyter notebook showing results of doc2vec experimentation which gets vectors for each review and runs a LogisticRegression model on the produced vectors.

