{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import numpy as np\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import and instantiate CountVectorizer (with the default parameters)\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# read text data set into pandas\n",
    "#path= 'train/pos'\n",
    "#review= pd.read_table(path, header=None, names=)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "positiveFiles = [x for x in os.listdir(\"train/pos/\") if x.endswith(\".txt\")]\n",
    "negativeFiles = [x for x in os.listdir(\"train/neg/\") if x.endswith(\".txt\")]\n",
    "testFiles = [x for x in os.listdir(\"test/\") if x.endswith(\".txt\")]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "positiveReviews, negativeReviews, testReviews = [], [], []\n",
    "for pfile in positiveFiles:\n",
    "    with open(\"train/pos/\"+pfile, encoding=\"ISO-8859-1\") as f:\n",
    "        positiveReviews.append(f.read())\n",
    "for nfile in negativeFiles:\n",
    "    with open(\"train/neg/\"+nfile, encoding=\"ISO-8859-1\") as f:\n",
    "        negativeReviews.append(f.read())\n",
    "for tfile in testFiles:\n",
    "    with open(\"test/\"+tfile, encoding=\"ISO-8859-1\") as f:\n",
    "        testReviews.append(f.read())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "reviews = pd.concat([\n",
    "    pd.DataFrame({\"review\":positiveReviews, \"label\":1, \"file\":positiveFiles}),\n",
    "    pd.DataFrame({\"review\":negativeReviews, \"label\":0, \"file\":negativeFiles}),\n",
    "    pd.DataFrame({\"review\":testReviews, \"label\":-1, \"file\":testFiles})\n",
    "], ignore_index=True).sample(frac=1, random_state=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Examine firts 10 rows\n",
    "reviews.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Examine the class ditribution \n",
    "reviews.label.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define X and y (from the reviews data ) for use with count vectorizer "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "reviews = reviews[[\"review\", \"label\"]].sample(frac=1, random_state=1)\n",
    "train = reviews[reviews.label!=-1].sample(frac=0.8, random_state=1)\n",
    "valid = reviews[reviews.label!=-1].drop(train.index)\n",
    "test = reviews[reviews.label==-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(train.shape)\n",
    "print(valid.shape)\n",
    "print(test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define X and y from the review dataset  for use with Countvectorizer\n",
    "X= reviews[reviews.label!=-1].review\n",
    "y= reviews[reviews.label!=-1].label\n",
    "print (X.shape)\n",
    "print (y.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split X and y into training and testing/validation sets\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)\n",
    "print(X_train.shape)\n",
    "print(X_test.shape)\n",
    "print(y_train.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Instanstiate the vectorizer\n",
    "vect= CountVectorizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Learn training data vocabulary fit and transform\n",
    "X_train_dtm= vect.fit_transform(X_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_dtm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Trasnform testing/validation data using fitted vocabulary \n",
    "X_test_dtm= vect.transform(X_test)\n",
    "X_test_dtm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Naives bayes from scikit learn for comparison purposes with scratch model\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "nb = MultinomialNB()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Train the model\n",
    "%time nb.fit(X_train_dtm, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Make class predictions \n",
    "y_pred= nb.predict(X_test_dtm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate accuracy\n",
    "from sklearn import metrics\n",
    "metrics.accuracy_score(y_test,y_pred)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2692,  389],\n",
       "       [ 575, 2594]])"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# print the confusion matrix\n",
    "metrics.confusion_matrix(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Logitic regression model\n",
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "logmodel=LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "          intercept_scaling=1, max_iter=100, multi_class='warn',\n",
       "          n_jobs=None, penalty='l2', random_state=None, solver='warn',\n",
       "          tol=0.0001, verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logmodel.fit(X_train_dtm, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_log=logmodel.predict(X_test_dtm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Now for the classification task we are going to import classification report\n",
    "from sklearn.metrics import classification_report\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.88      0.87      0.87      3081\n",
      "           1       0.87      0.89      0.88      3169\n",
      "\n",
      "   micro avg       0.88      0.88      0.88      6250\n",
      "   macro avg       0.88      0.88      0.88      6250\n",
      "weighted avg       0.88      0.88      0.88      6250\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,y_pred_log))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.87664"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "metrics.accuracy_score(y_test,y_pred_log)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Trying K nearest neigbors \n",
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree= DecisionTreeClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,\n",
       "            max_features=None, max_leaf_nodes=None,\n",
       "            min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "            min_samples_leaf=1, min_samples_split=2,\n",
       "            min_weight_fraction_leaf=0.0, presort=False, random_state=None,\n",
       "            splitter='best')"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dtree.fit(X_train_dtm, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_tree=dtree.predict(X_test_dtm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.7016\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.70      0.69      0.70      3081\n",
      "           1       0.70      0.71      0.71      3169\n",
      "\n",
      "   micro avg       0.70      0.70      0.70      6250\n",
      "   macro avg       0.70      0.70      0.70      6250\n",
      "weighted avg       0.70      0.70      0.70      6250\n",
      "\n",
      "\n",
      "\n",
      "[[2132  949]\n",
      " [ 916 2253]]\n"
     ]
    }
   ],
   "source": [
    "print(metrics.accuracy_score(y_test,y_pred_tree))\n",
    "print(classification_report(y_test,y_pred_tree))\n",
    "print('\\n')\n",
    "print(metrics.confusion_matrix(y_test,y_pred_tree))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Trying SVM \n",
    "from sklearn.svm import SVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "svmmodel = SVC()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "svmmodel.fit(X_train_dtm, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_svm=svmmodel.predict(X_test_dtm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(metrics.accuracy_score(y_test,y_pred_svm))\n",
    "print(classification_report(y_test,y_pred_svm))\n",
    "print('\\n')\n",
    "print(metrics.confusion_matrix(y_test,y_pred_svm))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
