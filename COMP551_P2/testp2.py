# COMP 551
# Boury Mbodj
# Humayun Khan
# Michael Segev
# Feb 14 2019

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import os


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer


# In[4]:


pos= [x for x in os.listdir("train/pos/") if x.endswith(".txt")]
neg= [x for x in os.listdir("train/neg/") if x.endswith(".txt")]
test= [x for x in os.listdir("test/") if x.endswith(".txt")]


# In[5]:


# need to add reference or change data processing with data grabber
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


# In[54]:


# end data processing 
reviews = pd.concat([
    pd.DataFrame({"review":posReviews, "label":1}),
    pd.DataFrame({"review":negReviews, "label":0}),
    pd.DataFrame({"review":testReviews, "label":-1})
], ignore_index=True).sample(frac=1, random_state=1)


# In[55]:


# Examine firts 10 rows
reviews.head(10)


# In[56]:


# Examine the class ditribution 
reviews.label.value_counts()


# In[57]:


# Define X and y from the review dataset  for use with Countvectorizer
X= reviews[reviews.label!=-1].review
y= reviews[reviews.label!=-1].label
print (X.shape)
print (y.shape)


# In[58]:


# Split X and y into training and testing/validation sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[59]:


# Instanstiate the tfidf transformer
vect = CountVectorizer(ngram_range=(1, 1),
                             stop_words= 'english',
                             min_df=1)
X_train_counts = vect.fit_transform(X_train)
trans = TfidfTransformer(smooth_idf=False,norm='l2',use_idf=True)


# In[60]:


#Learn training data vocabulary fit and transform
X_train_dtm= trans.fit_transform(X_train_counts)


# In[61]:


X_train_dtm


# In[62]:


print("Vocabulary size: {}".format(len(vect.vocabulary_)))


# In[63]:


# Trasnform testing/validation data using fitted vocabulary 
X_test_dtm= vect.transform(X_test)
X_test_dtm


# In[64]:


# Naives bayes from scikit learn for comparison purposes with scratch model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[65]:


#Train the model
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')


# In[66]:


#Make class predictions 
y_pred= nb.predict(X_test_dtm)


# In[67]:


# Calculate accuracy
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)


# In[68]:


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred)


# In[69]:


# Logitic regression model
from sklearn.linear_model import LogisticRegression


# In[70]:


logmodel=LogisticRegression()


# In[44]:


logmodel.fit(X_train_dtm, y_train)


# In[45]:


y_pred_log=logmodel.predict(X_test_dtm)


# In[46]:


#Now for the classification task we are going to import classification report
from sklearn.metrics import classification_report


# In[47]:


print(classification_report(y_test,y_pred_log))


# In[48]:


metrics.accuracy_score(y_test,y_pred_log)


# In[49]:


# Trying K nearest neigbors 
from sklearn.tree import DecisionTreeClassifier


# In[50]:


dtree= DecisionTreeClassifier()


# In[51]:


dtree.fit(X_train_dtm, y_train)


# In[52]:


y_pred_tree=dtree.predict(X_test_dtm)


# In[53]:


print(metrics.accuracy_score(y_test,y_pred_tree))
print(classification_report(y_test,y_pred_tree))
print('\n')
print(metrics.confusion_matrix(y_test,y_pred_tree))


# In[41]:


# Trying SVM 
from sklearn.svm import SVC


# In[ ]:


svmmodel = SVC()


# In[ ]:


svmmodel.fit(X_train_dtm, y_train)


# In[ ]:


y_pred_svm=svmmodel.predict(X_test_dtm)


# In[ ]:


print(metrics.accuracy_score(y_test,y_pred_svm))
print(classification_report(y_test,y_pred_svm))
print('\n')
print(metrics.confusion_matrix(y_test,y_pred_svm))


# In[ ]:




