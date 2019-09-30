# COMP 551
# Boury Mbodj
# Humayun Khan
# Michael Segev
# Feb 14 2019

#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd 
import numpy as np
import os


# In[14]:


import mglearn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


# import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.model_selection import GridSearchCV


# In[16]:


pos= [x for x in os.listdir("train/pos/") if x.endswith(".txt")]
neg= [x for x in os.listdir("train/neg/") if x.endswith(".txt")]
test= [x for x in os.listdir("test/") if x.endswith(".txt")]


# In[17]:


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


# In[18]:


# end data processing 
reviews = pd.concat([
    pd.DataFrame({"file":pos,"review":posReviews, "label":1}),
    pd.DataFrame({"file":neg,"review":negReviews, "label":0}),
    pd.DataFrame({"file":test,"review":testReviews, "label":-1})
], ignore_index=True).sample(frac=1, random_state=1)


# In[19]:


# Examine firts 10 rows
reviews["file"]= reviews["file"].str.split("_", n = 1, expand = True)
reviews["file"]= reviews["file"].str.split(".", n = 1, expand = True)
#reviews.set_index('file',inplace=True)
reviews.head(10)


# In[20]:


# Examine the class ditribution 
reviews.label.value_counts()


# In[21]:


# Define X and y from the review dataset  for use with Countvectorizer
X= reviews[reviews.label!=-1].review
y= reviews[reviews.label!=-1].label
print (X.shape)
print (y.shape)


# In[22]:


# Split X and y into training and testing/validation sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


# Instanstiate the tfidf transformer
vect = TfidfVectorizer(min_df=4, ngram_range=(1, 2))
X_train_counts = vect.fit_transform(X_train)
trans = TfidfTransformer(smooth_idf=False,norm='l2',use_idf=True)


# In[24]:


#Learn training data vocabulary fit and transform
X_train_dtm= trans.fit_transform(X_train_counts)


# In[69]:


X_train_dtm


# In[70]:


print("Vocabulary size: {}".format(len(vect.vocabulary_)))


# In[71]:


# Trasnform testing/validation data using fitted vocabulary 
X_test_dtm= vect.transform(X_test)
X_test_dtm


# In[72]:


# Naives bayes from scikit learn for comparison purposes with scratch model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[73]:


#Train the model
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')


# In[74]:


#Make class predictions 
y_pred= nb.predict(X_test_dtm)


# In[75]:


# Calculate accuracy
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)


# In[76]:


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred)


# In[77]:


# Logitic regression model
from sklearn.linear_model import LogisticRegression


# In[78]:


logmodel=LogisticRegression()


# In[79]:


logmodel.fit(X_train_dtm, y_train)


# In[80]:


y_pred_log=logmodel.predict(X_test_dtm)


# In[81]:


#Now for the classification task we are going to import classification report
from sklearn.metrics import classification_report


# In[82]:


print(classification_report(y_test,y_pred_log))


# In[83]:


metrics.accuracy_score(y_test,y_pred_log)


# In[84]:


# Trying K nearest neigbors 
from sklearn.tree import DecisionTreeClassifier


# In[85]:


dtree= DecisionTreeClassifier()


# In[86]:


dtree.fit(X_train_dtm, y_train)


# In[87]:


y_pred_tree=dtree.predict(X_test_dtm)


# In[88]:


print(metrics.accuracy_score(y_test,y_pred_tree))
print(classification_report(y_test,y_pred_tree))
print('\n')
print(metrics.confusion_matrix(y_test,y_pred_tree))


# In[89]:


feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))


# In[ ]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(penalty="l2", C=1), param_grid, cv=5)
grid.fit(X_train_dtm, y_train)


# In[91]:


y_pred_grid=grid.predict(X_test_dtm)


# In[92]:


print(metrics.accuracy_score(y_test,y_pred_grid))
print(classification_report(y_test,y_pred_grid))
print('\n')
print(metrics.confusion_matrix(y_test,y_pred_grid))


# In[93]:


# Define X and y from the review dataset  for use with Countvectorizer
y_prime= reviews[reviews.label==-1].review


# In[94]:


# Trasnform testing/validation data using fitted vocabulary 
X_prime_dtm= vect.transform(y_prime)


# In[95]:


y_predi=grid.predict(X_prime_dtm)


# In[96]:


submission = pd.DataFrame()


# In[97]:


#submission1["id"] = test_df["id"]
#submission1["labels"] = pred

#submission1=submission1.sort("id")
y_predi


# In[99]:


submission['Id']= reviews[reviews.label==-1].file


# In[100]:


submission['Category']=y_predi


# In[101]:


submission.head()


# In[108]:



submission.Id = pd.to_numeric(submission.Id, errors='coerce')

submission.sort_values(by=['Id'], inplace=True)


# In[109]:


submission.to_csv('solution1.csv', index=False)


# In[ ]:




