#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[3]:


#reading the data
variable = pd.read_csv(r"Documents\news.csv")
# get the shape and haed(top 5 tows and columns )
variable.shape
variable.head()


# In[4]:


# Get the labels
labels=variable.label
labels.head()


# In[5]:


#dataset(news) - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(variable['text'], labels, test_size=0.2, random_state=7)


# In[6]:


#train and test set
#Dataset - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[7]:


#Dataset - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Dataset - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[8]:


confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:





# In[ ]:




