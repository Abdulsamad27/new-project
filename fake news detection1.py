#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix_matrix


# In[3]:


#reading the data
variable = pd.read_csv(r"Documents\news.csv")
# get the shape and haed(top 5 tows and columns )
variable.shape
variable.head()
print(variable)

#training set
train_df= pd.read_csv(r"Documents\news.csv")
#test set
test_df = pd.read_csv(r"Documents\news.csv")

#some sample 
sample_submission = pd.read_csv(r"Documents\news.csv")
#
sample_submission.head()

#describe the dataset
train_df.describe()

train_df.info(verbose = True)


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

#Dataset - Fit and transform train set, transform test set
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

# here we get the accurecy of 92.82% and confusion matrix as [[589, 49
#                                                              42, 587]] and datatype as int64]
# so with this model we conclude that 589 as true positive and 587 as false negative and 42 as false positive and 49 as false negative.

confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


#plot
sns.countplot(x = train_df.columns.values)

#
test_df.describe()

#
def plot_feature_distribution(df1, df2, label1, label2, features): 
    i = 0 
    sns.set_style('whitegrid') 
    ax = plt.subplots(10,10,figsize=(18,22)),
    plt.figure()
    
#
def features():
    return np.randomrandint(3000,4888)
  
#
for featuresin features:
    i += 1
    plt.subplot(10,10,i)
    sns.displot(df1[feature], label=label1)
    sns.displot(df2[feature], label=label2)
    plt.xlabel(feature, fontsize=9)
    locs, labels = plt.xticks()
    plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
    plt.tick_params(axis='y', which='major', labelsize=6)
plt.show();
  
    


