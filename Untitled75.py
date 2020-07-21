#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[87]:


dataset = pd.read_csv("C:/Users/ANIKET/Downloads/Cough_combined.csv")
dataset


# In[88]:


dataset['Conditions'][0]


# In[89]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[90]:


corpus = []
for i in range(0,1000):
    Symptoms =  re.sub('[^a-zA-Z]', ' ', dataset['Symptoms'][i])
    Symptoms
    Symptoms = Symptoms.lower()
    Symptoms
    Symptoms = Symptoms.split()
    Symptoms
    ps = PorterStemmer()
    Symptoms = [ps.stem(word) for word in Symptoms if not word in set(stopwords.words('english'))]
    Symptoms
    Symptoms = ' '.join(Symptoms)
    Symptoms
    corpus.append(Symptoms)


# In[91]:


corpus


# In[92]:


X = dataset.iloc[:, 1]
X
X.shape
X = dataset['Symptoms'].values
X


# In[93]:


y = dataset.iloc[1:, 2]
y
y.shape
y = dataset['Conditions'].values
y


# In[ ]:





# In[94]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[166]:


X_train


# In[ ]:





# In[97]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[98]:


classifier = Sequential()


# In[163]:


classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# In[164]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# In[ ]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


cm

