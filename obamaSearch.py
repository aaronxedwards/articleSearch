#!/usr/bin/env python
# coding: utf-8

# In[4]:


#authour: aaron edwards
#project: news story search engine based on [ tf/idf ]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


story = pd.read_csv("articles.csv", sep=',', header=None)
story.head()


# In[6]:


story.columns


# In[7]:


# [ Clean Data ]


# In[8]:


story.rename(columns = {story.columns[2]: "Titles", 
                       story.columns[3]: "Publication",
                       story.columns[5]: "Date", 
                       story.columns[9]: "Content"}, inplace = True)


# In[9]:


story = story.drop([0])


# In[10]:


story.head()


# In[11]:


clstory = story[['Date', 'Titles', 'Content', 'Publication']]
clstory.head()


# In[ ]:





# In[12]:


#Create TFIDF object 
tfvector = TfidfVectorizer()


# In[13]:


#fit transform news stories 
X = tfvector.fit_transform(story["Content"])


# In[14]:


#View who has the most publications 
clstory.groupby(["Publication"]).count()


# In[15]:


labels  = ["Atlantic", "Breitbar","Business Insider", "CNN", "New York Times" ]
sizes = [171, 23781, 6757, 11488, 7803]

plt.title("% of News Publicators in Database:\n")
plt.pie(sizes, labels=labels, autopct='%1.2f%%')
plt.show()


# In[17]:


print(" Search News stories here... ")
qwery = input()


# In[18]:


search_vec = tfvector.transform([qwery])

#oT Rank
oT = cosine_similarity(X, search_vec).reshape((-1,))
clstory.columns


# In[19]:


#Print results
print("Search results : \n")
for i in oT.argsort()[-10:][::-1]:
    print( "\n", "\n", clstory.iloc[i, 3],"|", clstory.iloc[i, 0] , "\n", "\n",  " \t ", clstory.iloc[i, 2])


# In[ ]:




