
# coding: utf-8

# In[1]:


import string
from lxml import html
import requests
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer


# https://www.nltk.org/book/ch01.html
# http://spacab.com/wp/using-python-to-perform-lexical-analysis-on-a-short-story/

# In[2]:


cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "i'd": "i would",
  "i'd've": "i would have",
  "i'll": "i will",
  "i'll've": "i will have",
  "i'm": "i am",
  "i've": "i have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}


# ## HARRY POTTER FIRST BOOK

# In[3]:


f = open('HP1 Sorcerer of Stone.txt','r')
text = f.read()
f.close()

characters_to_remove = ["''",'``','...',':','--','限','¬','’', '’','said', 'told','asked',"'s"]
wordnet_lemmatizer = WordNetLemmatizer()

def cleaning_features(text):
    #Replace contractions
    for k,v in cList.items():    
        text = text.replace(k, v)
        
    words = word_tokenize(text)
    #Convert all text to lower case
    words = [word.lower() for word in words]
    words = [word for word in words if word not in set(characters_to_remove)]
    #Remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    #Remove punctuation expect !
    words = [word for word in words if word not in set(string.punctuation)]
    #Removes numbers 
    numbers_to_remove =[word for word in words if any(int(s) for s in word.split() if s.isdigit())] 
    words = [word for word in words if word not in set(numbers_to_remove)]

    Cleaned_lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in words]
    return Cleaned_lemma_list

def lexical_diversity(text):
    return len(set(text)) / len((text))

lexical_diversity(cleaning_features(text)) * 100


# In[ ]:


import nltk
from nltk.probability import FreqDist

fdist1 = FreqDist((cleaning_features(text)))
fdist1.most_common(50)


# In[10]:


fdist2 = FreqDist((nltk.bigrams(cleaning_features(text))))
fdist2.most_common(50)


# In[11]:


import nltk
from nltk.probability import FreqDist

fdist3 = FreqDist((nltk.trigrams(cleaning_features(text))))
fdist3.most_common(50) 


# In[127]:


V = set(cleaning_features(text))
long_words = [w for w in V if len(w) > 15]
sorted(long_words)


# In[31]:


# word counts can be useful in analyzing text data
# we can transform a list of words into a dictionary object
# containing word counts

# create a dictionary that will contain word & count as key/value pairs
wordcount={}

# iterate through the list of words, increment the word count for each word in the list

for word in lemma_list:
    if word not in wordcount:
        wordcount[word] = 1
    else:
        wordcount[word] += 1

        
# print the contents of the dictionary to screen

for key in wordcount.keys():
    print ("%s %s " %(key , wordcount[key]))


# ## HARRY POTTER BOOK 2

# In[34]:


f = open('HP2 Chamber of Secrets.txt','r',encoding='utf8')
text2 = f.read()
f.close()

lexical_diversity(cleaning_features(text2)) * 100


# In[11]:


fdist1 = FreqDist((cleaning_features(text2)))
fdist1.most_common(10)


# In[14]:


fdist2 = FreqDist((nltk.bigrams(cleaning_features(text2))))
fdist2.most_common(10)


# In[15]:


fdist3 = FreqDist((nltk.trigrams(cleaning_features(text2))))
fdist3.most_common(10) 


# In[19]:


V = set(cleaning_features(text2))
long_words = [w for w in V if len(w) > 15]
sorted(long_words)


# ## HARRY POTTER BOOK 3

# In[35]:


f = open('HP3 Prisoner of Azkaban.txt','r',encoding='utf8')
text3 = f.read()
f.close()

lexical_diversity(cleaning_features(text3)) * 100


# In[21]:


fdist1 = FreqDist((cleaning_features(text3)))
fdist1.most_common(10)


# In[22]:


fdist2 = FreqDist((nltk.bigrams(cleaning_features(text3))))
fdist2.most_common(10)


# In[23]:


fdist3 = FreqDist((nltk.trigrams(cleaning_features(text3))))
fdist3.most_common(10) 


# In[25]:


V = set(cleaning_features(text3))
long_words3 = [w for w in V if len(w) > 15]
sorted(long_words3)


# ## HARRY POTTER BOOK 4

# In[36]:


#Reading Text Files Example

f = open('HP4 Goblet of Fire.txt','r',encoding='utf8')
text4 = f.read()
f.close()

lexical_diversity(cleaning_features(text4)) * 100


# In[37]:


fdist1 = FreqDist((cleaning_features(text4)))
fdist1.most_common(10)


# In[38]:


fdist2 = FreqDist((nltk.bigrams(cleaning_features(text4))))
fdist2.most_common(10)


# In[29]:


fdist3 = FreqDist((nltk.trigrams(cleaning_features(text4))))
fdist3.most_common(10) 


# In[73]:


V = set(lemma_list4)
long_words4 = [w for w in V if len(w) > 15]
sorted(long_words4)


# ## HARRY POTTER BOOK 5

# In[30]:


#Reading Text Files Example

f = open('HP5 Order of Phoenix.txt','r',encoding='utf8')
text5= f.read()
f.close()

lexical_diversity(cleaning_features(text5)) * 100


# In[31]:


fdist1 = FreqDist((cleaning_features(text5)))
fdist1.most_common(10)


# In[39]:


fdist2 = FreqDist((nltk.bigrams(cleaning_features(text5))))
fdist2.most_common(10)


# In[41]:


fdist3 = FreqDist((nltk.trigrams(cleaning_features(text5))))
fdist3.most_common(10) 


# In[68]:


V = set(lemma_list5)
long_words5 = [w for w in V if len(w) > 15]
sorted(long_words5)


# ## HARRY POTTER BOOK 6

# In[42]:


#Reading Text Files Example

f = open('HP6 Half Blood Prince.txt','r',encoding='utf8')
text6 = f.read()
f.close()

lexical_diversity(cleaning_features(text6)) * 100


# In[43]:


fdist1 = FreqDist((cleaning_features(text6)))
fdist1.most_common(10)


# In[44]:


fdist2 = FreqDist((nltk.bigrams(cleaning_features(text6))))
fdist2.most_common(10)


# In[46]:


fdist3 = FreqDist((nltk.trigrams(cleaning_features(text6))))
fdist3.most_common(10) 


# In[64]:


V = set(lemma_list6)
long_words6 = [w for w in V if len(w) > 15]
sorted(long_words6)


# ## HARRY POTTER BOOK 7

# In[47]:


#Reading Text Files Example

f = open('HP7 Deathly Hallows.txt','r',encoding='utf8')
text7 = f.read()
f.close()

lexical_diversity(cleaning_features(text7)) * 100


# In[48]:


fdist1 = FreqDist((cleaning_features(text7)))
fdist1.most_common(10)


# In[49]:


fdist2 = FreqDist((nltk.bigrams(cleaning_features(text7))))
fdist2.most_common(10)


# In[50]:


fdist3 = FreqDist((nltk.trigrams(cleaning_features(text7))))
fdist3.most_common(10) 


# In[61]:


V = set(lemma_list7)
long_words7 = [w for w in V if len(w) > 15]
sorted(long_words7)


# In[53]:


fdist7.plot(50, cumulative=True)

