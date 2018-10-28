
# coding: utf-8

# ## HARRY POTTER LEXICAL DIVERSITY PT2
# 
# ## Number of Unique Words and Readability Scores

# In[1]:


#Import libraries as needed
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


# In[2]:


import textstat


# just some links 
# https://www.nltk.org/book/ch01.html
# http://spacab.com/wp/using-python-to-perform-lexical-analysis-on-a-short-story/

# In[3]:


#Try to expand contractions, thanks Github
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

# In[4]:


#Open text file 
f = open('HP1 Sorcerer of Stone.txt','r')
text = f.read()
f.close()

#Define cleaning function that tokenizes and cleans all the words into a lemma list
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
#define a lexical diversity to get a set of unique words and lexical diversity score

len(cleaning_features(text))


# In[39]:


#Open text file 
f = open('HP1 Sorcerer of Stone.txt','r')
text = f.read()
f.close()

#Define cleaning function that tokenizes and cleans all the words into a lemma list
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
#define a lexical diversity to get a set of unique words and lexical diversity score


# In[42]:


width = [len(w) for w in cleaning_features(text)]


# In[44]:


average = sum(width)/len(width)


# In[46]:


print(average)


# In[33]:


textstat.flesch_reading_ease(text)


# In[40]:


textstat.automated_readability_index(text)


# In[35]:


textstat.lexicon_count(text, removepunct=True)


# In[22]:


textstat.syllable_count(text, lang='en_US')


# In[82]:


textstat.text_standard(text)


# In[9]:


len(set(cleaning_features(text)))


# In[7]:


#Find the length of cleaned text that has more than 15 letters
V = set(cleaning_features(text))
long_words = [w for w in V if len(w) > 15]
sorted(long_words)


# ## HARRY POTTER BOOK 2

# In[83]:


f = open('HP2 Chamber of Secrets.txt','r',encoding='utf8')
text2 = f.read()
f.close()


# In[86]:


width2 = [len(w) for w in cleaning_features(text2)]
average = sum(width2)/len(width2)
print(average)


# In[84]:


textstat.automated_readability_index(text2)


# In[85]:


textstat.flesch_reading_ease(text2)


# In[4]:


f = open('HP2 Chamber of Secrets.txt','r',encoding='utf8')
text2 = f.read()
f.close()

len(cleaning_features(text2))


# In[6]:


len(set(cleaning_features(text2)))


# In[12]:


V = set(cleaning_features(text2))
long_words = [w for w in V if len(w) > 15]
sorted(long_words)


# ## HARRY POTTER BOOK 3

# In[87]:


f = open('HP3 Prisoner of Azkaban.txt','r',encoding='utf8')
text3 = f.read()
f.close()
textstat.automated_readability_index(text3)
textstat.flesch_reading_ease(text3)


# In[89]:


width3 = [len(w) for w in cleaning_features(text3)]
average = sum(width3)/len(width3)
print(average)


# In[88]:


textstat.automated_readability_index(text3)


# In[10]:


f = open('HP3 Prisoner of Azkaban.txt','r',encoding='utf8')
text3 = f.read()
f.close()

len(cleaning_features(text3))


# In[11]:


len(set(cleaning_features(text3)))


# In[17]:


V = set(cleaning_features(text3))
long_words3 = [w for w in V if len(w) > 15]
sorted(long_words3)


# ## HARRY POTTER BOOK 4

# In[90]:


f = open('HP4 Goblet of Fire.txt','r',encoding='utf8')
text4 = f.read()
f.close()


# In[93]:


width4 = [len(w) for w in cleaning_features(text4)]
average = sum(width4)/len(width4)
print(average)


# In[92]:


textstat.flesch_reading_ease(text4)


# In[91]:


textstat.automated_readability_index(text4)


# In[12]:


#Reading Text Files Example

f = open('HP4 Goblet of Fire.txt','r',encoding='utf8')
text4 = f.read()
f.close()

len(cleaning_features(text4))


# In[13]:


len(set(cleaning_features(text4)))


# In[23]:


V = set(cleaning_features(text4))
long_words4 = [w for w in V if len(w) > 15]
sorted(long_words4)


# ## HARRY POTTER BOOK 5

# In[96]:


f = open('HP5 Order of Phoenix.txt','r',encoding='utf8')
text5= f.read()
f.close()


# In[97]:


textstat.automated_readability_index(text5)


# In[98]:


textstat.flesch_reading_ease(text5)


# In[99]:


width5 = [len(w) for w in cleaning_features(text5)]
average = sum(width5)/len(width5)


# In[100]:


print(average)


# In[16]:


#Reading Text Files Example

f = open('HP5 Order of Phoenix.txt','r',encoding='utf8')
text5= f.read()
f.close()

len(cleaning_features(text5))


# In[17]:


len(set(cleaning_features(text5)))


# In[29]:


V = set(cleaning_features(text5))
long_words5 = [w for w in V if len(w) > 15]
sorted(long_words5)


# ## HARRY POTTER BOOK 6

# In[101]:


#Reading Text Files Example

f = open('HP6 Half Blood Prince.txt','r',encoding='utf8')
text6 = f.read()
f.close()


# In[105]:


width61 = [len(w) for w in (text6)]
average = sum(width61)/len(width61)
print(average)


# In[102]:


textstat.automated_readability_index(text6)


# In[103]:


textstat.flesch_reading_ease(text6)


# In[104]:


width6 = [len(w) for w in cleaning_features(text6)]
average = sum(width6)/len(width6)
print(average)


# In[18]:


#Reading Text Files Example

f = open('HP6 Half Blood Prince.txt','r',encoding='utf8')
text6 = f.read()
f.close()

len(cleaning_features(text6))


# In[19]:


len(set(cleaning_features(text6)))


# In[34]:


V = set(cleaning_features(text6))
long_words6 = [w for w in V if len(w) > 15]
sorted(long_words6)


# ## HARRY POTTER BOOK 7

# In[106]:


#Reading Text Files Example

f = open('HP7 Deathly Hallows.txt','r',encoding='utf8')
text7 = f.read()
f.close()


# In[109]:


width7 = [len(w) for w in cleaning_features(text7)]
average = sum(width7)/len(width7)
print(average)


# In[107]:


textstat.automated_readability_index(text7)


# In[108]:


textstat.flesch_reading_ease(text7)


# In[20]:


#Reading Text Files Example

f = open('HP7 Deathly Hallows.txt','r',encoding='utf8')
text7 = f.read()
f.close()

len(cleaning_features(text7))


# In[21]:


len(set(cleaning_features(text7)))


# In[39]:


V = set(cleaning_features(text7))
long_words7 = [w for w in V if len(w) > 15]
sorted(long_words7)


# In[41]:


fdist1.plot(50, cumulative=True)

