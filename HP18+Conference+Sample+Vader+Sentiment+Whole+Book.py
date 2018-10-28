
# coding: utf-8

# In[1]:


from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
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
from nltk.tokenize.moses import MosesDetokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[2]:


with open('HP6 Half Blood Prince.txt', 'r') as inFile:
    # Read the file
    contents = inFile.read()
    # This will store the different 256 character bits
    groups = []
    # while the contents contain something
    while contents:
        # Add the first 256 characters to the grouping
        groups.append(contents[:700])
        # Set the contents to everything after the first 256
        contents = contents[700:]
#print(groups)


# In[3]:


values = " ".join(str(x) for x in groups)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def print_sentiment_scores(values):
    snt = analyser.polarity_scores(values)
    print("{:-<40} {}".format(values, str(snt)))


# In[4]:


print_sentiment_scores(values)

