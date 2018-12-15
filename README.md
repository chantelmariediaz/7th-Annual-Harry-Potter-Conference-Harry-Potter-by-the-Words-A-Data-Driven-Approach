## 7th Annual Harry Potter Conference 2018: Harry Potter by the Words: A Data-Driven Approach Project

Goal: Find lexical diversity scores, sentiment scores, and more in each book and compare them throughout the series using Python

## Data Source:
J.K. Rowling's books! Via text files. See online source: http://www.glozman.com/textpages.html

## Main Libraries used: 
NLTK (Natural Language Toolkit), Textstat

## Analysis 1 & 2

Lexical Diversity (unique words, Automated Readability Index (ARI), Average Word Lengths, and Fine Grained Words (W>15). In the process, define the Fine Grained Words as Potter-Specific or Not-Potter Specific (labeled with Excel, seen in dataset). Lastly, compare frequent unigrams, bigrams, and trigrams and see which characters are mentioned most together in each book.

## Analysis 3 (different than in presentation, which I expanded further)

Find sentiment scores (positive, negative, neutral, and compound) using Vader.sentiment library in NLTK. This was the most challenging since vader is used primarily for analyzing sentiment of social media text, a.k.a line by line. Courtesy of the good people of stackoverflow, I was able to figure out a way to seperate the text by characters into different list, join and convert them into string into one list, and then run vader sentiment to get an overall sentiment score. Thankfully, after many trial and error, it worked! 

Side note: This could be done sentence by sentence, but the sentiment score is not accurate since each book gets longer and longer with more sentences. Essentially, the sentiment gets higher scores (positive and negative) as the series progresses, which is not the case.

Tableau Public for Corresponding Visualizations: https://public.tableau.com/profile/chantel.diaz#!/
