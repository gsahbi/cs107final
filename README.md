![](http://i.imgur.com/wYyRKNn.png?1)

# Project Overview
***

## Motivation :
Sentiment Analysis, is receiving a big attention these days, because of its huge spectrum of applications ranging from product review analysis, campaign feedback, competition bench-marking, customer profiles, political trends, etc...

<center><img src="http://i.imgur.com/hFwQEH3.jpg?1"></center>

There is a huge flow of information going through the internet and social networks. Online discussions are only relevant to people for a couple of days. Nobody actually goes in past to tweets that are older than maybe a week, for instance. This entire humanity archive of discussion could help in many applications if we train machines to understand the sentiment of people towards a specific theme at a specific time. 

## Background
“Sentiment analysis is the computational study of people's opinions, sentiments, emotions, and attitudes.” [Excerpt From: Bing Liu. Sentiment Analysis: Mining Opinions, Sentiments, and Emotions.]. This book is an excellent survey of NLP and SA research.

Given the large amount of data available on the Web, it is now possible to investigate high-level Information Retrieval tasks like user's intentions and feelings about facts or objects discussed. [Pang, B., Lee, L., 2008. Opinion mining and sentiment analysis. Foundations and Trends in Information Retrieval] 


## Challenge :
There are several things to take into consideration when approaching a Sentiment Analysis task. In general, there are two main approaches: 

- **Sentiment lexicons using Natural Language Processing (NLP) techniques**. A Sentiment lexicon is a list of words that are associated to polarity values (positive or negative). NLP techniques offer a deep level of analysis since they take into account the context words in the sentence. 

- **Machine Learning classification algorithms**. Because sentiment classification is a text classification problem, any existing supervised learning method can be directly applied [Bing Liu]. For example,  naive Bayes classification , logistic regression, support vector machines (SVM), etc..

In this work we'll work on ML classification and then try to get into the NLP and experience some of the basic techniques used.

## The Data

In this work we'll use a data-set that we obtained thankfully from [Julian McAuley](http://cseweb.ucsd.edu/~jmcauley/), at the University of San Diego [here](http://jmcauley.ucsd.edu/data/amazon/).

It is worth mentioning, their excellent work in SIGIR and KDD papers (listed on the above page).

This data-set contains a NJSON formatted product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014. We have decided to use electronics reviews for this work. Because electronics are not perfect so create a lot of contrasted opinions.
 
# Conclusion

Some of the basic techniques of NLP can enhance the accuracy of sentiment scoring. We went from 68% to 74% with a simple identification of negation-of-adjective (NOA) and negation-of-verb (NOV). And using a randomForest algorithm we achieved nearly 78% accuracy.

But, some language forms require a deep analysis. For example :

Sarcasm : "Great sound when working"
Deep inverter : "Not such a good value after all", polarity inverter adverb is 3-words faraway from its adjective.
