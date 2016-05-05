# Introduction
## Motivation :
Sentiment Analysis, is receiving a big attention these days, because of its huge spectrum of applications ranging from product review analysis, campaign feedback, competition bench-marking, customer profiles, political trends, etc...

There is a huge flow of information going through the internet and social networks. Online discussions are only relevant to people for a couple of days. Nobody actually goes in past to tweets that are older than maybe a week, for instance. This entire humanity archive of discussion could help in many applications if we train machines to understand the sentiment of people towards a specific theme at a specific time. 

## Background
“Sentiment analysis is the computational study of people's opinions, sentiments, emotions, and attitudes.” [Excerpt From: Bing Liu. Sentiment Analysis: Mining Opinions, Sentiments, and Emotions.]. This book is an excellent survey of NLP and SA research.

Given the large amount of data available on the Web, it is now possible to investigate high-level Information Retrieval tasks like user's intentions and feelings about facts or objects discussed. [Pang, B., Lee, L., 2008. Opinion mining and sentiment analysis. Foundations and Trends in Information Retrieval] 

# Sentiment Analyis

## Challenge :
There are several things to take into consideration when approaching a Sentiment Analysis task. In general, there are two main approaches: 

- Sentiment lexicons using Natural Language Processing (NLP) techniques. A Sentiment lexicon is a list of words that are associated to polarity values (positive or negative). NLP techniques offer a deep level of analysis since they take into account the context words in the sentence. 

- Machine Learning classification algorithms. Because sentiment classification is a text classification problem, any existing supervised learning method can be directly applied [Bing Liu]. For example,  naive Bayes classification , logistic regression, support vector machines (SVM), etc..

In this work we'll work on ML classification and then try to get into the NLP and experience some of the basic techniques used.


## Data Wrangling and Preparation

### Load data

In this work we'll use a data-set that we obtained thankfully from Julian McAuley, at the University of San Diego (here)[http://jmcauley.ucsd.edu/data/amazon/] 

We'll be also inspired by their SIGIR and KDD papers (listed on the above page) as a baseline for our accuracy bench-marking.

This data-set contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014. 
We have decided to use electronics reviews for this work. Because electronics are not perfect so create a lot of contrasted opinions.
 

The file format is **NDJSON** (Newline Delimited JSON), So `stream_in` is the appropriate way of to load the data into a data frame.



## Exploratory Data Analysis
### Basic numbers
Let's explore some facts about our data.

Checking how many users are there. Almost we have 46K users for 50K reviews. So each user has done 1 unique review per product.



Checking how many products are there using the distinct ASIN (Amazon Standard Identification Number) amazon's unique product identifier. 



Let's look at at the the number of ratings per product. It's quite skewed with some extreme best sellers (a headphone from Koss) having 3000 reviews. 



### Simplification of data

Prepare the text for processing, convert it to lower case, and keep only relevant columns.
We then garbage collect the old `dat`




### Rating system in Amazon

The rating system used in Amazon is as follows : 

* emotional positive (5 stars)
* rational positive  (4 stars)
* neutral            (3 stars)
* rational negative  (2 stars)
* emotional negative (1 star )

**The user's star rating of his own review description as a subjective human interpretation of opinion. So, we consider that as the ground truth.** 

Let's see the distribution of these ratings in our case



> The ratings are very skewed towards positive feedback. Which is an indication that Amazon is not selling junk at least but it's not going to help in our modeling. We have to have equal likelihood of each class of the ratings. 

> In the next sections we'll solve this.

## Design decisions on reviews

In this work we will have a binary classification. Either Positive or Negative.
We objectively chose to demarcate each rating at the 2.x level divide, such that star ratings above this level would be marked as “1” and star ratings below this level would be marked as “0” 

We will collapse the 3, 4 and 5 stars ratings into “1” value, and the 1 and 2 stars ratings into Negative “0”

So at the end we'll have only 2 opinions : negative/positive
The simplest way to do this is by joining a mapping matrix



Let's check again the distribution of opinions



Even after this mapping the class proportions need to be corrected manually.
We calculate the skew (disproportion rate)



we have 3x more Positive than Negative. Let's remove 2/3 of Positive to adjust the proportions



Check the numbers, we are good to go :



# Machine Learning Classification

## Bag of Words

One of the simpler things to do with text is to treat each text as a "bag of words". We have used the `tm` package in order to construct a Term Document Matrix but the computer couldn't handle such  huge dimensions. So let's go with `tidytext`

Let's discover the top words for both positive and negative ratings. We use the `wordcloud` package to have a nice display. 






## Text tidiying

Do a series of transformations :

- lowercase
- remove punctuation
- strip white space



Use `nrc` lexicon as a bag of words that have sentiments but we're not going to look into these sentiments for the time being.




## Creating features from Bag of Words

Use `unrest_tokens` and join function in order to convert our reviews into a sparse Matrix.




## TF-IDF

Term Frequency - Inverse Document Frequency is term count within a document weighted against the term's ubiquity within the corpus. This weight is based on the principle that terms occurring in almost every document are therefore less specific to an individual document and should be scaled down. 
So a tf-idf value represents the term's relative importance within a document.

Compute tf-idf, inverse document frequency, and relative term frequency on document-feature matrices

$$tf(t,d) = \frac{f_{d}(t)}{\underset{w \in d}{max}}$$
$$idf(t,D) = log \left (\frac{|D|}{|d \in D : t \in d|}  \right )$$
$$tfidf(t,d,D) = tf(t,d)\cdot idf(t,D)$$
$$f_d(t) := freqency\ of\ term\ t\ in\ document\ d$$
$$D : corpus\ of\ documents$$
$$|D| : number\ of\ documents\ where\ the term\ t\appears $$
$$|d \in D : t \in d|\ :\ number\ of\ documents\ where\ the\ term\ t\ appears$$




## Reduce dimensionality (Single Vector Decomposition)

Suppose we have m words (features) and n documents 
Single vector decomposition (SVD) of an m*n real or complex matrix X is a factorization of the form :


$$\mathbf{ X = U \Sigma V^{T} }$$
$$ U\ is\ m x r\ matrix,\ columns\ of\ U\ contain\ the\ Eigenvectors\ of\ X \cdot X^{T} $$
$$ V\ is\ an\ r x n,\ columns\ of\ V\ contain\ the\ Eigenvectors\ of\ X^{T} \cdot X $$
$$ \Sigma\ is\ a\ diagonal\ r x r\ matrix.\ diagonal\ values\ are\ eigenvalues\ of\ X \cdot X^{T}$$

$$\begin{bmatrix} x_{1,1} & x_{1,2}  & \cdots & x_{1,n} \\ x_{2,1}& \ddots & & \vdots \\ \vdots &  & \ddots & \vdots \\ 
 x_{m,1}& \cdots &  \cdots & x_{m,n} \\ \end{bmatrix}= \begin{bmatrix} u_{1,1} & u_{1,2}  & \cdots & u_{1,r} \\ u_{2,1}& \ddots & & \vdots \\ \vdots &  & \ddots & \vdots \\u_{m,1}& \cdots &  \cdots & u_{m,r} \\\end{bmatrix} \cdot diag \begin{bmatrix}d_{1}\\\vdots \\  \vdots  \\  d_{r} \\ \end{bmatrix}\cdot \begin{bmatrix} v_{1,1} & v_{1,2}  & \cdots & v_{1,n} \\ v_{2,1}& \ddots & & \vdots \\ \vdots &  & \ddots & \vdots \\ v_{r,1}& \cdots &  \cdots & v_{r,n} \\ \end{bmatrix}$$

It's basically a PCA but without mean shifting.But SVD works better in SA because it is able to detect and extract small signals from noisy data. Noisy data here means words that are not significant for prediction.

In this context, it is known as latent semantic analysis (LSA).




The diagonal vector `d` generated is ordered by importance of each dimension.
Let's graph a cumulative variance:  



> Design decision
>
>We have to decide how many dimensions to keep. Looking at the graph we can keep 10 dimensions which represent almost 99% of the features.
This is curious and a puzzling choice to make. But let's aim for fast processing and later on see if adding more dimensions improves our prediction.

In order to rotate into our new space of reduced dimensions we have to limit the number of eigenvectors in `U` then transpose it an multiply it by the original X :

$$ \hat{X} = U^{T}\cdot X $$



Transform it back to a dataframe and bring back the truth y




If we consider the first 2 dimensions as predictors let's see how it looks








## Supervised training

Now that we know some basic facts about our data set, 
let's randomly split the data into training and test data. 
We set the seed `set.seed(755)` and use `sample()` function to select 
your test index to create two separate data frames
called: `train` and `test` from the original `ratings` data frame. 
 `test` contains a randomly selected 20% of the rows and training the other 80%. We will 
use these data frames to do the rest of the analyses in the problem set.




As we go along we will be comparing different approaches. Let's start by creating a benchmark table:



### Naive Bayes supervised training 

The basic idea to find the probabilities of categories given a text document by using the joint probabilities of words and categories. It is based on the assumption of word independence.
The starting point is the Bayes theorem for conditional probability, stating that, for a given data point x and class C: 

$$P(C/x) = \frac{P(x/C) \cdot P(C)}{P(x)}$$

By making the assumption that for a data point x = {x1,x2,…xj}, the probability of each of its attributes occurring in a given class is independent, we can estimate the probability of x as follows :

$$P(C/x) = P(C) \cdot \prod_{j} P(x_{j}/C)$$


Now, we can train the naive Bayes model with the training set. We'll be using e1071 package made by David Meyer from TU Wien.
The naiveBayes function requires a 




### Support Vector Machine 

SVMs were developed by Cortes & Vapnik (1995) [1] for binary classification. Their approach may be roughly sketched as follows:
> Class separation: basically, we are looking for the optimal separating hyperplane between the two classes by maximizing the margin between the classes’ closest points —the points lying on the boundaries are called support vectors, and the middle of the margin is our optimal separating hyperplane;

[1] : Cortes, C. & Vapnik, V. (1995). Support-vector network. Machine Learning, 20, 1–25



### Logistic regression



In the previous model we were assuming a less optimal cut-off. Let's use LDA assuming of course that our covariates are bivariate normal





### Cross-Validation of Logistic Regression




### Randon Forest 

The `randomForest` prediction function works similarly to decision trees


 

 



## Conclusion

In this section we used a supervised machine learning approach and tried several classification models. From the table below, the algorithmic approach using randomForest gives the best accuracy. we are 25% better than flipping a coin with only a classified bag of words.





# Sentiment lexicons using Natural Language Processing

## Feature Extraction and Classification


The first step is to extract 2 types of phrases :
- Verbal phrases  that may imply opinions. Example : "I didn't like the design"
- Noun phrases that may describe the product. Example : The design was not good""

In our work, we use the OpenNLP chunker. This chunker will split a given text into a sequence of semantically correlated phrases but does not specify their internal structure, nor their role in the main sentence.

openNLP `Maxent_Chunk_Annotator` requires a pre-made models. These are conveniently available to R by installing the respective `openNLPmodels.language` package from the repository at http://datacube.wu.ac.at

To install English language model (a heavy download 74MB) :
`install.packages("openNLPmodels.en", repos = "http://datacube.wu.ac.at")`




In the following, we'll experiment with a 1-star rated wireless charger from Amazon. And to evaluate this chuncker we selected a text with a lot of grammar and orthographic mistakes:

Fix punctuation issues in this example "best.I" using `gsub` from the `stringr` package.



## Sentence chunking and Part-of-Speech tagging

The part-of-speech tags meaning is found in the [Penn Treebank Project](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

The chunk tags contain the name of the chunk type, for example I-NP for noun phrase words and I-VP for verb phrase words. Most chunk types have two types of chunk tags, B-CHUNK for the first word of the chunk and I-CHUNK for each other word in the chunk




Transform it to  [word, pos-tag, chunk-tag] dataframe :



The first column contains the current word.

The second its part-of-speech tag

The third its chunk tag.

### Feature identification

Now we want to identify features. For the purpose of this work we won't go for a full analysis using penn tree. Rather we'll simply identify the following words inside a verbal phrase:

- [VB] : verb w/ positive/negative sentiment : like, hate , etc
- [JJ] : adjective w/ positive/negative sentiment : bad, junk
- [RB] : adverb polarity inverter such as : n't, or incr/decrementers such as : too, very, more etc.

First, let's use tidytext sentiments to construct such mapping table



## Scoring algorithm

Now comes the scoring algorithm part. The idea is to find a VB or JJ and look for a surrounding RB multiplier:

For example : n't like : will compute 2 x -1



Now we need to wrap all above into a function 



Apply it to a number of review texts .



