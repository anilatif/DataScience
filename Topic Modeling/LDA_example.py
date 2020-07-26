#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 12:25:31 2019

@author: aneekalatif
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:41:37 2019

@author: jerem
"""

## CODE FOR LDA - the first section is what we covered in class. 
## Also - you will may wish to create your own data

# -*- coding: utf-8 -*-

###################################################
##
## LDA for Topic Modeling
##
###################################################

## DATA USED IS FROM KAGGLE
##
## https://www.kaggle.com/therohk/million-headlines/version/7

## Tutorial and code taken from 
## https://towardsdatascience.com/
## topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

## Other good tutorials
## https://nlpforhackers.io/topic-modeling/
# https://www.kaggle.com/meiyizi/spooky-nlp-and-topic-modelling-tutorial

#%%

import pandas as pd

#data = pd.read_csv('DATA/abcnews_date_text_Kaggle.csv', error_bad_lines=False);
data_small=pd.read_csv('data/abcnews-date-text.csv', error_bad_lines=False);
print(data_small.head())
## headline_text is the column name for the headline in the dataset
#data_text = data[['headline_text']]
data_text_small = data_small[['headline_text']]
print(data_text_small)

#data_text['index'] = data_text.index
data_text_small['index'] = data_text_small.index
#print(data_text_small.index)
#print(data_text_small['index'])

#documents = data_text
documents = data_text_small
print(documents)

print("The length of the file - or number of docs is", len(documents))
print(documents[:5])


#%%
###################################################
###
### Data Prep and Pre-processing
###
###################################################
#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python

import gensim
## IMPORTANT - you must install gensim first ##
## conda install -c anaconda gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer

import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')
from nltk import PorterStemmer
from nltk.stem import PorterStemmer 
stemmer = PorterStemmer()

from nltk.tokenize import word_tokenize 
from nltk.stem.porter import *

#NOTES
##### Installing gensim caused my Spyder IDE no fail and no re-open
## I used two things and did a restart
## 1) in cmd (if PC)  psyder --reset
## 2) in cmd (if PC) conda upgrade qt

######################################
## function to perform lemmatize and stem preprocessing
############################################################
## Function 1
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

## Function 2
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

#Select a document to preview after preprocessing
doc_sample = documents[documents['index'] == 50].values[0][0]
print(doc_sample)
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

#%%

## Preprocess the headline text, saving the results as ‘processed_docs’
processed_docs = documents['headline_text'].map(preprocess)
print(processed_docs[:10])

#%%

## Create a dictionary from ‘processed_docs’ containing the 
## number of times a word appears in the training set.

dictionary = gensim.corpora.Dictionary(processed_docs)

## Take a look ...you can set count to any number of items to see
## break will stop the loop when count gets to your determined value
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 5:
        break
    
    
#%%    
#print(processed_docs)   
## Filter out tokens that appear in
## - - less than 15 documents (absolute number) or
## - - more than 0.5 documents (fraction of total corpus size, not absolute number).
## - - after the above two steps, keep only the first 100000 most frequent tokens
 ############## NOTE - this line of code did not work with my small sample
## as it created blank lists.....       
#dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

for doc in processed_docs:
    print(doc)

print(dictionary)

#%%
#######################
## For each document we create a dictionary reporting how many
##words and how many times those words appear. Save this to ‘bow_corpus’
##############################################################################
#### bow: Bag Of Words
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(bow_corpus[3:5])


#%%
#################################################################
### TF-IDF
#################################################################
##Create tf-idf model object using models.TfidfModel on ‘bow_corpus’ 
## and save it to ‘tfidf’, then apply transformation to the entire 
## corpus and call it ‘corpus_tfidf’. Finally we preview TF-IDF 
## scores for our first document.

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
## pprint is pretty print
from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    ## the break will stop it after the first doc
    break

#%%

#############################################################
### Running LDA using Bag of Words
#################################################################
    
# ~ 12 minutes
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=3)
    
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())

#doc_lda = lda_model[bow_corpus]


# Compute Perplexity
perplx = lda_model.log_perplexity(bow_corpus)
print('\nPerplexity: ', perplx )  # a measure of how good the model is. lower the better.

#%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
#%% 
# Compute Coherence Score
from gensim.models import CoherenceModel
#coherence_model_lda = CoherenceModel(model=lda_model, texts=data_text_small, dictionary=dictionary, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)

#%%
import pyLDAvis
import pyLDAvis.sklearn as LDAvis

import pyLDAvis.gensim 
import matplotlib.pyplot as plt
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.show(vis)

# Topic 8 -> Accidents
# Topic 3 -> Crime
# Topic 7 -> Weather
# Topic 5 -> Trade
# Topic 1 -> Sports
# Topic 6 (2) -> Election Topics

#%%

################################################################
## sklearn
###################################################################3
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

## From previous example with Authors!!
NUM_TOPICS = 3

filenames = ['data/110_baldwin_x_wi.txt', 'data/110_bean_x_il.txt', 
             'data/110_berkley_x_nv.txt', 'data/110_boyda_x_ks.txt']
#%%
MyVectLDA=CountVectorizer(input='filename',encoding = "ISO-8859-1" )
##path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\SmallTextDocs"
data_vectorized = MyVectLDA.fit_transform(filenames)
#%%
ColumnNamesLDA=MyVectLDA.get_feature_names()
CorpusDF_LDA=pd.DataFrame(data_vectorized.toarray(),columns=ColumnNamesLDA)
CorpusDF_LDA = CorpusDF_LDA[CorpusDF_LDA.columns.drop(list(CorpusDF_LDA.filter(regex='\d+')))]
CorpusDF_LDA = CorpusDF_LDA[CorpusDF_LDA.columns.drop(list(CorpusDF_LDA.filter(regex='\_+')))]
print(CorpusDF_LDA)


lda_model = LatentDirichletAllocation(n_topics=NUM_TOPICS, max_iter=10, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
lda_Z = lda_model.fit_transform(data_vectorized)
lda_Z_DF = lda_model.fit_transform(CorpusDF_LDA)
print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Build a Non-Negative Matrix Factorization Model
nmf_model = NMF(n_components=NUM_TOPICS)
nmf_Z = nmf_model.fit_transform(CorpusDF_LDA)
print(nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
 
# Build a Latent Semantic Indexing Model
lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
lsi_Z = lsi_model.fit_transform(CorpusDF_LDA)
print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
 
 
# Let's see how the first document in the corpus looks like in
## different topic spaces
print(lda_Z_DF[0])
print(nmf_Z[0])
print(lsi_Z[0])

## implement a print function 
## REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
 
print("LDA Model:")
print_topics(lda_model, MyVectLDA)
#print("=" * 20)
 
print("NMF Model:")
print_topics(nmf_model, MyVectLDA)
#print("=" * 20)
 
print("LSI Model:")
print_topics(lsi_model, MyVectLDA)
#print("=" * 20)
#%%


#########################################
## Try sklean LDA with political data
##########################################
import os
all_file_names = []

path="/Users/aneekalatif/Textmining/Data/polpapers"

FileNameList=os.listdir(path)
#%%
#print(FileNameList)
ListOfCompleteFiles=[]
for name in os.listdir(path):
    print(path+ "/" + name)
    next=path+ "/" + name
    ListOfCompleteFiles.append(next)
#print("DONE...")
print("full list...")
print(ListOfCompleteFiles)
#%%

text_stopwords = list(stopwords.words("English"))
text_stopwords.extend(["mr", "doc", "docno", "text","representative","senate","legislation",
                       "speaker", "house", "would", "us", "american", "people","going", "dsdb", "bud1",
                       "bill", "representatives", "congress", "today", "support", "important",
                       "one", "also", "want", "madam", "year", "colleagues", "make",
                       "like", "president", "time", "2007", "2008", "chairman", "think",
                       "act", "know", "years", "many", "mrs", "may", "new"])

MyVectLDA_DH=TfidfVectorizer(input='filename',encoding = "ISO-8859-1", stop_words=text_stopwords,
                             strip_accents="unicode")
##path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\SmallTextDocs"
Vect_DH = MyVectLDA_DH.fit_transform(ListOfCompleteFiles)
ColumnNamesLDA_DH=MyVectLDA_DH.get_feature_names()
CorpusDF_DH=pd.DataFrame(Vect_DH.toarray(),columns=ColumnNamesLDA_DH)
print(CorpusDF_DH)



lda_model_DH = LatentDirichletAllocation(n_topics=5, max_iter=10, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
LDA_DH_Model = lda_model_DH.fit_transform(Vect_DH)

print("SIZE: ", LDA_DH_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in
## different topic spaces
print("First Doc political data docs...")
print(LDA_DH_Model[0])
print("Seventh Doc in political data docs...")
print(LDA_DH_Model[6])

## Print LDA using print function from above
print("LDA Dog and Hike Model:")
print_topics(lda_model_DH, MyVectLDA_DH)


#%%
####################################################
##
## VISUALIZATION
##
####################################################
import pyLDAvis.sklearn as LDAvis
import pyLDAvis
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
panel = LDAvis.prepare(lda_model_DH, Vect_DH, MyVectLDA_DH, mds='tsne')
pyLDAvis.show(panel)
#%%


import matplotlib.pyplot as plt
import numpy as np

word_topic = np.array(lda_model_DH.components_)
word_topic = word_topic.transpose()
vocab = MyVectLDA_DH.get_feature_names() 
num_top_words = 10
vocab_array = np.asarray(vocab)

fontsize_base =50 / np.max(word_topic) # font size for word with largest share in corpus

for t in range(5):
    plt.subplot(1, 5, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 1.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base*share)

plt.tight_layout()
plt.show()
#%%



# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=text_stopwords,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

#topics = np.array(lda_model_DH.components_)
#topics = word_topic.transpose()

topics = lda_model.show_topics(formatted=False)
fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
