#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:15:41 2019

@author: aneekalatif
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 09:36:58 2019

@author: aneekalatif
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:23:12 2019

@author: aneekalatif
"""

from matplotlib import pyplot as plt
import nltk
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer 
#from wordcloud import WordCloud
import csv
import re
from collections import Counter, defaultdict 

#%%



#starting the tokenization of text into words
#Read in text 

#read the annotations csv as a dataframe 
spot_annot= pd.read_csv('C:\\Users\\User\\Desktop\\Syracuse\\Summer19\\IST 736\\spotify_annotations2.csv')
#get the first 25 reviews 



#split the ratings by rater 
rater1=spot_annot['Genre (Aneeka)']
rater2=spot_annot['Genre (Samantha)']
rater3=spot_annot['Genre (Michael) ']

#remove missing values 
rater1=rater1.dropna()
rater2=rater2.dropna()
rater3=rater3.dropna()

#having indexing issues so reinstantializing as list
rater1=list(rater1)
rater2=list(rater2)
rater3=list(rater3)


#calculate pairwise kappa scores
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(rater1, rater2)
cohen_kappa_score(rater1, rater3)
cohen_kappa_score(rater2, rater3)
#%%
##############################################

stop_words = ["Aneeka", "dtype", "object", "length", "Name", "Michael", "Samantha", "Genre",
              "\'", "B"]

wordcloudaneeka = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                stopwords=stop_words,
                min_font_size = 10).generate(str(rater1)) 
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloudaneeka) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

wordcloudsamantha = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                stopwords=stop_words,
                min_font_size = 10).generate(str(rater2)) 
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloudsamantha) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

wordcloudmichael = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                stopwords=stop_words,
                min_font_size = 10).generate(str(rater3)) 
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloudmichael) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
#%%
#rater1[17]
#rater2[17]
#rater3[17]

#rater1[0]==rater2[0]


final_annot = []
for i in range(0, len(rater1)):
    if rater1[i] == rater2[i]:
        final_annot.append(rater1[i])
    elif rater1[i] == rater3[i]:
        final_annot.append(rater1[i])
    elif rater2[i] == rater3[i]:
        final_annot.append(rater2[i])
    else:
        final_annot.append("deliberation needed")
        
#find indexes for those where deliberation was needed 
index = final_annot.index("deliberation needed")
print(index)

#change the values based on the final decision, after comparing the three labels 
#using rater1[index], rater2[index], rater3[index]
#after this process, there should be no more elements in the list that are "deliberation needed"
final_annot[14]="Dance"
final_annot[15]="Hip-Hop"
final_annot[19]="Dance"
final_annot[25]="Country"
final_annot[29]="EDM" 
final_annot[34]="EDM" 
final_annot[40]="EDM"
final_annot[41]="Funk"
final_annot[51]="EDM"
final_annot[52]="House"
final_annot[71]="Indie"
final_annot[76]="House"
final_annot[82]="EDM"

print (final_annot)


#wordcloudannot = WordCloud(width = 800, height = 800, 
               ## background_color ='white',  
#                stopwords=stop_words,
     #           min_font_size = 10).generate(str(final_annot)) 
#plt.figure(figsize = (8, 8), facecolor = None) 
#plt.imshow(wordcloudannot) 
#plt.axis("off") 
#plt.tight_layout(pad = 0)

#%%
 ################ look at the lyrics
lyrics = spot_annot.dropna()
lyrics=list(lyrics["Lyrics"])

print(len(lyrics))


#text formatting cleanup 


#clean the formatting

lyrics=[re.sub(r'[\n]', '', x) for x in lyrics]
lyrics=[re.sub(r'[/]', '', x) for x in lyrics]
lyrics=[re.sub(r'[\\]', '', x) for x in lyrics]
lyrics=[re.sub(r'[\\]', '', x) for x in lyrics]
lyrics=[re.sub(r'[!@#$-.?]', '', x) for x in lyrics]
#remove anything thats not alphabetic characters, removes digits too 
lyrics = [re.sub(r'[^a-zA-Z\s:]', '', x) for x in lyrics] 


#set all characters to lowercase (use a temp variable)
temptext=[]
for i in range(0,len(lyrics)):
    temptext.append(
    lyrics[i].lower())
print(temptext)
#set temp variable back to main variable 
lyrics=temptext

#vectorization 
nltk.download('stopwords')
text_stopwords = list(stopwords.words("English"))
text_stopwords.extend(("abajo", "zip","rows", "columns", "abraza"))


MycountVect = CountVectorizer(
        input=lyrics, 
        analyzer='word',
        stop_words = list(text_stopwords))
CV=MycountVect.fit_transform(list(lyrics))
mycolumnnames=MycountVect.get_feature_names()
vectorizedf_text = pd.DataFrame(CV.toarray(),columns=mycolumnnames)
print (vectorizedf_text)
#%%
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = text_stopwords, 
                min_font_size = 10).generate(str(vectorizedf_text)) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

wordfreq = nltk.FreqDist(str(vectorizedf_text))
print (wordfreq)
wordfreq.most_common(5)
#%%


#################### SUBLISTS For wordcloud generation by genre##################################
newdf = spot_annot
newdf = spot_annot.dropna()
newdf['Genre']=final_annot
hiphop=(newdf.loc[newdf['Genre'] == 'Hip-Hop'])
hiphoplyrics = hiphop['Lyrics']

hiphopcv = CountVectorizer(
        input=hiphoplyrics, 
        analyzer='word',
        stop_words = list(text_stopwords))
CV=MycountVect.fit_transform(list(hiphoplyrics))
mycolumnnames=MycountVect.get_feature_names()
hiphop_text = pd.DataFrame(CV.toarray(),columns=mycolumnnames)
print (hiphop_text)

wordcloudhiphop = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = text_stopwords, 
                min_font_size = 10).generate(str(hiphop_text)) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloudhiphop) 
plt.axis("off") 
plt.tight_layout(pad = 0) 




pop=(newdf.loc[newdf['Genre'] == 'Pop'])
poplyrics = pop['Lyrics']

popcv = CountVectorizer(
        input=poplyrics, 
        analyzer='word',
        stop_words = list(text_stopwords))
CV=MycountVect.fit_transform(list(poplyrics))
mycolumnnames=MycountVect.get_feature_names()
pop_text = pd.DataFrame(CV.toarray(),columns=mycolumnnames)
print (pop_text)

wordcloudpop = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = text_stopwords, 
                min_font_size = 10).generate(str(pop_text)) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloudpop) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

edm=(newdf.loc[newdf['Genre'] == 'EDM'])
edmlyrics = edm['Lyrics']

edmcv = CountVectorizer(
        input=edmlyrics, 
        analyzer='word',
        stop_words = list(text_stopwords))
CV=MycountVect.fit_transform(list(edmlyrics))
mycolumnnames=MycountVect.get_feature_names()
edm_text = pd.DataFrame(CV.toarray(),columns=mycolumnnames)
print (edm_text)

wordcloudedm = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = text_stopwords, 
                min_font_size = 10).generate(str(edm_text)) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloudedm) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
#%%
import gensim
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer
from nltk.stem import PorterStemmer 
stemmer = PorterStemmer()

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
#%%
cleaned_df = pd.DataFrame(lyrics, columns = ['Lyrics'])
processed_docs = cleaned_df['Lyrics'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
#%%
from pprint import pprint
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=10)
    
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())

#doc_lda = lda_model[bow_corpus]


# Compute Perplexity
perplx = lda_model.log_perplexity(bow_corpus)
print('\nPerplexity: ', perplx )
#%%
import pyLDAvis.sklearn as LDAvis
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.show(vis)
#%%
import sklearn
from sklearn.cluster import KMeans
MyMatrix = vectorizedf_text.values

kmeans_object = sklearn.cluster.KMeans(n_clusters=5)
print(kmeans_object)
kmeans_object.fit(MyMatrix)
# Get cluster assignment labels
labels = kmeans_object.labels_
print(labels)
spot_annot_clean = spot_annot.dropna()
Myresults = pd.DataFrame([spot_annot_clean.index,labels]).T
print(Myresults)
#%%
#len(set(final_annot))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(lyrics, final_annot, test_size=0.5, random_state=1)

X_train_vec=MycountVect.transform(X_train)
X_test_vec=MycountVect.transform(X_test)
#%%
import numpy as np
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, (counts/len(y_train))*100)))
#%%
from sklearn.naive_bayes import MultinomialNB

# initialize the MNB model
nb_clf= MultinomialNB()

nb_clf.fit(X_train_vec,y_train)
nb_clf.score(X_test_vec,y_test)
#%%
from sklearn.metrics import confusion_matrix
y_pred = nb_clf.fit(X_train_vec, y_train).predict(X_test_vec)
cm=confusion_matrix(y_test, y_pred, labels=['Dance', 'EDM', 'Hip-Hop', 'Pop', 'R&B'])
print(cm)
from sklearn.metrics import classification_report
target_names = ['Dance', 'EDM', 'Hip-Hop', 'Pop', 'R&B']
print(classification_report(y_test, y_pred, target_names=target_names))
#%%
del final_annot[41]
del lyrics[41]
#%%
del final_annot[75]
del lyrics[75]

del final_annot[70]
del lyrics[70]

del final_annot[51]
del lyrics[51]

del final_annot[45]
del lyrics[45]

del final_annot[36]
del lyrics[36]

del final_annot[25]
del lyrics[25]

#%%
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
svm_clf3 = LinearSVC(C=0.5)
svm_clf3.fit(CV,final_annot)
print(cross_val_score(svm_clf3, CV, final_annot, cv=3))
