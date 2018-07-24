
# coding: utf-8

# In[ ]:

#
#get_ipython().system('python -m pip install --upgrade pip')
#get_ipython().system('pip install msgpack')
#get_ipython().system('pip install ipython')
#get_ipython().system('pip install pyspark==2.3.0')
#get_ipython().system('pip install pyLDAvis')


# In[1]:


import os
# os.environ["JAVA_HOME"] = "C:\Java\jdk1.8.0_172"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--master local[2] pyspark-shell"
os.environ["JAVA_HOME"] = "C:/jdk1.8.0_171"
#USATO PER RISOLVEREE PROBLEMA JAVA GATEWAY ..... (ho anche installato java e inserito manualmente java home tra le variabili ambiente)


# In[2]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.sql.functions import monotonically_increasing_id
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.corpus import stopwords
import pyspark
import string
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, Tokenizer, RegexTokenizer, StopWordsRemover
nltk.download("stopwords")
import pyLDAvis

#%%
#In[3]:


spark = SparkSession.builder.appName('NLP_topicModel').getOrCreate()

sc = spark.sparkContext
# In[4]:


def read_data_from_json(filename, multiline = True):
    df = spark.read.json(filename, multiLine = multiline)

    return df
def read_our_json_dataset (filename, multiline = True):
    df = read_data_from_json(filename, multiline)
    df = df.fillna({'testo_txt_it': ''})
    df = df.withColumn("uid", monotonically_increasing_id())
    df = df.withColumn("year_month", df.published_dt.substr(1,7))
    return(df)

# In[6]:


def myStemmer (record, array=False):
    stemmer = SnowballStemmer("italian")
    if array:
        text_out = [stemmer.stem(word) for word in record] 
    else:    
        text = record [2]#9
        text_out = [stemmer.stem(word) for word in text]
    return(text_out)

# def myLemmatizer (record):
#     tagger = ttw.TreeTagger(TAGLANG='it')
#     if array:
#         tags = tagger.tag_text(record)  
#         tags2 = ttw.make_tags(tags)
#         final_tags = [tags2[i][2] for i in range (len(tags2))]
#     else:
#         text = record[2]
#         tags = tagger.tag_text(text)
#         tags2=ttw.make_tags(tags)
#         final_tags = [tags2[i][2] for i in range (len(tags2))]
#     return(final_tags)

def removePunctuation(column, name_new_col = "sentence"):
    return trim(lower(regexp_replace(column, '[^A-Za-z]', ' '))).alias(name_new_col)


# In[ ]:


def get_token(df, input_col = "sentence", output_col = "tokenized"):
    tokenizer = Tokenizer(inputCol = input_col, outputCol = output_col)#testo_txt_it
    wordsDataFrame = tokenizer.transform(df)#rawdata
    return(wordsDataFrame)

def get_stemmed_words (df,input_col = "tokenized", output_col = "stemmed"):
    remover = StopWordsRemover(inputCol = input_col , outputCol="tokenizedNew", stopWords = [" ", ""])
    df = remover.transform(df)
    udf_myStemmer = udf(myStemmer, (ArrayType(StringType()))) # if the function returns an int
    df = df.withColumn(output_col, udf_myStemmer(struct([df[x] for x in df.columns])))
    return (df)

def words_widely_used_and_short (df, input_col = "stemmed", number_of_words = 100):
    cv_tmp = CountVectorizer(inputCol = input_col, outputCol="tmp_vectors")
    cv_tmp_model = cv_tmp.fit(df)
    top_words = list(cv_tmp_model.vocabulary[0:number_of_words])
    less_then_3_charachters = [word for word in cv_tmp_model.vocabulary if len(word) <= 3 ]
    return(top_words , less_then_3_charachters)

def collect_stopwords (df, input_col = "stemmed", number_of_words = 100):
    top_words, less_then_3_charachters = words_widely_used_and_short(df, input_col, number_of_words)
    stopWordsNLTK = list(set(stopwords.words('english')))+list(set(stopwords.words('italian')))
    stopWordsCustom = [" ","", "dal", "al","davan","avev","qualc", "qualcuno", "qualcosa", "avevano", "davanti", "aveva","e","avere", "fare","la","li", "lo", "gli", "essere", "solo", "per", "cosa", "ieri","disponibile", "anno", "detto", "quando","fatto", "sotto", "alcuna", "quali"]
#Add additional stopwords in th, is list
    stopWordsPySpark = StopWordsRemover.loadDefaultStopWords("italian")
#Combine all the stopwords
    stpw = top_words + stopWordsNLTK  + stopWordsCustom +stopWordsPySpark+ less_then_3_charachters 
    stem_stopw = myStemmer(stpw, True) #stemming the stopwords
    return (stpw+stem_stopw)

def remove_stopwords_train (df, input_col = "stemmed", output_col = "final", number_of_words = 100):
    stopwords = collect_stopwords(df, input_col, number_of_words )
    removerNew = StopWordsRemover(inputCol = input_col, outputCol = output_col, stopWords = stopwords) #Remove stopwords from the tokenized list
    new_df= removerNew.transform(df)#dropping the stemmed stopwords from the stemmed word 
    return(new_df, stopwords)


def remove_stopwords_test (df, stopwords, input_col = "stemmed", output_col = "final"):
    removerNew = StopWordsRemover(inputCol = input_col, outputCol = output_col, stopWords = stopwords) #Remove stopwords from the tokenized list
    new_df= removerNew.transform(df)#dropping the stemmed stopwords from the stemmed word 
    return(new_df)
    
def check_and_remove_null_string (df):
    pandasdf = df.toPandas()
    indexes = [i for i in range (len (pandasdf)) if pandasdf.iloc[i][0] == []]
    if len(indexes)!= 0:
        dfnew = spark.createDataFrame(pandasdf.drop(index = indexes ))
        return(dfnew)
    else: 
        return (df)
    
def pulizia_df_train (df, text_col = "testo_txt_it", number_of_words =100):
    pulito = df.select(removePunctuation(col(text_col), "sentence"))
#Tokenize the text in the text column
    wordsDataFrame = get_token(pulito,"sentence","tokenized" )    
    wordsDataFrame = get_stemmed_words(wordsDataFrame, "tokenized", "stemmed")
    df_clean, stopwords = remove_stopwords_train (wordsDataFrame, "stemmed", "final", number_of_words)
    clean_text = check_and_remove_null_string(df_clean)
    return(clean_text, stopwords)
 
def clean_test (df, stopwords, text_col = "testo_txt_it"):
    pulito = df.select(removePunctuation(col(text_col), "sentence"))
    wordsDataFrame = get_token(pulito,"sentence","tokenized" )    
    wordsDataFrame = get_stemmed_words(wordsDataFrame, "tokenized", "stemmed")
    removerNew = StopWordsRemover(inputCol = "stemmed", outputCol = "final", stopWords = stopwords) #Remove stopwords from the tokenized list
    new_df2= removerNew.transform(wordsDataFrame)
    clean_text = check_and_remove_null_string(new_df2)
    return(clean_text)
    
#%%
def train_and_test (df, perc_train = 0.95, perc_test = 0.05):
    train, test = df.select("final").randomSplit([perc_train, perc_test])
    return (train, test)


#%%
    

def tf_train (df):
    cv_train = CountVectorizer(inputCol="final", outputCol="rawFeatures", vocabSize = 4000, minDF = 3, minTF = 2)
    cvmodel = cv_train.fit(df)
    cvDatasetTrain = cvmodel.transform(df)
    return (cvmodel, cvDatasetTrain)

def idf_train (cvDatasetTrain):
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(cvDatasetTrain)
    rescaledDataTrain = idfModel.transform(cvDatasetTrain)
    return (idfModel, rescaledDataTrain)

def tf_idf_train (df):
    cvmodel, cvDatasetTrain = tf_train(df)
    idfModel, rescaledDataTrain = idf_train(cvDatasetTrain)
    return (cvmodel, cvDatasetTrain, idfModel, rescaledDataTrain)

def tf_test (df, cvmodel):    
    return (cvmodel.transform(df))

def idf_test (idfModel, cvDatasetTest):
    return (idfModel.transform(cvDatasetTest))

def tf_idf_test (df, cvmodel, idfModel):
    cvDatasetTest = tf_test (df, cvmodel)
    rescaledDataTest = idf_test(idfModel, cvDatasetTest)
    return (cvDatasetTest, rescaledDataTest)

#%%

def training_model (train, k = 10, maxiter = 120, features_name = "features", optimizer_type = "online",seed = 123):
    lda = LDA(k=k, seed=123, optimizer = optimizer_type , featuresCol= features_name, subsamplingRate = 0.1, 
          learningDecay = 0.5, optimizeDocConcentration= True, maxIter = maxiter)
    ldamodel = lda.fit(train)
    predictionTrain = ldamodel.transform(train)
    return (ldamodel, predictionTrain)


def testing_model (test, ldamodel):
    prediction = ldamodel.transform(test)
    
#%%
  
    
def doclengths_and_termfreq (countVectorizer_transf,column, df):
    rows_contents = countVectorizer_transf.select(column).take(df.count())
    doc_lengths=[]
    term_frequencies_list = []
    for i in range(len(rows_contents)):
        sparse_vector = rows_contents[i].asDict()[column] #è uno sparse vector se printato visualizza solo elem nonzero e
                                                    #rappresenta l'indice parola e la sua freq all'interno del documento
        doc_lengths.append(sparse_vector.numNonzeros()) #numero elementi non zero all'interno dell'array (senza le parentesi da 
                                                        # l'indice degli elem nonzero)
        term_frequencies_list.append(sparse_vector.toArray()) #toArray() mostra il vettore per intero, compresi gli zeri
    term_frequency = list(np.sum(term_frequencies_list, axis = 0)) #andiamo a sommare i vettori dei vari documenti, per ottenere le freq nel corpus
    to_drop = [i for i in range(len(doc_lengths)) if doc_lengths[i] == 0]       
    doc_lengths = [doc_lengths[i] for i in range(len(doc_lengths)) if i not in to_drop]  
    return(term_frequency, doc_lengths, to_drop)
    
def extract_data (ldamodel, countVectorizer_transf, countVectorizer_fit, df, results_pred, column = "rawFeatures"):
    vocab = countVectorizer_fit.vocabulary
    term_frequency, doc_lengths, to_drop = doclengths_and_termfreq (countVectorizer_transf, column, df)
    word_dists_topic = np.asmatrix(ldamodel.topicsMatrix().toArray())
    #qui abbiamo per ogni parola la distribuzione nei vari topic (1000, 10), noi vogliamo l'opposto (10, 1000)
    #quindi andiamo a trasporre la matrice appena creata e a trasformarla in una lista di liste (formato desiderato)
    topic_term_dists = word_dists_topic.transpose().tolist()
    pred_top_dists_pd = results_pred.select("topicDistribution").toPandas()
    doc_topic_dists = [list(pred_top_dists_pd.iloc[i][0]) for i in range(results_pred.count())]
    doc_topic_dists = [doc_topic_dists[i] for i in range(len(doc_topic_dists)) if i not in to_drop]       
#     results_visual = {"vocab": vocab, 'doc_lengths': doc_lengths , 'term_frequency': term_frequency, 'doc_topic_dists': doc_topic_dists, 'topic_term_dists': topic_term_dists}
    return(vocab, doc_lengths, term_frequency, topic_term_dists, doc_topic_dists)
 