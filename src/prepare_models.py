import pandas as pd
import numpy as np
import requests
from text_cleaned import TextCleaned
from tsne_model_plot import TsneModelPlot
from model_metrics import ModelMetrics
from reading_data import ReadingData
from time import time
import logging
from gensim.models import word2vec
import gensim
import sklearn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

#class PrepareModels:

#def prepare_model(self):

# Reading dataset

file_name = "SMSSpamCollection.txt"
num_symbols = 20000000
rd = ReadingData()
df = rd.reading_data(file_name, num_symbols)

# Cleaning dataset

tc = TextCleaned()
df['text_cleaned'] = list(map(tc.cleaning_text, df.text))
text_cleaned = df['text_cleaned']

# Word2vec when data is not labeled

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# setting parameters https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html

num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 3
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

print("Training model (takes a minute)")
model_word2vec = word2vec.Word2Vec(text_cleaned, workers=num_workers, size=num_features, min_count = min_word_count,
              window = context, sample = downsampling)

# saving the model and later to load use Word2Vec.load()

model_name = "word2vec_model_vol2"
model_word2vec.save(model_name)

model_word2vec = gensim.models.Word2Vec.load("word2vec_model_vol2")
print(model_word2vec.most_similar("free"))
'''
#tsne_model = TsneModelPlot()
#tsne_model.tsne_plot(model_word2vec)

# Models when data is labeled

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df['text_cleaned'], df["target"],
                                                                train_size=0.7, random_state=42)

bow_transform = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[3, 3], lowercase=False)
X_train_bow = bow_transform.fit_transform(X_train)
X_test_bow = bow_transform.transform(X_test)

tfidf_transform = TfidfTransformer(norm=None)
X_train_tfidf = tfidf_transform.fit_transform(X_train_bow)
X_test_tfidf = tfidf_transform.transform(X_test_bow)

def logistic_classify(X_train, y_train, X_test, y_test, description, _C=1.0):

    model = LogisticRegression(C=_C).fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print('Test Score with', description, 'features', score)
    return model

model_bow = logistic_classify(X_train_bow, y_train, X_test_bow, y_test, 'bow')
model_tfidf = logistic_classify(X_train_tfidf, y_train, X_test_tfidf, y_test, 'tf-idf')

model_bow.fit(X_train_bow, y_train)

y_predicted = model_bow.predict(X_test_bow)

model_metrics = ModelMetrics()
model_metrics(y_test, y_predicted)

# Predict custom message
'''

'''
# ---- check words freq -----
bow_data = []
for text in text_cleaned:
    words = word_tokenize(text)
    for w in words:
        bow_data.append(w)

bow_data = nltk.FreqDist(bow_data)

bow_data.plot(20, title='Most Common Words, top 20').savefig('results/bow_data_plot.png')
'''
