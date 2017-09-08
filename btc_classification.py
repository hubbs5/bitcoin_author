# Author: Christian Hubbs
# 08.09.2017
# Bitcoin Text Classification V1.0

import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import string
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import snowball
# Custom functions
from clf_functions import clf_score, clf_comp

path = os.getcwd() + "/btc_texts"
i = 0
df = pd.DataFrame(columns=['author', 'text'])
for x, file in enumerate(os.listdir(path)):
    text = open(os.path.join(path, file), "r", encoding="utf8")
    print(file)
    for line in text.readlines():      
        # Store author name and text in df
        author = file.split("_")[0]
        df = df.append(pd.DataFrame({'author': author, 'text': line}, index=[i]))            
        i += 1
    
    text.close()
    
# Separate Nakamoto's texts from the suspects
df_suspects = df[df['author']!='nakamoto'].reset_index(drop=True)
df_nakamoto = df[df['author']=='nakamoto'].reset_index(drop=True)

# Map author names to unique letters

names = df_suspects['author'].unique()
letters = string.ascii_uppercase
name_dict = {x: letters[i] for i, x in enumerate(names)}
df_suspects['author_id'] = [name_dict[i] for i in df_suspects['author']]

word_count_sus = df_suspects.text.apply(lambda x: len(x.split()))
word_count_nak = df_nakamoto.text.apply(lambda x: len(x.split()))

plt.figure(figsize=(12,8))
plt.subplot(211)
plt.hist(word_count_sus, bins='auto')
plt.title("Histogram of Suspect Document Lengths")

plt.subplot(212)
plt.hist(word_count_nak, bins='auto')
plt.title("Nakamoto's Document Lengths")
plt.show()

# Add bitcoin to stop_words
my_stop_words = text.ENGLISH_STOP_WORDS.union(['bitcoin'])

# Define vectorizer
vectorizer = TfidfVectorizer(analyzer='word',
                             lowercase=True,
                             ngram_range=(1,1),
                             stop_words=my_stop_words,
                             use_idf=True,
                             norm='l1',
                             tokenizer=None)

# Define classifier pipelines
# Note that OneVsRestClassifier() is called in helper function 
# so does not appear in the pipeline.
bayes_pipe = Pipeline([('vect', vectorizer),
                      ('clf', MultinomialNB()),
                      ])

sgd_pipe = Pipeline([('vect', vectorizer),
                    ('clf', SGDClassifier(loss='squared_hinge')),
                    ])

svc_pipe = Pipeline([('vect', vectorizer),
                    ('clf', LinearSVC()),
                    ])

rforest_pipe = Pipeline([('vect', vectorizer),
                        ('clf', RandomForestClassifier()),
                        ])

perc_pipe = Pipeline([('vect', vectorizer),
                     ('clf', Perceptron()),
                     ])

# List of pipelines and pipeline names to iterate through
pipes = [bayes_pipe, sgd_pipe, svc_pipe, rforest_pipe, perc_pipe]
pipe_names = ['Naive Bayes', 'SGD', 'Linear SVC', 'Random Forest',
             'Perceptron']

# Call custom function for traning and testing
clf_comp(pipes, df_suspects['text'], df_suspects['author_id'],
        pipe_names, 5, True)

pred_pipe = Pipeline([('vect', vectorizer),
                     ('clf', OneVsRestClassifier(Perceptron()))
                     ])

# Binarize outputs
mlb = MultiLabelBinarizer()
y_bin = mlb.fit_transform(df_suspects['author_id'])
pred_pipe.fit(df_suspects['text'], y_bin)
y_predict = pred_pipe.predict(df_nakamoto['text'])

pred_class = np.argmax(y_predict, axis=1)
pred_df = pd.DataFrame({'author': names,
                       'count': np.bincount(pred_class)})
pred_df['percentage'] = (pred_df['count'] / 
                         pred_df['count'].sum() * 100).round(1)

print(pred_df)