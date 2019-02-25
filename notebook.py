#%%
import os
import sys
os.chdir('python/QuoraQuestionPairs')

#%% [markdown]
## import
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

#%% [markdown]
## read file
#%%
df_train = pd.read_csv('./input/train.csv', encoding='utf-8')
df_test = pd.read_csv('./input/test.csv', encoding='utf-8').loc[0:2345795, :]
#%% [markdown]
## data check
df_train.head()

#%%
df_test.head()

#%%
# length of data
len(df_train)

#%%
len(df_test)

#%%
# the number of duplicate
sum(df_train['is_duplicate'])
#%% 
# the rate of duplicate
sum(df_train['is_duplicate'])/len(df_train)

#%% [markdown]
## check null
#%%
df_train.isnull().apply(sum)

#%%
df_test.isnull().apply(sum)

#%% [markdown]
# some null in question columns
# -> check test data with null

#%%
df_train[df_train['question1'].isnull() | df_train['question2'].isnull()]
#%%
df_test[df_test['question1'].isnull() | df_test['question2'].isnull()]

#%% [markdown]
# rows with null  -> duplicate = 0

#%%
# drop rows containing at least a null
df_train = df_train.dropna()

#%%
# check same word in a raws
df_train['q1_token'] = [nltk.word_tokenize(s) for s in df_train['question1']]
df_train['q2_token'] = [nltk.word_tokenize(s) for s in df_train['question2']]
df_train.head()

#%%
common_token = [sum([w2 in df_train.at[i, 'q1_token'] \
        for w2 in df_train.at[i, 'q2_token']]) for i in df_train.index]
df_train['common_token'] = common_token
df_train.head()

#%%
g = sns.FacetGrid(df_train, row="is_duplicate", )
g = g.map(sns.distplot, "common_token", kde=False, bins=20, norm_hist=True)

#%%
# remove stop word at common token counting
nltk.download('stopwords')
stop = stopwords.words('english')

#%%
common_token = [sum([w2 in df_train.at[i, 'q1_token'] and w2 not in stop \
        for w2 in df_train.at[i, 'q2_token']]) for i in df_train.index]
df_train['common_token'] = common_token

#%%
df_train.head()
#%%
g = sns.FacetGrid(df_train, row="is_duplicate", )
g = g.map(sns.distplot, "common_token", kde=False, bins=15, norm_hist=True)


#%% [markdown]
## tf-idf vectorize

#%%
vect = TfidfVectorizer()
vect.fit(pd.concat([df_train['question1'], df_train['question2']]))
tfidf_q1= vect.transform(df_train['question1'])
tfidf_q2 = vect.transform(df_train['question2'])

#%%
df_train['similarity_tfidf'] = \
        [np.dot(tfidf_q1[i], tfidf_q2[i].T).toarray()[0, 0] for i in range(tfidf_q1.shape[0])]

#%%
## Word Vectorise
# download the GloVe vectors from http://nlp.stanford.edu/data/glove.840B.300d.zip
# , and unzip
#%%
embeddings_index = {}
f = open("./model/glove.840B.300d.txt", encoding='utf-8')
for line in f:
    values = line.replace('\xa0', '').split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#%%
# creat vectors for the whole sentences 
def sent2vec(s):
    words = str(s).lower()
    words = nltk.word_tokenize(words)
    #words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


#%%\
wv_q1 = [sent2vec(x) for x in df_train['question1']]
wv_q2 = [sent2vec(x) for x in df_train['question2']]

#%%
df_train['similarity_w2v'] = [np.dot(x1, x2) for x1, x2, in zip(wv_q1, wv_q2)]

#%%

g = sns.FacetGrid(df_train, row="is_duplicate", )
g = g.map(sns.distplot, "similarity", kde=False, bins=20)

#%%
g = sns.FacetGrid(df_train, col="is_duplicate",)
g.map(sns.jointplot, "similarity_w2v", 'similarity_tfidf', kind='hex')

#%%
# calcurate product of q1 & q2 tfidf vector for each element
X_trian = tfidf_q1.multiply(tfidf_q2)
y = df_train['is_duplicate']


#%%
#params = {'C': [1, 10, 100]}
params = {"max_depth": [2, 3, None],
        "n_estimators":[10, 100] }
#model = LogisticRegression(random_state=0)
model = RandomForestClassifier(random_state=0)
clf = GridSearchCV(model, params, cv=4)
clf.fit(X_trian, y) 

#%%
print(clf.best_params_)
print(clf.best_score_)

#%%
# pre processing for test data
df_test[df_test['question1'].isnull() | df_test['question2'].isnull()]

#%% [markdown]
# drop rows containing at least a null
df_test_wona = df_test.dropna()
#%%
# tf-idf vectorise for test data
tfidf_q1_test= vect.transform(df_test_wona['question1'])
tfidf_q2_test = vect.transform(df_test_wona['question2'])
X_test = tfidf_q1_test.multiply(tfidf_q2_test)

#%%
y_pred = clf.predict_proba(X_test)

#%%
df_submit = pd.DataFrame({'test_id': df_test_wona['test_id'],
                        'is_duplicate': [y_pred[i][1] for i in range(len(y_pred))]})

#%%
df_submit_withna = pd.DataFrame({'test_id': df_test[df_test['question1'].isnull() | df_test['question2'].isnull()]['test_id'],
                                'is_duplicate': 0})

#%%
df_submit = df_submit.append(df_submit_withna, ignore_index=True)
#%%
df_submit.to_csv('./output/submission.csv', index=False)


#%%
