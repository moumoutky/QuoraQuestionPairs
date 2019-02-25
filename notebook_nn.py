#%%
import os
os.chdir('python/QuoraQuestionPairs')
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, \
    GlobalAveragePooling1D, concatenate, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras import backend as K

#%%
df_train = pd.read_csv('./input/train.csv', encoding='utf-8')
df_test = pd.read_csv('./input/test.csv', encoding='utf-8').loc[0:2345795,:]

#%%
df_train = df_train.dropna()

#%%
# tokenize
num_words = 20000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(list(pd.concat([df_train['question1'], df_train['question2']])))

#%%
list_seq1_train = tokenizer.texts_to_sequences(df_train['question1'])
list_seq2_train = tokenizer.texts_to_sequences(df_train['question2'])

#%%
# check # of words in a question
num_words_per_question = [len(x) for x in list_seq1_train]
plt.hist(num_words_per_question)
num_words_per_question = [len(x) for x in list_seq2_train]
plt.hist(num_words_per_question)

#%%
# padding, maxlen=50くらい、kernelに従い60に
maxlen = 60
X_train1 = pad_sequences(list_seq1_train, maxlen=maxlen)
X_train2 = pad_sequences(list_seq2_train, maxlen=maxlen)

#%%
y = df_train['is_duplicate']

#%%
# Prepare GloVe embedding model (matrix)
embedding_dim=300
f = open(".//model/glove.840B.300d.txt", encoding='utf-8')

embedding_index = {}
for i, line in enumerate(f):
    try:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vec
    except ValueError:
        print(i, word, values[1], values[2])
f.close()

#%%
embedding_matrix = np.zeros((num_words, embedding_dim))
word_index = {v-1:k for k, v in tokenizer.word_index.items() if v <= num_words}
for i, word in word_index.items():
    if word in embedding_index.keys():
        embedding_matrix[i] = embedding_index[word]
embedding_matrix

#%%
# make model
def model_conv1D(embedding_matrix, maxlen):
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=False
    )

    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Define inputs
    seq1 = Input(shape=(maxlen,))
    seq2 = Input(shape=(maxlen,))

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(emb2)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    conv5b = conv5(emb2)
    glob5b = GlobalAveragePooling1D()(conv5b)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    conv6b = conv6(emb2)
    glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different measure of equalness
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

    '''
    # Add the magic features
    magic_input = Input(shape=(5,))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    # Add the distance features (these are now TFIDF (character and word), Fuzzy matching, 
    # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence vector)
    distance_input = Input(shape=(20,))
    distance_dense = BatchNormalization()(distance_input)
    distance_dense = Dense(128, activation='relu')(distance_dense)
    '''
    # Merge the Magic and distance features with the difference layer
    # merge = concatenate([diff, mul, magic_dense, distance_dense])
    merge = concatenate([diff, mul])

    # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model

#%%
# Learning
batch_size = 128
epochs = 10
model = model_conv1D(embedding_matrix, maxlen)
model.fit([X_train1, X_train2], y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

#%%
# Prediction
df_test_wona = df_test.dropna()

list_seq1_test = tokenizer.texts_to_sequences(df_test_wona['question1'])
list_seq2_test = tokenizer.texts_to_sequences(df_test_wona['question2'])

#%%
# check # of words in a question
num_words_per_question = [len(x) for x in list_seq1_test]
plt.hist(num_words_per_question)
num_words_per_question = [len(x) for x in list_seq2_test]
plt.hist(num_words_per_question)

#%%
X_test1 = pad_sequences(list_seq1_test, maxlen=maxlen)
X_test2 = pad_sequences(list_seq2_test, maxlen=maxlen)

#%%
y_pred = model.predict([X_test1, X_test2])

#%%
# make submission file
df_submit = pd.DataFrame({'test_id': df_test_wona['test_id'],
                        'is_duplicate': y_pred[:, 0]})

#%%
df_submit_withna = pd.DataFrame({'test_id': df_test[df_test['question1'].isnull() | df_test['question2'].isnull()]['test_id'],
                                'is_duplicate': 0})

#%%
df_submit = df_submit.append(df_submit_withna, ignore_index=True)

#%%
df_submit.to_csv('./output/submission_nn.csv', index=False)
