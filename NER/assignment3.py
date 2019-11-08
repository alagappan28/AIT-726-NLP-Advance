'''
Author: Alagappan, Ganesh, Pranav, Tejasvi
Description: AIT 726 Homework #
Before running the code please set the respective file paths for the
Text file
    1. train.txt
    2. valid.txt
    3. text.txt
Embeddings
    GoogleNews-vectors-negative300.bin

Command to run the file: python assignment3.py


Flow of the program
---------------------
1. load the text files
2. split sentences and tags
3. Normalize the text(UC->LC if the whole word is not completely UC)
4. Vectorize the text
5. pad the text based on the size of the largest sentence in the text file
6. Apply embeddings on the vectorized text
7. Vectorize the tags
8. pad the tags based on the size of the largest sentence in the text file
9. Convert tags to one hot encodings
10. Finally applying 7 models on the preprocessed text file
11. Results are stored as text file which can be check by conlleval.py file


Results:
rnn.txt
processed 45905 tokens with 5477 phrases; found: 4117 phrases; correct: 2355.
accuracy:  44.59%; (non-O)
accuracy:  89.89%; precision:  57.20%; recall:  43.00%; FB1:  49.09
              LOC: precision:  61.97%; recall:  64.01%; FB1:  62.97  1688
             MISC: precision:  56.35%; recall:  16.42%; FB1:  25.43  197
              ORG: precision:  50.32%; recall:  34.25%; FB1:  40.76  1111
              PER: precision:  57.00%; recall:  41.63%; FB1:  48.12  1121

------------------------------------------------------------------------------
------------------------------------------------------------------------------
bi-rnn.txt
processed 46178 tokens with 5586 phrases; found: 4551 phrases; correct: 2839.
accuracy:  52.81%; (non-O)
accuracy:  91.15%; precision:  62.38%; recall:  50.82%; FB1:  56.01
              LOC: precision:  73.35%; recall:  67.81%; FB1:  70.47  1531
             MISC: precision:  65.53%; recall:  32.86%; FB1:  43.77  351
              ORG: precision:  53.74%; recall:  46.97%; FB1:  50.13  1442
              PER: precision:  57.95%; recall:  45.00%; FB1:  50.66  1227
------------------------------------------------------------------------------
------------------------------------------------------------------------------
lstm.txt
processed 46421 tokens with 5643 phrases; found: 5464 phrases; correct: 3280.
accuracy:  63.58%; (non-O)
accuracy:  91.37%; precision:  60.03%; recall:  58.13%; FB1:  59.06
              LOC: precision:  67.72%; recall:  68.53%; FB1:  68.12  1685
             MISC: precision:  44.25%; recall:  58.12%; FB1:  50.25  922
              ORG: precision:  50.85%; recall:  52.17%; FB1:  51.50  1703
              PER: precision:  74.96%; recall:  53.53%; FB1:  62.45  1154
------------------------------------------------------------------------------
------------------------------------------------------------------------------
bidir-lstm.txt
processed 46391 tokens with 5631 phrases; found: 5075 phrases; correct: 3477.
accuracy:  65.35%; (non-O)
accuracy:  92.57%; precision:  68.51%; recall:  61.75%; FB1:  64.95
              LOC: precision:  78.36%; recall:  77.18%; FB1:  77.77  1636
             MISC: precision:  71.79%; recall:  64.53%; FB1:  67.97  631
              ORG: precision:  58.97%; recall:  60.18%; FB1:  59.57  1694
              PER: precision:  66.70%; recall:  46.21%; FB1:  54.59  1114
------------------------------------------------------------------------------
------------------------------------------------------------------------------
gru.txt
processed 45983 tokens with 5452 phrases; found: 4559 phrases; correct: 2343.
accuracy:  42.55%; (non-O)
accuracy:  89.34%; precision:  51.39%; recall:  42.98%; FB1:  46.81
              LOC: precision:  59.52%; recall:  67.62%; FB1:  63.31  1870
             MISC: precision:  67.46%; recall:  36.96%; FB1:  47.75  378
              ORG: precision:  47.00%; recall:  28.20%; FB1:  35.25  932
              PER: precision:  38.94%; recall:  34.36%; FB1:  36.51  1379

------------------------------------------------------------------------------
------------------------------------------------------------------------------
bigru.txt
processed 46390 tokens with 5640 phrases; found: 5275 phrases; correct: 3533.
accuracy:  67.85%; (non-O)
accuracy:  92.65%; precision:  66.98%; recall:  62.64%; FB1:  64.74
              LOC: precision:  76.12%; recall:  72.33%; FB1:  74.18  1583
             MISC: precision:  59.29%; recall:  64.10%; FB1:  61.60  759
              ORG: precision:  58.76%; recall:  58.19%; FB1:  58.47  1644
              PER: precision:  70.75%; recall:  56.58%; FB1:  62.87  1289

-----------------------------------------------------------------------------------

'''

# -*- coding: utf-8 -*-

# import packages
from keras.preprocessing.sequence import pad_sequences
from gensim import models
from numpy import zeros
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation,TimeDistributed,InputLayer,Bidirectional,LSTM,GRU
from keras import optimizers
from keras.optimizers import Adam
from keras.layers import Embedding
from keras import backend as K
import numpy as np
from keras import backend as K
import time
start_time = time.time()

# list of tags
tag_list_prime=['<pad>','O','B-LOC', 'B-PER','B-ORG', 'B-MISC', 'I-LOC','I-PER','I-ORG', 'I-MISC']

# function to pad tags based on max len
def pad_tags(tags_list,max_len):
    #Pad the tags

    for i in range(0,len(tags_list)):
        tags_list[i] += ['<pad>']  * (max_len - len(tags_list[i]))

    return tags_list

# function to pad sentences based on max len
def pad_seq(line_list,max_len):
    #Pad each line
    sent_list = pad_sequences(line_list, maxlen=max_len,padding='post')
    return sent_list

# formating a word to lower case if the whole word is not in Upper case
def normalize_case(s):    
    '''
    Paramaeter: Word to be normalized
    Converts words with capitalized first letters in to lower case.
    '''
    if(not s.isupper()):
        return s.lower()
    else:
        return s

# function to read the input files
vocab=set([])
def openandread(path):
    #load the file and create list for sentences and tags
    tag_list=[]
    with open(path) as f:
        sent=[]
        tag=[]
        line_list=[]
        for line in f:
            content=line.split()

            if (line in ['\n', '\r\n']):
                if(len(sent)>0):
                    line_list.append(sent)
                    tag_list.append(tag)
                    sent = []
                    tag = []
            else:
                if content[0] != '-DOCSTART-':
                    token = normalize_case(content[0])
                    sent.append(token)
                    vocab.add(token)
                    tag.append(content[3])



    return line_list,tag_list

# vectorizing given text based on the vocabulary
def vectorize_line(line_list,words):
    #Convert word to index
    word2index = {w: i for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs
    train_list=[]
    for s in line_list:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
        train_list.append(s_int)
    # embeddings_index = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # line_vec = [[w[word] for word in line if word in w] for line in line_list]
    return train_list,word2index

# vectorizing given tags based on the available tag list
def vectorize_tag(tags,tag_list):
    #Convert tags to index
    tag2index = {t: i for i, t in enumerate(list(tags))}
    test_tags_y=[]
    for s in tag_list:
        test_tags_y.append([tag2index[t] for t in s])
    return test_tags_y

# Loading the embedding matrix
def embed(word2index):
    # Create embedding matrix using word2vec
    # embeddings_index = models.KeyedVectors.load_word2vec_format(r'D:\bin\AIT-726\Assignemnts\conll2003\GoogleNews-vectors-negative300.bin', binary=True)
    # please set your respective path for the embedding
    embeddings_index = models.KeyedVectors.load_word2vec_format(r'GoogleNews-vectors-negative300.bin', binary=True)
    embedding_matrix = np.zeros((len(word2index)+2, 300))
    embeddings_index['-DOCSTART-']=np.zeros(300)
    for word, i in word2index.items():
        if(word in embeddings_index):
            embedding_vector = embeddings_index[word]
        else:
            embedding_vector =np.zeros(300)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    return embedding_matrix


# function to convert tags into one hot encodings
def to_categorical(sequences, categories):
    #Create onehot encoding of y variable(tags)
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            x=np.zeros(categories)
            x[item] = 1.0
            cats.append(x)
        cat_sequences.append(cats)
    return np.array(cat_sequences)

# special accuracy metric used to evaluate the model, as we are padding our sentences and tags this affects
# normal accuracy metric (source :https://nlpforhackers.io/lstm-pos-tagger-keras/?fbclid=IwAR0mWM0udAkTfxZImkdS6tewPC0OzrCep6f1XzGgCbql4NPpAlzrUCLAN5U)
def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return ignore_accuracy

# Keras vanilla rnn model
def vanilla_rnn(max_len,embedding_matrix):
    #Create vannila rnn model
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len, )))
    model.add(Embedding(len(vocab), 300))
    model.add(SimpleRNN(256, return_sequences=True))
    model.add(TimeDistributed(Dense(10)))
    model.add(Activation('softmax'))
     
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.0001),
                  metrics=['accuracy',ignore_class_accuracy(0)])
    print(model.summary())
    return model

# Keras vanilla lstm model
def lstm_model(max_len,embedding_matrix):
    #Create vannila rnn model
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len, )))
    model.add(Embedding(len(vocab), 300))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(10)))
    model.add(Activation('softmax'))
     
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy',ignore_class_accuracy(0)])
    print(model.summary())
    return model

# Keras vanilla gru model
def gru_model(max_len,embedding_matrix):
    #Create vannila rnn model
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len, )))
    model.add(Embedding(len(vocab), 300))
    model.add(GRU(256, return_sequences=True))
    model.add(TimeDistributed(Dense(10)))
    model.add(Activation('softmax'))
     
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.0001),
                  metrics=['accuracy',ignore_class_accuracy(0)])
    print(model.summary())
    return model

# Keras vanilla bi-directional rnn model
def Bidirectional_rnn(max_len,embedding_matrix):
    #Create vannila rnn model
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len, )))
    model.add(Embedding(len(vocab), 300))
    model.add(Bidirectional(SimpleRNN(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(10)))
    model.add(Activation('softmax'))
     
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.0001),
                  metrics=['accuracy',ignore_class_accuracy(0)])
    print(model.summary())
    return model

# Keras vanilla bi-directional LSTM model
def Bidirectional_LSTM(max_len,embedding_matrix):
    #Create vannila rnn model
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len, )))
    model.add(Embedding(len(vocab), 300))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(10)))
    model.add(Activation('softmax'))
     
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy',ignore_class_accuracy(0)])
    print(model.summary())
    return model

# Keras vanilla bi-directional GRU model
def Bidirectional_GRU(max_len,embedding_matrix):
    #Create vannila rnn model
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len, )))
    model.add(Embedding(len(vocab), 300))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(10)))
    model.add(Activation('softmax'))
     
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.0001),
                  metrics=['accuracy',ignore_class_accuracy(0)])
    print(model.summary())
    return model


# Keras vanilla bi-directional GRU model
def bestModel(max_len, embedding_matrix):
    # Create vannila rnn model
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len,)))
    model.add(Embedding(len(vocab), 300,trainable=True))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(10)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.0001),
                  metrics=['accuracy', ignore_class_accuracy(0)])
    print(model.summary())
    return model
# function to map outputs to tags and store outputs as text file
def packAndStore(x,filename):
    y=[]
    for i in x:
        sub_y=[]
        for j in i:
            sub_y.append(tag_list_prime[np.argmax(j, axis=0)])
        y.append(sub_y)

    result=[]
    for i in range(0,len(line_list_test)):
        for j in range(0,len(line_list_test[i])):
            try:
                y[i][j]
                if(y[i][j]!='<pad>'):
                    result.append(line_list_test[i][j]+" "+tag_list_test[i][j]+" "+y[i][j])
            except IndexError:
                continue

    with open(filename, 'w') as f:
        for item in result:
            f.write("%s\n" % item)

if __name__ == "__main__":

    # loading train,test and val text files
    # please set your respective path
    line_list_train = []
    tag_list_train = []
    # line_list_train, tag_list_train = openandread(r'D:\bin\AIT-726\Assignemnts\conll2003\train.txt')
    line_list_train, tag_list_train = openandread(r'train.txt')

    line_list_valid = []
    tag_list_valid = []
    # line_list_valid, tag_list_valid = openandread(r'D:\bin\AIT-726\Assignemnts\conll2003\valid.txt')
    line_list_valid, tag_list_valid = openandread(r'valid.txt')

    line_list_test = []
    tag_list_test = []
    # line_list_test, tag_list_test = openandread(r'D:\bin\AIT-726\Assignemnts\conll2003\test.txt')
    line_list_test, tag_list_test = openandread(r'test.txt')

    # converting text to vector format and padding the text sequences
    max_len_train = len(max(line_list_train, key=len))
    line_vec_train,word2index=vectorize_line(line_list_train,vocab)
    line_vec_train = pad_seq(line_vec_train,max_len_train)

    line_vec_valid,word2index=vectorize_line(line_list_valid,vocab)
    line_vec_valid = pad_seq(line_vec_valid,max_len_train)

    line_vec_test,word2index=vectorize_line(line_list_test,vocab)
    line_vec_test = pad_seq(line_vec_test,max_len_train)

    # loading the pre-trained embeddings
    embedding_matrix=embed(word2index)

    # padding and vectorizing the tags
    padded_tag_train = pad_tags(tag_list_train,max_len_train)
    tag_vec_train=vectorize_tag(tag_list_prime,padded_tag_train)
    
    padded_tag_valid = pad_tags(tag_list_valid,max_len_train)
    tag_vec_valid=vectorize_tag(tag_list_prime,padded_tag_valid)
    
    padded_tag_test = pad_tags(tag_list_test,max_len_train)
    tag_vec_test=vectorize_tag(tag_list_prime,padded_tag_test)

    # converting tags to one hot encodings
    tag_vec_one_hot_train = to_categorical(tag_vec_train, 10)
    tag_vec_one_hot_valid = to_categorical(tag_vec_valid,10)
    tag_vec_one_hot_test = to_categorical(tag_vec_test,10)

    '''
    # RNN
    model=vanilla_rnn(max_len_train,embedding_matrix)
    model.fit(line_vec_train, tag_vec_one_hot_train, batch_size=150, epochs = 20,  validation_data=(line_vec_valid, tag_vec_one_hot_valid))
    x=model.predict(line_vec_test)
    packAndStore(x, r'output\rnn.txt')
        
    # bi-directional RNN
    model=Bidirectional_rnn(max_len_train,embedding_matrix)
    model.fit(line_vec_train, tag_vec_one_hot_train, epochs = 20, batch_size=150,  validation_data=(line_vec_valid, tag_vec_one_hot_valid))
    x=model.predict(line_vec_test)
    packAndStore(x,r'output\bi-rnn.txt')
   
    # LSTM
    model=lstm_model(max_len_train,embedding_matrix)
    model.fit(line_vec_train, tag_vec_one_hot_train, epochs = 5,  validation_data=(line_vec_valid, tag_vec_one_hot_valid))
    x=model.predict(line_vec_test)
    packAndStore(x,r'output\lstm.txt')
    
    # bi-directional LSTM
    model=Bidirectional_LSTM(max_len_train,embedding_matrix)
    model.fit(line_vec_train, tag_vec_one_hot_train, epochs = 10,  validation_data=(line_vec_valid, tag_vec_one_hot_valid))
    x=model.predict(line_vec_test)
    packAndStore(x,r'output\bi-lstm.txt')
    
    model=gru_model(max_len_train,embedding_matrix)
    model.fit(line_vec_train, tag_vec_one_hot_train, epochs = 20, batch_size=150, validation_data=(line_vec_valid, tag_vec_one_hot_valid))
    x=model.predict(line_vec_test)
    packAndStore(x,r'output\gru.txt')

    # bi-directional GRU
    model=Bidirectional_GRU(max_len_train,embedding_matrix)
    model.fit(line_vec_train, tag_vec_one_hot_train, epochs = 20,  validation_data=(line_vec_valid, tag_vec_one_hot_valid))
    x=model.predict(line_vec_test)
    packAndStore(x,r'output\bi-gru.txt')
    '''

    # bi-directional GRU
    model = bestModel(max_len_train, embedding_matrix)
    model.fit(line_vec_train, tag_vec_one_hot_train, epochs=20, validation_data=(line_vec_valid, tag_vec_one_hot_valid))
    x = model.predict(line_vec_test)
    packAndStore(x, r'output\bi-gru.txt')


    print("--- %s seconds ---" % (time.time() - start_time))


    

        