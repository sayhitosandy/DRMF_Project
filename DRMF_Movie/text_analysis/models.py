'''
Created on June 20, 2017

@author: Hao Wu
'''
from keras.layers import Input, TimeDistributed
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.layers.core import Reshape, Flatten, Dropout, Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import Average, Concatenate
from keras.layers.recurrent import GRU
from keras.models import Model

import numpy as np


np.random.seed(1337)

# from keras.preprocessing.sequence import pad_sequences

class CNN_module():
    '''
    classdocs
    '''
    batch_size = 128
    epochs = 1

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W=None):

        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200

        self.filter_lengths = [3, 4, 5]

        '''Embedding Layer'''
        in_seq = Input(shape=(max_len,))
        if init_W is None:
            seq_emb = Embedding(max_features, emb_dim, input_length=max_len, trainable=True)(in_seq)
        else:
            seq_emb = Embedding(max_features, emb_dim, input_length=max_len, trainable=False, weights=[init_W / 20])(in_seq)

        '''Convolution Layer & Max Pooling Layer'''
        tmp_list = []
        for ws in self.filter_lengths:
#            cnn = Reshape((self.max_len, emb_dim, 1), input_shape=(self.max_len,))(seq_emb)
#            cnn = Convolution2D(nb_filters, ws, emb_dim, activation="relu")(cnn)
#            cnn = MaxPooling2D(pool_size=(self.max_len - ws + 1, 1))(cnn)
#            cnn = Flatten()(cnn)
            cnn = Convolution1D(nb_filters, ws, border_mode='valid', activation='relu', subsample_length=1)(seq_emb)
            cnn = MaxPooling1D(pool_length=self.max_len - ws + 1)(cnn)
            cnn = Flatten()(cnn)
            tmp_list.append(cnn)
        cnn_con = Concatenate()(tmp_list)
        
        '''Dropout Layer'''
        seq_dropout = Dense(vanila_dimension, activation='tanh')(cnn_con)
        seq_dropout = Dropout(dropout_rate)(seq_dropout)
        
        '''Projection Layer & Output Layer'''
        out_seq = Dense(output_dimesion, activation='tanh')(seq_dropout)

        # Output Layer
        self.model = Model(in_seq, out_seq)
        self.model.compile(optimizer='rmsprop', loss='mse')
        
        print (self.model.summary())

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def qualitative_CNN(self, vocab_size, emb_dim, max_len, nb_filters):
        self.max_len = max_len
        max_features = vocab_size

        self.filter_lengths = [3, 4, 5]
        print("Build model...")
        
        self.qual_conv_set = {}
        '''Embedding Layer'''
        input_sequence = Input(shape=(max_len,), dtype=int)
        seq_embedding = Embedding(max_features, emb_dim, input_length=max_len, weights=self.model.nodes['sentence_embeddings'].get_weights())(input_sequence)

        '''Convolution Layer & Max Pooling Layer'''
        seq_reshape = Reshape((max_len, emb_dim, 1), input_shape=(max_len, emb_dim))(seq_embedding)
        ws = 3 
        self.qual_conv_set[ws] = Convolution1D(nb_filters, ws, emb_dim, activation="relu", subsample_length=1, weights=self.model.nodes[
                                                  'unit_' + str(ws)].layers[1].get_weights())
        cnn_3 = self.qual_conv_set[ws](seq_reshape)
        cnn_3 = MaxPooling1D(pool_size=max_len - ws + 1)(cnn_3)
        output_3 = Flatten()(cnn_3)
        ws = 4
        self.qual_conv_set[ws] = Convolution1D(nb_filters, ws, emb_dim, activation="relu", subsample_length=1, weights=self.model.nodes[
                                                  'unit_' + str(ws)].layers[1].get_weights())
        cnn_4 = self.qual_conv_set[ws](seq_reshape)
        cnn_4 = MaxPooling1D(pool_size=max_len - ws + 1)(cnn_4)
        output_4 = Flatten()(cnn_4)
        
        ws = 5
        self.qual_conv_set[ws] = Convolution1D(nb_filters, ws, emb_dim, activation="relu", subsample_length=1, weights=self.model.nodes[
                                                  'unit_' + str(ws)].layers[1].get_weights())
        cnn_5 = self.qual_conv_set[ws](seq_reshape)
        cnn_5 = MaxPooling1D(pool_size=max_len - ws + 1)(cnn_5)
        output_5 = Flatten()(cnn_5)
        
        self.model = Model(input=input_sequence, outputs=[output_3, output_4, output_5])
        self.model.compile(optimizer='rmsprop', loss={'output_4': 'mse', 'output_4': 'mse', 'output_5': 'mse'})
        

    def train(self, X_train, V, weight_of_sample, seed):
        # X_train = pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        weight_of_sample = np.random.permutation(weight_of_sample)

        print("Train CNN_module...")
        history = self.model.fit(X_train, V,
                                 verbose=1, batch_size=self.batch_size, epochs=self.epochs, sample_weight=weight_of_sample)
        return history

    def get_projection_layer(self, X_train):
        # X_train = pad_sequences(X_train, maxlen=self.max_len)
        theta = self.model.predict(X_train, batch_size=self.batch_size)
        ## add my own np file
        return theta
    


class CNN_GRU_module():
    epochs = 1
    batch_size = 128
    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, gru_outdim, maxlen_doc, maxlen_sent, nb_filters, init_W=None):
        self.filter_lengths = [3, 4, 5]
        if init_W is not None:
            self.vocab_size, self.emb_dim = init_W.shape
        else:
            self.vocab_size = vocab_size
            self.emb_dim = emb_dim
        self.maxlen_doc = maxlen_doc
        self.maxlen_sent = maxlen_sent
        self.nb_filters = nb_filters
        self.gru_outdim = gru_outdim

        print ("Build model...")
        """Embedding Layers"""
        in_seq = Input(shape=(self.maxlen_doc * self.maxlen_sent,))
        if init_W is None:
            seq_emb = Embedding(self.vocab_size, self.emb_dim, trainable=True)(in_seq)
        else:
            seq_emb = Embedding(self.vocab_size, self.emb_dim, weights=[init_W / 20], trainable=False)(in_seq)
        seq_emb = Reshape((self.maxlen_doc, self.maxlen_sent, self.emb_dim, 1))(seq_emb)
        """CNN Layers"""
        tmp_list = []
        for ws in self.filter_lengths:
            cnn = TimeDistributed(Convolution2D(self.nb_filters, ws, emb_dim, border_mode='valid', activation='relu'))(seq_emb)
            cnn = TimeDistributed(MaxPooling2D(pool_size=(self.maxlen_sent - ws + 1, 1), border_mode='valid'))(cnn)
            cnn = TimeDistributed(Flatten())(cnn)
            cnn = TimeDistributed(Activation('tanh'))(cnn)
            tmp_list.append(cnn)
        cnn_con = Concatenate()(tmp_list)
        """GRNN Layers"""
        h_forward = GRU(self.gru_outdim) (cnn_con)
        h_backward = GRU(self.gru_outdim, go_backwards=True)(cnn_con)
        h = Concatenate()([h_forward, h_backward])
        """Output Layer"""
        seq_dropout = Dropout(dropout_rate)(h)
        out_seq = Dense(output_dimesion, activation='tanh')(seq_dropout)
        # build and compile model
        self.model = Model(in_seq, out_seq)
        self.model.compile(optimizer='rmsprop', loss='mse')
        print (self.model.summary())
        
    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def train(self, X_train, V, weight_of_sample, seed):
        np.random.seed(seed) 
        X_train = np.random.permutation(X_train)
        np.random.seed(seed) 
        V = np.random.permutation(V) 
        np.random.seed(seed)
        weight_of_sample = np.random.permutation(weight_of_sample)

        print("Train CNN_GRU_module...")
        history = self.model.fit(X_train, V,
                                 verbose=1, batch_size=self.batch_size, epochs=self.epochs, sample_weight=weight_of_sample)
        return history


    def get_projection_layer(self, X_train):
        theta = self.model.predict(X_train, batch_size=self.batch_size)
        return theta
    
    
    
#class GRU_ATTEN_module(): 
#to be continued
    
