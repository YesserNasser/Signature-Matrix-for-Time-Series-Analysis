# -*- coding: utf-8 -*-
"""
Updated on Sun Sep 1 19:35:07 2019
@author: Yesser H. Nasser

'''==================== the work still on progress ======================== '''
"""

import pandas as pd
from pandas_datareader import data as web 
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import bs4 as bs
import pickle
import requests
import datetime as dt

import keras
from keras.layers import Conv2D, ConvLSTM2D, Conv3D, Deconv2D, Dense, Activation, BatchNormalization, Dropout, Input
from keras.models import Model

'''========================================================================='''
'''=========================== collecting symbols =========================='''
'''============================ for time series ============================'''
'''========================================================================='''

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        if (ticker != 'BRK.B'and ticker!= 'BF.B' and ticker!= 'CTVA'):
            tickers.append(ticker)
    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers,f)   
    print(tickers)  
    return tickers
tickers = save_sp500_tickers()

symbols = tickers[:30]
# extract the size of symbols (number of time series) 
# used to condition the dimension of seq_corr the signature matrix
n = np.array(symbols).shape[0]

start = dt.datetime(2000,1,3)
end = dt.datetime(2019,9,1)

volume= []
closes = []
for symbol in symbols:
    print (symbol)
    vdata = web.DataReader(symbol, 'yahoo', start, end)
    cdata= vdata[['Close']]
    closes.append(cdata)
    vdata = vdata[['Volume']]
    volume.append(vdata)
volume = pd.concat(volume, axis=1).dropna()
volume.columns = symbols
closes = pd.concat(closes, axis=1).dropna()
closes.columns = symbols

print(volume.head())
print(closes.head())

volume.plot(figsize=(12,6))
plt.show()

# building charcaterization status with signature matrics
ws = [5,10,15]
step = 1

# extract shape of ws to be used to condition the dimension of seq_corr 
'''================== creating signature matrix ======================= '''
'''========================================================================='''
s = np.array(ws).shape[0]

sequential_data = []
n_seq = int(len(volume)/min(ws)) # max number of sequence given min w
for i in range(0,len(volume)-max(ws),step):
    seq_corr = []
    for w in ws:
        if i+w<len(volume):
            corr_w = np.array((volume[i:i+w].corr())/w)
            if len(seq_corr)==0:
                seq_corr = corr_w
            else:
                seq_corr = np.dstack((seq_corr,corr_w))
    if np.array(seq_corr).shape == (n,n,s):
        sequential_data.append(seq_corr)
Signature_Matrix = np.array(sequential_data).reshape(-1,n,n,s)

# ========= tune number of sequences for reshaping the data later for 
h = 5
Signature_Matrix = Signature_Matrix[0:(len(Signature_Matrix)-len(Signature_Matrix)%h)]
# ================== reshape signature matrix for ConvLSTM2D ===================================

Signature_Matrix_convlstm2d = np.array(Signature_Matrix).reshape(-1,h,n,n,s)

#=============================== test =========================================
import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization, LSTM, Input, Conv2D, MaxPool2D
from attention_wrapper import TemporalPatternAttentionCellWrapper
from keras.models import Sequential, Model
from attention_decoder import AttentionDecoder
from AttentionWithContext import AttentionWithContext
from attention import Attention
from AttentionConvolutionalLSTM import AttenXConvLSTM2D
# =============================================================================

input_layer = Input(shape=Signature_Matrix_convlstm2d.shape[1:])
x = Conv3D(32, (3,3,3), strides=(1,1,1), name='Conv1')(input_layer)
x = AttenXConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=keras.regularizers.l2(l=0.01), recurrent_regularizer=keras.regularizers.l2(l=0.01),
                      return_sequences=True, name='axclstm2d_1')(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)

x = Dense(32, activation = 'relu')(x)
x = Dense(3, activation='softmax', name='Final_output_layer')(x)
model = Model(input_layer, x, name='model0')

model.summary()


# define the 4 Convolutional operators Conv1-Conv4
# filter size is (3,3,3)
def Conv1(x):
    x = Conv2D(32, (3,3,3), stride=(1,1), input_shape =(len(Signature_Matrix),n,n,s), name='Conv1')(x)
    x = Activation('SELU')(x)
    x = BatchNormalization(x)
    return x
def Conv2(x):
    x = Conv2D(64, (3,3,32), stride=(2,2), name='Conv2')(x)
    x = Activation('SELU')(x)
    x = BatchNormalization(x)
    return x
def Conv3(x):
    x = Conv2D(128,(2,2,64), stride=(2,2), name='Conv3')(x)
    x = Activation('SELU')(x)
    x = BatchNormalization(x)
    return x
def Conv4(x):
    x = Conv2D(256,(2,2,128), stride=(2,2), name='Conv4')(x)
    x = Activation('SELU')(x)
    x = BatchNormalization(x)
    return x

# building ConvLSTM
'''
ConvLSTM is a recurrent layer, just like LSTM but intrenal matrix multiplications are exchanged 
with convolution operations. As a result the data that flows through the ConvLSTM cells keeps
the input dimension (3D in our case).
the ConvLSTM layer input is a set of images as a 5D tensor with shape (samples, time_steps, rows, cols, channels)

the ConvLSTM layer output is a combination of a convolution and a LSTM output, Just like LSTM
if return_sequences = True, then it returns a sequence as a 5D tensor with shape (samples, time_steps, rows, columns, filters)
if return_sequences=false returns 4D tensor (samples, rows, columns, filters).

parameters:
    filters : number of output filters in convolution
    kernel_size : specifing the height and width of the convolution window.
    padding : valid, same
    data format : 'channels_first' or 'channels_last'
    activation function
    reccurent activation : activation function 'hard_sigmoid'
    retun sequences: True or False
'''
'''
if data_format='channels_last'
5D tensor with shape: (samples, time, rows, cols, channels)
in our case:
    samples = len(signature_matrix)
    time = 5 (We tune step length h (the number of previous segments) and set 
    it as 5 due to the best empirical performance.
    rows = n
    cols = n
    channels = 

'''
def ConvLSTM2D_1(x):
    x = ConvLSTM2D(filters=32, 
                   kernel_size=(3,3,3), 
                   data_format='channels_last', 
                   recuurent_activation='hard_sigmoid', 
                   activation='tanh', 
                   padding = 'same', 
                   return_sequences=False,
                   name='ConvLSTM1')(x)
    x = BatchNormalization(x)
    return x
def ConvLSTM2D_2(x):
    x = ConvLSTM2D(filters=64, 
                   kernel_size=(3,3,32), 
                   data_format='channels_last', 
                   recuurent_activation='hard_sigmoid', 
                   activation='tanh', 
                   padding = 'same', 
                   return_sequences=False,
                   name='ConvLSTM2')(x)
    x = BatchNormalization(x)
    return x
def ConvLSTM2D_3(x):
    x = ConvLSTM2D(filters=128, 
                   kernel_size=(2,2,64), 
                   data_format='channels_last', 
                   recuurent_activation='hard_sigmoid', 
                   activation='tanh', 
                   padding = 'same', 
                   return_sequences=False,
                   name='ConvLSTM3')(x)
    x = BatchNormalization(x)
    return x
def ConvLSTM2D_4(x):
    x = ConvLSTM2D(filters=128, 
                   kernel_size=(2,2,128), 
                   data_format='channels_last', 
                   recuurent_activation='hard_sigmoid', 
                   activation='tanh', 
                   padding = 'same', 
                   return_sequences=False,
                   name='ConvLSTM4')(x)
    x = BatchNormalization(x)
    return x
# building Decoder
def Deconv_1(x):
    x = Deconv2D(128, (2,2,256), stride=(2,2), name='Deconv1')(x)
    return x
def Deconv_2(x):
    x = Deconv2D(64, (2,2,128), stride=(2,2), name='Deconv2')(x)
    return x
def Deconv_3(x):
    x = Deconv2D(32, (3,3,64), stride=(2,2), name='Deconv3')(x)
    return x
def Deconv_4(x):
    x = Deconv2D(3, (3,3,64), stride=(1,1), name='Deconv4')(x)
    return x
''' =================== work still on progress ============================='''
''' ======================== ........... ==================================='''