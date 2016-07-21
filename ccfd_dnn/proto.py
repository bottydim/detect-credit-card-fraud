import pandas as pd
import matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import plotly.tools as tls
import pandas as pd
from sqlalchemy import create_engine # database connection
import datetime as dt
import io

import plotly.plotly as py # interactive graphing
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Bar, Scatter, Marker, Layout 
from heraspy.model import HeraModel
np.random.seed(1337)
import theano
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, GRU, LSTM, TimeDistributed, Masking
from model import *


def compile_seq2seq_RNN(rnn = 'lstm', hidden_dim = 300, num_layers = 3, 
	lbl_pad_val = 2, pad_val = -1,
	optimizer = keras.optimizers.RMSprop(lr=1e-3, rho=0.9, epsilon=1e-08)):

	input_layer = Input(shape=(int(seq_len_param), 44),name='main_input')
	mask = Masking(mask_value=0)(input_layer)
	if rnn == 'gru':
	    prev = GRU(hidden_dim,#input_length=50,
	                        return_sequences=True,go_backwards=False,stateful=False,
	                        unroll=False,consume_less='gpu',
	                        init='glorot_uniform', inner_init='orthogonal', activation='tanh',
	                inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None,
	                b_regularizer=None, dropout_W=0.0, dropout_U=0.0)(mask)
	else:
	    prev = LSTM(hidden_dim, return_sequences=True,go_backwards=False,stateful=False,
	        init='glorot_uniform', inner_init='orthogonal', 
	        forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid',
	        W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)(mask)
	for i in range(num_layers-1):
	    if rnn == 'gru':
	        prev = GRU(hidden_dim,#input_length=50,
	                            return_sequences=True,go_backwards=False,stateful=False,
	                            unroll=False,consume_less='gpu',
	                            init='glorot_uniform', inner_init='orthogonal', activation='tanh',
	                    inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None,
	                    b_regularizer=None, dropout_W=0.0, dropout_U=0.0)(prev)
	    else:
	        prev = LSTM(hidden_dim, return_sequences=True,go_backwards=False,stateful=False,
	            init='glorot_uniform', inner_init='orthogonal', 
	            forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid',
	            W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)(prev)
	output_layer = TimeDistributed(Dense(3,activation='softmax'))(prev)
	model = Model(input=[input_layer],output=[output_layer])
	model.compile(optimizer=optimizer,
	                loss='sparse_categorical_crossentropy',
	                metrics=['accuracy'])
	return model


if __name__ == "__main__":
    print "Commencing..."
    data_dir = './data/'
    evt_name = 'Featurespace_events_output.csv'
    auth_name = 'Featurespace_auths_output.csv'
    db_name = 'c1_agg.db'
    
    disk_engine = create_engine('sqlite:///'+data_dir+db_name,convert_unicode=True)
    disk_engine.raw_connection().connection.text_factory = str

    table = 'data_trim'
    # user_gen = user_generator(disk_engine,table=table)
    # next(user_gen)
    encoders = populate_encoders_scale(table,disk_engine)
    print 'Populated!!'