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


if __name__ == "__main__":
    print "Commencing..."
    data_dir = './data/'
    evt_name = 'Featurespace_events_output.csv'
    auth_name = 'Featurespace_auths_output.csv'
    db_name = 'c1_agg.db'
    
    disk_engine = create_engine('sqlite:///'+data_dir+db_name,convert_unicode=True)
    disk_engine.raw_connection().connection.text_factory = str

    table = 'auth'
    print trans_num_table(table,disk_engine,mode='train')
    # encoders = populate_encoders_scale(table,disk_engine)
    # print 'Populated!!'
	