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
import logging
import plotly.plotly as py # interactive graphing
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Bar, Scatter, Marker, Layout 
from heraspy.model import HeraModel
np.random.seed(1337)
import theano
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,model_from_yaml
from keras.layers import Input, Dense, GRU, LSTM, TimeDistributed, Masking,merge
from model import *
import os


def get_engine(address = "postgresql+pg8000://script@localhost:5432/ccfd"):

    # disk_engine = create_engine('sqlite:///'+data_dir+db_name,convert_unicode=True)
    # disk_engine.raw_connection().connection.text_factory = str
    disk_engine = create_engine(address)
    return disk_engine

class ModelLoader:

    archs = ['/home/botty/Documents/CCFD/data/archs/bi_GRU_320_4_DO-0.3']
    w_path = '/home/botty/Documents/CCFD/data/models/data_little/'
    ws = [w_path+'Bidirectional_Class400.0_GRU_320_4_RMSprop_0.00025_epochs_10_DO-0.3.09-0.01.hdf5']

    def __init__(self, arch_path, w_path):
        self.arch_path = arch_path
        self.w_path = w_path
        self.model = load_model(arch_path, w_path)

class Evaluator(ModelOperator):
    'Evaluates models'
    def __init__(self, *args, **kwargs):
        ModelOperator.__init__(self, *args, **kwargs)
        # self.disk_engine = disk_engine
        # self.model = model
        # self.table = table
        # self.auc_list = []
        # self.encoders = load_encoders()
        # if 'events_tbl' in kwargs:
        #     self.events_tbl = kwargs['events_tbl']
        # else:
        #     self.events_tbl = None

    def evaluate_model(self,user_mode,trans_mode,title,add_info=''):
        disk_engine = self.disk_engine
        model = self.model
        table = self.table
        events_tbl = self.events_tbl
        auc_list = self.auc_list
        encoders = self.encoders
        batch_size = 5000
        #calcualte # samples
        val_samples = trans_num_table(table,disk_engine,mode=user_mode,trans_mode=trans_mode)
        print '# samples',val_samples

        plt_filename = './figures/GS/'+table+'/'+'ROC_'+user_mode+'_'+trans_mode+'_'+title+'_'+add_info+".png"

        eval_gen = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                         batch_size=batch_size,usr_ratio=80, class_weight=None,lbl_pad_val = 2, pad_val = -1,events_tbl=events_tbl)

        eval_list  = eval_auc_generator(model, eval_gen, val_samples, max_q_size=10000,plt_filename=plt_filename)
        auc_val = eval_list[0]
        clc_report = eval_list[1]
        acc = eval_list[2]
        print "AUC:", auc_val
        print 'Classification report'
        print clc_report
        print 'Accuracy'
        print acc
        auc_list.append(str(auc_val))

if __name__ == "__main__":

    table = 'data_little'
    disk_engine = get_engine()
    ml = ModelLoader(ModelLoader.archs[0], ModelLoader.ws[0])
    model = ml.model
    title = 'BiRNN-DO3'
    data_little_eval = Evaluator(model, table, disk_engine)

    options = ['train','test']
    for user_mode in options:
        for trans_mode in options:
            print '################## USER:{user_mode}--------TRANS:{trans_mode}###############'.format(user_mode=user_mode,trans_mode=trans_mode)
            data_little_eval.evaluate_model(user_mode,trans_mode,title)
            print '#########################################################################################################'