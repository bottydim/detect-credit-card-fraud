import progressbar
import pandas as pd
import matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, accuracy_score,classification_report
import plotly.tools as tls
import plotly.exceptions
import pandas as pd
from sqlalchemy import create_engine # database connection
import datetime as dt
import time
import io
import cPickle as pickle
import plotly.plotly as py # interactive graphing
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Bar, Scatter, Marker, Layout, Figure
from heraspy.model import HeraModel
np.random.seed(1337)
import theano
import keras
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,model_from_yaml
from keras.layers import Input, Dense, GRU, LSTM, TimeDistributed, Masking
from keras.engine.training import *
from IPython.display import display
from utils import *
import sys


class ModelOperator(object):

    def __init__(self, model, table, disk_engine=None,**kwargs):
        if disk_engine == None:
            self.disk_engine = get_engine()
        else:
            self.disk_engine = disk_engine
        self.model = model
        self.table = table
        self.auc_list = []
        self.encoders = load_encoders()
        if 'events_tbl' in kwargs:
            self.events_tbl = kwargs['events_tbl']
        else:
            self.events_tbl = None


def save_object(path):
    with open(path, 'w') as data:
        obj = pickle.load(data)
        return obj

def load_object(path):
    with open(path, 'rb') as data:
        obj = pickle.load(data)
        return obj
def load_encoders(path_encoders=None):
    if path_encoders == None:
        tbl_src = 'auth'
        tbl_evnt = 'event'
        path_encoders ='./data/encoders/{tbl_src}+{tbl_evnt}'.format(tbl_src=tbl_src,tbl_evnt=tbl_evnt)    
    encoders = load_object(path_encoders)
    print 'encoders LOADED!'
    return encoders

def load_model(arch_path,w_path=None):

    yaml_string = load_object(arch_path)
    print 'Model Architecture LOADED!' 
    model = model_from_yaml(yaml_string)
    lr= 2.5e-4
    optimizer = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'],
                    sample_weight_mode="temporal")
    if w_path != None:
        model.load_weights(w_path)
    else:
        print '*************************IMPORTANT**************************'
        print '***********************Ws are not loaded - empty model'
    return model

time_cols = ['authzn_rqst_proc_tm']
time_cols = [x.lower() for x in time_cols]
date_cols = ['PREV_ADR_CHNG_DT','PREV_PMT_DT','PREV_CARD_RQST_DT','FRD_IND_SWT_DT']
date_cols = [x.lower() for x in date_cols]


def encode_column(df_col):
    print 'Total incoming types:',df_col.shape
    le = preprocessing.LabelEncoder()
    le.fit(df_col)
    print "Produced classes"
    print '#unique:', len(le.classes_)
    return le

def populate_encoders(table,disk_engine):
    df = pd.read_sql_query('select * from {table}'.format(table=table),disk_engine)
    df.head()
    encoders = {}
    for c,r in enumerate(df):
        tp = df.dtypes[c]
    #     print tp
        if tp == 'object':
            if r not in time_cols:
                encoders[r] = encode_column(df[r])
    return encoders

def populate_encoders_scale(table,disk_engine,events_tbl=None):
    df = pd.read_sql_query('select * from {table} limit 5'.format(table=table),disk_engine)
    col_names = df.columns.values
    encoders = {}
    # time_cols = ['AUTHZN_RQST_PROC_TM','PREV_ADR_CHNG_DT','PREV_PMT_DT','PREV_CARD_RQST_DT','frd_ind_swt_dt']
    for c,name in enumerate(col_names):
        tp = df.dtypes[c]
        # print tp

        if tp == 'object':
            # print 'ORIGINAL NAME:',name
            if name not in time_cols:
                print name
                df_cols = pd.read_sql_query('select distinct {col_name} from {table}'.format(col_name=name,table=table),disk_engine,chunksize=100000)
                arr = []
                progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                            progressbar.Percentage(), ' ',
                                            progressbar.ETA()]).start()
                for c,df_col in enumerate(df_cols): 
                    # arr = np.vstack((arr,np.array(df_col)))
                    arr.extend(np.array(df_col))
                    progress.update(c+1)
                if events_tbl != None:
                    df_cols = pd.read_sql_query('select distinct {col_name} from {table}'.format(col_name=name,table=events_tbl),disk_engine,chunksize=100000)
                    for c,df_col in enumerate(df_cols): 
                        arr.extend(np.array(df_col))
                        progress.update(c+1)
                progress.finish()
                arr = np.array(arr)
                encoders[name] = encode_column(np.array(arr).ravel())
    return encoders



def encode_df(df,encoders):
    for col in encoders.keys():
        try: 

            df[col] = encoders[col].transform(df[col])
        except:
            print 'EXCEPTION'
            print col 
            raise
    for col in time_cols:
        df[col] = pd.to_numeric(pd.to_datetime(df[col],errors='coerce'))
    for col in date_cols:
        df[col] = pd.to_numeric(pd.to_datetime(df[col],errors='coerce',format='%d%b%Y'))

def get_user_info(user,table,disk_engine):
    if user == '.':
        user = "'.'"
    df_u = pd.read_sql_query('select * from {table} where acct_id = {user}'.format(table=table,user=user),disk_engine)
    return df_u

def get_last_date(cutt_off_date,table,disk_engine):

    query = '''select authzn_rqst_proc_tm from {table} where frd_ind_swt_dt >='{cutt_off_date}' order by authzn_rqst_proc_tm limit 1'''
    

    # query = ''.join(query)
    query = query.format(table=table,cutt_off_date=cutt_off_date)
    dataFrame = pd.read_sql_query(query
                       .format(table=table), disk_engine)
    print query
    print 'last date'
    print pd.to_numeric(pd.to_datetime(dataFrame['authzn_rqst_proc_tm']))
    return pd.to_numeric(pd.to_datetime(dataFrame['authzn_rqst_proc_tm']))[0]


def get_col_id(col,df):
    col_list = list(df.columns.values)
    col_list.remove('index')
    col_list.index(col)


def generate_sequence(user,table,encoders,disk_engine,lbl_pad_val,pad_val,last_date,events_tbl=None,discard_id = -2):
    t0 = time.time()
    df_u = get_user_info(user,table,disk_engine)
    t1 = time.time()
    # print 'time user info:', str(t1-t0)  

    if events_tbl:
        df_u_events = get_user_info(user,events_tbl,disk_engine)
        df_u = pd.concat([df_u,df_u_events])
        ####possible issues with index 
        print df_u
        # sys.exit()
    '''
    serious bug: ordering has to occur before time shift! otherwise the shift occurs in random fashion
    '''
    t0 = time.time()
    df_u = df_u.sort_values('authzn_rqst_proc_tm',ascending=True)
    t1 = time.time()
    # print 'time sort:', str(t1-t0)  
#     display(df_u[df_u['frd_ind_swt_dt'].isnull()])
    df_u = df_u.drop('index', axis=1)
    #columns which are not available at this time step
    unav_cols = ['AUTHZN_APPRL_CD','TSYS_DCLN_REAS_CD','AUTHZN_RESPNS_CD','AUTHZN_APPRD_AMT',]
    unav_cols = [x.lower() for x in unav_cols]
    nan_rpl = ['AUTHZN_APPRL_CD',]
    nan_rpl = [x.lower() for x in nan_rpl]

    # print 'null values prior ',df_u.isnull().sum()
    for col in unav_cols:
        df_u[col] = df_u[col].shift(1)
        loc = list(df_u.columns.values).index(col)
        # if(col in nan_rpl):
        #     df_u.iloc[0,loc] = 'nan'
        # else:
        df_u.iloc[0,loc] = pad_val
    # print 'null values posterior',df_u.isnull().sum()
#     print df_u.count()
#     display(df_u.head())
#     display(df_u.sort_values('authzn_rqst_proc_tm',ascending=True))


##########REMOVE ENCODING######################
    # t0 = time.time()
    # encode_df(df_u,encoders)
    # t1 = time.time()
    # print 'time encoding:', str(t1-t0)  
#############################################




#     print df_u.count()
#     display(df_u.head())
#     display(df_u.sort_values('authzn_rqst_proc_tm',ascending=True))

#     display(df_u[df_u['frd_ind_swt_dt'] < pd.to_numeric(pd.Series(pd.to_datetime(cuttoff_date)))[0]].head(8))
### This is the last date, before which transaction will be used for trainning. 
### It coresponds to the date when the last knwon fraudulent transaction was confirmed
    t0 = time.time()
    last_date_num = last_date

    df_train = df_u[df_u['authzn_rqst_proc_tm'] < last_date_num]
    # display(df_train.head())
    df_test = df_u[df_u['authzn_rqst_proc_tm'] >= last_date_num]
    # display(df_test.head())
    train = np.array(df_train)
    test = np.array(df_test)

    if df_train.empty:
        return [],test[:,0:discard_id],[],test[:,discard_id]
    if df_test.empty:
        # print 'train/test sequence split:',np.array(df_train).shape[0],np.array(df_test).shape[0]
        return train[:,0:discard_id],[],train[:,discard_id],[]
#     display(df_train)
#     display(df_test)

    t1 = time.time()
    # print 'time remaining:', str(t1-t0)      

    # print 'test shape in sequencer',test.shape
    return train[:,0:discard_id],test[:,0:discard_id],train[:,discard_id],test[:,discard_id]

def chunck_seq(seq_list,seq_len=seq_len_param):
    split_seq = map(lambda x: np.array_split(x,math.ceil(len(x)/seq_len)) if len(x)>seq_len else [x],seq_list)
    flattened = [sequence for user_seq in split_seq for sequence in user_seq]
    assert sum(map(lambda x: len(x),flattened)) == sum(map(lambda x: len(x),seq_list))
    chunks_lens = map(lambda x: len(x),flattened)
    for cnk in chunks_lens:
        assert cnk <= seq_len_param, 'Sequence chunks are exceeding the max_len of {} \n {}'.format(seq_len_param,chunks_lens)
    return flattened

def generate_sample_w(y_true,class_weight):
    shps = y_true.shape
    sample_w = []
    for i in range(shps[0]):
        sample_w.append([])
        for j in range(shps[1]):
            sample_w[i].append(class_weight[y_true[i,j,0]])
    return np.asarray(sample_w)
def sequence_generator(users,encoders,disk_engine,lbl_pad_val,pad_val,last_date,mode='train',table='data_trim',class_weight=None,events_tbl=None,discard_id = -2):
    X_train_S = []
    X_test_S = []
    y_train_S =[]
    y_test_S = []
    print "Number of users:",len(users)
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',
                            progressbar.Percentage(), ' ',
                            progressbar.ETA()],maxval=len(users)).start()
    t0 = time.time()
    c = 0
    for user in users:
    #     if user != '337018623': 
    #         continue
        X_train,X_test,y_train,y_test = generate_sequence(user,table,encoders,disk_engine,lbl_pad_val,pad_val,last_date,events_tbl,discard_id=discard_id)
        if type(X_train) != list:
            assert X_train.shape[0] == y_train.shape[0], 'Sequence mismatch for user {user}: X_Train.shape {x_shape}'\
                    ' : y_train.shape {y_shape} \n'.format(user=user,x_shape=X_train.shape,y_shape=y_train.shape)
        c+=1
        progress.update(c)
       
        # print 'shapes:',X_train.shape[0],":",y_train.shape[0]
        # if X_test != []:
        #     print 'shape in generator',X_test.shape
        X_train_S.append(X_train)
        X_test_S.append(X_test) 
        y_train_S.append(y_train)
        y_test_S.append(y_test)
    #     break

    t1 = time.time()
    progress.finish()
    print 'sequence acquired in:', str(t1-t0)   
    X_train_S = filter(lambda a: len(a) != 0, X_train_S)
    y_train_S = filter(lambda a: len(a) != 0, y_train_S)
    X_test_S = filter(lambda a: len(a) != 0, X_test_S)
    y_test_S = filter(lambda a: len(a) != 0, y_test_S)

    # ds_list = [X_train_S,y_train_S,X_test_S,y_test_S]
    # for x in ds_list:
    #     if len(x) ==

    if mode =='train':
        # chuncked = chunck_seq(X_train_S)
        # assert 
        X_train_pad = keras.preprocessing.sequence.pad_sequences(chunck_seq(X_train_S), maxlen=int(seq_len_param),dtype='float32',value=pad_val)
        y_train_S = keras.preprocessing.sequence.pad_sequences(np.array(chunck_seq(y_train_S)), maxlen=int(seq_len_param),dtype='float32',value=lbl_pad_val)
        y_train_S = np.expand_dims(y_train_S, -1)
        # print 'xs shape',X_train_pad.shape
        # print 'labels shape',y_train_S.shape
        if class_weight != None:

            sample_w = generate_sample_w(y_train_S,class_weight)
            return X_train_pad,y_train_S,sample_w
#         print y_train_S
#         print y_train_S.shape
#         y_train_S = to_categorical(y_train_S,3)
        return X_train_pad,y_train_S
    else:
        X_test_S_pad = keras.preprocessing.sequence.pad_sequences(chunck_seq(X_test_S), maxlen=int(seq_len_param),dtype='float32',value=pad_val)
        y_test_S = keras.preprocessing.sequence.pad_sequences(np.array(chunck_seq(y_test_S)),maxlen=int(seq_len_param),dtype='float32',value=lbl_pad_val)
        y_test_S = np.expand_dims(y_test_S, -1)
        print 'labels shape',y_test_S.shape
        if class_weight != None:
            sample_w = generate_sample_w(y_test_S,class_weight)
            return X_test_S_pad,y_test_S,sample_w
        return X_test_S_pad,y_test_S


def get_count_table(table,disk_engine,cutt_off_date,trans_mode):
    query = ['select acct_id,count(*) '
        'as num_trans from {table} '
        'where authzn_rqst_proc_tm <= '
        '(select authzn_rqst_proc_tm '
        'from {table} '
        'where frd_ind_swt_dt >=' 
             ''' ' ''',
        str(cutt_off_date),
             ''' ' '''
        'order by authzn_rqst_proc_tm limit 1) '
        'group by acct_id order by num_trans']
    query = ''.join(query)
    query = query.format(table=table)
    if trans_mode == 'test':
        query = query.replace('<=','>')
    dataFrame = pd.read_sql_query(query
                       .format(table=table), disk_engine)
    return dataFrame

def trans_num_table(table,disk_engine,mode='train',cutt_off_date='2014-05-11',trans_mode='train'):
    cutt_off_date = pd.to_numeric(pd.to_datetime(pd.Series([cutt_off_date])))[0]
    last_date = get_last_date(cutt_off_date,table,disk_engine)
    print 'last_date calculated!'
    dataFrame = get_count_table(table,disk_engine,last_date,trans_mode)
    u_list = set(dataFrame.acct_id)
    
    user_tr,user_ts = train_test_split(list(u_list), test_size=0.33, random_state=42)

    total_t =0
    if mode == 'train':
        users = user_tr
    else:
        users = user_ts
    print '#users in sequence counting:',len(users)
    total_t = total_trans_batch(users,dataFrame)
    return math.ceil(total_t)


def total_trans_batch(users,dataFrame_count):
    num_trans = 0
    users = set(users)
    for user in users:
        num_trans+=get_num_trans(user,dataFrame_count)
    return num_trans

def get_num_trans(user,dfc):
    try:
        df = dfc[dfc['acct_id']==user]
        if df.empty:
            print " user not existing in the table",user
            seq_len = 0
        else:
            seq_len = dfc[dfc['acct_id']==user].values[0][1]
    except:
        print 'EXCEPTION IN SEQUENCE COUNTING'
        print dfc[dfc['acct_id']==user]
        raise
    return math.ceil(1.0*seq_len/seq_len_param)

def add_user(index,u_list,dataFrame_count,users):
    cnt_trans = 0
    user = u_list[index]
    if user not in users:
        users.add(user)
        return get_num_trans(user,dataFrame_count)
    else:
        return 0
def user_generator(disk_engine,table='data_trim',batch_size=50,usr_ratio=80,
                   mode='train',cutt_off_date='2014-05-11',trans_mode='train',sub_sample=None):

    
    dataFrame_count = get_count_table(table,disk_engine,cutt_off_date,trans_mode)
    
#     display(dataFrame_count.head(5)) 
    print "User List acquired"
    u_list = list(dataFrame_count.acct_id)
#     u_list.extend(list(dataFrame_Y.acct_id))
    print 'total # users:',len(u_list)
    u_set = set(u_list)
    print 'e.g. users', u_list[0:10]
    print 'total # unique users:',len(u_set) 
    user_tr,user_ts = train_test_split(list(u_set), test_size=0.33, random_state=42)
    print 'total # sequences:',total_trans_batch(list(u_set),dataFrame_count)
    if mode == 'train':
        u_list =  user_tr
    else:
        u_list =  user_ts
    if trans_mode == 'test':
        print 'used # sequences: value is inaccurate, please implement'
    print 'used # sequences:',total_trans_batch(u_list,dataFrame_count)                         
#     display(dataFrame.acct_id)
    
    u_list = list(set(u_list))
    print 'return set cardinality:',len(u_list)
    cnt = 0
    head = 0
    tail = len(u_list)-1
    u_list_all = u_list
    batch_size_temp = batch_size
     

    while True:
        users = set()
        cnt_trans = 0
        if sub_sample != None:
            print 'SUBSAMPLING'
            assert sub_sample<=len(u_list_all), 'sub_sample size select is {sub_sample}, but there are only {us} users'.format(sub_sample=sub_sample,us=len(u_list_all))
            u_list = np.random.choice(u_list_all, sub_sample,replace=False)
            print 'indeed they have been generated'
            ### reset tail value, to avoid outof bounds exception
            tail = len(u_list)-1
            #####if using subsample the batch should be no larger than the total number of sequences
            to_be_used = total_trans_batch(u_list,dataFrame_count)  
            print 'batch_size: {bs} : to_be_used {tbu} : user_sub_sample {usr}'.format(bs=batch_size,tbu=to_be_used,usr=len(u_list))
            
            if batch_size_temp > to_be_used:
                batch_size = to_be_used
            else:
                batch_size = batch_size_temp
        #cover all users before subsampling again! 
        while head != tail+1:        
            while cnt_trans<batch_size:
                
                if cnt<usr_ratio:
                    cnt_trans+=add_user(head,u_list,dataFrame_count,users)
                    cnt+=1
                    head+=1

                else:
                    cnt_trans+=add_user(tail,u_list,dataFrame_count,users)
                    tail-=1
                    cnt=0
    #             print 'head',head
    #             print 'tail',tail
    #             print 'cnt_trans',cnt_trans
                if head == tail+1:
                        head = 0
                        tail = len(u_list)-1
                        cnt_trans = 0
                        cnt = 0
                        #if you have go through all users - return in order not to overfill epoch
                        #the same logic could have been achieved with break and without the yield line
                        print "##########ALL COVERED##########"
                        # yield users
                        # users = set()
                        break
                        
    #                     print len(users)
    #         print head
    #         print tail
            # print 'return list length:',len(users)
    #         print '# users expiriencing both', len(u_list)-len(users)

###############
            #sOtherwise there is an infinite loop with the same users all the time - head and tail cannot be changed! 
            ####
            print 'USERGEN: user set sequence length: ',cnt_trans
            print 'USERGEN: user set cardinality: ',len(users)
            cnt_trans = 0
            yield users
            users = set()
# def eval_trans_generator(disk_engine,encoders,table='data_trim',batch_size=512,usr_ratio=80,class_weight=None,lbl_pad_val = 2, pad_val = -1):
#     user_gen = user_generator(disk_engine,usr_ratio=usr_ratio,batch_size=batch_size,table=table)
#     print "Users generator"
#     while True:
#         users = next(user_gen)
#         yield sequence_generator(users,encoders,disk_engine,lbl_pad_val,pad_val,mode='test',table=table,class_weight=class_weight)

# def eval_users_generator(disk_engine,encoders,table='data_trim',batch_size=512,usr_ratio=80,class_weight=None,lbl_pad_val = 2, pad_val = -1):
#     user_gen = user_generator(disk_engine,usr_ratio=usr_ratio,batch_size=batch_size,table=table,mode='test')
#     print "Users generator"
#     while True:
#         users = next(user_gen)
#         yield sequence_generator(users,encoders,disk_engine,lbl_pad_val,pad_val,mode='train',table=table,class_weight=class_weight)   


def data_generator(user_mode,trans_mode,disk_engine,encoders,table,
                   batch_size=512,usr_ratio=80,class_weight=None,lbl_pad_val = 2,
                   pad_val = -1,cutt_off_date='2014-05-11',sub_sample=None,epoch_size=None,events_tbl=None,discard_id = -2):
    '''

    :param user_mode:
    :param trans_mode:
    :param disk_engine:
    :param encoders:
    :param table:
    :param batch_size:
    :param usr_ratio:
    :param class_weight:
    :param lbl_pad_val:
    :param pad_val:
    :param cutt_off_date:
    :param sub_sample:
    :param epoch_size:
    :param events_tbl:
    :param discard_id:  the number of features @ end of the table to truncate
    :return:
    '''
    cutt_off_date = pd.to_numeric(pd.to_datetime(pd.Series([cutt_off_date])))[0]
    last_date = get_last_date(cutt_off_date,table,disk_engine)
    print 'last_date calculated!'
    user_gen = user_generator(disk_engine,usr_ratio=usr_ratio,batch_size=batch_size,table=table,mode=user_mode,trans_mode=trans_mode,sub_sample=sub_sample,cutt_off_date=last_date)
    print "Users generator"
    x_acc = []
    y_acc = []
    sample_w = []
    total_eg = 0
    while True:
        print 'new users'
        users = next(user_gen)
        outs = sequence_generator(users,encoders,disk_engine,lbl_pad_val,pad_val,last_date,mode=trans_mode,table=table,class_weight=class_weight,events_tbl=events_tbl,discard_id=discard_id)
        
        if not(epoch_size == None):
            while True:
                num_seq = outs[0].shape[0]
                print 'DATAGEN: num_Seq - != USERGEN due to chuncking',num_seq
               
                remain = epoch_size - (total_eg + num_seq)
                print '{remain} = {epoch_size} - ({total_eg}+{num_seq})'.format(remain=remain,epoch_size=epoch_size,total_eg=total_eg,num_seq=num_seq)   
                print 'remain',remain
                if remain >=0:
                    total_eg +=num_seq
                    yield outs
                else:
                    ### remain <0 => num_seq - remain
                    cutline = num_seq + remain
                    temp = []
                    for i in range(len(outs)):
                        temp.append(outs[i][0:cutline])
                    yield tuple(temp)
                    ####end of epoch!

                    total_eg = 0
                    temp = []
                    for i in range(len(outs)):
                        temp.append(outs[i][cutline:])
                    outs =  tuple(temp) 
                if remain >=0:
                    break
        else:    
            yield outs

def eval_auc_generator(model, generator, val_samples, max_q_size=10000,plt_filename=None,acc=True,tr_percent=0.0136):
    '''Generates predictions for the input samples from a data generator.
    The generator should return the same kind of data as accepted by
    `predict_on_batch`.

    # Arguments
        generator: generator yielding batches of input samples.
        val_samples: total number of samples to generate from `generator`
            before returning.
        max_q_size: maximum size for the generator queue

    # Returns
        Numpy array(s) of predictions.
    '''


    processed_samples = 0
    wait_time = 0.01
    all_outs = []
    all_y_r = []
    all_y_hat = []
    data_gen_queue, _stop = generator_queue(generator, max_q_size=max_q_size)

    while processed_samples < val_samples:
        generator_output = None
        while not _stop.is_set():
            if not data_gen_queue.empty():
                generator_output = data_gen_queue.get()
                break
            else:
                time.sleep(wait_time)

        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, y = generator_output
                sample_weight = None
            elif len(generator_output) == 3:
                x, y, sample_weight = generator_output
            else:
                _stop.set()
                raise Exception('output of generator should be a tuple '
                                '(x, y, sample_weight) '
                                'or (x, y). Found: ' + str(generator_output))
        else:
            _stop.set()
            raise Exception('output of generator should be a tuple '
                                '(x, y, sample_weight) '
                                'or (x, y). Found: ' + str(generator_output))

        try:
            if x.size != 0:
                y_hat = model.predict_on_batch(x)
                y_r = y.ravel()
                y_hat_r = y_hat[:,:,1].ravel()
                pad_ids = np.where(y_r!=2)
                all_y_r.extend(y_r[pad_ids])
                all_y_hat.extend(y_hat_r[pad_ids])
        except:
            _stop.set()
            raise
        nb_samples = x.shape[0]   

        processed_samples += nb_samples

    _stop.set()


    all_y_r = np.array(all_y_r,dtype=np.dtype(float))
    all_y_hat = np.array(all_y_hat,dtype=np.dtype(float))
    print all_y_r.shape
    print all_y_hat.shape
    print '# fraud transactions',all_y_r[np.where(all_y_r==1)].shape
    print '# total transactions', all_y_r.shape
    print '# total sequences',processed_samples
    #######ROC CURVE
    fpr,tpr,tresholds = roc_curve(all_y_r,all_y_hat)
    # print all_y_hat
    # print tresholds
    print 'False Positive Rates'
    print fpr
    print 'True Positive Rates'
    print tpr
    ####################DETERMINE CUTOFF TRESHOLD############
    tr_shape = tresholds.shape
    print 'Tresholds shape:', tr_shape
    if tr_shape[0]>3:
        cutt_off_tr = tresholds[3]
        id_tr, fpr_perc = min(enumerate(fpr), key=lambda x: abs(x[1]-tr_percent))
        print 'FPR[{}] = {}'.format(id_tr,fpr_perc)
        print 'TPR[{}] = {}'.format(id_tr,tpr[id_tr])
        cutt_off_tr = tresholds[id_tr]


    else:
        cutt_off_tr = tresholds[-1]



    auc_val = auc(fpr, tpr)
    print 'AUC:',auc_val
    ############CLASSIFICATION REPORT########################
    target_names = ['Genuine', 'Fraud']
    #########Need to determine treshold 
    all_y_hat[np.where(all_y_hat>=cutt_off_tr)] = 1
    all_y_hat[np.where(all_y_hat<cutt_off_tr)]  = 0
    # print '############hat#####################'
    # print all_y_hat
    # print '#############y_r####################'
    # print all_y_r
    clc_report = classification_report(all_y_r, all_y_hat, target_names=target_names,digits=7)
    ############Accuracy
    acc = accuracy_score(all_y_r,all_y_hat)
    if plt_filename != None and not np.isnan(auc_val):
        trace = Scatter(x=fpr,y=tpr)
        data = [trace]
        title = 'ROC'
        layout = Layout(title=title, width=800, height=640)
        fig = Figure(data=data, layout=layout)
        print plt_filename
        try:
            py.image.save_as(fig,filename=plt_filename)
        except Exception, e:
            print 'PLOTLY CRASH.......'
            for i in range(10):
                print >> sys.stderr, 'PLOTLY!'
            print e
            print type(e)
        
    return [auc_val,clc_report,acc]



def export_generator(models, generator, val_samples, max_q_size=1000,remove_pad=False):
    processed_samples = 0
    wait_time = 0.01
    all_outs = []
    all_y_r = []
    all_y_hat = []
    x_exp = []
    y_exp = []
    y_hat_dic = []
    for c,m in enumerate(models):
        y_hat_dic.append([])
    data_gen_queue, _stop = generator_queue(generator, max_q_size=max_q_size)

    while processed_samples < val_samples:
        generator_output = None
        while not _stop.is_set():
            if not data_gen_queue.empty():
                generator_output = data_gen_queue.get()
                break
            else:
                time.sleep(wait_time)

        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, y = generator_output
                sample_weight = None
            elif len(generator_output) == 3:
                x, y, sample_weight = generator_output
            else:
                _stop.set()
                raise Exception('output of generator should be a tuple '
                                '(x, y, sample_weight) '
                                'or (x, y). Found: ' + str(generator_output))
        else:
            _stop.set()
            raise Exception('output of generator should be a tuple '
                                '(x, y, sample_weight) '
                                'or (x, y). Found: ' + str(generator_output))

        try:
            if x.size != 0:
                print 'x',x.shape
                print 'y',y.shape
                x_re = x.reshape(-1, x.shape[-1])
                y_re = y.reshape(-1, y.shape[-1])
                if remove_pad:
                    pad_ids = np.where(y_re[:,0]!=2)[0]
                    x_re = x_re[pad_ids,:]
                    y_re = y_re[pad_ids,0]


                x_exp.extend(x_re)
                #ravel into 1D array 
                ### fails for MIO(Multi-Input-Output)
                # y_re = y_re.ravel()
                y_exp.extend(y_re)

                for c,model in enumerate(models):

                    y_hat = model.predict_on_batch(x)


                    ###BACKEND IMPLEMENTATION!!!!!#########
                    # get_output = K.function([model.layers[0].input,K.learning_phase()],
                    #               [model.layers[-1].output])
                    # #first 0 for test mode
                    # y_hat = get_output([x,0])[0]

                    y_hat_re = y_hat.reshape(-1, y_hat.shape[-1])
                    if remove_pad:
                        y_hat_re = y_hat_re[pad_ids,:]
                    y_hat_exp = y_hat_dic[c]
                    y_hat_exp.extend(y_hat_re)
                    print c,' y_hat',y_hat_re.shape



                    
                print 'x',x_re.shape
                print 'y',y_re.shape

                
        except:
            _stop.set()
            raise
        nb_samples = x.shape[0]
        print 'PROCESSED SAMPLES'   
        print processed_samples
        print 'val_samples'
        print val_samples
        processed_samples += nb_samples

    _stop.set()
    x_exp = np.array(x_exp,dtype=np.dtype(float))
    y_exp = np.array(y_exp,dtype=np.dtype(float))
    for c,model in enumerate(models):
        y_hat_exp = y_hat_dic[c]
        y_hat_exp = np.array(y_hat_exp,dtype=np.dtype(float))
        y_hat_dic[c] = y_hat_exp
        print y_hat_exp.shape
    print x_exp.shape
    print y_exp.shape
    
    return (x_exp, y_exp, y_hat_dic)









if __name__ == "__main__":
    print "Commencing..."
    data_dir = './data/'
    evt_name = 'Featurespace_events_output.csv'
    auth_name = 'Featurespace_auths_output.csv'
    db_name = 'c1_agg.db'
    
    disk_engine = create_engine('sqlite:///'+data_dir+db_name,convert_unicode=True)
    disk_engine.raw_connection().connection.text_factory = str
    #######################Settings#############################################
    hidden_dim = 400
    num_layers = 1
    optimizer=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08)
    samples_per_epoch = 485
    nb_epoch = 20
    table = 'data_trim'
    lbl_pad_val = 2
    pad_val = -1
    class_weight = {0 : 1.,
                1: 10.,
                2: 0.}



    ####################################DATA SOURCE################################
    table = 'data_trim'
    ################################################################################

    encoders = populate_encoders(table,disk_engine)
    print "Encoders populated"



    ####################################TEST################################
    print 'Verifying Dimensions'
    data_gen =  data_generator(disk_engine,encoders,table='data_trim',class_weight=class_weight)
    X_train_pad,y_train_S,sample_w = next(data_gen)
    print X_train_pad.shape
    print y_train_S.shape
    print sample_w.shape
    print 'Completed'
    ################################################################################

    ######################SOMEHOW this is working or not really
    # input_layer = Input(shape=(50, 44),name='main_input')
    # mask = Masking(mask_value=0)(input_layer)
    # prev = GRU(hidden_dim,#input_length=50,
 #                      return_sequences=True,go_backwards=False,stateful=False,
 #                      unroll=False,consume_less='gpu',
 #                      init='glorot_uniform', inner_init='orthogonal', activation='tanh',
 #              inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None,
 #              b_regularizer=None, dropout_W=0.0, dropout_U=0.0)(mask)
    # # for i in range(num_layers-1):
    # #     prev = GRU(output_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh',
    # #            inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None,
    # #            b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
    # output_layer = TimeDistributed(Dense(1))(prev)
    # model = Model(input=[input_layer],output=[output_layer])
    # model.compile(optimizer=optimizer,
 #                  loss='binary_crossentropy',
 #                  metrics=['accuracy'])
    # data_gen =  data_generator(disk_engine,encoders,table=table)
    # input_layer = Input(shape=(50, 44),name='main_input')
    # mask = Masking(mask_value=pad_val)(input_layer)
    # prev = GRU(hidden_dim,#input_length=50,
    #                 return_sequences=True,go_backwards=False,stateful=False,
    #                 unroll=False,consume_less='gpu',
    #                 init='glorot_uniform', inner_init='orthogonal', activation='tanh',
    #         inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None,
    #         b_regularizer=None, dropout_W=0.0, dropout_U=0.0)(mask)
    # # for i in range(num_layers-1):
    # #     prev = GRU(output_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh',
    # #            inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None,
    # #            b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
    # output_layer = TimeDistributed(Dense(3,activation='softmax'))(prev)
    # model = Model(input=[input_layer],output=[output_layer])
    # model.compile(optimizer=optimizer,
    #             loss='sparse_categorical_crossentropy',
    # #               metrics=['accuracy','hinge','squared_hinge','binary_accuracy','binary_crossentropy'])
    #             metrics=['accuracy'],
    #             sample_weight_mode="temporal")
    # data_gen =  data_generator(disk_engine,encoders,table=table,class_weight=class_weight)
    # # data_gen =  data_generator(disk_engine,encoders,table=table,class_weight=class_weight)
    # history = model.fit_generator(data_gen, samples_per_epoch, nb_epoch, verbose=1, callbacks=[],validation_data=None, nb_val_samples=None, class_weight=class_weight, max_q_size=10000)



    # hera_model = HeraModel(
    #     {
    #         'id': '200 epochs' # any ID you want to use to identify your model
    #     },
    #     {
    #         # location of the local hera server, out of the box it's the following
    #         'domain': 'localhost',
    #         'port': 4000
    #     }
    # )
    # history = model.fit_generator(data_gen, samples_per_epoch, nb_epoch, verbose=1, callbacks=[hera_model.callback],validation_data=None, nb_val_samples=None, class_weight=None, max_q_size=10)
    # py.sign_in('bottydim', 'o1kuyms9zv') 
    # title = 'Training_Loss'
    # fig = {
    #     'data': [Scatter(
    #         x=history.epoch,
    #         y=history.history['loss'])],
    #     'layout': {'title': title}
    #     }
    # py.image.save_as(fig,filename='./figures/'+title+".png")
    # # iplot(fig,filename='figures/'+title,image='png')
    # title = 'Training Acc'
    # fig = {
    #     'data': [Scatter(
    #         x=history.epoch,
    #         y=history.history['acc'])],
    #     'layout': {'title': title}
    #     }
    # py.image.save_as(fig,filename='./figures/'+title+".png")
    # iplot(fig,filename='figures/'+title,image='png')



    #############GRID SEARCH #####################
    rsl_file = './data/gs_results_trim.csv'
    hid_dims = [512,1024]
    num_l = [1,2,3,4,5]
    lr_s = [5e-4]
    # lr_s = [1e-2,1e-3,1e-4]
    # lr_s = [1e-1,1e-2,1e-3]
    opts = lambda x,lr:[keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08),
                    keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                    keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)][x]
    add_info = 'cont'                
    table = 'data_trim'
    samples_per_epoch = 444
    # samples_per_epoch = 1959
    # table = 'data_trim'
    # samples_per_epoch = 485
    nb_epoch = 30
    lbl_pad_val = 2
    pad_val = -1

    encoders = populate_encoders(table,disk_engine)
    gru_dict = {}
    lstm_dict = {}
    for hidden_dim in hid_dims:
    # gru
        for opt_id in range(3):
            for lr in lr_s:
                optimizer = opts(opt_id,lr)
                for num_layers in num_l:
                    for rnn in ['gru','lstm']:
                        title = 'Training_Loss'+'_'+rnn.upper()+'_'+str(hidden_dim)+'_'+str(num_layers)+'_'+str(type(optimizer).__name__)+'_'+str(lr)
                        print title
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
                        data_gen =  data_generator(disk_engine,encoders,table=table)
                        history = model.fit_generator(data_gen, samples_per_epoch, nb_epoch, verbose=1, callbacks=[],validation_data=None, nb_val_samples=None, class_weight=None, max_q_size=10000)
                        py.sign_in('bottydim', 'o1kuyms9zv') 

                        with io.open(rsl_file, 'a', encoding='utf-8') as file:
                            title_csv = title.replace('_',',')+','+str(history.history['acc'][-1])+','+str(history.history['loss'][-1])+'\n'
                            file.write(unicode(title_csv))
                        fig = {
                            'data': [Scatter(
                                x=history.epoch,
                                y=history.history['loss'])],
                            'layout': {'title': title}
                            }
                        py.image.save_as(fig,filename='./figures/GS/'+table+'/'+title+'_'+table+'_'+add_info+".png")
                        # iplot(fig,filename='figures/'+title,image='png')
                        title = title.replace('Loss','Acc')
                        fig = {
                            'data': [Scatter(
                                x=history.epoch,
                                y=history.history['acc'])],
                            'layout': {'title': title}
                            }
                        py.image.save_as(fig,filename='./figures/GS/'+table+'/'+title+'_'+table+'_'+add_info+".png")    
