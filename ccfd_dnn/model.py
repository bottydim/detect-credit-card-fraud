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
import pandas as pd
from sqlalchemy import create_engine # database connection
import datetime as dt
import io

import plotly.plotly as py # interactive graphing
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Bar, Scatter, Marker, Layout, Figure
from heraspy.model import HeraModel
np.random.seed(1337)
import theano
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, GRU, LSTM, TimeDistributed, Masking
from keras.engine.training import *

time_cols = ['AUTHZN_RQST_PROC_TM','PREV_ADR_CHNG_DT','PREV_PMT_DT','PREV_CARD_RQST_DT','FRD_IND_SWT_DT']
seq_len_param = 60.0
def encode_column(df_col):
    print df_col.shape
    le = preprocessing.LabelEncoder()
    le.fit(df_col)
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

def populate_encoders_scale(table,disk_engine):
    df = pd.read_sql_query('select * from {table} limit 5'.format(table=table),disk_engine)
    col_names = df.columns.values
    encoders = {}
    time_cols = ['AUTHZN_RQST_PROC_TM','PREV_ADR_CHNG_DT','PREV_PMT_DT','PREV_CARD_RQST_DT','FRD_IND_SWT_DT']
    for c,name in enumerate(col_names):
        tp = df.dtypes[c]
    #     print tp

        if tp == 'object':
            if name not in time_cols:
                print name
                df_col = pd.read_sql_query('select distinct {col_name} from {table}'.format(col_name=name,table=table),disk_engine)
                encoders[name] = encode_column(np.array(df_col).ravel())
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


def get_user_info(user,table,disk_engine):
    if user == '.':
        user = '"."'
    df_u = pd.read_sql_query('select * from {table} where acct_id = {user}'.format(table=table,user=user),disk_engine)
    return df_u

def get_last_date(df_u,cuttoff_date):
#     print "Before Trim"
#     display(df_u)
    df_trim = df_u[df_u['FRD_IND_SWT_DT'] >= pd.to_numeric(pd.Series(pd.to_datetime(cuttoff_date)))[0]]
#     print "After Trim"
#     display(df_trim)
    ### a historicly later transaction may have been confirmed earlier than a historicly preceeding T
    df_trim = df_trim.sort_values('AUTHZN_RQST_PROC_TM',ascending=True,inplace=False)
    df_trim = df_trim.reset_index(drop=True)
#     print "After Reorder"
#     display(df_trim)
#     display(df_trim)
    if not df_trim.empty:
#         print 'value to be returned',df_trim['AUTHZN_RQST_PROC_TM'][0]
        return df_trim['AUTHZN_RQST_PROC_TM'][0]
    else:
        return None


def get_col_id(col,df):
    col_list = list(df.columns.values)
    col_list.remove('index')
    col_list.index(col)
    
def generate_sequence(user,table,encoders,disk_engine,lbl_pad_val,pad_val,cuttoff_date='2014-05-11'):
    df_u = get_user_info(user,table,disk_engine)
    unav_cols = ['AUTHZN_APPRL_CD','TSYS_DCLN_REAS_CD','AUTHZN_RESPNS_CD','AUTHZN_APPRD_AMT',]
    nan_rpl = ['AUTHZN_APPRL_CD',]
    for col in unav_cols:
        df_u[col] = df_u[col].shift(1)
        loc = list(df_u.columns.values).index(col)
        if(col in nan_rpl):
            df_u.iloc[0,loc] = 'nan'
        else:
            df_u.iloc[0,loc] = pad_val
#     print df_u.count()
#     display(df_u.head())
#     display(df_u.sort_values('AUTHZN_RQST_PROC_TM',ascending=True))
    encode_df(df_u,encoders)
#     print df_u.count()
#     display(df_u.head())
#     display(df_u.sort_values('AUTHZN_RQST_PROC_TM',ascending=True))
    df_u = df_u.sort_values('AUTHZN_RQST_PROC_TM',ascending=True)
#     display(df_u[df_u['FRD_IND_SWT_DT'].isnull()])
    df_u = df_u.drop('index', axis=1)
#     display(df_u[df_u['FRD_IND_SWT_DT'] < pd.to_numeric(pd.Series(pd.to_datetime(cuttoff_date)))[0]].head(8))
### This is the last date, before which transaction will be used for trainning. 
### It coresponds to the date when the last knwon fraudulent transaction was confirmed
    last_date_num = get_last_date(df_u,cuttoff_date)
    if last_date_num == None:
        train = np.array(df_u)
#         print "No cutt offs"
#         print train[:,0:-2].shape
#         print "labels"
#         print train[:,-2].shape
        return train[:,0:-2],[],train[:,-2],[]
    else:
        df_train = df_u[df_u['AUTHZN_RQST_PROC_TM'] < last_date_num]
        df_test = df_u[df_u['AUTHZN_RQST_PROC_TM'] >= last_date_num]
        print 'train/test sequence split:',np.array(df_train).shape[0],np.array(df_test).shape[0]
        print user
#     display(df_train)
#     display(df_test)

        
    train = np.array(df_train)
    test = np.array(df_test)
    print 'test shape in sequencer',test.shape
    return train[:,0:-2],test[:,0:-2],train[:,-2],test[:,-2]

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
def sequence_generator(users,encoders,disk_engine,lbl_pad_val,pad_val,mode='train',table='data_trim',class_weight=None):
    X_train_S = []
    X_test_S = []
    y_train_S =[]
    y_test_S = []
    print "Number of users:",len(users)
    for user in users:
    #     if user != '337018623': 
    #         continue
        X_train,X_test,y_train,y_test = generate_sequence(user,table,encoders,disk_engine,lbl_pad_val,pad_val)
        if X_test != []:
            print 'shape in generator',X_test.shape
        X_train_S.append(X_train)
        X_test_S.append(X_test) 
        y_train_S.append(y_train)
        y_test_S.append(y_test)
    #     break
    X_test_S = filter(lambda a: a != [], X_test_S)
    y_test_S = filter(lambda a: a != [], y_test_S)
    if mode =='train':
        # chuncked = chunck_seq(X_train_S)
        # assert 
        X_train_pad = keras.preprocessing.sequence.pad_sequences(chunck_seq(X_train_S), maxlen=int(seq_len_param),dtype='float32',value=pad_val)
        y_train_S = keras.preprocessing.sequence.pad_sequences(np.array(chunck_seq(y_train_S)), maxlen=int(seq_len_param),dtype='float32',value=lbl_pad_val)
        y_train_S = np.expand_dims(y_train_S, -1)
        print 'labels shape',y_train_S.shape
        if class_weight != None:

            sample_w = generate_sample_w(y_train_S,class_weight)
            return X_train_pad,y_train_S,sample_w
#         print y_train_S
#         print y_train_S.shape
#         y_train_S = to_categorical(y_train_S,3)
        return X_train_pad,y_train_S
    else:
        print 'len test',len(X_test_S)
        X_test_S_pad = keras.preprocessing.sequence.pad_sequences(chunck_seq(X_test_S), maxlen=int(seq_len_param),dtype='float32',value=pad_val)
        y_test_S = keras.preprocessing.sequence.pad_sequences(np.array(chunck_seq(y_test_S)),maxlen=int(seq_len_param),dtype='float32',value=lbl_pad_val)
        y_test_S = np.expand_dims(y_test_S, -1)
        if class_weight != None:
            sample_w = generate_sample_w(y_train_S,class_weight)
            return X_train_pad,y_train_S,sample_w
        return X_test_S_pad,y_test_S


def trans_num_table(table,disk_engine,mode='train'):
    dataFrame_acc = pd.read_sql_query('select distinct acct_id, FRD_IND '
                       'from {table} '
                       'order by FRD_IND'.format(table=table), disk_engine)
    dataFrame = pd.read_sql_query('select acct_id, count(*) as num_trans '
                       'from {table} '
                       'group by acct_id '
                        'order by num_trans'
                       .format(table=table), disk_engine)
    u_list = set(dataFrame_acc.acct_id)
    
    user_tr,user_ts = train_test_split(list(u_list), test_size=0.33, random_state=42)
    print 'Distinct users:',len(u_list)
    total_t =0
    if mode == 'train':
        users = user_tr
    else:
        users = user_ts
    print 'Using:',len(users)
    total_t = total_trans_batch(users,dataFrame)
    return math.ceil(total_t)

def total_trans_batch(users,dataFrame_count):
    num_trans = 0
    for user in users:
        num_trans+=get_num_trans(user,dataFrame_count)
    return num_trans
def get_num_trans(user,dfc):
    try:
        seq_len = dfc[dfc['acct_id']==user].values[0][1]
    except:
        display(dataFrame_count.head(5))
    return math.ceil(1.0*seq_len/seq_len_param)

def add_user(index,u_list,dataFrame_count,users):
    cnt_trans = 0
    user = u_list[index]
    if user not in users:
        users.add(user)
        return get_num_trans(user,dataFrame_count)
    else:
        return 0
def user_generator(disk_engine,table='data_trim',sample_size=50,usr_ratio=80,mode='train'):
#     dataFrame_Y = pd.read_sql_query('select distinct acct_id, FRD_IND '
#                        'from {table} '
#                        'where FRD_IND="Y"'.format(table=table), disk_engine)
#     dataFrame_N = pd.read_sql_query('select distinct acct_id, FRD_IND '
#                        'from {table} '
#                        'where FRD_IND="N"'.format(table=table), disk_engine)
    dataFrame = pd.read_sql_query('select distinct acct_id, FRD_IND '
                       'from {table} '
                       'order by FRD_IND'.format(table=table), disk_engine)
    dataFrame_count = pd.read_sql_query('select acct_id, count(*) as num_trans '
                       'from {table} '
                       'group by acct_id '
                        'order by -num_trans'
                       .format(table=table), disk_engine)
    
#     display(dataFrame_count.head(5)) 
    print "User List acquired"
    u_list = list(dataFrame.acct_id)
#     u_list.extend(list(dataFrame_Y.acct_id))
    print 'total # users:',len(u_list)
    u_set = set(u_list)
    print 'total # unique users:',len(u_set) 
    user_tr,user_ts = train_test_split(list(u_set), test_size=0.33, random_state=42)
    print 'total # sequences:',total_trans_batch(list(u_set),dataFrame_count)
    if mode == 'train':
        u_list =  user_tr
    else:
        u_list =  user_ts     
    print 'used # sequences:',total_trans_batch(u_list,dataFrame_count)                         
#     display(dataFrame.acct_id)
    u_list = list(set(u_list))
    print 'return set cardinality:',len(u_list)
    cnt = 0
    head = 0
    tail = len(u_list)-1
    while True:
        users = set()
        cnt_trans = 0
        while cnt_trans<sample_size:
            
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
                    yield users
                    users = set()
                    
#                     print len(users)
#         print head
#         print tail
        # print 'return list length:',len(users)
#         print '# users expiriencing both', len(u_list)-len(users)
        yield users
def eval_trans_generator(disk_engine,encoders,table='data_trim',sample_size=400,usr_ratio=80,class_weight=None,lbl_pad_val = 2, pad_val = -1):
    user_gen = user_generator(disk_engine,usr_ratio=usr_ratio,sample_size=sample_size,table=table)
    print "Users generator"
    while True:
        users = next(user_gen)
        yield sequence_generator(users,encoders,disk_engine,lbl_pad_val,pad_val,mode='test',table=table,class_weight=class_weight)

def eval_users_generator(disk_engine,encoders,table='data_trim',sample_size=400,usr_ratio=80,class_weight=None,lbl_pad_val = 2, pad_val = -1):
    user_gen = user_generator(disk_engine,usr_ratio=usr_ratio,sample_size=sample_size,table=table,mode='test')
    print "Users generator"
    while True:
        users = next(user_gen)
        yield sequence_generator(users,encoders,disk_engine,lbl_pad_val,pad_val,mode='train',table=table,class_weight=class_weight)   


def eval_generator(user_mode,trans_mode,disk_engine,encoders,table='data_trim',
                   sample_size=400,usr_ratio=80,class_weight=None,lbl_pad_val = 2, pad_val = -1):
    user_gen = user_generator(disk_engine,usr_ratio=usr_ratio,sample_size=sample_size,table=table,mode=user_mode)
    print "Users generator"
    while True:
        users = next(user_gen)
        yield sequence_generator(users,encoders,disk_engine,lbl_pad_val,pad_val,mode=trans_mode,table=table,class_weight=class_weight)

def eval_auc_generator(model, generator, val_samples, max_q_size=10000,plt_filename=None,acc=True):
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
    print '# fraud transaction',all_y_r[np.where(all_y_r==1)].shape
    print '# total transactions',processed_samples
    #######ROC CURVE
    fpr,tpr,tresholds = roc_curve(all_y_r,all_y_hat)
    print all_y_hat
    print tresholds
    print tresholds.shape
    auc_val = auc(fpr, tpr)
    print auc_val
    ############CLASSIFICATION REPORT########################
    target_names = ['Genuine', 'Fraud']
    #########Need to determine treshold 
    all_y_hat[np.where(all_y_hat>=tresholds[2])] = 1
    all_y_hat[np.where(all_y_hat<tresholds[2])]  = 0
    clc_report = classification_report(all_y_r, all_y_hat, target_names=target_names)
    ############Accuracy
    acc = accuracy_score(all_y_r,all_y_hat)
    if plt_filename != None:
        trace = Scatter(x=fpr,y=tpr)
        data = [trace]
        title = 'ROC'
        layout = Layout(title=title, width=800, height=640)
        fig = Figure(data=data, layout=layout)
        py.image.save_as(fig,filename=plt_filename)
    return [auc_val,clc_report,acc]

def data_generator(disk_engine,encoders,table='data_trim',sample_size=400,usr_ratio=80,class_weight=None,lbl_pad_val = 2, pad_val = -1):
    user_gen = user_generator(disk_engine,usr_ratio=usr_ratio,sample_size=sample_size,table=table)
    print "Users generator"
    while True:
        users = next(user_gen)
        yield sequence_generator(users,encoders,disk_engine,lbl_pad_val,pad_val,mode='train',table=table,class_weight=class_weight)


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
