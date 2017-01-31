import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
import argparse
import sys



if __name__ == "__main__":


    t_start = dt.datetime.now()
    parser = argparse.ArgumentParser(prog='Weighted Model')
    parser.add_argument('-t','--table',required=True)
    args = parser.parse_args()

    ####################################DATA SOURCE################################
    table = vars(args)['table']
    # table = 'data_trim'
    # rsl_file = './data/gs_results_trim.csv'
    # rsl_file = './data/psql_data_trim.csv'

    # table = 'data_little_enc'
    # rsl_file = './data/gs_results_little.csv'
    
    # table = 'data_more'
    # rsl_file = './data/gs_results_more.csv'
    # table = 'auth'
    # rsl_file = './data/auth.csv'

    events_tbl = 'event'
    events_tbl = None
    rsl_file = './data/psql_{table}.csv'.format(table=table)
    ################################################################################




    print "Commencing..."
    data_dir = './data/'
    evt_name = 'Featurespace_events_output.csv'
    auth_name = 'Featurespace_auths_output.csv'
    db_name = 'c1_agg.db'
    

    address = "postgresql+pg8000://script@localhost:5432/ccfd"
    # disk_engine = create_engine('sqlite:///'+data_dir+db_name,convert_unicode=True)
    # disk_engine.raw_connection().connection.text_factory = str
    disk_engine = create_engine(address)



    #######################Settings#############################################
    samples_per_epoch = trans_num_table(table,disk_engine,mode='train',trans_mode='train')
    # epoch_limit = 10000
    # samples_per_epoch = epoch_limit
    # user_sample_size = 8000


    epoch_limit = samples_per_epoch
    user_sample_size = None

    nb_epoch = 100
    fraud_w_list = [1000.]
    
    ##########ENCODERS CONF
    tbl_src = 'auth'
    # tbl_src = table
    tbl_evnt = 'event'
    ##################################
   
    batch_size = 300  #user sample size!
    batch_size_val = 1000
    seq_len_param = 60.0
    print "SAMPLES per epoch:",samples_per_epoch
    print "User sample size:",user_sample_size
    print 'sequence length size',batch_size
    # samples_per_epoch = 1959
    # table = 'data_trim'
    # samples_per_epoch = 485
    


    lbl_pad_val = 2
    pad_val = -1

    dropout_W_list = [0.3]
    # dropout_W_list = [0.4,0.5,0.6,0.7]
    # dropout_W_list = [0.15,0.3,0.4,0.8]
    




    n = 70
    discard_val = n-7
    input_dim = 63
    hid_dims = [320]
    num_l = [7]
    lr_s = [2.5e-4]
    # lr_s = [1.25e-4,6e-5]
    # lr_s = [1e-2,1e-3,1e-4]
    # lr_s = [1e-1,1e-2,1e-3]

    ###Optimizer Set-up
    num_opt = 1
    opts = lambda x,lr:[keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08),
                    # keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                    # keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
                    ][x]

    ### Additional parameters
    patience = 30  # how long to wait before to terminate due to Early Stopping

    # add_info = str(int(seq_len_param))+'_class_w_'+str(fraud_w)                
                  
    

    print 'Populating encoders'

    path_encoders ='./data/encoders/{tbl_src}+{tbl_evnt}'.format(tbl_src=tbl_src,tbl_evnt=tbl_evnt) 
    if os.path.exists(path_encoders):
        encoders = load_encoders(path_encoders)
    else:
        encoders = populate_encoders_scale(tbl_src,disk_engine,tbl_evnt)
        with open(path_encoders, 'wb') as output:
            pickle.dump(encoders, output, pickle.HIGHEST_PROTOCOL)
            print 'ENCODERS SAVED to {path}!'.format(path=path_encoders)

    # sys.exit()
    gru_dict = {}
    lstm_dict = {}
    for fraud_w in fraud_w_list:
        add_info = 'Mask=pad_class_w_'+str(fraud_w)+'ES-OFF'  
        class_weight = {0 : 1.,
            1: fraud_w,
            2: 0.}
        for dropout_W in dropout_W_list:
            for hidden_dim in hid_dims:
            # gru
                for opt_id in range(num_opt):
                    for lr in lr_s:
                        optimizer = opts(opt_id,lr)
                        for num_layers in num_l:
                            for rnn in ['gru']:

                                short_title = 'bi_'+rnn.upper()+'_'+str(hidden_dim)+'_'+str(num_layers)+'_DO-'+str(dropout_W)+'_w'+str(class_weight[1])
                                title = 'Bidirectional_Class'+str(class_weight[1])+'_'+rnn.upper()+'_'+str(hidden_dim)+'_'+str(num_layers)+'_'+str(type(optimizer).__name__)+\
                                        '_'+str(lr)+'_epochs_'+str(nb_epoch)+'_DO-'+str(dropout_W)+'_'+add_info
                                print title
                                input_layer = Input(shape=(int(seq_len_param), input_dim),name='main_input')
                                mask = Masking(mask_value=pad_val)(input_layer)
                                x = mask
                                for i in range(num_layers):
                                    if rnn == 'gru':
                                        prev_frw = GRU(hidden_dim,#input_length=50,
                                                            return_sequences=True,go_backwards=False,stateful=False,
                                                            unroll=False,consume_less='gpu',
                                                            init='glorot_uniform', inner_init='orthogonal', activation='tanh',
                                                    inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None,
                                                    b_regularizer=None, dropout_W=dropout_W, dropout_U=0.0)(x)
                                        prev_bck = GRU(hidden_dim,#input_length=50,
                                                            return_sequences=True,go_backwards=True,stateful=False,
                                                            unroll=False,consume_less='gpu',
                                                            init='glorot_uniform', inner_init='orthogonal', activation='tanh',
                                                    inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None,
                                                    b_regularizer=None, dropout_W=dropout_W, dropout_U=0.0)(x)
                                    else:
                                        prev_frw = LSTM(hidden_dim, return_sequences=True,go_backwards=False,stateful=False,
                                            init='glorot_uniform', inner_init='orthogonal', 
                                            forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid',
                                            W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=dropout_W, dropout_U=0.0)(x)
                                        prev_bck = LSTM(hidden_dim, return_sequences=True,go_backwards=True,stateful=False,
                                            init='glorot_uniform', inner_init='orthogonal', 
                                            forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid',
                                            W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=dropout_W, dropout_U=0.0)(x)
                                    x = merge([prev_frw, prev_bck], mode='concat')
                                output_layer = TimeDistributed(Dense(3,activation='softmax'))(x)
                                model = Model(input=[input_layer],output=[output_layer])
                                model.compile(optimizer=optimizer,
                                                loss='sparse_categorical_crossentropy',
                                                metrics=['accuracy'],
                                                sample_weight_mode="temporal")

                                ########save architecture ######
                                arch_dir = './data/models/archs/'+short_title+'.yml'
                                yaml_string = model.to_yaml()
                                with open(arch_dir, 'wb') as output:
                                    pickle.dump(yaml_string, output, pickle.HIGHEST_PROTOCOL)
                                    print 'model saved!'
                                ##############        

                                user_mode = 'train'
                                trans_mode = 'train'
                                data_gen = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                                                 batch_size=batch_size,usr_ratio=80,class_weight=class_weight,lbl_pad_val = lbl_pad_val, pad_val = pad_val,
                                                 sub_sample=user_sample_size,epoch_size=epoch_limit,events_tbl=events_tbl,discard_id = discard_val)
                                                 # sub_sample=user_sample_size,epoch_size=samples_per_epoch)
                                ########validation data
                                print 'Generating Validation set!'
                                user_mode = 'test'
                                trans_mode = 'test'
                                val_gen = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                                                 batch_size=batch_size_val,usr_ratio=80,class_weight=class_weight,lbl_pad_val = lbl_pad_val, pad_val = pad_val,
                                                 sub_sample=None,epoch_size=None,events_tbl=events_tbl,discard_id = discard_val)
                                validation_data = next(val_gen)
                                print '################GENERATED#######################'
                                ###############CALLBACKS

                                early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto')

                                save_path = './data/models/'+table+'/'
                                var_name = '.{epoch:02d}-{val_loss:.5f}.hdf5'
                                checkpoint = keras.callbacks.ModelCheckpoint(save_path+short_title+var_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

                                root_url = 'http://localhost:9000'
                                remote_log = keras.callbacks.RemoteMonitor(root=root_url)

                                # callbacks = [early_Stop,checkpoint]
                                callbacks = [early_Stop,checkpoint,remote_log]
                                callbacks = [checkpoint,remote_log]
                                history = model.fit_generator(data_gen, samples_per_epoch, nb_epoch, verbose=1, callbacks=callbacks,validation_data=validation_data, nb_val_samples=None, class_weight=None, max_q_size=10000)

                                py.sign_in('bottydim', 'o1kuyms9zv') 

                                auc_list = []
                                print '#########################TRAIN STATS################'
                                user_mode = 'train'
                                trans_mode = 'train'
                                val_samples = trans_num_table(table,disk_engine,mode=user_mode,trans_mode=trans_mode)
                                print '# samples',val_samples
                                plt_filename = './figures/GS/'+table+'/'+'ROC_'+user_mode+'_'+trans_mode+'_'+title+'_'+add_info+".png"

                                data_gen = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                                                 batch_size=batch_size,usr_ratio=80,class_weight=None,lbl_pad_val = lbl_pad_val, pad_val = pad_val,events_tbl=events_tbl,discard_id = discard_val) 

                                eval_list  = eval_auc_generator(model, data_gen, val_samples, max_q_size=10000,plt_filename=plt_filename)
                                auc_val = eval_list[0]
                                clc_report = eval_list[1]
                                acc = eval_list[2]
                                print "AUC:",auc_val 
                                print 'CLassification report'
                                print clc_report
                                print 'Accuracy'
                                print acc
                                auc_list.append(str(auc_val))
                                print '##################EVALUATION USERS#########################'

                                user_mode = 'test'
                                trans_mode = 'train'
                                val_samples = trans_num_table(table,disk_engine,mode=user_mode,trans_mode=trans_mode)
                                print '# samples',val_samples
                                plt_filename = './figures/GS/'+table+'/'+'ROC_'+user_mode+'_'+trans_mode+'_'+title+'_'+add_info+".png"

                                eval_gen = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                                                 batch_size=batch_size,usr_ratio=80,class_weight=None,lbl_pad_val = lbl_pad_val, pad_val = pad_val,events_tbl=events_tbl,discard_id = discard_val) 

                                eval_list  = eval_auc_generator(model, eval_gen, val_samples, max_q_size=10000,plt_filename=plt_filename)
                                auc_val = eval_list[0]
                                clc_report = eval_list[1]
                                acc = eval_list[2]
                                print "AUC:",auc_val 
                                print 'CLassification report'
                                print clc_report
                                print 'Accuracy'
                                print acc
                                auc_list.append(str(auc_val))
                                print '#####################################################'
                                print '##################EVALUATION Transactions#########################'

                                user_mode = 'train'
                                trans_mode = 'test'
                                val_samples = trans_num_table(table,disk_engine,mode=user_mode,trans_mode=trans_mode)
                                print '# samples',val_samples
                                plt_filename = './figures/GS/'+table+'/'+'ROC_'+user_mode+'_'+trans_mode+'_'+title+'_'+add_info+".png"

                                eval_gen = data_generator(user_mode,trans_mode, disk_engine,encoders, table=table,
                                                 batch_size=batch_size,usr_ratio=80,class_weight=None, lbl_pad_val = lbl_pad_val, pad_val = pad_val,events_tbl=events_tbl,discard_id = discard_val)

                                eval_list  = eval_auc_generator(model, eval_gen, val_samples, max_q_size=10000, plt_filename=plt_filename)
                                auc_val = eval_list[0]
                                clc_report = eval_list[1]
                                acc = eval_list[2]
                                print "AUC:",auc_val 
                                print 'CLassification report'
                                print clc_report
                                print 'Accuracy'
                                print acc
                                auc_list.append(str(auc_val))
                                print '#####################################################'
                                print '##################EVALUATION Pure#########################'

                                user_mode = 'test'
                                trans_mode = 'test'
                                val_samples = trans_num_table(table,disk_engine,mode=user_mode,trans_mode=trans_mode)
                                print '# samples',val_samples
                                plt_filename = './figures/GS/'+table+'/'+'ROC_'+user_mode+'_'+trans_mode+'_'+title+'_'+add_info+".png"
                                
                                eval_gen = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                                                 batch_size=batch_size,usr_ratio=80,class_weight=None,lbl_pad_val = lbl_pad_val, pad_val = pad_val,events_tbl=events_tbl,discard_id = discard_val)
                                
                                eval_list  = eval_auc_generator(model, eval_gen, val_samples, max_q_size=10000,plt_filename=plt_filename)
                                auc_val = eval_list[0]
                                clc_report = eval_list[1]
                                acc = eval_list[2]
                                print "AUC:",auc_val 
                                print 'CLassification report'
                                print clc_report
                                print 'Accuracy'
                                print acc
                                auc_list.append(str(auc_val))
                                print '#####################################################'
                                with io.open(rsl_file, 'a', encoding='utf-8') as file:
                                    auc_string = ','.join(auc_list)
                                    title_csv = title.replace('_',',')+','+str(history.history['acc'][-1])+','+str(history.history['loss'][-1])+','+str(auc_val)+','+str(acc)+','+auc_string+'\n'
                                    file.write(unicode(title_csv))
                                    print 'logged @ {file}'.format(file=rsl_file)
                                trim_point = -15
                                fig = {
                                    'data': [Scatter(
                                        x=history.epoch[trim_point:],
                                        y=history.history['loss'][trim_point:])],
                                    'layout': {'title': title}
                                    }
                                py.image.save_as(fig,filename='./results/figures/'+table+'/'+short_title+'_'+'LOSS'+'_'+add_info+".png")
                                trim_point = 0
                                fig = {
                                    'data': [Scatter(
                                        x=history.epoch[trim_point:],
                                        y=history.history['loss'][trim_point:])],
                                    'layout': {'title': title}
                                    }
                                py.image.save_as(fig,filename='./results/figures/'+table+'/'+short_title+'_'+'LOSS'+'_'+'FULL'+".png")                            

                                # iplot(fig,filename='figures/'+title,image='png')
                                # title = title.replace('Loss','Acc')
                                fig = {
                                    'data': [Scatter(
                                        x=history.epoch[trim_point:],
                                        y=history.history['acc'][trim_point:])],
                                    'layout': {'title': title}
                                    }
                                filename_val='./results/figures/'+table+'/'+short_title+'_'+'ACC'+'_'+add_info+".png"
                                py.image.save_as(fig,filename=filename_val)    
                                print 'exported @',filename_val
                                fig = {
                                    'data': [Scatter(
                                        x=history.epoch[trim_point:],
                                        y=history.history['val_loss'][trim_point:])],
                                    'layout': {'title': title}
                                    }
                                py.image.save_as(fig,filename='./results/figures/'+table+'/'+short_title+'_'+'VAL LOSS'+'_'+add_info+".png")   
                                print 'time taken: {time}'.format(time=days_hours_minutes_seconds(dt.datetime.now()-t_start))