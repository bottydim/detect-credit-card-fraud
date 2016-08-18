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
from keras.layers import Input, Dense, GRU, LSTM, TimeDistributed, Masking,merge
from model import *


if __name__ == "__main__":
    print "Commencing..."
    data_dir = './data/'
    evt_name = 'Featurespace_events_output.csv'
    auth_name = 'Featurespace_auths_output.csv'
    db_name = 'c1_agg.db'
    
    disk_engine = create_engine('sqlite:///'+data_dir+db_name,convert_unicode=True)
    disk_engine.raw_connection().connection.text_factory = str
   

    ####################################DATA SOURCE################################
    # table = 'data_trim'
    # rsl_file = './data/gs_results_trim.csv'
    table = 'data_little'
    rsl_file = './data/gs_results_little.csv'

    # table = 'data_more'
    # rsl_file = './data/gs_results_more.csv'
    # table = 'auth'
    # rsl_file = './data/auth.csv'
    ################################################################################

    #######################Settings#############################################
    samples_per_epoch = trans_num_table(table,disk_engine,mode='train',trans_mode='train')
    epoch_limit = 1000
    samples_per_epoch = epoch_limit
    print "SAMPLES per epoch:",samples_per_epoch
    user_sample_size = 800
    print "User sample size:",user_sample_size
    batch_size = 512
    # samples_per_epoch = 1959
    # table = 'data_trim'
    # samples_per_epoch = 485
    nb_epoch = 10
    lbl_pad_val = 2
    pad_val = -1
    dropout_W_list = [0.3]
    # dropout_W_list = [0.15,0.3,0.4,0.8]
    fraud_w = 400.
    class_weight = {0 : 1.,
            1: fraud_w,
            2: 0.}





    hid_dims = [320]
    num_l = [4]
    lr_s = [2.5e-4]
    # lr_s = [1e-2,1e-3,1e-4]
    # lr_s = [1e-1,1e-2,1e-3]
    num_opt = 1
    opts = lambda x,lr:[keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08),
                    # keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                    # keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
                    ][x]
    add_info = str(int(seq_len_param))+'_class_w_'+str(fraud_w)                
    

    encoders = populate_encoders_scale(table,disk_engine)
    gru_dict = {}
    lstm_dict = {}
    for dropout_W in dropout_W_list:
        for hidden_dim in hid_dims:
        # gru
            for opt_id in range(num_opt):
                for lr in lr_s:
                    optimizer = opts(opt_id,lr)
                    for num_layers in num_l:
                        for rnn in ['gru']:
                            title = 'Bidirectional_Class'+str(class_weight[1])+'_'+rnn.upper()+'_'+str(hidden_dim)+'_'+str(num_layers)+'_'+str(type(optimizer).__name__)+'_'+str(lr)+'_epochs_'+str(nb_epoch)+'_DO'+str(dropout_W)
                            print title
                            input_layer = Input(shape=(int(seq_len_param), 44),name='main_input')
                            mask = Masking(mask_value=0)(input_layer)
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
                            user_mode = 'train'
                            trans_mode = 'train'
                            data_gen = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                                             batch_size=batch_size,usr_ratio=80,class_weight=class_weight,lbl_pad_val = 2, pad_val = -1,
                                             sub_sample=user_sample_size,epoch_size=epoch_limit)
                                             # sub_sample=user_sample_size,epoch_size=samples_per_epoch)
                            ########validation data
                            user_mode = 'test'
                            trans_mode = 'test'
                            val_gen = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                                             batch_size=batch_size,usr_ratio=80,class_weight=class_weight,lbl_pad_val = 2, pad_val = -1,
                                             sub_sample=None,epoch_size=None)
                            validation_data = next(val_gen)
                            ###############CALLBACKS
                            patience = 30
                            early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto')

                            save_path = './data/models/'+table+'/'
                            checkpoint = keras.callbacks.ModelCheckpoint(save_path+title, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

                            root_url = 'http://localhost:9000'
                            remote_log = keras.callbacks.RemoteMonitor(root=root_url)

                            callbacks = [early_Stop,checkpoint]
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
                                             batch_size=batch_size,usr_ratio=80,class_weight=None,lbl_pad_val = 2, pad_val = -1) 

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
                                             batch_size=batch_size,usr_ratio=80,class_weight=None,lbl_pad_val = 2, pad_val = -1) 

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

                            eval_gen = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                                             batch_size=batch_size,usr_ratio=80,class_weight=None,lbl_pad_val = 2, pad_val = -1) 

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
                            print '##################EVALUATION Pure#########################'

                            user_mode = 'test'
                            trans_mode = 'test'
                            val_samples = trans_num_table(table,disk_engine,mode=user_mode,trans_mode=trans_mode)
                            print '# samples',val_samples
                            plt_filename = './figures/GS/'+table+'/'+'ROC_'+user_mode+'_'+trans_mode+'_'+title+'_'+add_info+".png"
                            
                            eval_gen = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                                             batch_size=batch_size,usr_ratio=80,class_weight=None,lbl_pad_val = 2, pad_val = -1)
                            
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
                                print 'logged'
                            trim_point = -15
                            fig = {
                                'data': [Scatter(
                                    x=history.epoch[trim_point:],
                                    y=history.history['loss'][trim_point:])],
                                'layout': {'title': title}
                                }
                            py.image.save_as(fig,filename='./figures/GS/'+table+'/'+title+'_'+table+'_'+add_info+".png")
                            # iplot(fig,filename='figures/'+title,image='png')
                            title = title.replace('Loss','Acc')
                            fig = {
                                'data': [Scatter(
                                    x=history.epoch[trim_point:],
                                    y=history.history['acc'][trim_point:])],
                                'layout': {'title': title}
                                }
                            py.image.save_as(fig,filename='./figures/GS/'+table+'/'+title+'_'+table+'_'+add_info+".png")    

