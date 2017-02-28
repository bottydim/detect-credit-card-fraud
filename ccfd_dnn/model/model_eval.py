import os
import argparse
import re
import numpy as np
from ccfd_dnn.model import *
np.random.seed(1337)

def get_engine(address = "postgresql+pg8000://script@localhost:5432/ccfd"):

    # disk_engine = create_engine('sqlite:///'+data_dir+db_name,convert_unicode=True)
    # disk_engine.raw_connection().connection.text_factory = str
    disk_engine = create_engine(address)
    return disk_engine

class ModelLoader:

    arch_dir = '/home/botty/Documents/CCFD/data/models/archs/'
    archs = ['/home/botty/Documents/CCFD/data/archs/bi_GRU_320_4_DO-0.3']
    w_dir = '/home/botty/Documents/CCFD/data/models/{table}/'
    __ws = [
        'Bidirectional_Class400.0_GRU_320_4_RMSprop_0.00025_epochs_10_DO-0.3.09-0.01.hdf5',
        'Bidirectional_Class400.0_GRU_320_4_RMSprop_0.00025_epochs_10_DO-0.3.05-0.04623.hdf5'
        ]

    def __init__(self,table,w_id=-1,arch_path=None,w_path=None):

        if arch_path == None:
            self.arch_path = self.archs[0]
        else:
            self.arch_path = ModelLoader.arch_dir+arch_path
        if w_path == None:

            self.w_path = ModelLoader.w_dir.format(table=table)+ModelLoader.__ws[w_id]
        else:
            self.w_path = ModelLoader.w_dir.format(table=table)+w_path

        self.model = load_model(self.arch_path, self.w_path)

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
        encoders = self.encoders
        batch_size = 2000
        #calcualte # samples
        val_samples = trans_num_table(table,disk_engine,mode=user_mode,trans_mode=trans_mode)
        # val_samples = 10
        print '# samples',val_samples


        eval_gen = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                         batch_size=batch_size,usr_ratio=80, class_weight=None,lbl_pad_val = 2, pad_val = -1,events_tbl=events_tbl)

        
        plt_filename = './results/figures/'+table+'/'+'ROC_'+user_mode+'_'+trans_mode+'_'+title+'_'+add_info+".png"
        eval_list  = eval_auc_generator(model, eval_gen, val_samples, max_q_size=10000,plt_filename=plt_filename)
        

        auc_val = eval_list[0]
        clc_report = eval_list[1]
        acc = eval_list[2]
        print "AUC:", auc_val
        print 'Classification report'
        print clc_report
        print 'Accuracy'
        print acc
        self.auc_list.append(str(auc_val))
        return eval_list

def find_best_val_file(name,files):
    pass

def extract_val_loss(f_name):
    m = re.search('-[0-9]+\.[0-9]+\.hdf5',f_name)
    start = m.span()[0]+1
    end = m.span()[1]-5
    return float(f_name[start:end])


def eval_best_val(table):
    #populate dictionary 
    directory = ModelLoader.w_dir.format(table=table)
    print 'evaluating path @'
    for root, dirs, files in os.walk(directory):
        names = {}
        for f in files:
            # print f
            m = re.search('\.[0-9]+-',f)
            name_end = m.span()[0]
            name = f[0:name_end]
            if 'Bidirectional_Class' in f:
                print 'skipping >>>>> '+f
                continue
            # print name
            if name in names.keys():
                temp = names[name]
                temp_val = extract_val_loss(temp)
                curr_val = extract_val_loss(f)
                if(curr_val<temp_val):
                    names[name] = f
            else:
                names[name] = f
    print 'TOTAL MODELS TO EVALUATE! - ',len(names) 
    for k,v in names.iteritems():
        auc_list = eval_model(table,arch=k+'.yml',w_path=v,add_info='BEST_VAL',title = k)
        rsl_file = '/home/botty/Documents/CCFD/results/psql_gs_{table}.csv'.format(table=table)
        with io.open(rsl_file, 'a', encoding='utf-8') as file:
            auc_string = ','.join(auc_list)
            title_csv = k+','+auc_string+'\n'
            file.write(unicode(title_csv))
            print 'logged @ {file}'.format(file=rsl_file)

def eval_model(*args, **kwargs):

    table = args[0]
    arch = None
    ws_path = None
    add_info = ''
    title = 'BiRNN-DO3-DLE'
    if 'add_info' in kwargs.keys():
        add_info = kwargs['add_info']


    if 'title' in kwargs.keys():
        title = kwargs['title']
    if 'arch' in kwargs.keys():
        arch = kwargs['arch']
    if 'w_path' in kwargs.keys():
        w_path = kwargs['w_path']
    else:
        w_id = kwargs['w_id']
    #########################################
    disk_engine = get_engine()
    ml = ModelLoader(table,arch_path=arch,w_path=w_path)
    model = ml.model
    
    data_little_enc_eval = Evaluator(model, table, disk_engine)
    table_auth = 'auth_enc'
    auth_enc_eval = Evaluator(model, table_auth, disk_engine)
    options = ['train','test']
    py.sign_in('bottydim', 'o1kuyms9zv') 
    print '======================={}============================'.format(table)
    for user_mode in options:
        for trans_mode in options:
            print '################## USER:{user_mode}--------TRANS:{trans_mode}###############'.format(user_mode=user_mode,trans_mode=trans_mode)
            eval_list = data_little_enc_eval.evaluate_model(user_mode,trans_mode,title,add_info=add_info)
            print '#########################################################################################################'
    print '=======================AUC============================'
    return data_little_enc_eval.auc_list
def parse_args():

    parser = argparse.ArgumentParser(prog='Model Evaluator')
    parser.add_argument('-t','--table', required=True)
    parser.add_argument('-i','--id', default=1)
    args = parser.parse_args()

    ####################################DATA SOURCE################################
    var_args = vars(args)
    table = var_args['table']
    w_id = var_args['id']
    return table, w_id

if __name__ == "__main__":

    table, w_id = parse_args()
    # eval_model(table,w_id = w_id):


    eval_best_val(table)

    # for user_mode in options:
    #     for trans_mode in options:
    #         print '################## USER:{user_mode}--------TRANS:{trans_mode}###############'.format(user_mode=user_mode,trans_mode=trans_mode)
    #         auth_enc_eval.evaluate_model(user_mode,trans_mode,title)
    #         print '#########################################################################################################'            