import h5py
import numpy as np
from model import *
import keras
import argparse
from model_eval import *

class Exporter(ModelOperator):


    export_dist = '/home/botty/Documents/CCFD/data/models/export/'

    def __init__(self, *args, **kwargs):
        ModelOperator.__init__(self, *args, **kwargs)



    def create_file(self,name):
        file_name ="{dir}/{title}/{name}.hdf5".format(title=self.model.name,dir=Exporter.export_dist,name=name)
        f = h5py.File(file_name, "w")
        return f


    def create_group(self,f,name):
        grp = f.create_group(name)
        return grp
    def create_DS(self,f,dset_name,arr,group=None):
        shape = arr.shape
        dtype = arr.dtype
        if group==None:
            dset = f.create_dataset(dset_name,shape, dtype=dtype)
        else:
            dset = group.create_dataset(dset_name,shape, dtype=dtype)
        dset = arr
        return dset
    def export_states(self,user_mode='train',trans_mode='train',batch_size=2000,val_samples=None,remove_pad=False):
        model = self.model
        inter_models = []
        for layer in model.layers:
            if type(layer) == keras.engine.topology.Merge:
                # print layer.__dict__
                inter_models.append(Model(input=model.inputs, output=layer.output))

        
        disk_engine = self.disk_engine
        model = self.model
        title = model.name
        table = self.table
        events_tbl = self.events_tbl
        auc_list = self.auc_list
        encoders = self.encoders
        #calcualte # samples
        if val_samples == None:
            val_samples = trans_num_table(table,disk_engine,mode=user_mode,trans_mode=trans_mode)
        print '# samples',val_samples


        generator = data_generator(user_mode,trans_mode,disk_engine,encoders,table=table,
                         batch_size=batch_size,usr_ratio=80, class_weight=None,lbl_pad_val = 2, pad_val = -1,events_tbl=events_tbl)
        
        (x_exp, y_exp, y_hat_dic) = export_generator(inter_models, generator, val_samples, max_q_size=10000,remove_pad=remove_pad)
        print 'states acquired!'
        f_train = self.create_file('features')
        self.create_DS(f_train,'train',x_exp)
        self.create_DS(f_train,'labels',y_exp)
        f_state = self.create_file('states')
        for c,y_hat in enumerate(y_hat_dic):
            self.create_DS(f_state,'layer_{}'.format(c),y_exp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Model Exporter')
    parser.add_argument('-t','--table',required=True)
    parser.add_argument('-i','--id',default=1)
    args = parser.parse_args()

    ####################################DATA SOURCE################################
    var_args = vars(args)
    table = var_args['table']
    w_id = var_args['id']
    #########################################

    disk_engine = get_engine()
    ml = ModelLoader(table,w_id)
    model = ml.model
    title = 'BiRNN-DO3-DLE'
    model.name = title

    exporter  = Exporter(model,table)
    exporter.export_states(user_mode='train',trans_mode='train',batch_size=100,val_samples=200,remove_pad=True)
    print 'ALL EXPORTED!'
  