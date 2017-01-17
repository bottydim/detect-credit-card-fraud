import h5py
import numpy as np
from model import *
import keras
import argparse
from model_eval import *

def encode_ascii(x):
    return x.encode("ascii", "ignore") 

class Exporter(ModelOperator):


    export_dist = '/home/botty/Documents/CCFD/data/models/export'

    def __init__(self, *args, **kwargs):
        ModelOperator.__init__(self, *args, **kwargs)



    def create_file(self,name):
        
        dir_name = '{dir}/{title}/'.format(title=self.model.name,dir=Exporter.export_dist)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name ="{dir_name}{name}.hdf5".format(dir_name=dir_name,name=name)

        f = h5py.File(file_name, "x")
        return f



    @classmethod
    def create_ds(self,f,dset_name,arr,group=None,dtype=None):

        # print dset_name

        shape = arr.shape
        if dtype ==None:
            dtype = arr.dtype
        else:
            # arr = arr.astype(dtype)
            # arr = np.vectorize(encode_ascii)(arr)
            arr = np.array(map(encode_ascii,arr))
        if group==None:
            dset = f.create_dataset(dset_name,shape, dtype=dtype,data=arr)
        else:
            dset = group.create_dataset(dset_name,shape, dtype=dtype,data=arr)
        print 'DATASET {} {}#!=0 @{}'.format(dset_name,np.count_nonzero(dset),dtype)
        return dset



    @classmethod
    def create_group(self,f,name):
        grp = f.create_group(name)
        return grp


    def create_DS(self,f,dset_name,arr,group=None):
        shape = arr.shape
        dtype = arr.dtype
        if group==None:
            dset = f.create_dataset(dset_name,shape, dtype=dtype,data=arr)
        else:
            dset = group.create_dataset(dset_name,shape, dtype=dtype,data=arr)
        print 'DATASET {} {}!=0@{}'.format(dset_name,np.count_nonzero(dset),dtype)
        return dset
    def export_states(self,user_mode='train',trans_mode='train',batch_size=2000,val_samples=None,remove_pad=False):
        model = self.model
        inter_models = []
        for layer in model.layers:
            if type(layer) == keras.engine.topology.Merge:
                # print layer.__dict__
                inter_models.append(Model(input=model.inputs, output=layer.output))

        inter_models.append(model)
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
        
        (x_exp, y_exp, y_hat_dic) = export_generator(inter_models, generator, val_samples, max_q_size=10,remove_pad=remove_pad)
        print 'non-zero x', np.count_nonzero(x_exp)
        print 'non-zero y', np.count_nonzero(y_exp)
        print 'states acquired!'
        f_train = self.create_file('train')
        self.create_DS(f_train,'features',x_exp)
        self.create_DS(f_train,'labels',y_exp)
        # f_train.flush()
        f_train.close()
        f_state = self.create_file('states')
        for c,y_hat in enumerate(y_hat_dic):
            print 'non-zero ws', np.count_nonzero(y_hat)
            self.create_DS(f_state,'layer_{}'.format(c),y_hat)
        # f_state.flush()
        f_state.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Model Exporter')
    parser.add_argument('-t','--table',required=True)
    parser.add_argument('-i','--id',default=1)
    parser.add_argument('-ns','--num_samples',default=600)
    args = parser.parse_args()

    ####################################DATA SOURCE################################
    var_args = vars(args)
    table = var_args['table']
    w_id = int(var_args['id'])
    num_samples = int(var_args['num_samples'])
    #########################################

    disk_engine = get_engine()
    ml = ModelLoader(table,w_id)
    model = ml.model
    
    title = 'birnn-do3-dle-{}'.format(num_samples)
    model.name = title

    print 'Commencing export...'
    exporter  = Exporter(model,table)
    exporter.export_states(user_mode='train',trans_mode='train',batch_size=200,val_samples=num_samples,remove_pad=True)
    print 'ALL EXPORTED!'
  