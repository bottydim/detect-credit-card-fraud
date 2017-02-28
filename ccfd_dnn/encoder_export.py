import argparse

import h5py
import numpy as np

from ccfd_dnn.model.model_export import Exporter
from db_operations import DbOperator
from model import load_encoders

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Encoder Exporter')
    parser.add_argument('-t','--table',required=True)
    parser.add_argument('-n','--model_name',required=True)
    args = parser.parse_args()

    ####################################DATA SOURCE################################
    # table = 'data_little_enc'
    var_args = vars(args)
    table = var_args['table']
    save_dir = var_args['model_name']
    address = 'postgresql://script@localhost:5432/ccfd'
    db_ops = DbOperator(address=address)
    col_names = list(db_ops.get_columns(table))
    encoders = load_encoders()
    # encoder_mapper = {}
    # encoder_mapper['columns'] = col_names
    # for c in encoders.keys():
    #     encoder_mapper[c] = encoders[c]._classes
        
    path = Exporter.export_dist+'/'+save_dir+'/'+'encoders.h5'
    # save_object(path)
    f = h5py.File(path, "x")
    col_names.remove('index') #!!!!!!
    skip_list = ['frd_ind']
    for x in skip_list:
        col_names.remove(x)
    arr = np.array(col_names)
    arr = np.char.array(col_names)
    dset_name = 'columns'
    col_names_ds = Exporter.create_ds(f,dset_name,arr,dtype='S50')

    group_name = 'encoders'
    group = Exporter.create_group(f,group_name)
    for dset_name in encoders.keys():
        if dset_name in skip_list:
            continue
        arr = encoders[dset_name].classes_
        arr = map(lambda x: x if x!=None else 'N/A',arr)
        arr = np.asarray(arr)
        print arr

        Exporter.create_ds(f,dset_name,arr,group=group,dtype='S50')
    f.close()