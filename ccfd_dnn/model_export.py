import h5py
import numpy as np
from model import *




class Exporter(ModelOperator):
	export_dist = '/home/botty/Documents/CCFD/data/models/export/'
	def __init__(self, *args, **kwargs):
		ModelOperator.__init__(self, *args, **kwargs)


	def create_group(f,name):
		grp = f.create_group(name)
		return grp
	def create_DS(f,dset_name,shape,dtype='f',group=None):
		if group==None:
			dset = f.create_dataset(dset_name,shape, dtype=dtype)
		else:
			dset = group.create_dataset(dset_name,shape, dtype=dtype)
		return dset
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

    file_name ="{dir}/{title}.hdf5".format(title=title,dir=Exporter.export_dist)
    f = h5py.File(file_name, "w")

    dset_name = "layer"
    shape = (100,)
    dtype = 'i'
    
    print 