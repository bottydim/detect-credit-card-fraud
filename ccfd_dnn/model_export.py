import h5py
import numpy as np
from model import *




class Exporter(ModelOperator):
	export_dist = '/home/botty/Documents/CCFD/data/models/export/'
	def __init__(self, *args, **kwargs):
		ModelOperator.__init__(self, *args, **kwargs)

if __name__ == "__main__":