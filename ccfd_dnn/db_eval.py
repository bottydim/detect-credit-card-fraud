
import pandas as pd
from sqlalchemy import create_engine  # database connection
import datetime as dt
from IPython.display import display

import plotly.plotly as py # interactive graphing
import plotly.graph_objs as go
from plotly.graph_objs import Bar, Scatter, Marker, Layout, Histogram, Box
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import psycopg2 as pg
#load python script that batch loads pandas df to sql
import cStringIO
import time

if __name__ == '__main__':

    address = 'postgresql://script@localhost:5432/ccfd'
    engine = create_engine(address)
    connection = engine.raw_connection()
    connection.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
    cursor = connection.cursor()

    table = 'data_little_new'

    df = pd.read_sql_query('select * from {table} limit 5'.format(table=table),engine)
    col_names = df.columns.values

    print 'count all rows'
    query = '''SELECT COUNT(*) FROM {table}'''.format(table=table)
    ######################################
    t0 = time.time()
    cursor.execute(query)
    t1 = time.time()
    print 'time taken:'+str(t1-t0)
    ######################################

    print 'PANDAS'
    ######################################
    t0= time.time()
    pd.read_sql_query(query,engine)
    t1 = time.time()
    print 'time taken:'+str(t1-t0)
    ######################################


    print 'count each column'
    ######################################
    # for name in col_names:
    #     print name
    #     t0 = time.time()
    #     cursor.execute('''SELECT COUNT({col}) FROM {table}'''.format(table=table,col=name))
    #     t1 = time.time()
    #     print 'time taken:'+str(t1-t0)
    ######################################

    print 'random user'
    ######################################
    user = '"."'
    user = 0
    t0= time.time()
    cursor.execute('select * from {table} where acct_id = {user}'.format(table=table,user=user))
    t1 = time.time()
    print 'time taken:'+str(t1-t0)
    ######################################


    print 'cutoff-date'
    ######################################
    cutt_off_date = '2014-05-11'
    t0= time.time()
    qry = '''select AUTHZN_RQST_PROC_TM
        from {table}
        where FRD_IND_SWT_DT >=
             "{cutt_off_date}"
        order by AUTHZN_RQST_PROC_TM limit 1 '''.format(table=table, cutt_off_date=cutt_off_date)
    cursor.execute(qry)
    t1 = time.time()
    print 'time taken:'+str(t1-t0)
    ######################################
