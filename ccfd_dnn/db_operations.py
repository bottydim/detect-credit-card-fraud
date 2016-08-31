import plotly.tools as tls
import pandas as pd
from sqlalchemy import create_engine # database connection
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


address = 'postgresql://script@localhost:5432/ccfd'
engine = create_engine(address)
connection = engine.raw_connection()
connection.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
cursor = connection.cursor()

table = 'auth'
###get col names
df = pd.read_sql_query('select * from {table} limit 5'.format(table=table),engine)
col_names = df.columns.values
print col_names


#count nan vals
table = 'data_little_enc'
for c,name in enumerate(col_names):
    if name =='index':
        continue
    print name
    qry = '''
    SELECT count(*) FROM {table}
    WHERE {col} = 'NaN'
    '''
    result = cursor.execute(qry.format(table=table,col=name))
    for row in result:
        print row
################CREATE IDXS##################################
# for c,name in enumerate(col_names):
#     if name =='index':
#         continue
#     t_mid = dt.datetime.now()
#     cursor.execute('''CREATE INDEX id_auth_{col} 
#                 ON {table} ({col})'''.format(table=table,col=name))
#     connection.commit()
#     print '{} index created in {}'.format(name,(dt.datetime.now() - t_mid).minutes)
# print 'idxs created!'

# cursor.execute('''CREATE INDEX id_{table}_acct_id_tm
#                 ON {table} (acct_id,AUTHZN_RQST_PROC_TM)'''.format(table=table))
# cursor.execute('''CREATE INDEX id_{table}_tm_frd 
#                 ON {table} (AUTHZN_RQST_PROC_TM,FRD_IND_SWT_DT)'''.format(table=table))
# connection.commit()
# print 'composite indeces created'
###########################################################################

# tbl_evnt = 'event'
# tbl_ath = 'data_trim'
# for c,name in enumerate(col_names):
#     if name =='index':
#         continue
#     t_mid = dt.datetime.now()
#     qry = '''
#     select count(*) as cnt from (select {tbl1}.{col} from {tbl1} where not exists (select {tbl2}.{col}  from {tbl2} where {tbl2}.{col}  = {tbl1}.{col} ) 
#     '''
#     result = cursor.execute(qry.format(tbl1=tbl_evnt,tbl2=tbl_ath,col=name))
#     not_in_auth = result[0]['cnt']
#     print not_in_auth
#     if (not_in_auth>0):
#         message = str(not_in_auth)
#     else:        
#         message = 'OK'
#     print '{} index - {} :created in {}'.format(name,message,(dt.datetime.now() - t_mid).minutes)