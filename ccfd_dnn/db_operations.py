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
import utils
import math

class DB_Operator():

    # def __init__(self, *args, **kwargs):
    def __init__(self,engine):
        self.engine = engine
        self.connection = engine.raw_connection()
        self.connection.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
        self.cursor = self.connection.cursor()



    def get_columns(self,tbl):
        engine = self.engine
        df = pd.read_sql_query('select * from {table} limit 1'.format(table=table),engine)
        col_names = df.columns.values
        return col_names


    def cf_db_uniq(self,t1,t2):

        col_names = self.get_columns(t1)
        assert utils.list_equal(col_names,self.get_columns(t2)),'Tables have different structure'


        for c,name in enumerate(col_names):
            if name =='index':
                continue
            t_mid = dt.datetime.now()
            # qry = '''
            # select count(*) as cnt from (select {tbl1}.{col} from {tbl1} where not exists (select {tbl2}.{col}  from {tbl2} where {tbl2}.{col}  = {tbl1}.{col} ) 
            # '''
            # result = cursor.execute(qry.format(tbl1=t1,tbl2=t2,col=name))

            qry ='select count(distinct {col}) as cnt from {table}'
            cursor.execute(qry.format(table=t1,col=name))
            result_t1 = cursor.fetchall()
            cursor.execute(qry.format(table=t2,col=name))
            result_t2 = cursor.fetchall()
            # print qry.format(table=t1,col=name)
            # print qry.format(table=t2,col=name)

            print result_t1
            print result_t2
            col_dif = abs(int(result_t1[0][0]) - int(result_t2[0][0]))
            if (col_dif>0):
                message = '{t1} - {r1} <<<<<>>>>> {t2} - {r2}'.format(t1=t1,t2=t2,r1=result_t1[0][0],r2=result_t2[0][0])
            else:        
                message = 'OK'
            print '{} ===== {} - {}'.format(name,message,utils.days_hours_minutes_seconds(dt.datetime.now() - t_mid))


##############################CREATE COMPOSITE IDXS####################################
    def create_idx_comp(self,table):
        cursor = self.cursor
        cursor.execute('''CREATE INDEX id_{table}_acct_id_tm
                        ON {table} (acct_id,AUTHZN_RQST_PROC_TM)'''.format(table=table))
        cursor.execute('''CREATE INDEX id_{table}_tm_frd 
                        ON {table} (AUTHZN_RQST_PROC_TM,FRD_IND_SWT_DT)'''.format(table=table))
        connection.commit()
        print 'composite indeces created'



if __name__ == "__main__":
    address = 'postgresql://script@localhost:5432/ccfd'
    engine = create_engine(address)
    connection = engine.raw_connection()
    connection.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
    cursor = connection.cursor()

    table = 'auth_enc'
    table_2 = 'data_little_enc'

    db_ops = DB_Operator(engine)
    db_ops.cf_db_uniq(table,table_2)

# ###get col names
# df = pd.read_sql_query('select * from {table} limit 5'.format(table=table),engine)
# col_names = df.columns.values
# print col_names


# #count nan vals
# table = 'data_little_enc'
# for c,name in enumerate(col_names):
#     if name =='index':
#         continue
#     print name
#     qry = '''
#     SELECT count(*) FROM {table}
#     WHERE {col} = 'NaN'
#     '''
#     result = cursor.execute(qry.format(table=table,col=name))
#     for row in result:
#         print row
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




####count difference##############
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






###########################################################
###########################################################
################CREATE IDXS##################################


# df = pd.read_sql_query('select * from {table} limit 5'.format(table=table),engine)
# col_names = df.columns.values
# print col_names
# for c,name in enumerate(col_names):
#     if name =='index':
#         continue
#     t_mid = dt.datetime.now()
#     cursor.execute('''CREATE INDEX IF NOT EXISTS id_{table}_{col} 
#                 ON {table} ({col})'''.format(table=table,col=name))
#     connection.commit()
#     print '{} index created in {}'.format(name,(dt.datetime.now() - t_mid))
# print 'idxs created!'



# df = pd.read_sql_query('select * from {table} limit 5'.format(table=table),engine)
# col_names = df.columns.values
# print col_names







###########################################################################



