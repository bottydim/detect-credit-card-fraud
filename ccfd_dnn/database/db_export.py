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


data_dir = '/home/botty/Documents/CCFD/data/'
evt_name = 'Featurespace_events_output.csv'
auth_name = 'Featurespace_auths_output.csv'
db_name = 'c1_agg.db'
db_name = 'ccfd.db'


address = 'postgresql://script@localhost:5432/ccfd'
engine = create_engine(address)
connection = engine.raw_connection()
connection.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
cursor = connection.cursor()

drop_qry = '''DROP TABLE IF EXISTS {table}; '''
create_qry = ''' 
CREATE TABLE {table} 
(
  index bigint,
  acct_id text,
  AUTHZN_RQST_PROC_TM timestamp without time zone,
  AUTHZN_APPRL_CD text,
  AUTHZN_AMT double precision,
  MRCH_NM text,
  MRCH_CITY_NM text,
  MRCH_PSTL_CD text,
  MRCH_CNTRY_CD text,
  MRCH_ID text,
  TRMNL_ID text,
  MRCH_CATG_CD text,
  POS_ENTRY_MTHD_CD bigint,
  POS_COND_CD bigint,
  TRMNL_CLASFN_CD bigint,
  TRMNL_CAPBLT_CD bigint,
  TRMNL_PIN_CAPBLT_CD bigint,
  TSYS_DCLN_REAS_CD bigint,
  MRCH_TMP_PRTPN_IND text,
  AUTHZN_MSG_TYPE_MODR_CD text,
  AUTHZN_ACCT_STAT_CD text,
  AUTHZN_MSG_TYPE_CD bigint,
  AUTHZN_RQST_TYPE_CD bigint,
  AUTHZN_RESPNS_CD bigint,
  ACCT_STAT_REAS_NUM bigint,
  RQST_CARD_SEQ_NUM text,
  PIN_OFST_IND bigint,
  PIN_VLDTN_IND text,
  CARD_VFCN_REJ_CD text,
  CARD_VFCN_RESPNS_CD text,
  CARD_VFCN2_RESPNS_CD text,
  CAVV_CD text,
  ECMRC_SCURT_CD text,
  ACQR_BIN_NUM text,
  ACQR_CRCY_CD bigint,
  CRCY_CNVRSN_RT bigint,
  AUTHZN_APPRD_AMT double precision,
  PRIOR_MONEY_AVL_AMT double precision,
  PRIOR_CASH_AVL_AMT double precision,
  ACCT_CL_AMT double precision,
  ACCT_CURR_BAL double precision,
  PREV_ADR_CHNG_DT timestamp without time zone,
  PREV_PMT_DT timestamp without time zone,
  PREV_PMT_AMT double precision,
  PREV_CARD_RQST_DT timestamp without time zone,
  FRD_IND text,
  FRD_IND_SWT_DT timestamp without time zone
);'''


table = 'data_trim'
cursor.execute(drop_qry.format(table=table))
connection.commit()
cursor.execute(create_qry.format(table=table))
connection.commit()
print 'table created'





start = dt.datetime.now()
chunksize = 10000
j = 0
index_start = 1
###################data source
file_loc = data_dir+auth_name
########################
dtFormat = "%d%b%Y %H:%M:%S.%f"
def getTime(x):
    dtString = "{} {}".format(x.AUTHZN_RQST_PROC_DT,x.AUTHZN_RQST_PROC_TM)
    return dt.datetime.strptime(dtString,dtFormat)

for df in pd.read_csv(file_loc, chunksize=chunksize, iterator=True,encoding='ISO-8859-1'):
    
    df = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) # Remove spaces from columns

#     df['AUTHZN_RQST_PROC_DT'] = pd.to_datetime(df['AUTHZN_RQST_PROC_DT'],format='%d%b%Y') # Convert to datetimes
#     df['AUTHZN_RQST_PROC_TM'] = df['AUTHZN_RQST_PROC_DT']+ pd.to_datetime(df.AUTHZN_RQST_PROC_TM).dt.time
    df['acct_id'] = df['acct_id'].astype(str)
    df['AUTHZN_RQST_PROC_TM'] = df.apply(lambda x: getTime(x),1)
    df['AUTHZN_APPRL_CD'] =pd.to_numeric(df['AUTHZN_APPRL_CD'], errors='coerce')
    df['AUTHZN_APPRL_CD'] =df['AUTHZN_APPRL_CD'].astype(str)
    df.MRCH_CNTRY_CD = df.MRCH_CNTRY_CD.astype(str)
    df.MRCH_CATG_CD = df.MRCH_CATG_CD.astype(str)
    df.AUTHZN_MSG_TYPE_MODR_CD = df.AUTHZN_MSG_TYPE_MODR_CD.astype(str)
    df.RQST_CARD_SEQ_NUM = df.RQST_CARD_SEQ_NUM.astype(str)
    df.ECMRC_SCURT_CD = df.ECMRC_SCURT_CD.astype(str)
    df.ACQR_BIN_NUM = df.ACQR_BIN_NUM.astype(str)
    df.PREV_ADR_CHNG_DT =pd.to_datetime(df.PREV_ADR_CHNG_DT,errors='coerce',format='%d%b%Y')
    df.PREV_PMT_DT = pd.to_datetime(df.PREV_PMT_DT,errors='coerce',format='%d%b%Y')
    df.PREV_CARD_RQST_DT = pd.to_datetime(df.PREV_CARD_RQST_DT,errors='coerce',format='%d%b%Y')
    df.FRD_IND_SWT_DT = pd.to_datetime(df.FRD_IND_SWT_DT,errors='coerce',format='%d%b%Y')
#     df['AUTHZN_RQST_PROC_TM'] = pd.to_datetime(df[['AUTHZN_RQST_PROC_DT','AUTHZN_RQST_PROC_TM']],format='%Y%m%d%H')
#     df['AUTHZN_RQST_PROC_TM'] = pd.to_datetime(df.AUTHZN_RQST_PROC_DT.dt.strftime('%Y-%m-%d') +' '+ df.AUTHZN_RQST_PROC_TM.dt.strftime('%H'))
#     df['PREV_ADR_CHNG_DT'] = pd.to_datetime(df['PREV_ADR_CHNG_DT'])
#     df['PREV_PMT_DT'] = pd.to_datetime(df['PREV_PMT_DT'])
#     df['PREV_CARD_RQST_DT'] = pd.to_datetime(df['PREV_CARD_RQST_DT'])
#     df['FRD_IND_SWT_DT'] = pd.to_datetime(df['FRD_IND_SWT_DT'])
    df.index += index_start

    # Remove the un-interesting columns
    columns = ['AUTHZN_RQST_PROC_DT','EXCSV_ACTVY_PARM_CD']

    for c in df.columns:
        if c in columns:
            df = df.drop(c, axis=1)    

    
    j+=1
    
    print '{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j*chunksize)
    
#     display(df)
#     print df.dtypes
    
#     break
#     table = 'data_trim'
    # df.to_sql(table, disk_engine, if_exists='append')
    output = cStringIO.StringIO()
    #ignore the index
    df.to_csv(output, sep='\t', header=False, index=True,encoding='utf-8')
    # print df[df['AUTHZN_RQST_PROC_TM']=='nan']
    #jump to start of stream
    output.seek(0)
    contents = output.getvalue()
    cur = connection.cursor()
    #null values become ''
    t_mid = dt.datetime.now()
    cur.copy_from(output, table, null="", size=200000)    
    connection.commit()
    cur.close()
    
    print '{} seconds: inserted {} rows'.format((dt.datetime.now() - t_mid).seconds, j*chunksize)
    index_start = df.index[-1] + 1
    break

df = pd.read_sql_query('select * from {table} limit 5'.format(table=table),engine)
col_names = df.columns.values
print col_names
for c,name in enumerate(col_names):
    if name =='index':
        continue
    t_mid = dt.datetime.now()
    cursor.execute('''CREATE INDEX id_{table}_{col} 
                ON {table} ({col})'''.format(table=table,col=name))
    connection.commit()
    print '{} index created in {}'.format(name,(dt.datetime.now() - t_mid))
print 'idxs created!'