import sys
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
from model import *
import psycopg2 as pg
#load python script that batch loads pandas df to sql
import cStringIO
import time
import re
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
  acct_id bigint,
  AUTHZN_RQST_PROC_TM bigint,
  AUTHZN_APPRL_CD bigint,
  AUTHZN_AMT double precision,
  MRCH_NM bigint,
  MRCH_CITY_NM bigint,
  MRCH_PSTL_CD bigint,
  MRCH_CNTRY_CD bigint,
  MRCH_ID bigint,
  TRMNL_ID bigint,
  MRCH_CATG_CD bigint,
  POS_ENTRY_MTHD_CD bigint,
  POS_COND_CD bigint,
  TRMNL_CLASFN_CD bigint,
  TRMNL_CAPBLT_CD bigint,
  TRMNL_PIN_CAPBLT_CD bigint,
  TSYS_DCLN_REAS_CD bigint,
  MRCH_TMP_PRTPN_IND bigint,
  AUTHZN_MSG_TYPE_MODR_CD bigint,
  AUTHZN_ACCT_STAT_CD bigint,
  AUTHZN_MSG_TYPE_CD bigint,
  AUTHZN_RQST_TYPE_CD bigint,
  AUTHZN_RESPNS_CD bigint,
  ACCT_STAT_REAS_NUM bigint,
  RQST_CARD_SEQ_NUM bigint,
  PIN_OFST_IND bigint,
  PIN_VLDTN_IND bigint,
  CARD_VFCN_REJ_CD bigint,
  CARD_VFCN_RESPNS_CD bigint,
  CARD_VFCN2_RESPNS_CD bigint,
  CAVV_CD bigint,
  ECMRC_SCURT_CD bigint,
  ACQR_BIN_NUM bigint,
  ACQR_CRCY_CD bigint,
  CRCY_CNVRSN_RT bigint,
  AUTHZN_APPRD_AMT double precision,
  PRIOR_MONEY_AVL_AMT double precision,
  PRIOR_CASH_AVL_AMT double precision,
  ACCT_CL_AMT double precision,
  ACCT_CURR_BAL double precision,
  PREV_ADR_CHNG_DT bigint,
  PREV_PMT_DT bigint,
  PREV_PMT_AMT double precision,
  PREV_CARD_RQST_DT bigint,
  FRD_IND bigint,
  FRD_IND_SWT_DT bigint
);'''


table = 'auth_enc'
cursor.execute(drop_qry.format(table=table))
connection.commit()
cursor.execute(create_qry.format(table=table))
connection.commit()
print 'table created'



encoders = load_encoders()



col_id = 'mrch_pstl_cd'
print len(encoders[col_id].classes_)
print encoders[col_id].classes_[0:20]
# print encoders[col_id].classes_[0:20]
# encoders[col_id].classes_ = [x.encode('utf-8') if x !=None else None for x in encoders[col_id].classes_]
encoders[col_id].classes_ = [str(x).replace('"','') if x !=None else None for x in encoders[col_id].classes_]

col_id = 'acqr_bin_num'
# print len(encoders[col_id].classes_)
# print encoders[col_id].classes_[0:20]
encoders[col_id].classes_ = [x.encode('utf-8') if x !=None else None for x in encoders[col_id].classes_]
encoders[col_id].classes_ = [str(x).replace('.0','') if x !=None else None for x in encoders[col_id].classes_]
# print encoders[col_id].classes_[0:20]
# sys.exit()

col_id = 'mrch_nm'
print encoders[col_id].classes_
# print np.nan in encoders['mrch_pstl_cd'].classes_
encoders[col_id].classes_ = [x.encode('utf-8') if x !=None else None for x in encoders[col_id].classes_]
encoders[col_id].classes_ = [str(x).replace('"','') if x !=None else None for x in encoders[col_id].classes_]

col_id = 'trmnl_id'
print encoders[col_id].classes_
# print np.nan in encoders['mrch_pstl_cd'].classes_
# encoders[col_id].classes_ = [x.encode('utf-8') if x !=None else None for x in encoders[col_id].classes_]
encoders[col_id].classes_ = [str(x).replace('"','') if x !=None else None for x in encoders[col_id].classes_]


col_id = 'mrch_city_nm'
print encoders[col_id].classes_
# print np.nan in encoders['mrch_pstl_cd'].classes_
encoders[col_id].classes_ = [x.encode('utf-8') if x !=None else None for x in encoders[col_id].classes_]
encoders[col_id].classes_ = [str(x).replace('"','') if x !=None else None for x in encoders[col_id].classes_]


tadex_orig = 'TADEX SKLEP RTV-AGD POM'
tadex_bug = '"""TADEX"" SKLEP RTV-AGD POM"'
# print encoders[col_id].classes_
print '"NATION OF SHOPKEEPERS' in encoders[col_id].classes_


for x in encoders[col_id].classes_:

    if re.search(r'.*TADEX.*',str(x)):
        print x

#####config vars#####
na_val_cc = encoders['cavv_cd'].classes_[1]
na_val = encoders['mrch_pstl_cd'].classes_[1]
##########


# sys.exit()
start = dt.datetime.now()
chunksize = 500000
copy_block = 500000
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
    

    t0 = time.time()
    df = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) # Remove spaces from columns

#     df['AUTHZN_RQST_PROC_DT'] = pd.to_datetime(df['AUTHZN_RQST_PROC_DT'],format='%d%b%Y') # Convert to datetimes
#     df['AUTHZN_RQST_PROC_TM'] = df['AUTHZN_RQST_PROC_DT']+ pd.to_datetime(df.AUTHZN_RQST_PROC_TM).dt.time
    df['acct_id'] = df['acct_id'].astype(str)
    df['AUTHZN_RQST_PROC_TM'] = df.apply(lambda x: getTime(x),1)
    df['AUTHZN_APPRL_CD'] =pd.to_numeric(df['AUTHZN_APPRL_CD'], errors='coerce')
    df['AUTHZN_APPRL_CD'] =df['AUTHZN_APPRL_CD'].astype(str)
    df['cavv_cd'.upper()] = df['cavv_cd'.upper()].astype(str)
    # df['mrch_nm'.upper()] = df['mrch_nm'.upper()].astype(str)
    df['mrch_nm'.upper()] = df['mrch_nm'.upper()].str.encode('utf-8')
    df['mrch_nm'.upper()] = df['mrch_nm'.upper()].replace({'\"':''},regex=True)

    # df['mrch_nm'.upper()] = df['mrch_nm'.upper()].replace({'\"':''},regex=True)

    col_id = 'trmnl_id'
    df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
    df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)

    col_id = 'mrch_city_nm'
    df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
    df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)

    col_id = 'mrch_pstl_cd'
    df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
    df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)

    col_id = 'acqr_bin_num'
    df[col_id.upper()] = df[col_id.upper()].astype(str)
    df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
    df[col_id.upper()] = df[col_id.upper()].replace({'\.0':''},regex=True)
    # df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)
    # a = df['mrch_nm'.upper()] 
    # print a.dtype
    # a.index = range(len(a))
    # print a[a.str.contains(r".*NATION OF.*",na=False)]

    # print df['mrch_nm'.upper()]
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

    t1 = time.time()
    print 'Preprocessing operations:', str(t1-t0)  
    j+=1

    print '{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j*chunksize)
    
    t0 = time.time()
    df.columns = [x.lower() for x in df.columns]
    df.fillna(value=na_val,inplace=True)

    rpl_dict = {'cavv_cd':{'nan':None,na_val:None},
                'mrch_nm':{na_val:None},
                'trmnl_id':{na_val:None},
                'card_vfcn_respns_cd':{na_val:None},
                'pin_vldtn_ind':{na_val:None},
                'mrch_city_nm':{na_val:None},
                'authzn_acct_stat_cd':{na_val:None},
                'card_vfcn_rej_cd':{na_val:None},
                'mrch_id':{na_val:None},
                'card_vfcn2_respns_cd':{na_val:None},
                'mrch_pstl_cd':{na_val:None},
                'trmnl_id':{na_val:None},

                # 'mrch_pstl_cd':{'':None},
                # 'mrch_pstl_cd':{'[]':None},
                # 'mrch_pstl_cd':{'[0]':None},

    }
    df.replace(rpl_dict,inplace=True)
    
    t1 = time.time()
    print 'time colums and na fil:', str(t1-t0)  

    # if j==2:
    #     classes = np.unique(df['mrch_pstl_cd'])
    #     other = encoders['mrch_pstl_cd'].classes_
    #     # print set(encoders['mrch_pstl_cd'].classes_).difference(classes)
    #     print np.setdiff1d(classes, np.intersect1d(classes, other))
    #     print len(np.intersect1d(classes, other))
    #     print len(classes)
    #     print 'official:',len(np.intersect1d(classes, other)) < len(classes)
    #     print 'SUBSET',set(df['mrch_pstl_cd']) <= set(encoders['mrch_pstl_cd'].classes_)
    #     diff = set(df['mrch_pstl_cd']).difference(set(encoders['mrch_pstl_cd'].classes_))
    #     print len(diff)
    #     print 'len of column',len(df['mrch_pstl_cd'])
    #     print 'len & ', len(set(df['mrch_pstl_cd']) & set(encoders['mrch_pstl_cd'].classes_))
    #     diff = np.setdiff1d(df['mrch_pstl_cd'],encoders['mrch_pstl_cd'].classes_)
    #     print len(diff)
    #     print len(set(df['mrch_pstl_cd']))
    #     # for i in diff:
    #     #     print type(i)
    #     # sys.exit()
    t0 = time.time()
    encode_df(df,encoders)
    t1 = time.time()
    print 'time encoding:', str(t1-t0)  
#     display(df)
#     print df.dtypes
    
#     break
#     table = 'data_trim'


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
    cur.copy_from(output, table, null="", size=copy_block)    
    connection.commit()
    cur.close()
    
    print '{} seconds: inserted {} rows'.format((dt.datetime.now() - t_mid).seconds, j*chunksize)
    index_start = df.index[-1] + 1
    # break

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



df = pd.read_sql_query('select * from {table} limit 5'.format(table=table),engine)
col_names = df.columns.values
print col_names



##############################CREATE COMPOSITE IDXS####################################

cursor.execute('''CREATE INDEX id_{table}_acct_id_tm
                ON {table} (acct_id,AUTHZN_RQST_PROC_TM)'''.format(table=table))
cursor.execute('''CREATE INDEX id_{table}_tm_frd 
                ON {table} (AUTHZN_RQST_PROC_TM,FRD_IND_SWT_DT)'''.format(table=table))
connection.commit()
print 'composite indeces created'




###########################################################################