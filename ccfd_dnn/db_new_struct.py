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
event_name = 'Featurespace_events_output.csv'
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
  AUTHZN_RQST_PROC_TM_hour smallint,
  AUTHZN_RQST_PROC_TM_minute smallint,
  AUTHZN_RQST_PROC_TM_second smallint,
  AUTHZN_RQST_PROC_TM_microsecond smallint,
  AUTHZN_RQST_PROC_DT_year smallint,
  AUTHZN_RQST_PROC_DT_month smallint,
  AUTHZN_RQST_PROC_DT_day smallint,
  AUTHZN_RQST_PROC_DT_dayofweek smallint,
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
  PREV_ADR_CHNG_DT_year smallint,
  PREV_ADR_CHNG_DT_month smallint,
  PREV_ADR_CHNG_DT_day smallint,
  PREV_ADR_CHNG_DT_dayofweek smallint,
  PREV_PMT_DT bigint,
  PREV_PMT_DT_year smallint,
  PREV_PMT_DT_month smallint,
  PREV_PMT_DT_day smallint,
  PREV_PMT_DT_dayofweek smallint,
  PREV_PMT_AMT double precision,
  PREV_CARD_RQST_DT bigint,
  PREV_CARD_RQST_DT_year smallint,
  PREV_CARD_RQST_DT_month smallint,
  PREV_CARD_RQST_DT_day smallint,
  PREV_CARD_RQST_DT_dayofweek smallint,
  FRD_IND bigint,
  FRD_IND_SWT_DT bigint,
  FRD_IND_SWT_DT_year smallint,
  FRD_IND_SWT_DT_month smallint,
  FRD_IND_SWT_DT_day smallint,
  FRD_IND_SWT_DT_dayofweek smallint
);'''

# table = 'auth_enc_v1'
table = 'auth_event'
cursor.execute(drop_qry.format(table=table))
connection.commit()
cursor.execute(create_qry.format(table=table))
connection.commit()
print 'table created'



encoders = load_encoders()


# for key in encoders.keys():
#   col_id = key
#   print col_id
#   print len(encoders[col_id].classes_)
#   print encoders[col_id].classes_[0:20]




col_names_order = ['acct_id', 'authzn_rqst_proc_tm', 'authzn_rqst_proc_tm_hour', 'authzn_rqst_proc_tm_minute', 'authzn_rqst_proc_tm_second', 'authzn_rqst_proc_tm_microsecond', 'authzn_rqst_proc_dt_year', 'authzn_rqst_proc_dt_month', 'authzn_rqst_proc_dt_day', 'authzn_rqst_proc_dt_dayofweek', 'authzn_apprl_cd', 'authzn_amt', 'mrch_nm', 'mrch_city_nm', 'mrch_pstl_cd', 'mrch_cntry_cd', 'mrch_id', 'trmnl_id', 'mrch_catg_cd', 'pos_entry_mthd_cd', 'pos_cond_cd', 'trmnl_clasfn_cd', 'trmnl_capblt_cd', 'trmnl_pin_capblt_cd', 'tsys_dcln_reas_cd', 'mrch_tmp_prtpn_ind', 'authzn_msg_type_modr_cd', 'authzn_acct_stat_cd', 'authzn_msg_type_cd', 'authzn_rqst_type_cd', 'authzn_respns_cd', 'acct_stat_reas_num', 'rqst_card_seq_num', 'pin_ofst_ind', 'pin_vldtn_ind', 'card_vfcn_rej_cd', 'card_vfcn_respns_cd', 'card_vfcn2_respns_cd', 'cavv_cd', 'ecmrc_scurt_cd', 'acqr_bin_num', 'acqr_crcy_cd', 'crcy_cnvrsn_rt', 'authzn_apprd_amt', 'prior_money_avl_amt', 'prior_cash_avl_amt', 'acct_cl_amt', 'acct_curr_bal', 'prev_adr_chng_dt', 'prev_adr_chng_dt_year', 'prev_adr_chng_dt_month', 'prev_adr_chng_dt_day', 'prev_adr_chng_dt_dayofweek', 'prev_pmt_dt', 'prev_pmt_dt_year', 'prev_pmt_dt_month', 'prev_pmt_dt_day', 'prev_pmt_dt_dayofweek', 'prev_pmt_amt', 'prev_card_rqst_dt', 'prev_card_rqst_dt_year', 'prev_card_rqst_dt_month', 'prev_card_rqst_dt_day', 'prev_card_rqst_dt_dayofweek', 'frd_ind', 'frd_ind_swt_dt', 'frd_ind_swt_dt_year', 'frd_ind_swt_dt_month', 'frd_ind_swt_dt_day', 'frd_ind_swt_dt_dayofweek'] 
rm_0_cols = ['acqr_bin_num','trmnl_id','mrch_tmp_prtpn_ind','rqst_card_seq_num','authzn_apprl_cd','authzn_msg_type_modr_cd','mrch_cntry_cd','mrch_catg_cd']
col_with_enc = ['acqr_bin_num','mrch_nm','mrch_id','mrch_city_nm']
for col_id in rm_0_cols:
  print col_id
  print len(encoders[col_id].classes_)
  print encoders[col_id].classes_[0:20]
  if col_id in col_with_enc:
    encoders[col_id].classes_ = [x.encode('utf-8') if x !=None else None for x in encoders[col_id].classes_]
    print 'encoded'
  encoders[col_id].classes_ = [str(x).replace('.0','') if x !=None else None for x in encoders[col_id].classes_]
  print len(encoders[col_id].classes_)
  #problem is that sometimes encoding is needed other time it is not

rm_quot_cols = ['mrch_pstl_cd','mrch_nm','trmnl_id','mrch_id','mrch_city_nm']
for col_id in rm_quot_cols:
  print col_id
  print len(encoders[col_id].classes_)
  print encoders[col_id].classes_[0:20]
  if col_id in col_with_enc:
    encoders[col_id].classes_ = [x.encode('utf-8') if x !=None else None for x in encoders[col_id].classes_]
    print 'encoded'
  encoders[col_id].classes_ = [str(x).replace('"','') if x !=None else None for x in encoders[col_id].classes_]
  print len(encoders[col_id].classes_)




col_id = 'mrch_city_nm'
print len(encoders[col_id].classes_)
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

    time_col = 'AUTHZN_RQST_PROC_TM'
    temp = pd.DatetimeIndex(df[time_col])
    df[time_col+'_hour'] = temp.hour
    df[time_col+'_minute'] = temp.minute
    df[time_col+'_second'] = temp.second
    df[time_col+'_microsecond'] = temp.microsecond
    df['AUTHZN_RQST_PROC_TM'] = df.apply(lambda x: getTime(x),1)
    df['AUTHZN_APPRL_CD'] =pd.to_numeric(df['AUTHZN_APPRL_CD'], errors='coerce')
    df['AUTHZN_APPRL_CD'] =df['AUTHZN_APPRL_CD'].astype(str)
    df['cavv_cd'.upper()] = df['cavv_cd'.upper()].astype(str)
    # df['mrch_nm'.upper()] = df['mrch_nm'.upper()].astype(str)



    # df['mrch_nm'.upper()] = df['mrch_nm'.upper()].str.encode('utf-8')
    # df['mrch_nm'.upper()] = df['mrch_nm'.upper()].replace({'\"':''},regex=True)

    # df['mrch_nm'.upper()] = df['mrch_nm'.upper()].replace({'\"':''},regex=True)
    for col_id in rm_quot_cols:
      print col_id
      df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
      df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)    

    # col_id = 'trmnl_id'
    # df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
    # df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)

    # col_id = 'mrch_city_nm'
    # df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
    # df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)

    # col_id = 'mrch_pstl_cd'
    # df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
    # df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)

  
   
    for col_id in rm_0_cols:
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
    date_cols = ['PREV_ADR_CHNG_DT','PREV_PMT_DT','PREV_CARD_RQST_DT','FRD_IND_SWT_DT','AUTHZN_RQST_PROC_DT']
    components = ['year','month','day','dayofweek']
    for col in date_cols:
      tc = df[col]
      tc = pd.to_datetime(tc,errors='coerce',format='%d%b%Y')
      temp = pd.DatetimeIndex(tc)
      df[col+'_year'] = temp.year
      # df[col+'_year'] = pd.to_numeric(df[col+'_year'],errors='coerce')
      # 
      # df.loc[df[col+'_year']=='nan',df[col+'_year']] = None
      df[col+'_month'] = temp.month
      # df[col+'_month'] = pd.to_numeric(df[col+'_month'],errors='coerce')
      df[col+'_day'] = temp.day
      # df[col+'_day'] = pd.to_numeric(df[col+'_day'],errors='coerce')
      df[col+'_dayofweek'] = temp.dayofweek
      # df[col+'_dayofweek'] = pd.to_numeric(df[col+'_dayofweek'],errors='coerce')
    # df.PREV_ADR_CHNG_DT =pd.to_datetime(df.PREV_ADR_CHNG_DT,errors='coerce',format='%d%b%Y')
    # df.PREV_PMT_DT = pd.to_datetime(df.PREV_PMT_DT,errors='coerce',format='%d%b%Y')
    # df.PREV_CARD_RQST_DT = pd.to_datetime(df.PREV_CARD_RQST_DT,errors='coerce',format='%d%b%Y')
    # df.FRD_IND_SWT_DT = 
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
                # 'cavv_cd':{'nan':encoders['cavv_cd'].transform(None),na_val:encoders['cavv_cd'].transform(None)},
                # 'mrch_pstl_cd':{'':None},
                # 'mrch_pstl_cd':{'[]':None},
                # 'mrch_pstl_cd':{'[0]':None},

    }


    # print df.ix[0:500,'prev_adr_chng_dt_year']
    for dc in date_cols:
      for comp in components:
        col_name = dc.lower()+'_'+comp
        # rpl_dict[col_name] = {np.nan:None}
        nan_rm = lambda x: -1 if str(x)=='nan' else x
        df[col_name] = df[col_name].apply(nan_rm)
        # if dc == 'PREV_PMT_DT': 
        df[col_name] = df[col_name].astype(int)

        # df[col_name] = df[col_name].map({'nan': None,})
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
    ######REORDER!!!!!!!!! 
    #THIS IS OF PARAMOUNT IMPORTANCE!!!!
    df = df[col_names_order]




    # print df.ix[0:500,'prev_adr_chng_dt_year']
    # print df.ix[499,'prev_adr_chng_dt_year']
    # print type(df.ix[499,'prev_adr_chng_dt_year'])
    #ignore the index
    df.to_csv(output, sep='\t', header=False, index=True,encoding='utf-8')
    # print df[df['AUTHZN_RQST_PROC_TM']=='nan']
    #jump to start of stream
    output.seek(0)
    contents = output.getvalue()
    cur = connection.cursor()
    t_mid = dt.datetime.now()
    #null values become ''
    cur.copy_from(output, table, null="", size=copy_block)    
    connection.commit()
    cur.close()
    
    print '{} seconds: inserted {} rows'.format((dt.datetime.now() - t_mid).seconds, j*chunksize)
    index_start = df.index[-1] + 1
    # break


print '##############EVENTS'
chunksize = 500000
file_loc = data_dir+event_name
########################

for df in pd.read_csv(file_loc, chunksize=chunksize, iterator=True,encoding='ISO-8859-1'):
    

    t0 = time.time()
    df = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) # Remove spaces from columns

#     df['AUTHZN_RQST_PROC_DT'] = pd.to_datetime(df['AUTHZN_RQST_PROC_DT'],format='%d%b%Y') # Convert to datetimes
#     df['AUTHZN_RQST_PROC_TM'] = df['AUTHZN_RQST_PROC_DT']+ pd.to_datetime(df.AUTHZN_RQST_PROC_TM).dt.time
    df['acct_id'] = df['acct_id'].astype(str)

    time_col = 'AUTHZN_RQST_PROC_TM'
    temp = pd.DatetimeIndex(df[time_col])
    df[time_col+'_hour'] = temp.hour
    df[time_col+'_minute'] = temp.minute
    df[time_col+'_second'] = temp.second
    df[time_col+'_microsecond'] = temp.microsecond
    df['AUTHZN_RQST_PROC_TM'] = df.apply(lambda x: getTime(x),1)
    df['AUTHZN_APPRL_CD'] =pd.to_numeric(df['AUTHZN_APPRL_CD'], errors='coerce')
    df['AUTHZN_APPRL_CD'] =df['AUTHZN_APPRL_CD'].astype(str)

    col_to_str_lc = ['cavv_cd']
    col_to_str_lc.extend(rm_quot_cols)
    col_to_str = [x.upper() for x in col_to_str_lc]
    for col in col_to_str:
      df[col] = df[col].astype(str)
    # df['mrch_nm'.upper()] = df['mrch_nm'.upper()].astype(str)



    # df['mrch_nm'.upper()] = df['mrch_nm'.upper()].str.encode('utf-8')
    # df['mrch_nm'.upper()] = df['mrch_nm'.upper()].replace({'\"':''},regex=True)

    # df['mrch_nm'.upper()] = df['mrch_nm'.upper()].replace({'\"':''},regex=True)
    for col_id in rm_quot_cols:
      print col_id  
      df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
      df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)    

    # col_id = 'trmnl_id'
    # df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
    # df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)

    # col_id = 'mrch_city_nm'
    # df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
    # df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)

    # col_id = 'mrch_pstl_cd'
    # df[col_id.upper()] = df[col_id.upper()].str.encode('utf-8')
    # df[col_id.upper()] = df[col_id.upper()].replace({'\"':''},regex=True)

  
   
    for col_id in rm_0_cols:
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
    col_id = 'mrch_pstl_cd'
    df[col_id.upper()] = df[col_id.upper()].astype(str)


    date_cols = ['PREV_ADR_CHNG_DT','PREV_PMT_DT','PREV_CARD_RQST_DT','FRD_IND_SWT_DT','AUTHZN_RQST_PROC_DT']
    components = ['year','month','day','dayofweek']
    for col in date_cols:
      tc = df[col]
      tc = pd.to_datetime(tc,errors='coerce',format='%d%b%Y')
      temp = pd.DatetimeIndex(tc)
      df[col+'_year'] = temp.year
      # df[col+'_year'] = pd.to_numeric(df[col+'_year'],errors='coerce')
      # 
      # df.loc[df[col+'_year']=='nan',df[col+'_year']] = None
      df[col+'_month'] = temp.month
      # df[col+'_month'] = pd.to_numeric(df[col+'_month'],errors='coerce')
      df[col+'_day'] = temp.day
      # df[col+'_day'] = pd.to_numeric(df[col+'_day'],errors='coerce')
      df[col+'_dayofweek'] = temp.dayofweek
      # df[col+'_dayofweek'] = pd.to_numeric(df[col+'_dayofweek'],errors='coerce')
    # df.PREV_ADR_CHNG_DT =pd.to_datetime(df.PREV_ADR_CHNG_DT,errors='coerce',format='%d%b%Y')
    # df.PREV_PMT_DT = pd.to_datetime(df.PREV_PMT_DT,errors='coerce',format='%d%b%Y')
    # df.PREV_CARD_RQST_DT = pd.to_datetime(df.PREV_CARD_RQST_DT,errors='coerce',format='%d%b%Y')
    # df.FRD_IND_SWT_DT = 
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
                'mrch_tmp_prtpn_ind':{na_val:None},
                'pos_entry_mthd_cd':{na_val:None},
                'pos_cond_cd':{na_val:None},
                'trmnl_clasfn_cd':{na_val:None}, 
                'trmnl_capblt_cd':{na_val:None},
                'trmnl_pin_capblt_cd':{na_val:None},
                'tsys_dcln_reas_cd':{na_val:None},
                # 'cavv_cd':{'nan':encoders['cavv_cd'].transform(None),na_val:encoders['cavv_cd'].transform(None)},
                # 'mrch_pstl_cd':{'':None},
                # 'mrch_pstl_cd':{'[]':None},
                # 'mrch_pstl_cd':{'[0]':None},

    }

    for col in ['acqr_crcy_cd']:
    # for col in []:
      rpl_dict[col] = {na_val:None}
    # print df.ix[0:500,'prev_adr_chng_dt_year']
    for dc in date_cols:
      for comp in components:
        col_name = dc.lower()+'_'+comp
        # rpl_dict[col_name] = {np.nan:None}
        nan_rm = lambda x: -1 if str(x)=='nan' else x
        df[col_name] = df[col_name].apply(nan_rm)
        # if dc == 'PREV_PMT_DT': 
        df[col_name] = df[col_name].astype(int)

        # df[col_name] = df[col_name].map({'nan': None,})
    df.replace(rpl_dict,inplace=True)
    

    nan_fl_int = lambda x: -1 if None ==x else int(x)
    col2int = ['pos_entry_mthd_cd','pos_cond_cd','trmnl_clasfn_cd','trmnl_capblt_cd','trmnl_pin_capblt_cd','tsys_dcln_reas_cd','acqr_crcy_cd']
    for col_id in col2int:
    # print df[col_id.upper()]
      df[col_id].fillna(-1,inplace=True)
      df[col_id] = df[col_id].apply(nan_fl_int)
    # df[col_id] = df[col_id].fillna(-1).astype(int)
      df_temp = df[col_id]
      # df.loc[df_temp==-1,col_id] = None 
      print df[col_id]


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
    ######REORDER!!!!!!!!! 
    #THIS IS OF PARAMOUNT IMPORTANCE!!!!
    df = df[col_names_order]




    # print df.ix[0:500,'prev_adr_chng_dt_year']
    # print df.ix[499,'prev_adr_chng_dt_year']
    # print type(df.ix[499,'prev_adr_chng_dt_year'])
    #ignore the index
    df.to_csv(output, sep='\t', header=False, index=True,encoding='utf-8')
    # print df[df['AUTHZN_RQST_PROC_TM']=='nan']
    #jump to start of stream
    output.seek(0)
    contents = output.getvalue()
    cur = connection.cursor()
    t_mid = dt.datetime.now()
    #null values become ''
    cur.copy_from(output, table, null="", size=copy_block)    
    connection.commit()
    cur.close()
    
    print '{} seconds: inserted {} rows'.format((dt.datetime.now() - t_mid).seconds, j*chunksize)
    index_start = df.index[-1] + 1








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