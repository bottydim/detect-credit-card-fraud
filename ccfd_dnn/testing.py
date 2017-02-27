import cPickle as pickle
import os


tbl_src = 'auth'
# tbl_src = table
# tbl_evnt = 'event'
#
# # encoders = None
# path_encoders = '/home/botty/Documents/CCFD/data/encoders/{tbl_src}+{tbl_evnt}'.format(tbl_src=tbl_src, tbl_evnt=tbl_evnt)
# if os.path.exists(path_encoders):
#     with open(path_encoders, 'rb') as enc_data:
#         encoders = pickle.load(enc_data)
#         print 'encoders LOADED!'
#
# le = encoders['mrch_catg_cd']
#
# print len(le.classes_)
#
# tbl_src = 'data_trim'
# # tbl_src = table
# tbl_evnt = 'event'
# path_encoders = '/home/botty/Documents/CCFD/data/encoders/{tbl_src}+{tbl_evnt}'.format(tbl_src=tbl_src, tbl_evnt=tbl_evnt)
# if os.path.exists(path_encoders):
#     with open(path_encoders, 'rb') as enc_data:
#         encoders = pickle.load(enc_data)
#         print 'encoders LOADED!'
#
# le = encoders['mrch_catg_cd']
#
# print len(le.classes_)
# table = 'data_trim'
# cutt_off_date = '2014-05-11'
# query = '''select acct_id,count(*) as num_trans from {table}
# where authzn_rqst_proc_tm <= (select authzn_rqst_proc_tm from {table} where frd_ind_swt_dt >='{cutt_off_date}'
# order by authzn_rqst_proc_tm limit 1) group by acct_id order by num_trans;'''
# print query.format(table=table,cutt_off_date=cutt_off_date)

