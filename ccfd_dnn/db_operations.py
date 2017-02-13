
import pandas as pd
from sqlalchemy import create_engine  # database connection
import datetime as dt
import utils
import time

class DbOperator():

    def __init__(self, *args, **kwargs):

        if 'engine' not in kwargs.keys():
            self.address = kwargs['address']
            self.engine = create_engine(self.address)
        else:
            self.engine = kwargs['engine']
        self.connection = engine.raw_connection()
        self.connection.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
        self.cursor = self.connection.cursor()



    def get_columns(self,tbl):
        engine = self.engine
        df = pd.read_sql_query('select * from {table} limit 1'.format(table=tbl), engine)
        col_names = df.columns.values
        return col_names


    def cf_db_uniq(self, t1, t2):
        cursor = self.cursor
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
            cursor.execute(qry.format(table=t1, col=name))
            result_t1 = cursor.fetchall()
            cursor.execute(qry.format(table=t2, col=name))
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
    def create_idx_missing(self, table):
        cursor = self.cursor
        col_names = self.get_columns(table)
        for c, name in enumerate(col_names):
            if name =='index':
                continue
            t_mid = dt.datetime.now()
            cursor.execute('''CREATE INDEX IF NOT EXISTS id_{table}_{col}
                        ON {table} ({col})'''.format(table=table, col=name))
            self.connection.commit()
            print '{} index created in {}'.format(name,(dt.datetime.now() - t_mid))
        print 'idxs created!'

    def create_idx_comp(self,table):
        cursor = self.cursor
        cursor.execute('''CREATE INDEX id_{table}_acct_id_tm
                        ON {table} (acct_id,AUTHZN_RQST_PROC_TM)'''.format(table=table))
        cursor.execute('''CREATE INDEX id_{table}_tm_frd 
                        ON {table} (AUTHZN_RQST_PROC_TM,FRD_IND_SWT_DT)'''.format(table=table))

        self.connection.commit()
        print 'composite indeces created'

    def create_idx_single(self, table, index):
        cursor = self.cursor
        cursor.execute('''CREATE INDEX id_{table}_{index}
                                ON {table} ({index})'''.format(table=table,index=index))

    def copy_rows(self,from_tbl, into_tbl, where_clause=None):
        copy_query = 'CREATE TABLE {dst} AS ' \
                     'SELECT * FROM {src} WHERE {filter}'
        qry = copy_query.format(src=from_tbl,dst=into_tbl,filter=where_clause)
        cursor = self.cursor
        print (qry)
        cursor.execute(qry)
        self.connection.commit()

    def count_unique_features(self, table, columns=None,skip_col=['index']):
        uf =self.unique_features(table,columns=columns,skip_col=skip_col)
        columns = list(self.get_columns(table))
        print("FEATURE REPORT TABLE {}".format(table))
        for col in skip_col:
            columns.remove(col)
        for col in columns:
            print("{col}:{cnt}".format(col=col, cnt=uf[col]))
        num_feat = 0
        for col in uf.keys():
            num_feat+=uf[col][1]
        print("Total number of features: {}".format(num_feat))
        return num_feat

    def unique_features(self, table,columns=None, skip_col=['index']):
        # type: (string, string) -> dict
        '''
            :param table:
            :type table:
            :param columns:
            :type List of which columns to count. By default None mans all:
            :return:
            :rtype:
            Notes http://stackoverflow.com/questions/11250253/postgresql-countdistinct-very-slow
            '''
        uniq_feat_count_list = {}
        if columns is None:
            columns = list(self.get_columns(table))

        for col in skip_col:
            if col in columns:
                columns.remove(col)
        for col in columns:
            qry = 'SELECT COUNT(*) FROM' \
                  '(SELECT DISTINCT {column_name} FROM {table_name})' \
                  ' AS temp;'
            qry = 'SELECT COUNT({column_name}), COUNT(DISTINCT {column_name}), MAX({column_name}),' \
                  'MIN({column_name}), AVG({column_name})' \
                  'FROM {table_name}'
            self.cursor.execute(qry.format(column_name=col, table_name=table))
            uniq_feat = self.cursor.fetchall()
            uniq_feat_count_list[col] = list(uniq_feat[0])
            # print("{col}:{cnt}".format(col=col, cnt=uniq_feat_count_list[col]))
        return uniq_feat_count_list

    def get_frad_users_qry(self, table):
        frad_users_qry = 'SELECT DISTINCT ({column_name}) ' \
              'FROM {table_name} ' \
              'WHERE {filter} = 1'.format(column_name='acct_id',
                                           filter='frd_ind', table_name=table)
        return frad_users_qry
    def get_fraud_users(self, table):
        return NotImplemented
        qry = 'SELECT * ' \
              'FROM {table_name} ' \
              'WHERE acct_id in (frad_users_qry)'
        self.cursor.execute(qry.format(frad_users_qry=frad_users_qry, table_name=table))
        #TODO

    def count_t_num(self, table, filter_qry):
        qry = 'SELECT Count(*) FROM {table_name} ' \
              'WHERE {filter_qry}'.format(table_name=table, filter_qry=filter_qry)
        self.cursor.execute(qry.format(table_name=table, filter_qry=filter_qry))
        count = self.cursor.fetchone()
        return long(count[0])

if __name__ == "__main__":
    address = 'postgresql://script@localhost:5432/ccfd'
    engine = create_engine(address)
    connection = engine.raw_connection()
    connection.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')

    # table = "auth_enc"
    # table_2 = "auth_enc_evt"
    # table_3 = "data_little_event"

    db_ops = DbOperator(engine=engine)
    # db_ops.cf_db_uniq(table,table_2)

    src_table = "auth_enc"
    dst_table = "data_fraud"
    # t0 = time.time()
    # frad_users = '{col} in (  {sub_qry})'.format(col='acct_id', sub_qry=db_ops.get_frad_users_qry(src_table))
    # db_ops.copy_rows(src_table, dst_table, frad_users)
    # t1 = time.time()
    # print 'time taken:'+str(t1-t0)
    # print ('Creating IDXs')
    # t0 = time.time()
    # db_ops.create_idx_missing(dst_table)
    # t1 = time.time()
    # print 'COLUMN time taken:' + str(t1 - t0)
    # db_ops.create_idx_comp(dst_table)
    # t1 = time.time()
    # print 'time taken:' + str(t1 - t0)
    #


    src_table = "auth_enc_evt"
    dst_table = "data_fraud_evt"
    t0 = time.time()
    frad_users = '{col} in (  {sub_qry})'.format(col='acct_id', sub_qry=db_ops.get_frad_users_qry(src_table))
    db_ops.copy_rows(src_table, dst_table, frad_users)
    t1 = time.time()
    print 'time taken:'+str(t1-t0)
    print ('Creating IDXs')
    t0 = time.time()
    db_ops.create_idx_missing(dst_table)
    t1 = time.time()
    print 'COLUMN time taken:' + str(t1 - t0)
    db_ops.create_idx_comp(dst_table)
    t1 = time.time()
    print 'time taken:' + str(t1 - t0)



    #count number of transaction for all users exhibiting fraud
    # print ("Number of transaction for fraud users in {table} = {num}".format(table=table,
    #                                                                          num=db_ops.count_t_num(table, frad_users)))


    # ####COUNT UNIQUE FEATURES
    # db_ops.count_unique_features(table)
    # db_ops.count_unique_features(table_2)
    # db_ops.count_unique_features(table_3)




    # print(db_ops.get_columns(table_3))


    # ####CREATE IDXs
    # print "create indecies for table {table}".format(table=table)
    # db_ops.create_idx_comp(table)
    # db_ops.create_idx_missing(table)
    #
    #


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
#     print '{} index created in {}'.format(name,(dt.datetime.now() - t_mid))
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
#     print '{} index - {} :created in {}'.format(name,message,(dt.datetime.now() - t_mid))






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




