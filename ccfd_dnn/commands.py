
	 THEANO_FLAGS=mode=FAST_RUN python ./ccfd_dnn/model_weight.py 1>./results/testing.out

	 docker run --hostname=quickstart.cloudera --privileged=true -t -i -p 8888 -p 7180 -p 80:3000 cloudera/quickstart /usr/bin/docker-quickstart

	 select acct_id,count(*) as num_trans from data_little where AUTHZN_RQST_PROC_TM >= (select AUTHZN_RQST_PROC_TM from data_little where FRD_IND_SWT_DT >= '2014-05-11' order by AUTHZN_RQST_PROC_TM limit 1) group by acct_id order by num_trans;

    if trans_mode = 'train':
            dataFrame = pd.read_sql_query('select acct_id,count(*) '
                                          'as num_trans from data_little '
                                          'where AUTHZN_RQST_PROC_TM < '
                                          '(select AUTHZN_RQST_PROC_TM '
                                          'from data_little '
                                          'where FRD_IND_SWT_DT >= "2014-05-11" '
                                          'order by AUTHZN_RQST_PROC_TM limit 1) '
                                          'group by acct_id order by num_trans'
                       .format(table=table), disk_engine)

select sum(num_trans) from (select acct_id,count(*) as num_trans from data_trim where AUTHZN_RQST_PROC_TM < (select AUTHZN_RQST_PROC_TM from data_trim where FRD_IND_SWT_DT >="2014-05-11" order by AUTHZN_RQST_PROC_TM limit 1) group by acct_id order by num_trans);

'select acct_id,count(*) as num_trans from {table} where authzn_rqst_proc_tm <= (select authzn_rqst_proc_tm from {table} where frd_ind_swt_dt >={cutt_off_date} order by authzn_rqst_proc_tm limit 1) group by acct_id order by num_trans;'

service postgresql restart

free -m

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu,cuda.root=/usr/local/cuda-7.5/ python ./ccfd_dnn/model_weight.py > ./results/data_little_enc.out

nvidia-smi

#return all rows containing a null

select * from data_little_enc where not(data_little_enc is not null)limit 20;