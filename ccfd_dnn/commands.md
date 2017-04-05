# SHELL
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
iostat -x -m 5

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu,cuda.root=/usr/local/cuda-7.5/ python ./ccfd_dnn/model_weight.py > ./results/data_little_enc.out

nvidia-smi

#return all rows containing a null

select * from data_little_enc where not(data_little_enc is not null)limit 20;


sudo find / -type f -size +50M -exec du -h {} \; | sort -n

Linux command to check disk space
df command - Shows the amount of disk space used and available on Linux file systems.
du command - Display the amount of disk space used by the specified files and for each subdirectory.
btrfs fi df /device/ -

apt-get install python-tk

install latex to convert notebooks to pdf
sudo apt-get install texlive-latex-base

sudo apt-get install xzdec


## Debugging in Notebooks
### Constant interactivity

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
### breakpoint
from IPython.core.debugger import Tracer; Tracer()()
###navigation
n(ext) line and run this one
s(tep): 
c(ontinue) running until next breakpoint
q(uit) the debugger
### Source
http://kawahara.ca/how-to-debug-a-jupyter-ipython-notebook/
https://iqbalnaved.wordpress.com/2013/10/17/how-to-debug-in-ipython-using-ipdb/
## Tip & Tricks for notebooks
https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/