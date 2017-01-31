import datetime as dt
from sqlalchemy import create_engine # database connection

def days_hours_minutes_seconds(td):
    return td.days, td.seconds//3600, (td.seconds//60)%60, td.seconds%60



def list_equal(L1,L2):
	return len(L1) == len(L2) and sorted(L1) == sorted(L2)


def get_engine(address = "postgresql+pg8000://script@localhost:5432/ccfd"):

    # disk_engine = create_engine('sqlite:///'+data_dir+db_name,convert_unicode=True)
    # disk_engine.raw_connection().connection.text_factory = str
    disk_engine = create_engine(address)
    return disk_engine
