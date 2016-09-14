import datetime as dt


def days_hours_minutes_seconds(td):
    return td.days, td.seconds//3600, (td.seconds//60)%60, td.seconds%60



def list_equal(L1,L2):
	return len(L1) == len(L2) and sorted(L1) == sorted(L2)