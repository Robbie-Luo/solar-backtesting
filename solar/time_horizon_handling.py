import datetime

def to_date(f):
    def _inner(s, *args, **kwargs):
        if isinstance(s,str):
            return f(datetime.datetime.strptime(s,"%Y-%m-%d"), *args, **kwargs)
        return f(s, *args, **kwargs)
    return _inner

@to_date
def subtract_one_month(d):
    if d.month==1:
        return datetime.datetime(d.year-1, 12, 1)
    return datetime.datetime(d.year, d.month-1, 1)

@to_date
def add_one_month(d):
    if d.month==12:
        return datetime.datetime(d.year+1, 1, 1)
    return datetime.datetime(d.year, d.month+1, 1)

def subtract_months(d, months):
    if months < 1:
        return d
    if months == 1:
        return subtract_one_month(d)
    return subtract_months(subtract_one_month(d), months-1)

def add_months(d, months):
    if months < 1:
        return d
    if months == 1:
        return add_one_month(d)
    return add_months(add_one_month(d), months-1)