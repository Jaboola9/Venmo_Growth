from datetime import timedelta
import pandas as pd
from pandas.plotting import scatter_matrix
import pickle
from sklearn.base import BaseEstimator, TransformerMixin 
from matplotlib import pyplot as plt
import seaborn as sns


# def convert_to_datetime(data, columns_to_convert):
#     """converts column to datetime"""
#     for col in columns_to_convert:
#         data[col] = pd.to_datetime(data[col])
#     return data


def date_sort_delay(data, datecolumns=None,
                    sorter='date', delay=6):
    """sorts transactions by date and adjusts timestamp from German time (GMT+1)
    to EST"""
    if datecolumns is None:
        datecolumns = ['date',
                       'payee_join_date',
                       'payer_join_date']
#                        'date_completed']
    for col in datecolumns:
        data[col] = pd.to_datetime(data[col])
    # sorted_df = convert_to_datetime(data, datecolumns)
    sorted_df = data.sort_values(sorter)
    sorted_df.date = sorted_df.date - timedelta(hours=delay)
    return sorted_df


def data_dates_viewer(data):
    """see all dates in dataset"""
    plt.figure(figsize=(20, 10))
    data.note.groupby(data['date'].dt.date).count().plot(kind="bar")
    plt.show()


def continuous_gather_check(data, date, size=(20, 10), plotkind='bar'):
    """define a function for checking hourly takes on venmo"""
    day = data[data.date.dt.date == pd.to_datetime(date)]
    plt.figure(figsize=size)
    day.note.groupby(day['date'].dt.hour).count().plot(kind=plotkind)
    plt.xlabel('hour')
    plt.ylabel('transactions_per_hour')
    plt.show()


def date_slicer(data, start_date, end_date):
    """slices subsection of transactions between two dates"""
    mask = (data['date'] > start_date) & (data['date'] <= end_date)
    sliced = data.loc[mask]
    return sliced


def age_finder(x):
    """determines how long the transaction happened after the earliest
    party joined"""
    if x['who_joined'] == 'payee':
        age = x.date - x.payer_join_date
    elif x.who_joined == 'payer':
        age = x.date - x.payee_join_date
    else:
        if x.payee_join_date > x.payer_join_date:
            age = x.date - x.payer_join_date
        else:
            age = x.date - x.payee_join_date
    return age.days


def create_time_variables(sorted_data):
    """creates target columns ('join', 'recruiting', 'who_joined') and
    input columns based on the first date in the slice"""
    first_transaction = sorted_data.date.iloc[0]
    new_payer = sorted_data.loc[sorted_data.payee_join_date > first_transaction]
    new_payee = sorted_data.loc[sorted_data.payer_join_date > first_transaction]
    new = pd.concat([new_payee, new_payer], axis=0)
    new_sorted = new.sort_values('date')
    joined = new_sorted.drop_duplicates(subset=['payee_name', 'payer_name'], keep='first')
    joined = joined.assign(join=1)
    joined['who_joined'] = joined.payee_join_date > joined.payer_join_date
    joined.replace({'who_joined': False}, 'payer', inplace=True)
    joined.replace({'who_joined': True}, 'payee', inplace=True)
    add_data = joined[['transaction_id', 'join', 'who_joined']]
    sorted_data = sorted_data.assign(recruiting=0)
    data_xy = pd.merge(sorted_data, add_data, on=['transaction_id'], how='left')
    data_xy.dropna(subset=['payee_join_date', 'payer_join_date', 'note'], inplace=True)
    data_xy.who_joined.fillna('neither', inplace=True)
    data_xy['age'] = data_xy.apply(age_finder, axis=1)
    data_xy['payer_count'] = data_xy.groupby('payer_name').cumcount() + 1
    data_xy['payee_count'] = data_xy.groupby('payee_name').cumcount() + 1
    print(data_xy.columns)
    return data_xy


def ready_slice(data, start_date, end_date):
    """does everything necessary time-wise before a split can occur"""
    sliced = date_slicer(data, start_date, end_date)
    data_xy = create_time_variables(sliced)
    return data_xy


def percent_splitter(sorted_data, percentile=70):
    """splits date-sorted data into a train and test group"""
    length = len(sorted_data)
    test = sorted_data.head(int(length * (percentile / 100)))
    train = sorted_data.tail(int(length * ((100 - percentile) / 100)))
    return test, train


def select_and_fill(data):
    """selects and fills in target variable ('join')"""
    data_slim = data[
        ['payee_name', 'payer_name', 'date', 'app_id', 'join', 'age', 'transaction_id', 'who_joined', 'note']].copy()
    data_slim['join'].fillna(0, inplace=True)
    data_slim.info()
    return data_slim


def clean_transform(data):
    """selects relevant variables, fills in target, and creates unique transaction
    identifier, weekday, highest usage, bins the apps into apple, android, or other,
    and resets the index"""
    # data.drop(columns=['Unnamed: 0', 'transaction_type'], inplace=True)
    data_slim = data[
        ['payee_name', 'payer_name', 'date', 'app_id', 'join', 'age',
         'transaction_id', 'who_joined', 'note', 'payee_count', 'payer_count']].copy()
    data_slim['join'].fillna(0, inplace=True)
    data_slim['pair'] = data_slim.payee_name + data_slim.payer_name
    data_slim['weekday'] = data_slim.date.apply(lambda x: x.day_name())
    data_slim['frequency'] = data_slim.apply(frequency, axis=1)
    data_slim['weekday'] = data_slim.weekday.astype(str)
    data_slim['app'] = data_slim.apply(app_bin, axis=1)
    data_ready = data_slim.reset_index()
    data_select = data_ready[['transaction_id', 'payee_name', 'payer_name', 'pair', 'date', 'weekday',
                              'age', 'frequency', 'app', 'note', 'join', 'who_joined']].copy()
    data_select.info()
    return data_select


class LabelAdder(BaseEstimator, TransformerMixin): 
  """creates target variables ('join', 'recruiting', and 'who_joined') based on dates"""
  def __init__(self): # no *args or **kargs 
    return None
  def fit(self, X, y=None): 
    return self # nothing else to do
  def transform(self, X, y=None): 
    first_transaction = X.date.iloc[0]
    new_payer = X.loc[X.payee_join_date > first_transaction]
    new_payee = X.loc[X.payer_join_date > first_transaction]
    new = pd.concat([new_payee, new_payer], axis=0)
    new_sorted = new.sort_values('date')
    joined = new_sorted.drop_duplicates(subset=['payee_name', 'payer_name'], keep='first')
    joined = joined.assign(join=1)
    joined['who_joined'] = joined.payee_join_date > joined.payer_join_date
    joined.replace({'who_joined': False}, 'payer', inplace=True)
    joined.replace({'who_joined': True}, 'payee', inplace=True)
    add_data = joined[['transaction_id', 'join', 'who_joined']]
    sorted_data = X.assign(recruiting=0)
    data_xy = pd.merge(sorted_data, add_data, on=['transaction_id'], how='left')
    data_xy.dropna(subset=['payee_join_date', 'payer_join_date', 'note'], inplace=True)
    data_xy.who_joined.fillna('neither', inplace=True)
    data_xy['join'].fillna(0, inplace=True)
    data_xy['age'] = data_xy.apply(age_finder, axis=1)
    data_xy['payer_count'] = data_xy.groupby('payer_name').cumcount() + 1
    data_xy['payee_count'] = data_xy.groupby('payee_name').cumcount() + 1
    print(data_xy.columns)
    return data_xy


def load_pickle(filename):
  with open(filename+'.pkl', 'rb') as p:
    loaded_pickle = pickle.load(p)
    p.close()
  return loaded_pickle