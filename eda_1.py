import bson
import emoji
import numpy as np
import pandas as pd
import missingno as msno
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix

# project_location = 'Flatiron/FinalProject/'
# root_path = 'gdrive/My Drive/' + project_location
# bson_location = root_path + 'venmo.bson'


def bson_decode(location):
    """decodes the bson file using and iterator and stores it as a dataframe"""
    data = bson.decode_file_iter(open(location, 'rb'))
    transactions = {'payee': [], 'payee_name': [], 'payer': [],
                    'payer_name': [], 'merchant': [],
                    'transaction_type': [], 'date': [], 'transaction_id': [],
                    'payee_join_date': [],'payer_join_date': [],
                    'date_completed': [], 'note': [], 'app_name': [],
                    'app_id': []}
#     count = -1
#     for c, d in enumerate(data):
#         count += 1
    

    for c, d in enumerate(data):
        try:
            payee = d['payment']['target']['user']['id']
        except:
            payee = np.nan
        transactions['payee'].append(payee)
        try:
            payee_name = d['payment']['target']['user']['username']
        except:
            payee_name = np.nan
        transactions['payee_name'].append(payee_name)
        try:
            payer = d['payment']['actor']['id']
        except:
            payer = np.nan
        transactions['payer'].append(payer)
        try:
            payer_name = d['payment']['actor']['username']
        except:
            payer_name = np.nan
        transactions['payer_name'].append(payer_name)
        try:
            merchant = d['payment']['target']['merchant']
        except:
            merchant = np.nan
        transactions['merchant'].append(merchant)
        date = (d['date_created'])
        transactions['date'].append(date)
        transaction_type = d['type']
        transactions['transaction_type'].append(transaction_type)
        transaction_id = d['_id']
        transactions['transaction_id'].append(transaction_id)
        try:
            payee_join_date = d['payment']['target']['user']['date_joined']
        except:
            payee_join_date = np.nan
        transactions['payee_join_date'].append(payee_join_date)
        try:
            payer_join_date = d['payment']['actor']['date_joined']
        except:
            payer_join_date = np.nan
        transactions['payer_join_date'].append(payer_join_date)
        try:
            date_completed = d['payment']['date_completed']
        except:
            date_completed = 'missing_date_completed'
        transactions['date_completed'].append(date_completed)
        note = d['note']
        transactions['note'].append(note)
        app_name = d['app']['name']
        transactions['app_name'].append(app_name)
        app_id = d['app']['id']
        transactions['app_id'].append(app_id)
        if c%1000000 == 0:  # count = 7076584
            print(f'{c} done')
    return pd.DataFrame(transactions)



# consecutive_sample = venmo_json_decode(bson_location)
# consecutive_sample.drop(columns=['Unnamed: 0', 'transaction_type'], inplace=True)


def convert_to_datetime(data, columns_to_convert):
    '''converts to datetime and sorts'''
    for col in columns_to_convert:
        data[col] = pd.to_datetime(data[col])
    data = data.sort_values('date')
    return data


# consecutive_sample = convert_to_datetime(consecutive_sample,
#                                          ['date',
#                                           'payee_join_date',
#                                           'payer_join_date',
#                                           'date_completed'])

# august_data = consecutive_sample[consecutive_sample.date.dt.month == 8]
# october_data = consecutive_sample[consecutive_sample.date.dt.month == 10]


def clean_format_transactions(data):
    '''removes self-loops and converts IDs to strings'''
    data = data[data['payee'] != data['payer']]
    data['payer'] = data['payer'].astype(str)
    data['payee'] = data['payee'].astype(str)
    data['pair'] = data.payee + data.payer
    return data


# august_data = clean_format_transactions(august_data)


def create_target_recruiting_transaction(data):
    first_transaction = data.date[0]
    new_payer = data[data.payee_join_date > first_transaction]
    print(new_payer.shape)
    new_payee = data[data.payer_join_date > first_transaction]
    print(new_payee.shape)
    new = pd.concat([new_payee, new_payer], axis=0)
    print(new.shape)
    new_sorted = new.sort_values('date')
    joined = new_sorted.drop_duplicates(subset=['payee', 'payer'], keep='first')
    joined['joined'] = 1
    joined['who_joined'] = joined.payee_join_date > joined.payer_join_date
    joined.replace({'who_joined': False}, 'payer', inplace=True)
    joined.replace({'who_joined': True}, 'payee', inplace=True)
    add_data = joined[['transaction_id', 'joined', 'who_joined']]
    data_xy = pd.merge(data, add_data, on=['transaction_id'], how='left')
    data_xy.drop(columns=['merchant'], inplace=True)
    print(data_xy.shape)
    data_xy.dropna(subset=['payee_join_date', 'payer_join_date', 'note'], inplace=True)
    print(data_xy.shape)
    data_xy.who_joined.fillna('neither', inplace=True)
    data_xy.joined.fillna(0, inplace=True)
    return data_xy


# august_xy = create_target_recruiting_transaction(august_data)


def age_finder(x):
    if x['who_joined'] == 'payee':
        age = x.date - x.payer_join_date
    elif x.who_joined == 'payer':
        age = x.date - x.payer_join_date
    else:
        if x.payee_join_date > x.payer_join_date:
            age = x.date - x.payer_join_date
        else:
            age = x.date - x.payee_join_date
    return age.days


def demoji(x):
    note = x['note']
    return emoji.demojize(note)


def weekday(x):
    weekday = x.date.weekday()
    return weekday


def frequency(x):
    if x.payer_count >= x.payee_count:
        top = x.payer_count
    else:
        top = x.payee_count
    return top


def app_bin(x):
    if x.app_id == 1:
        app = 'iphone'
    elif x.app_id == 4:
        app = 'android'
    else:
        app = 'other'
    return app


def create_recruiting_x_variables(data):
    data['demoji'] = data.apply(demoji, axis=1)
    data['weekday'] = data.apply(weekday, axis=1)
    data['weekday'] = data.weekday.astype(str)
    data['frequency'] = data.apply(frequency, axis=1)
    data['app'] = data.apply(app_bin, axis=1)
    to_dummy = ['app', 'weekday']
    data = pd.get_dummies(data, columns=to_dummy,
                          drop_first=True)
    return data


# august_ready = create_recruiting_x_variables(august_xy)
# august_xy.to_pickle('august_ready.pkl')

# august_ready.groupby(['note']).count().sort_values(['payee'], ascending=False)

# X_list = ['age', 'frequency', 'app_iphone', 'app_other',
#           'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
#           'weekday_5', 'weekday_6']


def split_X_from_y(data, X_list, y_variable):
    X = data[X_list]
    y = data[y_variable]
    return X, y


# X, y = split_X_from_y(august_ready, X_list, 'joined')

# X_train, X_test, y_train, y_test = train_test_split(X, y)

# np.mean(y_train)


def nice_confusion(model):
    """Creates a nice looking confusion matrix"""
    plt.figure(figsize=(10, 10))
    plt.xlabel('Predicted Class', fontsize=18)
    plt.ylabel('True Class', fontsize=18)
    #     plt.xticks(labels=[''])
    viz = ConfusionMatrix(
        model,
        cmap='PuBu', fontsize=18)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.poof()

