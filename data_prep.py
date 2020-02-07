import missingno as msno
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


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


def app_bin(x):
    """bins apps into one of three categories"""
    if x.app_id == 1:
        app = 'iphone'
    elif x.app_id == 4:
        app = 'android'
    else:
        app = 'other'
    return app


def frequency(x):
    """returns highest frequency payer's usage for within the time period"""
    if x.payer_count >= x.payee_count:
        top = x.payer_count
    else:
        top = x.payee_count
    return top


class DropVariables(BaseEstimator, TransformerMixin):
    """Drops unnecessary and irrational values"""
    def __init__(self):
        return None
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if X[X.age == -1].shape[0]/X.shape[0] < 0.001:
            X = X[X.age != -1].copy()
            msno.matrix(X)
        else:
            print('check -1 age values')
        full_ratio = X.dropna().shape[0]/X.shape[0]
        if full_ratio> 0.9:
            X = X.dropna()
        else:
            print(f'{1-full_ratio}% missing values!')
        # add and drop variables
        return X
    

class AddVariables(BaseEstimator, TransformerMixin):
    """Adds 'app', 'frequency' and 'weekday' variables"""
    def __init__(self):
        return None
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        to_drop = ['app_id', 'payer_count', 'payee_count', 'date']
        X['app'] = X.apply(app_bin, axis=1)
        X['frequency'] = X.apply(frequency, axis=1)
        X['weekday'] = X.date.apply(lambda x: x.day_name())
        X.drop(to_drop, axis=1, inplace=True)
        return X
    

class TopicCreator(BaseEstimator, TransformerMixin):
    """CAUTION: THIS TAKES SEVERAL HOURS! Runs through the full NLP GSDMM steps to sort notes into 20 topics using all functions below"""
    def __init__(self, k=20):
        self.k = k
        
    def fit(self, X, y=None):
        
        self.mpg_ = mpg_tester(clean_notes(X), self.k)
        return self

    def transform(self, X, y=None):
        try:
            getattr(self, "mpg_")
        except AttributeError:
            raise RuntimeError("You must train classifer before creating topic feature!")
        return topic_builder(self.mpg_, self.k)
    
    
def data_prep(raw_data, complete_num=['age', 'frequency', 'certainty_20'], complete_cat=['app', 'weekday', 'topic_cluster_20'], nlp_needed=False,):
    """Prepares data for model selection with optional NLP classification"""
    if nlp_needed: # runs GSDMM unsupervised classification within pipeline
        topiccreator = TopicCreator()
        topiccreator.fit_transform(raw_data.note)
        complete_reset = raw_data.reset_index()
        complete_nlp = pd.concat([complete_reset, add_on_20], axis=1)
    else:
        pass
    of_interest = ['payer_count', 'payee_count', 'date', 'app_id',
               'certainty_20', 'topic_cluster_20', 'age', 'join'] # 'note'
    dv = DropVariables()
    complete_set = dv.transform(raw_data[of_interest])
    
    y = complete_set['join']
    X = complete_set.drop('join', axis=1)
    
    adv = AddVariables()
    X = adv.transform(X)
    print(X.columns)

#     complete_num = ['age', 'frequency', 'certainty_20'] #
#     complete_cat = ['app', 'weekday', 'topic_cluster_20'] #
    column_pipeline = ColumnTransformer([
        ("num", StandardScaler(), complete_num),
        ("cat", OneHotEncoder(), complete_cat),
    ])

    completely_prepared = column_pipeline.fit_transform(X)
    return completely_prepared, y #should be a numpy array and a series