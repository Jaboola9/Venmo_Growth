from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn import metrics
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, learning_curve
import joblib


def model_fit(model, X, y, cv_no=3):
    """fits model and prints precision, recall, and F1_score"""
    mdl = model
    mdl.fit(X, y)
    y_pred = cross_val_predict(mdl, X, y, cv=cv_no)
    print(confusion_matrix(y, y_pred))
    print(f'precision: {precision_score(y, y_pred)}')
    print(f'recall: {recall_score(y, y_pred)}')
    print(f'F1_score: {f1_score(y, y_pred)}')
    return mdl, y_pred


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend()
    plt.show()
    

def precision_recall_probcurve(probs, y):
    probs  = probs[:, 1]
    print('AUPRC = {}'.format(average_precision_score(y, probs)))
    precision, recall, _ = precision_recall_curve(y, probs)
    no_skill = len(y[y==1]) / len(y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='model')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    

def evaluate_classifier_probs(classifier, X, y, c=3):
    probs = cross_val_predict(classifier, X, y, cv=c, method="predict_proba")
    precision_recall_probcurve(probs, y)
    precisions, recalls, thresholds = precision_recall_curve(y, probs[:, 1])
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    return probs


def transform_test_data(test):
    """Replication of all training data transforms plus inputs missing columns"""
    labeled_test = attr_adder.transform(test, train, nlp=mpg_20, k=20)
    #NLP
    full_chunk_test, full_dic_test = create_chunk_and_diction(labeled_test) # clean and split notes into tokens and dictionary
    big_copy_test = blank_filler(full_chunk_test, full_dic_test)
    add_on_20_test = topic_builder(nlp, big_copy_test, k)
    labeled_reset_test = labeled_test.reset_index()
    labeled_nlp_test = pd.concat([labeled_reset_test, add_on_20_test], axis=1)
    ##other
    complete_test = add_var.transform(labeled_nlp_test)
    ##clean
    complete_clean_test = complete_test[complete_test.age != -1]
    ##selec
    complete_relevant_test = complete_clean_test[['age', 'topic_cluster_20',
                                                  'certainty_20', 'frequency',
                                                  'weekday', 'app', 'join']].copy()
    ##dummies
    completely_ready_test = pd.get_dummies(complete_relevant_test, columns=['topic_cluster_20',
                                                     'weekday', 'app'])
    completely_ready_test['weekday_Monday'] = pd.Series([0 for x in range(len(completely_ready_test.index))],
                                                        index=completely_ready_test.index)
    completely_ready_test['weekday_Tuesday'] = pd.Series([0 for x in range(len(completely_ready_test.index))],
                                                        index=completely_ready_test.index)
    completely_ready_test['weekday_Wednesday'] = pd.Series([0 for x in range(len(completely_ready_test.index))],
                                                        index=completely_ready_test.index)
    completely_ready_test.columns = train.columns
    return completely_ready_test


def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds, metrics='aucpr', early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='aucpr')
        
#     #Predict training set:
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
#     #Print model report:
#     print("\nModel Report")
#     print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
#     print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
#     print('AUPRC = {}'.format(average_precision_score(dtrain[target].values, \
#                                               dtrain_predprob)))             
#     feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
#     feat_imp.plot(kind='bar', title='Feature Importances')
#     plt.ylabel('Feature Importance Score')
    return alg  