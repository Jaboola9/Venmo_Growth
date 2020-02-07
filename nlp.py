import numpy as np
import pandas as pd
import nltk
import tqdm
from nltk.corpus import stopwords
from tqdm.notebook import trange, tqdm
import string
import re
import spacy
from spacy import load as spacy_load
import emoji
import pickle
from gsdmm import MovieGroupProcess
from nltk.stem.porter import *
import heapq
from sklearn.base import BaseEstimator, TransformerMixin


def pickle_this(data, filename):
    """pickle the data"""
    with open(filename+'.pkl', 'wb') as p:
        pickle.dump(data, p)
        p.close()
    
    
def load_pickle(filename):
    with open(filename+'.pkl', 'rb') as p:
        loaded_pickle = pickle.load(p)
        p.close()
    return loaded_pickle


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

#     def fit(self, X, y=None):
#         filled = clean_notes(X)
#         print('finished cleaning notes')
#         model = mpg_tester(filled, 20)
#         return model
#     def transform(self, X, y=None):
#         filled = clean_notes(X)
#         print('finished cleaning notes')
#         model = mpg_tester(filled, 20)
#         print('generated model')
#         topics = topic_builder(model, 20)
#         return topics

    
    
    
def clean_notes(notes):
    """cleans notes data for mpg fitting using create_big_chunk"""
#     demoji = data.note.apply(lambda x: emoji.demojize(x))
    demoji = notes.apply(lambda x: emoji.demojize(x))
    stop_words = set(stopwords.words('english'))
    test = demoji.apply(lambda x: x.replace(':',' '))
    test2 = test.apply(lambda x: x.strip())
    full_chunk, full_dic = create_big_chunk(test2, 75)
    filled = blank_filler(full_chunk, full_dic)
    return filled


def doc_to_spans(list_of_texts, join_string=' ||| '):
    """changes individual notes to an iterable single doc to speed up the process"""
    nlp = spacy_load('en_core_web_lg')
    all_docs = nlp(' ||| '.join(list_of_texts))
    # print(all_docs)
    split_inds = [i for i, token in enumerate(all_docs) if token.text == '|||'] + [len(all_docs)]
    new_docs = [all_docs[(i + 1 if i > 0 else i):j] for i, j in zip([0] + split_inds[:-1], split_inds)]
    return new_docs      


def clean4vec(list_of_spans, test_dic={}):
    """stems spans and creates list"""
    stemmer = PorterStemmer()
    ram_break = []
    for span in list_of_spans:
        # result = [i for i in span if not i in stop_words]
        result = [i for i in span if len(i) > 1]
        result = [stemmer.stem(i.text) for i in result]
        result = [re.sub(r'(.)\1+', r'\1\1', i) for i in result]
        for i in result:
            if i in test_dic:
                test_dic[i] += 1
            else:
                test_dic[i] = 1
        ram_break.append(result)
    return ram_break


def create_big_chunk(notes, chunks_no=15, test_dic={}):
    """creates """
    stop_words = set(stopwords.words('english'))
    lat_list = []
    chunk_list = []
    counter = 0
    chunks_no = chunks_no
    chunk_length = int(len(notes)/chunks_no) + 1
    for i in range(chunks_no):
        filtered_small = []
        really = doc_to_spans(notes[counter:counter+chunk_length])
        really = clean4vec(really, test_dic)
        # print(len(really))
        for note in really:
            filtered = [i for i in note if not i in stop_words]
            filtered_small.append(filtered)
        # print(len(really))
        counter += chunk_length
        if not (i+1)%25:
            print(f"finished chunk {i+1} and {counter} notes, {chunks_no - i - 1} chunks to go")
        else:
            pass
        chunk_list.append(filtered_small)
    lat_list = [item for sublist in chunk_list for item in sublist]
    print(f"finished {len(lat_list)} notes in total")
    return lat_list, test_dic


def blank_filler(data, data_dic):
    """get rid of blanks and singles"""
    foodict = {k: v for k, v in data_dic.items() if v==1}
    print(len(foodict))
    singles = list(foodict.keys())
    big_copy = []
    for note in tqdm(data):
        note = [i for i in note if i not in singles]
        if len(note) == 0:
#             print('blank note found')
            big_copy.append('-1')
        else:
            big_copy.append(note)
    return big_copy


def topic_builder(model, data, number):
    """builds a dataframe of topics"""
    mgp = model
    doc_count = np.array(mgp.cluster_doc_count)
    print('Number of documents per topic :', doc_count)
    print('*'*number)
    top_index = doc_count.argsort()[-number:][::-1]
    print('Most important clusters (by number of docs inside):', top_index)
    print('*'*number)
    # Show the top 50 words in term frequency for each cluster 
    for i in range(number):
        print(i)
        A = mgp.cluster_word_distribution[i]
        print(heapq.nlargest(50, A, key=A.get))
    topic_cluster = []
    certainty = []
#     topic_guess = []
    for i in tqdm(range(len(data))):
        pred = mgp.choose_best_label(data[i])
        topic_cluster.append(pred[0])
        certainty.append(pred[1])
#         topic_guess.append(topic_dict[pred[0]])
    add_on = pd.DataFrame({'topic_cluster_'+str(number): topic_cluster,
                           'certainty_'+str(number): certainty})
    return add_on


def mpg_tester(data, k=5, a=0.1, b=0.1, iters=30):
    "fits GSDMM model to create k clusters of topics out of data"
    vocab = set(x for doc in data for x in doc)
    n_terms = len(vocab)
    mgpk = MovieGroupProcess(K=k, alpha=a, beta=b, n_iters=iters)
    y = mgpk.fit(data, n_terms)
    doc_count = np.array(mgpk.cluster_doc_count)
    print('Number of documents per topic :', doc_count)
    print('*'*20)
    top_index = doc_count.argsort()[-10:][::-1]
    print('Most important clusters (by number of docs inside):', top_index)
    print('*'*20)
    # Show the top 20 words in term frequency for each cluster 
    for i in range(k):
        print(i)
        A = mgpk.cluster_word_distribution[i]
        print(heapq.nlargest(20, A, key=A.get))
    return mgpk


topic_names_20 = ['halloween',
               'food_and_transport', 
               'love',
               'star',
               'food', 
               'rent',
               'celebration',
               'people_emojis', 
               'thanks', 
               'entertainment', 
               'rent_2',
               'short_words',
               'strange',
               'sport',
               'transport',
               'hearts',
               'friend',
               'night_out',
               'utilities',
               'food_2']


def create_topic_dictionary(topic_names, number_of_topics):
    """creates dictionary or topics to match with the numbers of topic builder (look at results of this first)"""
    if len(topic_names) != number_of_topics:
        return "list is not the correct length"
    else:
        topic_dict = {}
        for i, topic_num in enumerate(top_index):
            topic_dict[topic_num]=topic_names[i]
        return topic_dict
    

def create_chunk_and_diction(complete):
    demoji = complete.note.apply(lambda x: emoji.demojize(x))
    print('done demojizing')
    stop_words = set(stopwords.words('english'))
    print('done with stop_words')
    test2 = demoji.apply(lambda x: x.replace(':',' '))
    print('replaced :')
    test3 = test2.apply(lambda x: x.strip())
    print('stripped notes')
    full_chunk, full_dic = create_big_chunk(test3, 75)
    return full_chunk, full_dic
# addon code for topic_builder topic is set
# topic_cluster = []
# certainty = []
# topic_guess = []
# for i in tqdm(range(len(filled))):
#     pred = mgp.choose_best_label(filled[i])
# #     print(topic_dict[pred[0]], pred[1], filled[i])
# #     print(pred)
#     topic_cluster.append(pred[0])
#     certainty.append(pred[1])
#     topic_guess.append(topic_dict[pred[0]])