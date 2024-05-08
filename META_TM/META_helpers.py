'''
@Author : ROMY BEAUTÉ
@Date creation : 07-05-2024
@Last modification : 07-05-2024
@Contact : r.beaut@sussex.ac.uk

#Contains META final and regrouped handmades functions to help with Topic Modelling pipeline for Dreamachine open reports data

'''



import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
import sys
import os
import re
import itertools

from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer


# BERTopic Libraries
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN


from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


#############################################################################
################ DEFINE PRE-PARAMS ##########################################
#############################################################################


custom_stopwords = {'trevor', 'test','eg','devtest','hello'}


# def hyperparams(len_dataset):
#     '''
#     Defined in helpers/BERT_helpers.py
#     '''
#     return {'umap_params': {
#             'n_components': [2,3,4,5,7,9,11], #default to 2 
#             'n_neighbors': [3,4,5,7,10,12],
#             'min_dist': [0.0,0.01,0.025,0.05], #default to 1.0
#         },
#         'hdbscan_params': {
#             #list of 3 values : 1% len_dataset,10 (default value), 5% len_dataset
#             'min_cluster_size': [int(len_dataset*0.01),10,int(len_dataset*0.05)],
#             'min_samples': [int(len_dataset*0.01),10,int(len_dataset*0.05),None] #default to None
#         }
#     }

def hyperparams(len_dataset):
    '''
    Defined in helpers/BERT_helpers.py
    '''
    return {'umap_params': {
            'n_components': [2,3,4,5,6,7,8,9,10,12], #default to 2 
            'n_neighbors': [5],
            'min_dist': [0.01], #default to 1.0
        },
        'hdbscan_params': {
            #list of 3 values : 1% len_dataset,10 (default value), 5% len_dataset
            'min_cluster_size': [15],
            'min_samples': [15] #default to None
        }
    }


class BERTopicGridSearchWrapper(BaseEstimator):
    def __init__(self, vectorizer_model, embedding_model, n_neighbors=15, n_components=5, min_dist=0.0, min_cluster_size=10, min_samples=None, top_n_words=10):
        self.vectorizer_model = vectorizer_model
        self.embedding_model = embedding_model
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.top_n_words = top_n_words
        self.model = None

    def fit(self, X, y=None):
        
        umap_model = UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components, min_dist=self.min_dist, random_state=77)
        hdbscan_model = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, prediction_data=True)

        self.model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                              embedding_model=self.embedding_model,
                              vectorizer_model=self.vectorizer_model,
                              top_n_words=self.top_n_words,
                              language='english',
                              calculate_probabilities=False,
                              verbose=True)
        self.model.fit_transform(X)
        return self

    def score(self, X, y=None):
        # Using coherence as scorer
        coherence_score = calculate_coherence(self.model, X)
        return coherence_score


#############################################################################
################ LOAD DATA ##################################################
#############################################################################


def get_data(HighSensory, Handwritten,remove_stopwords=False,extended_stopwords=False):
    '''
    Defined in BERT_DM/BERTopic_hypertuned_multiprocessing.py
    '''

    reports = load_DM_data(HighSensory, Handwritten)
    
    # ********** Pre-process data **********
    #Apply very basic pre-processing of data from clean_text function in BERT_helpers.py : (remove stopwords if set to True), punctuation, numbers, convert to lowercase
    reports_filtered = get_cleaned_data(reports,
                                        remove_stopwords=remove_stopwords,
                                        extended_stopwords=extended_stopwords,
                                        apply_vectorizer=False)
    return reports_filtered



def load_DM_data(HighSensory=True,Handwritten=False):
    '''
    Defined in helpers/get_DM_data.py
    '''

    dataset_name = "SensoryTool_CombinedData.csv"
    if Handwritten:
        dataset_name = "Handwritten_CombinedTranslation.csv"

    metaproject_name = 'TopicModelling_META'

    condition = 'handwritten' if Handwritten else 'highsensory' if HighSensory else 'deeplistening'
    print(f'Condition : "{condition}"')

    PROJDIR = os.path.expanduser(f"~/projects/{metaproject_name}")
    DATADIR = os.path.join(PROJDIR,f'DATA/{dataset_name}')

    df = pd.read_csv(DATADIR)

    if not Handwritten:
        dataset = df[df['meta_HighSensory'] == HighSensory]['reflection_answer']
        reports = dataset[dataset.notna() & (dataset != '')]
    else:
        dataset = df['reflection_answer']
        reports = dataset[dataset.notna() & (dataset != '')]
    print('N={} reports (condition : {})'.format(len(reports),condition))

    return reports






def clean_text(text,remove_stopwords, stop_words=stopwords.words('english')):
    '''
    Defined in helpers/BERT_helpers.py
    Basic text cleaning and stopwords removal
    '''
    text = text.replace('\n', ' ') # remove newline characters
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.lower()  # convert to lowercase
    text = ' '.join(text.split())  # remove extra whitespaces

    if remove_stopwords:
        text = ' '.join([word for word in text.split() if word not in stop_words])
        print(f"Number of stopwords removed: {len(set(text.split()).intersection(stop_words))}")
        print(f"Number of custom stopwords removed: {len(set(text.split()).intersection(custom_stopwords))}")
        print(f"Extended stopwords: {set(text.split()).intersection(custom_stopwords)}")

    return text



def get_cleaned_data(reports,remove_stopwords,extended_stopwords,apply_vectorizer=False):
    '''
    Defined in helpers/get_DM_data.py
    '''

    #Define stopwords
    stop_words = set(stopwords.words('english'))
    if extended_stopwords:
        stop_words = stop_words.union(custom_stopwords) #load custom stopwords from BERT_helpers.py

    reports_cleaned = reports.apply(lambda x: clean_text(x, remove_stopwords=remove_stopwords, stop_words=stop_words))
    reports_filtered = reports_cleaned[reports_cleaned.apply(lambda x: len(x.split()) > 1)]
    print('N={} cleaned reports'.format(len(reports_filtered))) 

    if apply_vectorizer:
        vectorizer_model = CountVectorizer(ngram_range=(1,2),stop_words='english')
        model = BERTopic(vectorizer_model=vectorizer_model,language='english',calculate_probabilities=True)
        topics, probs = model.fit_transform(reports)
        freq = model.get_topic_info()  # Access the freq topics that were generated
        vec_dict = {'model':model,'topics':topics,'probs':probs,'freq':freq}
        model.visualize_hierarchy()
        model.visualize_barchart()
        return reports_filtered,vec_dict
    else:
        return reports_filtered





#############################################################################
################ COHERENCE METRICS ##########################################
#############################################################################


# calculate coherence using BERTopic's model
def calculate_coherence(model, data):
    tokens = [doc.split() for doc in data]
    dictionary = Dictionary(tokens)
    topics = [[word for word, _ in model.get_topic(topic_id)] for topic_id in model.get_topics()]
    coherence_model = CoherenceModel(topics=topics, texts=tokens, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()


#############################################################################
################ BERTOPIC ##########################################
#############################################################################


def get_params_grid(len_dataset):
    hyperparams_dict = hyperparams(len_dataset)
    hyperparams_umap = hyperparams_dict['umap_params']
    hyperparams_hdbscan = hyperparams_dict['hdbscan_params']

    param_grid = {
        'n_neighbors': hyperparams_umap['n_neighbors'],
        'n_components': hyperparams_umap['n_components'],
        'min_dist': hyperparams_umap['min_dist'],
        'min_cluster_size': hyperparams_hdbscan['min_cluster_size'],
        'min_samples': hyperparams_hdbscan['min_samples']}
    
    #print number of total combinations
    print("Total number of combinations:", len(param_grid['n_neighbors'])*len(param_grid['n_components'])*len(param_grid['min_dist'])*len(param_grid['min_cluster_size'])*len(param_grid['min_samples']))
    return param_grid




def run_bertopic_hyperparams(data):
    '''
    Defined in BERT_DM/BERTopic_hypertuned_multiprocessing.py
    ''' 
    
    hyperparams_umap, hyperparams_hdbscan, param_grid = get_params_grid(len(data))
    hyper_neighbours = hyperparams_umap['n_neighbors']
    hyper_components = hyperparams_umap['n_components']
    hyper_dist = hyperparams_umap['min_dist']

    hyper_cluster_size = hyperparams_hdbscan['min_cluster_size']
    hyper_min_samples = hyperparams_hdbscan['min_samples']

    print("Hyperparameters for UMAP: ", hyper_neighbours, hyper_components, hyper_dist)
    print("Hyperparameters for HDBSCAN: ", hyper_cluster_size, hyper_min_samples)


    #VECTORIZER
    #will remove stopwords with the vectorizer model instead of the clean_text function
    stopwords_list = list(stopwords.words('english')) + list(custom_stopwords)
    print("Additional stopwords: ", custom_stopwords)
    vectorizer_model = CountVectorizer(ngram_range=(1,2), stop_words=stopwords_list)

    #EMBEDDING
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    #GRID SEARCH using hyperparams_umap, hyperparams_hdbscan, param_grid on main function
    grid_search = GridSearchCV(BERTopicGridSearchWrapper(vectorizer_model, embedding_model),
                           param_grid,
                           cv=3,
                           scoring=make_scorer(calculate_coherence, greater_is_better=True),
                           verbose=10)
     
    grid_search.fit(data)
    


    # umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, random_state=77)
    # hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True)
    
    # model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
    #                  embedding_model=embedding_model, vectorizer_model=vectorizer_model)
    # topics, _ = model.fit_transform(data)
    # print("Type of topics:", type(topics))
    
    print("Best parameters:", grid_search.best_params_)
    print("Best coherence score:", grid_search.best_score_)
    
    return grid_search

    # return {'params': params, 'coherence_score': coherence_score}




# def main(data,
#          vectorizer_model,
#          embedding_model,
#          n_neighbors, 
#          n_components, 
#          min_dist, 
#          min_cluster_size, 
#          min_samples,
#          top_n_words = 10,
#          ):
    
#     '''
#     Defined in BERT_DM/BERTopic_hypertuned_multiprocessing.py
#     '''
#     print(f"Received parameters: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}, min_cluster_size={min_cluster_size}, min_samples={min_samples}")


#     # ********** Instanciate BERTOPIC **********



#     umap_model = UMAP(n_neighbors=n_neighbors,
#                       n_components=n_components,
#                       min_dist=min_dist,
#                       random_state=77) # rdm seed for reportability
    
#     hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, 
#                             min_samples=min_samples,
#                             gen_min_span_tree=True,
#                             prediction_data=True) 
    
#     model = BERTopic(
#         umap_model=umap_model,
#         hdbscan_model=hdbscan_model,
#         embedding_model=embedding_model,
#         vectorizer_model=vectorizer_model,
#         top_n_words=top_n_words,
#         language='english',
#         calculate_probabilities=True,
#         verbose=True)

#     # ********** Fit BERTOPIC **********
#     topics, _ = model.fit_transform(data)
#     coherence_score = calculate_coherence(model, data)
#     print("Coherence Score:", coherence_score)


#     return topics,model,coherence_score