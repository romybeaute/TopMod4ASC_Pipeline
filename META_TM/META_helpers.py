'''
@Author : ROMY BEAUTÉ
@Date creation : 07-05-2024
@Last modification : 15-05-2024
@Contact : r.beaut@sussex.ac.uk

#Contains META final and regrouped handmades functions to help with Topic Modelling pipeline for Dreamachine open reports data

'''



import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
import re
import itertools
import time

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import nltk
nltk.download('wordnet')

from gensim.corpora.dictionary import Dictionary
import gensim.corpora as corpora
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



custom_stopwords = {'trevor', 'test','eg','devtest','hello','like','felt','sometimes','end','also','ho','un','che','really','would','almost'}
stop_words = set(stopwords.words('english'))
extended_stop_words = stop_words.union(custom_stopwords) #load custom stopwords from BERT_helpers.py



def hyperparams(len_dataset):
    '''
    Defined in helpers/BERT_helpers.py
    '''
    return {'umap_params': {
            'n_components': [2,3,4,5,7,9,11], #default to 2 
            'n_neighbors': [3,4,5,7,10,12],
            'min_dist': [0.0,0.01,0.025,0.05], #default to 1.0
        },
        'hdbscan_params': {
            #list of 3 values : 1% len_dataset,10 (default value), 5% len_dataset
            'min_cluster_size': [int(len_dataset*0.01),10,int(len_dataset*0.05)],
            'min_samples': [int(len_dataset*0.01),10,int(len_dataset*0.05),None] #default to None
        }
    }

# def hyperparams(len_dataset):
#     '''
#     Defined in helpers/BERT_helpers.py
#     '''
#     return {'umap_params': {
#             'n_components': [3,4], #default to 2 
#             'n_neighbors': [3,5],
#             'min_dist': [0.01], #default to 1.0
#         },
#         'hdbscan_params': {
#             #list of 3 values : 1% len_dataset,10 (default value), 5% len_dataset
#             'min_cluster_size': [5,10],
#             'min_samples': [None] #default to None
#         }
#     }


class BERTopicGridSearchWrapper(BaseEstimator):
    def __init__(self, vectorizer_model, embedding_model, n_neighbors=10, n_components=5, min_dist=0.01, min_cluster_size=10, min_samples=None, top_n_words=5):
        self.vectorizer_model = vectorizer_model
        self.embedding_model = embedding_model
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.top_n_words = top_n_words
        self.model = None

    def fit(self, X):
        
        umap_model = UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components, min_dist=self.min_dist, random_state=77)
        hdbscan_model = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, prediction_data=True)

        self.model = BERTopic(umap_model=umap_model, 
                              hdbscan_model=hdbscan_model,
                              embedding_model=self.embedding_model,
                              vectorizer_model=self.vectorizer_model,
                              top_n_words=self.top_n_words,
                              language='english',
                              calculate_probabilities=True,
                              verbose=True)
        self.model.fit_transform(X)
        return self

    def score(self, X):
        coherence_score = calculate_coherence(self.model, X)
        return coherence_score


#############################################################################
################ LOAD DATA ##################################################
#############################################################################


def get_data(HighSensory, Handwritten,remove_stopwords=[],tokenize=False,lemmatize=False):
    '''
    Defined in BERT_DM/BERTopic_hypertuned_multiprocessing.py
    '''

    reports = load_DM_data(HighSensory, Handwritten)
    
    # ********** Pre-process data **********
    #Apply very basic pre-processing of data from clean_text function in BERT_helpers.py : (remove stopwords if set to True), punctuation, numbers, convert to lowercase
    reports_filtered = get_cleaned_data(reports,
                                        remove_stopwords=remove_stopwords,
                                        tokenize=tokenize,
                                        lemmatize=lemmatize)
    
    return reports_filtered
s


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



#############################################################################
################ CLEAN DATA ##################################################
#############################################################################


def get_cleaned_data(reports,remove_stopwords=[],tokenize=False,lemmatize=False):
    '''
    Defined in helpers/get_DM_data.py
    Apply the clean_text function to list of reports.
    Parameters:
    - reports (list): List of textual reports.
    - remove_stopwords (list): Stopwords to remove.
    - tokenize (bool): Flag to tokenize the text.
    - lemmatize (bool): Flag to lemmatize the text; requires tokenization.
    '''
    print("Cleaning reports with options - Remove stopwords:", bool(remove_stopwords), "Tokenize:", tokenize, "Lemmatize:", lemmatize)
    if not tokenize and not lemmatize:
        reports_cleaned = reports.apply(lambda x: clean_text(x, remove_stopwords=remove_stopwords)) 
        reports_filtered = reports_cleaned[reports_cleaned.apply(lambda x: len(x.split()) > 1)] #remove empty reports and reports < 1 1word
    else:
        reports_list = reports.tolist() if isinstance(reports, pd.Series) else reports
        cleaned_reports = [clean_text(report, remove_stopwords, tokenize, lemmatize) for report in reports_list]
        reports_filtered = [report for report in cleaned_reports if report and (len(report.split()) > 1 if not tokenize else len(report) > 1)]
        
    print('N={} cleaned and filtered reports'.format(len(reports_filtered)))

    return reports_filtered 





def clean_text(text,remove_stopwords=[],tokenize=False,lemmatize=False):
    '''
    Defined in helpers/BERT_helpers.py
    Clean and process text by removing unwanted characters, stopwords, tokenizing, and optionally lemmatizing.
    Parameters:
    - text (str): The text to be processed.
    - remove_stopwords (list): List of stopwords to remove; if None, no removal is performed.
    - tokenize (bool): Whether to tokenize the text.
    - lemmatize (bool): Whether to lemmatize the text; tokenization is required to lemmatize.
    '''

    text = text.replace('\n', ' ') # remove newline characters
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.lower()  # convert to lowercase
    text = ' '.join(text.split())  # remove extra whitespaces

    # if remove_stopwords not emoty, remove stopwords :
    if remove_stopwords:
        text = ' '.join([word for word in text.split() if word not in remove_stopwords])

    if tokenize:
        tokenizer = WordPunctTokenizer()
        tokens = tokenizer.tokenize(text)
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        text = tokens if tokenize and not lemmatize else ' '.join(tokens)

    return text







#############################################################################
################ COHERENCE METRICS ##########################################
#############################################################################


# calculate coherence using BERTopic's model
def calculate_coherence(topic_model, data):

    topics, _ = topic_model.fit_transform(data)
    # Preprocess Documents
    documents = pd.DataFrame({"Document": data,
                          "ID": range(len(data)),
                          "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    
    #Extracting the vectorizer and embedding model from BERTopic model
    vectorizer = topic_model.vectorizer_model #CountVectorizer of BERTopic model 
    tokenizer = vectorizer.build_tokenizer()
    analyzer = vectorizer.build_analyzer() #allows for n-gram tokenization
    
    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names_out()
    tokens = [tokenizer(doc) for doc in data]
    # tokens = [analyzer(doc) for doc in data]

    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    topic_words = [[word for word, _ in topic_model.get_topic(topic_id)] for topic_id in range(len(set(topics))-1)]

    print("Topics:", topic_words)
    coherence_model = CoherenceModel(topics=topic_words, 
                                     texts=tokens, 
                                     corpus=corpus,
                                     dictionary=dictionary, 
                                     coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    return coherence_score



def test_wrapper(reports_filtered, vectorizer_model, embedding_model):
    '''
    funtion to test the BERTopicGridSearchWrapper class with only few hyperparameters
    '''
    wrapper = BERTopicGridSearchWrapper(vectorizer_model=vectorizer_model, embedding_model=embedding_model)
    wrapper.fit(reports_filtered)  
    return wrapper.score(reports_filtered)



#############################################################################
################ BERTOPIC POSTRAINED ########################################
#############################################################################



def redmerge_topics_pt(data,topic_model,nr_topics=False,topics_to_merge_list=[]):
    '''
    Reduce or merge topics from the BERTopic model
    '''

    # if topics_to_merge not empty list, merge topics
    if topics_to_merge_list:
        topic_model.merge_topics(data, topics_to_merge_list)
    else:
        # if nr_topics not False, reduce topics
        if nr_topics:
            topic_model.reduce_topics(data, nr_topics=nr_topics)
    topics = topic_model.topics_
    print("Topics after merging:", topics)
    print("Number of topics after merging:", len(set(topics))-1)

    coherence_pt = calculate_coherence_postrained(data,topic_model, topics)
    print("Coherence Score after merging:", coherence_pt)
    return topic_model, topics, coherence_pt




def calculate_coherence_postrained(data,topic_model, topics):

    
    #Extracting the vectorizer and embedding model from BERTopic model
    vectorizer = topic_model.vectorizer_model #CountVectorizer of BERTopic model 
    tokenizer = vectorizer.build_tokenizer()
    
    # Extract features for Topic Coherence evaluation
    tokens = [tokenizer(doc) for doc in data]

    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    topic_words = [[word for word, _ in topic_model.get_topic(topic_id)] for topic_id in range(len(set(topics))-1)]

    print("Topics:", topic_words)
    coherence_model = CoherenceModel(topics=topic_words, 
                                     texts=tokens, 
                                     corpus=corpus,
                                     dictionary=dictionary, 
                                     coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    return coherence_score


#############################################################################
################ BERTOPIC ###################################################
#############################################################################




def get_params_grid(len_dataset):
    hyperparams_dict = hyperparams(len_dataset)

    # Generate all combinations of hyperparameters
    umap_combinations = list(itertools.product(
    hyperparams_dict['umap_params']['n_components'],
    hyperparams_dict['umap_params']['n_neighbors'],
    hyperparams_dict['umap_params']['min_dist']))

    hdbscan_combinations = list(itertools.product(
    hyperparams_dict['hdbscan_params']['min_cluster_size'],
    hyperparams_dict['hdbscan_params']['min_samples']))

    print("Total number of combinations:", len(umap_combinations)*len(hdbscan_combinations))
    return umap_combinations, hdbscan_combinations




def run_grid_search(data, vectorizer_model, embedding_model,HighSensory,Handwritten=False,store_results=False):

    umap_combinations, hdbscan_combinations =get_params_grid(len(data))

    start_time = time.time()
    
    # Nested loop to iterate over each combination of UMAP and HDBSCAN parameters
    results = []

    for umap_config in tqdm(umap_combinations):
        for hdbscan_config in hdbscan_combinations:
            try:
                # Unpack the parameter sets
                n_components, n_neighbors, min_dist = umap_config
                min_cluster_size, min_samples = hdbscan_config
                
                # Execute the main function using unpacked parameters
                model, _, coherence_score = main(
                    data=data,
                    vectorizer_model=vectorizer_model,
                    embedding_model=embedding_model,
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=min_dist,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples
                )
                # Store results
                results.append({
                    'n_components': n_components,
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist,
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'coherence_score': coherence_score,
                    'model': model
                })
            except Exception as e:
                print(f"Error with parameters {umap_config}, {hdbscan_config}: {e}")
                continue
    results_df = pd.DataFrame(results).sort_values(by='coherence_score', ascending=False)
    print(f"Grid search completed in {time.time() - start_time:.2f} seconds")

    if store_results:
        name_file = f'META_DATA/grid_search_results_{"HighSensory" if HighSensory else "DeepListening" if not Handwritten else "HandWritten"}.csv'
        if os.path.isfile(name_file):
            results_df.to_csv(name_file, mode='a', header=False, index=False)
            # re sort the file by coherence score
            results_df = pd.read_csv(name_file).sort_values(by='coherence_score', ascending=False)

        else:
            results_df.to_csv(name_file, index=False)

    return results_df






# def run_bertopic_hyperparams(data):
#     '''
#     Defined in BERT_DM/BERTopic_hypertuned_multiprocessing.py
#     ''' 
    
#     hyperparams_umap, hyperparams_hdbscan, param_grid = get_params_grid(len(data))
#     hyper_neighbours = hyperparams_umap['n_neighbors']
#     hyper_components = hyperparams_umap['n_components']
#     hyper_dist = hyperparams_umap['min_dist']

#     hyper_cluster_size = hyperparams_hdbscan['min_cluster_size']
#     hyper_min_samples = hyperparams_hdbscan['min_samples']

#     print("Hyperparameters for UMAP: ", hyper_neighbours, hyper_components, hyper_dist)
#     print("Hyperparameters for HDBSCAN: ", hyper_cluster_size, hyper_min_samples)


#     #VECTORIZER
#     #will remove stopwords with the vectorizer model instead of the clean_text function
#     stopwords_list = list(stopwords.words('english')) + list(custom_stopwords)
#     print("Additional stopwords: ", custom_stopwords)
#     vectorizer_model = CountVectorizer(ngram_range=(1,2), stop_words=stopwords_list)

#     #EMBEDDING
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#     #GRID SEARCH using hyperparams_umap, hyperparams_hdbscan, param_grid on main function
#     grid_search = GridSearchCV(BERTopicGridSearchWrapper(vectorizer_model, embedding_model),
#                            param_grid,
#                            cv=3,
#                            scoring=make_scorer(calculate_coherence, greater_is_better=True),
#                            verbose=10)
     
#     grid_search.fit(data)
    
#     print("Best parameters:", grid_search.best_params_)
#     print("Best coherence score:", grid_search.best_score_)
    
#     return grid_search




def main(data,
         vectorizer_model,
         embedding_model,
         n_neighbors, 
         n_components, 
         min_dist, 
         min_cluster_size, 
         min_samples=None,
         top_n_words = 5,
         nr_topics = 'auto'):
    
    '''
    Defined in BERT_DM/BERTopic_hypertuned_multiprocessing.py

    Return : 
    - model : implemented BERTopic model with fine-tuned hyperparameters
    - topics : contains a one-to-one mapping of inputs to their modeled topic (or cluster)
    - probs : contains the probability of each document belonging to their assigned topic

    '''
    print(f"Received parameters: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}, min_cluster_size={min_cluster_size}, min_samples={min_samples}")


    # ********** Instanciate BERTOPIC **********

    umap_model = UMAP(n_neighbors=n_neighbors,
                      n_components=n_components,
                      min_dist=min_dist,
                      random_state=77) # rdm seed for reportability
    
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, 
                            min_samples=min_samples,
                            gen_min_span_tree=True,
                            prediction_data=True) 
    
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words,
        nr_topics= nr_topics,#default to None
        language='english',
        calculate_probabilities=True,
        verbose=True)

    # ********** Fit BERTOPIC **********
    topics,_ = model.fit_transform(data) 
    coherence_score = calculate_coherence(model, data)
    print("Coherence Score:", coherence_score)

    return model, topics, coherence_score