
import os 
import sys
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


#import helpers functions manually created
project_path = os.path.abspath('/Users/rb666/projects/TopicModelling_META')
if project_path not in sys.path:
    sys.path.append(project_path)
from helpers.BERT_helpers import *



def load_DM_data(HighSensory=True,Handwritten=False):

    #Load data
    dataset_name = "SensoryTool_CombinedData.csv"
    if Handwritten:
        dataset_name = "Handwritten_CombinedTranslation.csv"

    metaproject_name = 'TopicModelling_META'
    subproject_name = 'TopMod_pipeline'

    # condition can be either 'highsensory','deeplistening', or 'handwritten'
    condition = 'handwritten' if Handwritten else 'highsensory' if HighSensory else 'deeplistening'
    print(f'Condition : "{condition}"')

    PROJDIR = os.path.expanduser(f"~/projects/{metaproject_name}")
    DATADIR = os.path.join(PROJDIR,f'DATA/{dataset_name}')
    CODEDIR = os.path.join(PROJDIR,f'{subproject_name}')

    df = pd.read_csv(DATADIR)

    if not Handwritten:
        dataset = df[df['meta_HighSensory'] == HighSensory]['reflection_answer']
        reports = dataset[dataset.notna() & (dataset != '')]
    else:
        dataset = df['reflection_answer']
        reports = dataset[dataset.notna() & (dataset != '')]
    print('N={} reports (condition : {})'.format(len(reports),condition))

    return reports




def get_cleaned_data(reports,extended_stopwords=True,apply_vectorizer=False):

    #Define stopwords
    stop_words = set(stopwords.words('english'))
    if extended_stopwords:
        stop_words = stop_words.union(custom_stopwords) #load custom stopwords from BERT_helpers.py

    # cleaned_reports = clean_text(reports,remove_stopwords=True, stop_words=custom_stopwords)
    reports_cleaned = reports.apply(clean_text,stop_words)
    reports_filtered = reports_cleaned[reports_cleaned.apply(lambda x: len(x.split()) > 1)]
    print('N={} cleaned reports'.format(len(reports_filtered))) 

    if apply_vectorizer:
        vectorizer_model = CountVectorizer(ngram_range=(1,2),stop_words='english')
        model = BERTopic(vectorizer_model=vectorizer_model,language='english',calculate_probabilities=True)
        topics, probs = model.fit_transform(reports)
        freq = model.get_topic_info()  # Access the freq topics that were generated
        vec_dict = {'model':model,'topics':topics,'probs':probs,'freq':freq}
        model.get_topic(0)
        model.visualize_hierarchy()
        model.visualize_barchart()
        return reports_filtered,vec_dict
    else:
        return reports_filtered



def get_hyperparams(len_dataset):
    return hyperparams(len_dataset)

def get_custom_stopwords():
    return custom_stopwords

def get_customed_BERTopic(umap_model,hdbscan_model,embedding_model,vectorizer_model):
    return customed_BERTopic(umap_model,hdbscan_model,embedding_model,vectorizer_model)

