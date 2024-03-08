#Handmade functions to help with BERT model


import re
from nltk.corpus import stopwords

# Basic text cleaning and stopwords removal
def clean_text(text,remove_stopwords=True, stop_words=stopwords.words('english')):
    text = text.replace('\n', ' ') # remove newline characters
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.lower()  # convert to lowercase
    text = ' '.join(text.split())  # remove extra whitespaces

    if remove_stopwords:
        text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


# Additional stopwords can be added based on dataset-specific jargon
custom_stopwords = {'trevor', 'test'}
