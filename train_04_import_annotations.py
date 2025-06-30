import pandas as pd

from imdb_absa.config import Config
from imdb_absa.db import DB
from imdb_absa.nlp import get_aspect_categories

if __name__ == "__main__":
    """ Read in doccano annotations and save as gold aspect labels to sqlite database """

    config = Config()    
    
    db = DB(config.database)
    db.assure_database()

    print('Reading in annotations')
    
    with open(config.import_annotations, encoding='utf-8') as file:
        annotations = pd.read_json(file, orient='records', lines=True)

    annotations = annotations.explode(['aspect', 'label'])

    annotations = annotations[annotations['label'].str[2] != 'candidate']
    
    annotations['polarity'] = annotations['label'].str[2]
    annotations['aspect_term'] = annotations['aspect'].str[2]
    annotations['aspect_context'] = annotations['aspect'].str[3]
    annotations['ordinal'] = annotations['aspect'].str[4]
 
    count = len(annotations)


    print('Adding aspect categories')
 
    aspect_terms = db.get_aspect_terms()

    annotations['category'] = annotations['aspect_context'].apply(get_aspect_categories,aspect_terms=aspect_terms)

    annotations = annotations.explode('category')

    if 'sentiment_term' not in annotations.columns:
        annotations['sentiment_term'] = ''

    annotations['verified'] = 1 # gold aspects


    print('Saving to database')

    annotations = annotations[['id', 'category', 'aspect_term', 'ordinal', 'polarity', 'sentiment_term', 'verified']]
    
    db.import_sentence_aspects(annotations)
    
    
    print(f'Imported {count} annotations.')