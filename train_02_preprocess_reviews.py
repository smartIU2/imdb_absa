import pandas as pd
import spacy

from imdb_absa.config import Config
from imdb_absa.db import DB
from imdb_absa.nlp import NLP


if __name__ == "__main__":
    """ preprocess reviews """

    config = Config()
    
    print('Assuring database')
    db = DB(config.database)
    db.assure_database()

    print('initializing NLP models')
    nlp = NLP(config.model_spacy, config.model_spacy_exclude, config.model_maverick, config.pre_coref_resolution)


    print('normalizing reviews')
    
    reviews = db.get_reviews_for_preprocess()
    
    count_reviews = len(reviews)
    count_sentences = 0
    
    reviews['normalizedText'] = nlp.normalize_reviews(reviews['originalText'], config.pre_normal)

    # update normalizedText in review table
    db.update_reviews(reviews)


    reviews_noTitle = reviews[reviews['title_id'].isnull()]

    if len(reviews_noTitle.index) != 0:

        print('handling reviews for unknown titles')

        #split into sentences
        reviews_noTitle['sentence'] = nlp.split_sentences(reviews_noTitle['normalizedText'])
        
        reviews_noTitle = reviews_noTitle.explode('sentence', ignore_index=True)
    
        reviews_noTitle.index += db.get_new_sentence_id()
    
        reviews_noTitle['sentence'] = nlp.add_aspect_term(reviews_noTitle['sentence'])
        
        reviews_noTitle = reviews_noTitle[['review_id','sentence']]
        
        
        #estimate sentence polarities
        polarity = nlp.estimate_polarity(reviews_noTitle['sentence'])

        reviews_noTitle = pd.concat([reviews_noTitle, polarity], axis=1)
            
        db.import_sentences(reviews_noTitle)
        
        
        #split into tokens with POS tags
        reviews_noTitle['tokens'] = nlp.get_tokens_from_sentences(reviews_noTitle['sentence'])
        
        reviews_noTitle = reviews_noTitle[['tokens']]
        
        reviews_noTitle = reviews_noTitle.explode('tokens')
      
        reviews_noTitle[['word','whitespace','POS']] = reviews_noTitle['tokens'].apply(pd.Series)
      
        reviews_noTitle['sentencePart'] = reviews_noTitle['word'].isin([',',':',';']).cumsum()
        
        reviews_noTitle.drop(columns=['tokens','whitespace'], inplace=True)
        
        db.import_words(reviews_noTitle)
    
    del reviews_noTitle


    # preprocess with metadata for each movie 
    titles = reviews['title_id'].unique()    
    for title_id in titles:
    
        print(f'processing reviews for {title_id} ...')
    
        reviews_title = reviews.query(f'title_id == "{title_id}"')

        metadata = db.get_metadata_replacements(title_id)
        
        reviews_title['normalizedText'] = nlp.replace_metadata(reviews_title['normalizedText'], metadata)
  
  
        print('splitting reviews into sentences')
        
        splits = nlp.split_sentences(reviews_title['normalizedText']) 

        reviews_title.drop(columns=['normalizedText'], inplace=True)


        print(f"tokenizing sentences and replacing proper names{' & coreferences' if config.pre_coref_resolution else ''}")

        reviews_title['tokens'] = nlp.replace_propernames_corefs(splits, metadata)
       
        reviews_title = reviews_title.explode('tokens', ignore_index=True)

        reviews_title.index += db.get_new_sentence_id()
        
        reviews_title['sentence'] = nlp.get_sentence_from_tokens(reviews_title['tokens'], metadata)
        
        reviews_title = reviews_title[~pd.isna(reviews_title['sentence'])]
        
        reviews_title['sentence'] = nlp.add_aspect_term(reviews_title['sentence'])
        
        tokens = reviews_title[['tokens']]
        
        reviews_title = reviews_title[['review_id','sentence']]
        
        count_sentences += len(reviews_title)
        
        
        print('estimating sentence polarity')
                
        polarity = nlp.estimate_polarity(reviews_title['sentence'])

        reviews_title = pd.concat([reviews_title, polarity], axis=1)
            
        db.import_sentences(reviews_title)
        
        
        print('saving words and POS tags')
      
        tokens = tokens.explode('tokens')
      
        tokens[['word','whitespace','POS']] = pd.DataFrame(tokens['tokens'].tolist(), index=tokens.index)
      
        tokens['sentencePart'] = tokens['word'].isin([',',':',';']).cumsum()
        
        tokens.drop(columns=['tokens','whitespace'], inplace=True)
        
        db.import_words(tokens)
  
  
    print('cleaning up database...')
    db.vacuum()
    
    print(f'processed {count_reviews} reviews with {count_sentences} sentences.')