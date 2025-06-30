import pandas as pd
import spacy

from imdb_absa.config import Config
from imdb_absa.db import DB
from imdb_absa.nlp import NLP

if __name__ == "__main__":
    """ Predict aspect based sentiment polarity and save to sqlite database """

    config = Config()    
    
    print('Assuring database')
    db = DB(config.database)
    db.assure_database()

    #TODO: make config
    genre_id = -1 #All
    chunk_size = 6000

    print('Deleting previous predictions from database')
    
    db.reset_predictions()
    

    print('Getting sentences')

    sentences = db.get_sentences_for_prediction(genre_id, chunk_size)
    
    if len(sentences.index) == 0:
        print('No sentences to predict.')
    
    else:
    
        print('Loading models')

        nlp = NLP(config.model_spacy, config.model_spacy_exclude, config.model_maverick, False, config.model_setfit, True)


        while len(sentences.index) != 0:
            
            print('Predicting aspect based sentiments for chunk of sentences...')
             
            sentence_ids = sentences['id'].unique()
            
            aspect_terms = db.get_aspect_terms()
            
            _, aspects = nlp.predict_absa(sentences['text'], aspect_terms)

            
            print('Saving to database')
            
            sentences['absa'] = [[(aspect.text, aspect.ordinal, aspect.categories, aspect.label) for aspect in doc_aspects] for doc_aspects in aspects]
            
            sentences = sentences.explode('absa')
            
            sentences = sentences[~sentences['absa'].isna()]
            
            sentences[['aspect_term','ordinal','category','polarity']] = pd.DataFrame(sentences['absa'].tolist(), index=sentences.index)
                    
            sentences = sentences.explode('category')
                    
            sentences['sentiment_term'] = ''
            
            sentences['verified'] = 0 # not gold aspects


            #save to sql

            sentences = sentences[['id', 'category', 'aspect_term', 'ordinal', 'polarity', 'sentiment_term', 'verified']]
            
            db.import_sentence_aspects(sentences)
            
            db.update_sentences_analyzed(sentence_ids)
            
            # get next chunk
            sentences = db.get_sentences_for_prediction(genre_id, chunk_size)
            
        print('Done.')