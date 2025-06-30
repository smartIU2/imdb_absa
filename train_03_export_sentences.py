import pandas as pd

from setfit.span.aspect_extractor import AspectExtractor

from imdb_absa.config import Config
from imdb_absa.db import DB


def aspects_to_char_spans(aspects):
    
    terms = []
    spans = []
    
    for aspect in aspects:
        span = aspect.doc[aspect.start:aspect.stop]
        spans.append([span.start_char, span.end_char, span.text, aspect.context.lower(), terms.count(span.text)])
        terms.append(span.text)
        
    return spans
    
    
def spans_to_labels(spans):

    return [[span[0], span[1], 'candidate'] for span in spans]


if __name__ == "__main__":
    """ Export sample sentences for annotation in doccano """

    config = Config()    
    
    db = DB(config.database)
    db.assure_database()

    print('Collecting sample sentences')

    #TODO: make config
    genre_id = 1 # Action

    sentences = db.get_sample_sentences(genre_id)


    print('Getting possible aspects')
    
    ae = AspectExtractor(config.model_spacy, ['lemmatizer', 'ner'])
    
    _, aspects = ae(sentences['text'].tolist())


    print('Reshaping to doccano format')
    
    sentences['aspect'] = [aspects_to_char_spans(doc_aspects) for doc_aspects in aspects]
    sentences['label'] = sentences['aspect'].apply(spans_to_labels)
    
    
    sentences.to_json(config.export_sentences, orient='records', lines=True)
  
    print(f'Exported {len(sentences)} sentences to {config.export_sentences}')