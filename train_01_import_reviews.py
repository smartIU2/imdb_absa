import pandas as pd
import sys

from imdb_absa.config import Config
from imdb_absa.db import DB

if __name__ == "__main__":
    """ Read in reviews from csv
        there's no check for identical reviews, so make sure to only import once
    """

    if len(sys.argv) < 2:
        print('No import file specified. Call like "train_01_import_reviews.py reviews.csv"')
    else:
        config = Config()
        
        db = DB(config.database)
        db.assure_database()
    
        csv = sys.argv[1]
        
        print('Reading in reviews')
        
        reviews = pd.read_csv(csv)
    
        if not all([(column in reviews.columns.values) for column in ('title_id', 'text', 'rating')]):
            print(f"{csv} does not contain the required columns 'title_id', 'text' and 'rating'")
            
        else:
            
            if not 'usage' in reviews.columns:
                reviews['usage'] = ''

            count_total = len(reviews)

            count_imported = db.import_reviews(reviews)
            
            print(f'Imported {count_imported} of {count_total} reviews.')
            
            if count_imported < count_total:
                print('Reviews for movies not in the database were discarded.')