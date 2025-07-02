import pandas as pd
import os

from imdb_absa.config import Config

if __name__ == "__main__":
    """ Convert aclImdb dataset to csv for import """

    config = Config()
    
    print('Converting aclImdb')
    
    splits = ('train', 'test')
    polarities = ('neg', 'pos')
    
    ratings = []
    reviews = []
    titles = []
    usage = []
    
    for split in splits:
        
        path = os.path.join(config.import_aclImdb, split)
        
        for pol in polarities:
  
            with os.scandir(os.path.join(path, pol)) as files:
                
                sorted_files = sorted(files, key=lambda f: int(f.name.split('_')[0]))
                for file in sorted_files:
                    ratings.append(int(file.name.split('_')[1].split('.')[0]))
                    with open(file.path, 'r', encoding='utf-8') as f:
                        reviews.append(f.read().replace('<br /><br />', '\n'))
              
            with open(os.path.join(path, f'urls_{pol}.txt'), 'r', encoding='utf-8') as file:
                urls = file.readlines()
                titles.extend([url.split('/')[4] for url in urls])
                usage.extend([split for url in urls])

    df = pd.DataFrame({'title_id':titles, 'text':reviews, 'rating':ratings, 'usage':usage})
    df.to_csv('aclImdb_reviews.csv', index=False)
    
    print('Done.')