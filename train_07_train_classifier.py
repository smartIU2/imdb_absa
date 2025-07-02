import argparse
import os
import pandas as pd
import pickle

from sklearn.svm import LinearSVC
from sklearn.model_selection import ShuffleSplit, validation_curve

from imdb_absa.config import Config
from imdb_absa.db import DB


if __name__ == "__main__":
    """ Train classifier based on aspect polarities """

    parser = argparse.ArgumentParser()
    parser.add_argument('--review_id', type=int, help='filter by review')
    parser.add_argument('--genre_id', type=int, help='filter by genre')
    parser.add_argument('--usage', type=str, help='filter by custom usage tag')
    parser.add_argument('--ratings', metavar='N', type=int, nargs='+', help='filter by ratings')
    filters = parser.parse_args()


    config = Config()    
    
    print('Assuring database')
    db = DB(config.database)
    db.assure_database()


    print('Getting review polarities')
    
    polarities = db.get_review_polarities_sparse(**vars(filters))
  
    models = (
        {'name':'multi-class', 'file_name':'SVC_5.pkl', 'col_features':3, 'col_label':'class', 'ds_factor':1},
        {'name':'binary', 'file_name':'SVC_2.pkl', 'col_features':3, 'col_label':'binary_class', 'ds_factor':1}
    )
    
    bst_Cs = []
    
    for args in models:
        
        print(f"Training {args['name']} model")
        
        # downsampling to most underrepresented class
        classes = polarities.groupby(args['col_label'])[args['col_label']]
        min_classCount = classes.transform('size').min() * args['ds_factor']
        classCounter = classes.cumcount()
        
        X = polarities[classCounter <= min_classCount].iloc[:,args['col_features']:].values
        y = polarities[classCounter <= min_classCount].iloc[:,polarities.columns.get_loc(args['col_label'])].values
         

        #use splits of dataset for param search
        model = LinearSVC(penalty="l1", loss="squared_hinge", dual=False)
        
        Cs = (0.024, 0.05, 0.12, 0.3, 0.5, 1.0, 1.8, 3.0, 5.0, 9.0, 21.0)
        shuffle_params = {
            "train_size": 0.7,
            "test_size": 0.2,
            "n_splits": 128,
            "random_state": 42
        }

        cv = ShuffleSplit(**shuffle_params)
        train_scores, test_scores = validation_curve(
            model,
            X,
            y,
            param_name="C",
            param_range=Cs,
            cv=cv,
            n_jobs=-1,
        )
        
        results = {'C': Cs, 'train_score': train_scores.mean(axis=1), 'test_score': test_scores.mean(axis=1)}
        
        results = pd.DataFrame(results)
        
        print(results)
      
        
        #fit model with best params to whole dataset
        bst_c = results.iloc[results['test_score'].idxmax()]['C']
        bst_Cs.append(bst_c)
        
        model = LinearSVC(penalty="l1", loss="squared_hinge", dual=False, C=bst_c)
        model.fit(X, y)
        
        print('Saving model as pickle for inference')
        
        model_path = os.path.join(config.model_classifier, args['file_name'])
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    print('Saving inputs to recreate models in different environment')
    
    header = '#SVC model inputs for imdb_absa_classifier\n'
    
    for model, c in zip(models, bst_Cs):
        header += f"#{model['name']} C:{c}\n"
        
    rows = polarities.to_csv(sep='\t', index=False)
    
    with open(os.path.join(config.model_classifier, 'inputs.tsv'), 'w') as f:
        f.write(header + rows)
    
    print('Done.')