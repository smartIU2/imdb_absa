import argparse
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from matplotlib import colormaps
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, ConfusionMatrixDisplay

from imdb_absa.config import Config
from imdb_absa.db import DB


if __name__ == "__main__":
    """ Export classifier results to .png """

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
  
    X = polarities.iloc[:,polarities.columns.get_loc('binary_class') + 1:].values

    models = ( 
        {'name':'multi-class', 'file':'SVC_5.pkl', 'col':'class',
         'values':(1,2,3,4,5), 'labels':('1-2','3-4','5-6','7-8','9-10'),
         'rotate_y_labels':False, 'ylabel':'Movie Rating', 'xlabel':'Predicted Rating'},
         
        {'name':'binary', 'file':'SVC_2.pkl', 'col':'binary_class',
         'values':(0,1), 'labels':('negative', 'positive'),
         'rotate_y_labels':True, 'ylabel':'Review Sentiment', 'xlabel':'Predicted Sentiment'}
    )

    for model in models:
        
        print(f"\nEvaluating {model['name']} classifier")

        with open(os.path.join(config.model_classifier, model['file']), 'rb') as f:
            clf = pickle.load(f)

        y = polarities[model['col']].values
        preds = clf.predict(X)

        # calculate metrics
        accuracy = balanced_accuracy_score(y, preds)
        mcc = matthews_corrcoef(y, preds)
        
        print(f'Balanced Accuracy: {accuracy}')
        print(f'Matthews Correlation Coefficient: {mcc}')
        
        # create confusion matrix
        disp = ConfusionMatrixDisplay.from_predictions(y, preds, labels=model['values'], normalize='true', cmap='cividis')
        
        # set custom labels
        disp.ax_.set_xticklabels(model['labels'])
        disp.ax_.set_yticklabels(model['labels'])
        
        if model['rotate_y_labels']:
            disp.ax_.tick_params(axis='y', labelrotation=90)
        
        plt.ylabel(model['ylabel'])
        plt.xlabel(f"{model['xlabel']}\n\nBalanced Accuracy: {accuracy:.3f}  MCC: {mcc:.3f}")

        # set fixed range for color values, to make the colors comparable between models
        disp.im_.set_clim(vmin=0.1, vmax=0.9)
        
        # save as png
        matrix_file = f"{model['file'].replace('.pkl', '_matrix.png')}"
        plt.savefig(matrix_file, dpi=300, bbox_inches='tight')
        
        print(f'Saved confusion matrix to {matrix_file}')