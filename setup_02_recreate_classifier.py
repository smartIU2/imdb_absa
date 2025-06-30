import pandas as pd
import os
import pickle

from sklearn.svm import LinearSVC

from imdb_absa.config import Config


if __name__ == "__main__":
    """ Recreate classifier from saved inputs
       (this is done to prevent sharing a pickle file, or having onnx dependencies)
    """

    config = Config()    

    path_inputs = os.path.join(config.model_classifier, 'inputs.tsv')
    
    Cs = []
    with open(path_inputs, 'r') as file:
        for i, line in enumerate(file):
            if i in (1, 2):
                Cs.append(float(line.split(':')[1].strip()))
            if i > 1:
                break

    polarities = pd.read_csv(path_inputs, sep='\t', skiprows=3)
    
    models = (
        {'name':'multi-class', 'file_name':'SVC_5.pkl', 'col_features':3, 'col_label':1, 'C':Cs[0]},
        {'name':'binary', 'file_name':'SVC_2.pkl', 'col_features':3, 'col_label':2, 'C':Cs[1]}
    )
    
    for args in models:
        
        print(f"Training {args['name']} model")
        
        X = polarities.iloc[:,args['col_features']:].values
        y = polarities.iloc[:,args['col_label']].values

        #fit model with saved param to whole dataset
        model = LinearSVC(penalty="l1", loss="squared_hinge", dual=False, C=args['C'])
        model.fit(X, y)
        
        print('Saving model as pickle for inference')
        
        model_path = os.path.join(config.model_classifier, args['file_name'])
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


    print('Done.')