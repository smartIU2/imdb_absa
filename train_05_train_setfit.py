import pandas as pd
import gc
import os
import sys
from pathlib import Path

from datasets import Dataset
from setfit import AbsaModel
from setfit import AbsaTrainer, TrainingArguments
from transformers import EarlyStoppingCallback
    
from imdb_absa.config import Config
from imdb_absa.db import DB


def train_model(database, ratings_csv, model_spacy, config):
    """ Train a new setfit absa model """

    print('Assuring database')
    db = DB(database)
    db.assure_database()

    print('Getting training dataset')

    # get annotated sentences
    df = db.get_absa_dataset(config.genre_id)

    # make sure the model understands ratings (4/10, ** out of ****, B-)
    ratings = pd.read_csv(ratings_csv)
    df = pd.concat([df, ratings], ignore_index=True)

    # split
    train_dataset = Dataset.from_pandas(df[df['dataset'] == 'train']).shuffle(seed=42)
    eval_dataset = Dataset.from_pandas(df[df['dataset'] == 'eval']).shuffle(seed=42)

    del db, ratings, df
    

    print('Preparing AbsaTrainer')

    model = AbsaModel.from_pretrained(config.model_base, spacy_model=model_spacy, spacy_disable_pipes=['ner','lemmatizer'])

    args = TrainingArguments(
        output_dir="models",
        num_epochs=(config.epochs_embedding, config.epochs_classifier),
        use_amp=config.use_amp,
        batch_size=config.batch_size,
        sampling_strategy=config.sampling_strategy,
        eval_strategy="steps",
        eval_steps=config.steps,
        save_steps=config.steps,
        load_best_model_at_end=True,
    )

    trainer = AbsaTrainer(
        model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)] if config.early_stopping_patience > 0 else None,
    )
    
    del train_dataset, eval_dataset
    
    gc.collect()
    
    
    print('Training model')

    trainer.train()

    # save models
    model.save_pretrained(config.save_as)

    print('Done.')


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('No model specified for training. Call like "train_model.py action"')
    else:
        config = Config()    
    
        model = sys.argv[1]
        if not model in config.train_models:
            print('No model "{model}" defined in config.json')
            
        else:
            train_model(config.database, config.train_ratings, config.model_spacy, config.train_models[model])
    