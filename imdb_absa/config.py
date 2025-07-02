import json

class Config:
     """ encapsulates .json config file """

     def __init__(self):

          with open('config.json') as file:
               conf = json.load(file)

               self.database = conf['database']

               self.import_chunk_size = conf['import_chunk_size']
               
               self.import_titles = conf['import_titles']
               self.import_names = conf['import_names']
               self.import_principals = conf['import_principals']
               self.import_ratings = conf['import_ratings']
               
               self.import_ambiguous = conf['import_ambiguous']
               self.import_aspects = conf['import_aspects']
               self.import_franchises = conf['import_franchises']
               
               self.import_aclImdb = conf['import_aclImdb']

               self.imdb_types = conf['imdb_types']
               self.imdb_incl_adult = conf['imdb_incl_adult']
               self.imdb_year_min = conf['imdb_year_min']
               self.imdb_runtime_min = conf['imdb_runtime_min']
               self.imdb_votes_min = conf['imdb_votes_min']
               
               self.pre_normal = conf['pre_normal']
               self.pre_coref_resolution = conf['pre_coref_resolution']
               
               self.export_sentences = conf['export_sentences']
               self.import_annotations = conf['import_annotations']
               
               self.model_spacy = conf['model_spacy']
               self.model_spacy_exclude = conf['model_spacy_exclude']
               self.model_maverick = conf['model_maverick']
               self.model_setfit = conf['model_setfit']
               self.model_classifier = conf['model_classifier']
               
               self.dash_highlight_with_context = conf['dash_highlight_with_context']
               self.dash_highlight_categories = conf['dash_highlight_categories']
               
               
               self.train_ratings = conf['train_ratings']
               
               self.train_models = dict()
               for model in conf['train_models']:
                    self.train_models[model] = TrainModel(conf['train_models'][model])
               
               
class TrainModel:

     def __init__(self, params):

          self.model_base = params['model_base']
          self.save_as = params['save_as']
          self.genre_id = params['genre_id']          
          self.use_amp = params['use_amp']          
          self.epochs_embedding = params['epochs_embedding']
          self.epochs_classifier = params['epochs_classifier']
          self.sampling_strategy = params['sampling_strategy']
          self.batch_size = params['batch_size']
          self.steps = params['steps']
          self.early_stopping_patience = params['early_stopping_patience']