import pandas as pd
import webbrowser 
import random
import spacy
import warnings

from spacy import displacy
from spacy.tokens.span import Span
from dash import Dash, html, dcc, ctx, callback, Output, Input, State
from itertools import count
from threading import Timer

from imdb_absa.db import DB
from imdb_absa.config import Config
from imdb_absa.nlp import NLP
from imdb_absa.utils import get_reviews


class imdb_absa_dash:
    """ dash app to visualize movie review sentiment analysis results
    
    IMPORTANT: designed for local deployment and a single user only
             , neither feasable nor safe to employ on a website
    """
    
    PORT = 8087
    
    COLORS = {
         'very negative': '#9b2727'
        ,'negative': '#da912b'
        ,'neutral': '#f4fb98'
        ,'positive': '#bef66a'
        ,'very positive': '#309929'
    }
    
    STYLES = {
        'title': {
            'textAlign': 'center',
            'margin-block-end': '0.2em',
            'color': '#486691'
        },
        'dropdown': {
            'width': '96%'
        },
        'loading': {
            'height':'auto'
        },
        'button': {
            'background-color': '#fff',
            'border': '1px solid #d5d9d9',
            'border-radius': '7px',
            'box-shadow': 'rgba(213, 217, 217, .5) 0 2px 5px 0',
            'box-sizing': 'border-box',
            'color': '#0f1111',
            'cursor': 'pointer',
            'line-height': '2',
            'padding': '0 10px 0 10px',
            'margin': '8px 8px 2px 0px',
            'vertical-align': 'middle'
        },
        'div': {
            'float':'left',
            'width':'40%',
            'min-width':'400px',
            'margin':'6px 25px 0px 0px'
        },
        'text_area': {
            'width': '100%',
            'height':'580px',
            'line-height':'2'
        },
        'rating': {
            'margin-top':'12px'
        },
        'aspect': {
            'display':'block'
        },
        'stars': {
            'width':'128px',
            'margin-left':'-6px'
        }
    }     
    
    PRESS_ANALYZE = "Press 'Analyze' to predict sentiments."
    
    
    def __init__(self, name, highlight_context, highlight_categories, gpu):
        
        self.highlight_context = highlight_context #determines size of highlights
        self.highlight_categories = highlight_categories #determines which aspects to highlight
        self.gpu = gpu #determines spacy model preference
        
        #TODO: make config
        genre_id = 1 #Action
        
        self.titles = db.get_titles_by_genre(genre_id) #filtered movies for selection
        self.aspect_terms = db.get_aspect_terms() #aspect terms to determine categories
        self.features = db.get_review_polarities_input() #DataFrame to use for classification
        
        self.genres = [] #genres for selected movie (there's always more than one)
        self.metadata = None #selected movie's title and principals
        
        self.counter = count(0) #counter to keep reloading gifs
        
        
        #dash app layout
        self.STYLES['disabled'] = self.STYLES['button'].copy()
        self.STYLES['disabled']['cursor'] = 'default'
        self.STYLES['disabled']['color'] = '#ced9eb'
        
        self._app = Dash(name)

        self._app.layout = html.Div([
            dcc.Location(id='url', refresh=False),
            html.Div(id='page-content')])


    def start_server(self):
        """ start local dash server and open in browser """
        Timer(3, webbrowser.open(f'http://localhost:{self.PORT}', new=0)).start()
        self._app.run_server(debug=False, port=self.PORT)


    @callback(Output('page-content', 'children'),[Input('url', 'pathname')])
    def generate_layout(url):
        
        return html.Div([
            html.H1(children='Aspect-Based Movie Review Sentiment Analysis', style=app.STYLES['title']),
            html.Label('Reviewed Movie:'),
            dcc.Dropdown(
                options=[{'label': t.title, 'value': t.id} for t in app.titles.itertuples(index=False)],
                style=app.STYLES['dropdown'], value='',
                id='ddTitle'
            ),
            html.Button('Get random movie review',
                style=app.STYLES['disabled'], n_clicks=0,
                id='btnRandom'),
            html.Button('Analyze',
                style=app.STYLES['disabled'], n_clicks=0,
                id='btnAnalyze'),
            html.Div([html.Div(style=app.STYLES['div'],
                children=[
                dcc.Loading(
                    id='loading_1',
                    type='default',
                    color='#486691',
                    delay_show=500,
                    show_initially=False,
                    parent_style=app.STYLES['loading'],
                    children=[dcc.Textarea(
                            style=app.STYLES['text_area'],
                            id='review_text'
                            )]
                    )]),
                html.Div(style=app.STYLES['div'],
                children=[
                dcc.Loading(
                    id='loading_2',
                    type='default',
                    color='#486691',
                    delay_show=500,
                    show_initially=False,
                    parent_style=app.STYLES['loading'],
                    children=[html.Div([html.Iframe(
                              sandbox='', srcDoc='',
                              style=app.STYLES['text_area'],
                              id='absa_output'
                            )])]
                    )])
                ]),
            html.Div([html.Div(id='ratings'),
                      html.Img(src=r'',alt='thumbs',hidden=True, id='thumb')
                    ])
            ])


    @callback(Output('review_text', 'value', allow_duplicate=True), Output('review_text', 'placeholder'),
              Output('btnRandom', 'disabled'), Output('btnAnalyze', 'disabled'),
              Output('btnRandom', 'style'), Output('btnAnalyze', 'style'),
              Input('ddTitle', 'value')
             ,prevent_initial_call='initial_duplicate')
    def set_title(title_id):
        """ load genres and metadata for selected title """
    
        if (title_id == None) or (title_id == ''):
            return '', 'Please select a movie.', True, True, app.STYLES['disabled'], app.STYLES['disabled']
    
        app.genres = db.get_genres_for_title(title_id)
        app.metadata = db.get_metadata_replacements(title_id)

        return '', 'Get a random review, or write your own.', False, False, app.STYLES['button'], app.STYLES['button']


    @callback(Output('review_text', 'value', allow_duplicate=True),
              Input('btnRandom', 'n_clicks'), State('ddTitle', 'value')
             ,prevent_initial_call=True)
    def get_random_review(click, title_id):
        """ get a random review from database, or fetch some online """
        
        reviews = db.get_reviews_for_title(title_id)
    
        if len(reviews.index) == 0:
            reviews = get_reviews('imdb', title_id)
            
            reviews = pd.DataFrame(reviews, columns=['originalText', 'rating'])
            reviews['title_id'] = title_id
            
            #save fetched reviews to sqlite database
            db.import_reviews(reviews)
            

        review_id = random.randint(0, len(reviews) - 1)
        
        return reviews.iloc[review_id]['originalText']


    @callback(Output('absa_output', 'srcDoc', allow_duplicate=True),
              Input('review_text', 'value')
             ,prevent_initial_call=True)
    def clear_output(review_text):
    
        if (not review_text) or (review_text == ''):
            return ''
    
        return app.PRESS_ANALYZE


    @callback(Output('absa_output', 'srcDoc', allow_duplicate=True),
              Input('btnAnalyze', 'n_clicks'), State('review_text', 'value')
             ,prevent_initial_call=True)
    def display_absa_output(click, review_text):
        """ display edited sentences from review text with polarity highlights """

        if not review_text:
            return ''

        # bugfix for 'RuntimeError: Expected all tensors to be on the same device' from torch inside spacy's pipe()
        # seems necessary for every callback, because they are running on different threads (?)
        if app.gpu:
            spacy.prefer_gpu()

        app.sentences = nlp.preprocess_text(review_text, app.metadata)
        
        docs, app.aspects = nlp.predict_absa(app.sentences['sentence'], app.aspect_terms)

        # convert aspects to spacy ents, to make use of displacy for rendering
        for doc, doc_aspects in zip(docs, app.aspects):
            if app.highlight_context:
                # highlight aspect plus context, as used for polarity prediction
                # needs trimming overlapping contexts
                ents = []
                max_stop = len(doc)
                for i in range(len(doc_aspects) - 1, -1, -1):
                    aspect = doc_aspects[i]
                    if any([cat in app.highlight_categories for cat in aspect.categories]):
                        ents.insert(0, Span(doc, aspect.context_start, min(aspect.context_stop, max_stop), label=aspect.label))
                        max_stop = aspect.context_start
                        
                doc.ents = ents
            else:
                # highlight aspect nouns only, as used for aspect prediction
                doc.ents = [Span(doc, aspect.start, aspect.stop, label=aspect.label) for aspect in doc_aspects
                           if any([cat in app.highlight_categories for cat in aspect.categories])]
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            # suppress spacy warnings for sentences without aspects
            
            html = displacy.render(docs, style='ent', options={"colors": app.COLORS}, page=False)

        return html.replace('style="line-height: 2.5;', 'style="line-height: 2;')
    
    
    @callback(Output('thumb', 'src'), Output('thumb', 'hidden'), Output('ratings', 'children'),
              Input('absa_output', 'srcDoc')
             ,prevent_initial_call=True)
    def display_ratings(absa_text):
        """ display rating from 1 to 5 stars for each predicted aspect category
          , plus a thumbs up or down from binary classification
        """

        if (not absa_text) or (absa_text == app.PRESS_ANALYZE):
            return '', True, ''
            
        preds, recommendation = nlp.predict_sentiments(app.genres, app.sentences, app.aspects, app.features.copy())
        
        ratings = [html.Div(children=[html.Label(children=pred.aspect, style=app.STYLES['aspect'])
                                    , html.Img(src=f'assets/stars_{pred.rating}.gif?id={next(app.counter)}', style=app.STYLES['stars']
                                    )], style=app.STYLES['rating'])
                    for pred in preds.itertuples() if pred.aspect in app.highlight_categories]

        up_down = 'up' if recommendation else 'down'
        
        return f'assets/thumbs_{up_down}.gif?id={next(app.counter)}', False, ratings
    

if __name__ == '__main__':

    config = Config()
    db = DB(config.database)

    print('loading models')
    nlp = NLP(config.model_spacy, config.model_spacy_exclude
            , config.model_maverick, config.pre_coref_resolution
            , config.model_setfit, True
            , config.model_classifier, True)


    print('starting server')
    running_on_gpu = config.model_spacy.endswith('trf')
    app = imdb_absa_dash(__name__, config.dash_highlight_with_context, config.dash_highlight_categories, running_on_gpu)
    app.start_server()