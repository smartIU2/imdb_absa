import pandas as pd
import re
import itertools
import logging
import warnings
import nltk
import spacy 
import os
import pickle

from spacy.language import Language
from spacy.tokens.span import Span
from nltk.sentiment.vader import SentimentIntensityAnalyzer

pd.options.mode.copy_on_write = True # to employ inplace replace

""" regex to find emojis """
# edited from: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1#gistcomment-3208085
EMOJIS = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
    "\U0001F300-\U0001F5FF"  # General Symbols & Pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002700-\U000027BF"  # Dingbats
    "\U00002300-\U000023FF"  # Technical Symbols
    "]+"
)


""" regex bounds for metadata replacement """
SEARCH_PREFIX = '((?<=^)|(?<=[ \(\'"/]))'
SEARCH_SUFFIX = '(?=[\.,;!\?\) \'"/]|$)'

""" abbreviations to replace conditionally """
ABBREVIATIONS = [('OG','original')
                ,('OMG','')
                ,('LOL', '')]

""" word combinations to remove / reduce  """
REPETITIONS_1 = [(r'director(\'s)? this movie','this movie')
               ,(r'(the|a|this) movie (the |this )?movie','this movie')
               ,(r'director (the )?director','the director')
               ,(r'composer (the )?composer','composer')
               ,(r'editor (the )?editor','editor')
               ,(r'character (the )?character','character')
               ,(r'actress (the )?actress','actress')
               ,(r'writer (the )?writer','writer')
               ,(r'actor (the )?actor','actor')
               ,(r'\(played by the actress\)','')
               ,(r'\(played by the actor\)','')
               ,(r'(Captain|Doctor|Dr|Mister|Mr|Lady|Ms) the (director|editor|writer|composer|character|actor|actress)',r'the \3')
               
               ,('this this','this')
               ,('this the','this')
               ,('the this','this')
               ,('the the','the')
               ]

REPETITIONS_2 = [('anotherperson','another person')
               ,('anotherfeature','another feature')
               ,('another another','another')
               ,('another feature-?another feature','another feature')
               ,('another person(\'s |-)?another person','another person')
               ,(r'(Captain|Doctor|Dr|Mister|Mr|Lady|Ms) another (feature|person)',r'another \3')
               ,(r'[\'\"]another (feature|person)[\'\"]','another feature')
               ,(r'another person(\'s)? (another feature|movie|film)','another feature')
               ,(r'(director )?another person the director','the director')
               ,(r'(writer )?another person the writer','the writer')
               ,(r'the director\'s? direction','the direction')
               ,(r'the writer\'s? writing','the writing')
               ,(r'the (actor|actress)\'s? acting','the acting')
               ,(r'another person ((the )?(character|actress|actor))',r'\2')
               ,(r'character another person','character')
               ,(r'another person character','character')
               ,(r'(?<!was )(?<!is )(?<!calls )(?<!called )(?<!calling )this movie another feature','another feature')
               ,(r'\((from )?another feature ?\)','')
               ,(r'\((by)? ?another person ?\)','')
               ,(r'10 (another person|website)', '10')
               ,(r'(another person)((, |, and | and )(another person))+','other persons')
               ,(r'(the actor|the actress|another person|other persons)((, |, and | and )(the actor|the actress|another person|other persons)){2,}','the actors')
               ,(r'(the character|another person|other persons)((, |, and | and )(the character|another person|other persons))+','the characters')
               ,(r'("?another feature"?)((, |, and | and )("?another feature"?))+','other features')
               ,(r'(another feature|other features) (films|movies|features)','other features')

               ,('the the','the')
               ,(r'(the|a|an) another', 'another')
               ]

""" maps the product of spacy's ent_iob and ent_type to its replacement """
TOKEN_MAPPING = {
                 380:'person' #PERSON
                ,388:'feature' #WORK_OF_ART
                ,1140:'another person'
                ,1164:'another feature'
                ,3800:''
                ,3880:''
                ,11400:'another'
                ,11640:'another'
                }

""" keep these entities from getting replaced """
NER_EXCLUSIONS = ['Oscar', 'Oscars', 'Shakespeare', 'Rocks', 'Awards', 'WHILST', 'Story', 'Storyline']   

""" whitelist for allowed coreference replacement (apart from 'this movie') """
COREF_SUBS = { 'the director': ['he', "he 's", 'she', "she 's", 'they', "they 're", 'his', 'her', 'their']
              ,'the actor': ['he', "he 's"]
              ,'the actress' : ['she', "she 's"]
              ,'the actors': ['they', "they 're"]
              ,'the composer' : ['he', "he 's", 'she', "she 's", 'they', "they 're"]
              ,'the writer' : ['he', "he 's", 'she', "she 's", 'they', "they 're"]
              ,'the writers' : ['they', "they 're"]
              ,'the editor' : ['he', "he 's", 'she', "she 's", 'they', "they 're"]
             }
 
""" 'sentences' to ignore """
NONE_SENTENCES = ['.','/.',' .','..','...','!','?','?!','(!)','(?)','*','**','***','website','website.']

""" split sentences into smaller parts, when exceeding this length """
SENTENCE_MAXLENGTH = 1000    

""" maps setfit labels to sentiment scores """
POLARITY_MAPPING = {
    'very negative': -1.0
   ,'negative': -0.5
   ,'neutral': 0.1
   ,'positive': 0.5
   ,'very positive': 1.0
   ,'none': 0.0
}


@Language.component("ner_split_fix")
def ner_split_fix(doc):
    """ spacy _sometimes_ includes leading or trailing characters for entities
        this component fixes this, to facilitate replacement
    """
    
    ents_changed = False
    
    new_ents = []
    for ent in doc.ents:
        if ((ent[0].text == '-') or 
            ((ent[0].text == '"') and not ent.text.endswith('"'))):
            new_ents.append(Span(doc, ent.start + 1, ent.end, label=ent.label))
            ents_changed = True
        elif ((ent[-1].text in ["'s", "-"]) or
            (ent[-1].text == '"') and not ent.text.startswith('"')):
            new_ents.append(Span(doc, ent.start, ent.end - 1, label=ent.label))
            ents_changed = True
        else:
            new_ents.append(ent)

    if ents_changed:
        doc.ents = new_ents
        
    return doc

    
def get_aspect_categories(aspect, aspect_terms):
    """ returns categories for a given aspect term """

    terms = set()
    categories = set()

    for a in aspect_terms.itertuples(index=False):
        if a.term in aspect and not any([a.term in term for term in terms]):
            terms.add(a.term)
            categories.add(a.category)

    if len(categories) > 1:
        categories.discard('Overall')
    elif len(categories) == 0:
        categories.add('Other')
        
    return categories
        

class NLP:
    """ encapsulates natural language processing tasks """

    def __init__(self, spacy_model, spacy_exclude
                     , coref_model=None, coref_active=False
                     , setfit_model=None, setfit_active=False
                     , clf_model=None, clf_active=False):
        """ load nlp models
        
          Arguments:
            spacy_model: spacy model to use for NER & POS tags
            spacy_exclude: pipes to exclude from spacy model (set to empty list when using _sm model)
            coref_model: maverick-coref model
            coref_active: activate coreference resolution and replacement (disable, if you can't build maverick-coref)
            setfit_model: local path to trained setfit absa models (without '-aspect' / '-polarity')
            setfit_active: load setfit model (disable for preprocessing only)
            clf_model: local path to folder containing pickled SVC models
            clf_active: load classifier model (disable for preprocessing only)
        """

        # SBD
        self._sent_detector = nltk.PunktTokenizer()

        # NER
        if spacy_model.endswith('trf'):
            spacy.prefer_gpu()
            
        self._nlp = spacy.load(spacy_model, exclude=spacy_exclude)
        self._nlp.add_pipe("ner_split_fix", after='ner')
        
        # SA
        self._sia = SentimentIntensityAnalyzer()

        # Coref
        if coref_active:
            self._coref = True
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
            
                from maverick import Maverick
                
                self._maverick = Maverick(coref_model)
        else:
            self._coref = False
            
        # Absa
        if setfit_active:
        
            from setfit import AbsaModel
            
            self._setfit = AbsaModel.from_pretrained(f'{setfit_model}-aspect', f'{setfit_model}-polarity', spacy_model=self._nlp
                          ,spacy_disable_pipes=['ner','ner_split_fix'])
       
        # SVC
        if clf_active:
            
            # 1 to 5 stars classification
            with open(os.path.join(clf_model, 'SVC_5.pkl'), 'rb') as f:
                self._clf_5 = pickle.load(f)
            
            # binary classification
            with open(os.path.join(clf_model, 'SVC_2.pkl'), 'rb') as f:
                self._clf_2 = pickle.load(f)


    def _check_name(self, name):
        doc = self._nlp(f'{name} once went to {name}.')
        
        if len(doc.ents) > 0:
            return doc.ents[0].label_ not in ['PERSON','ORG']

        return False

    def check_names(self, pdSeries):
        """ check if spacy would recognize names as another NER type
            (used to create ambiguous_names.csv)
        
            Returns:
                True, if potentially conflicting
        """
        
        return pdSeries.apply(self._check_name)
        

    def normalize_reviews(self, pdSeries, normalForm = 'NFKC'):
        """ cleanup common review irregularities
          , make some replacements to facilitate tokenization
          , and normalize unicode characters
        """

        return pdSeries.str.replace(
                                r'(([A-Z]\. )([A-Z]\. )+([A-Z]\.))', lambda a: a.group(0).replace(' ',''), regex=True #spaced abbreviations
                                ).replace(regex={
                                r'(https?://|www\.).*?(?=$|[\s\)\'"])':'website' #url
                               ,r'([\s\(\'"])[^\s\(\'"]+\.(com|net|html?)(?=$|[\s\)\'"])':r'\1website' #TODO: Exclude websites in movie titles
                               ,r'(^|[\s\(])[#@][^\d\'" ].*?(?=$|[\s\)])':r'\1' #tags
                               ,r'([^\.,:;!\?]) ?\n ?([A-Z])':r'\1. \2' #new line missing full stop
                               ,r'([^\.,:;!\?]) ?\n ?[>-]+':r'\1. ' # bullet list
                               ,r'([\.,:;!\?]) ?\n ?[>-]+':r'\1 '
                               ,r'([^\.,:;!\?]) ?\n ?[\d]{1,2}[\.\)]+(?!\d)':r'\1. ' # numbered list
                               ,r'([\.,:;!\?]) ?\n ?[\d]{1,2}[\.\)]+(?!\d)':r'\1 '
                               ,r'^\.\.\.':'' #continuation from review title
                               ,r'\\':'/' # \
                               ,r'==+':' ' # 'lines'
                               ,r'-{4,}':' '
                               ,r'\+{4,}':' '
                               ,r'\*{6,}':' '
                               ,r'\?[\?\.]+':'?' # multiples
                               ,r'![!\.]+':'!'
                               ,r',,+':','
                               ,r' ?\.( ?\. ?)+\.':'...'
                               ,EMOJIS:'. '
                               ,r' ?[:;] ?[\)\|]+':'. '
                               ,r' ?:[\(]+(?=$|[\s\.])':'. '
                               ,r' \^[-_]?\^ ?':'. '
                               ,r':?=\)':''
                               ,r'(<\)|<3)':''
                               ,r'\*(sigh|cough|yawn|rolls eyes)\*':' ' #emotions
                               ,r'[Ff][*#@Uu-][*#@Cc-][*#@Kk]':'f*ck' #censorship (to remove @ )
                               ,r'[Cc][Rr]@[Pp]':'cr*p'
                               ,r'[Bb]@[Ll][Ll][Ss]':'balls'
                               ,r'[Ss][Hh]@[Tt]':'sh*t'
                               ,r'[”¨“]':'"' #unusual characters
                               ,r"(’|´|''|`|‘)":"'"
                               ,r'[★☆⭐]':'*'
                               ,r' @ ':' at '
                               ,r'\(=':'('
                               }).replace(regex={
                                r'[_¡~\{\}><°♥﻿￼]':' ' #unused characters (includes invisible characters!)
                               ,r'\n\(?website\)?$':'' #advertising
                               ,r'(^|[\s\(\'"])[a-zA-Z][a-zA-Z0-9]*@[a-zA-Z0-9]*(?=$|[\s\)\'"])':r'\1website'
                               ,r'(^|\s)[*\']([^\s\*]+?)[*\']($|[\.,:;\s!\?])':r'\1\2\3' # 'quotes' around single words
                               ,r'[\r\n\t\f\v]':' ' #other white spaces
                               ,r'\( ?\d{4}[\!\? ]?\)':'' #info in parentheses
                               ,r'\(R\.I\.P\.\)':''
                               ,r'[Tt][Ll];?[Dd][Rr](:| - )?':'In summary, '
                               ,r'([a-zA-Z])\(([a-zA-Z])\)':r'\1\2' # suffix letter(s)
                               ,r'\(([d-zD-Z])\)([a-zA-Z])':r'\1\2' # prefix (le)tters
                               ,r'\(([a-zA-Z]{2})\)([a-zA-Z])':r'\1\2'
                               }).replace(regex={
                                r'website website':'website'
                               ,r',([a-zA-Z])':r', \1' #missing space
                               ,r'([a-z])\.([A-Z])':r'\1. \2'
                               ,r'\.\.\.([^ \)\'"])':r'... \1'
                               ,r';([^ ])':r'; \1'
                               ,r'([!\?])([^!\?\)\'" ])':r'\1 \2'
                               ,r'([^ ])(\()':r'\1 \2'
                               ,r'(\)[,:;\.]?)([^ ,:;\.])':r'\1 \2'
                               ,r'(\D:)([^ ])':r'\1 \2'
                               ,r'([eE]\.g\.|[iI]\.e\.)[ :]':r'\1, ' #missing comma
                               ,r'(etc\.) ([a-z])':r'\1, \2'
                               ,r' ?--+ ?':', '
                               ,r' Im ':" I'm " # common misspellings
                               ,r' isnt ':" isn't "
                               ,r' arnt ':" are not "
                               ,r' didnt ':" didn't "
                               ,r' back ground ':' background '
                               ,r'[Mm]ake-[Uu]p':'makeup'
                               ,r'[Ss]low-[Mm]o(tion)?':'slow motion'
                               ,r'[Ss](cript|creen)( -)?[Ww]riter':'writer'
                               ,r'master piece':'masterpiece'
                               ,r' alround ':' all-round '
                               ,r'[Cc]a?pt\.':'Captain' # abbreviations
                               ,r'Dr\.':'Doctor'
                               ,r'Mr\.':'Mister'
                               ,r'Mr?s\.':'Lady'
                               ,r'[Cc]\.[Gg]\.[Ii]\.?':'CGI'
                               ,r'([ \(\'":])[Vv]ol\. ?':r'\1Volume '
                               ,r'R\.I\.P\.(?!D\.)':'farewell,'
                               ,r'[\., ]+b/c ':', because '
                               ,r'( |\()w/ ':r'\1with '
                               ,r'( |\()w/o ':r'\1without '
                               ,r' [Vv]/?[Ss]\.? ':' versus '
                               ,r'[Cc]ontd\.':'continued'
                               ,r' [Nn][or]\. (?=\d)':' number '
                               ,r'appr\.':'approximately'
                               ,r'( |\()pp\.':r'\1pages'
                               }).replace(regex={
                                r' -(?=[^ \d])':' ' # random hyphen
                               ,r'(?<=[^ A-F])-( |$)':' '
                               ,r'\.?\.\. +([A-Z])':r'. \1' # unnecessary ellipsis
                               ,r'\. (\. )+':'. ' #extra white spaces
                               ,r' , ':', '
                               ,r'  +':' '
                               }).str.normalize(normalForm).str.strip()
                               
                               
    def _replace_searchStr(self, pdSeries, strSearch, strReplace, replace_quotes, remove_parentheses, noCase=False):

        # replace with quotes
        if replace_quotes:
            pdSeries = pdSeries.str.replace(f'[\'\"]{strSearch}[\'\"]', strReplace, case=False, regex=True)
            
        # remove in parentheses
        if remove_parentheses:
            pdSeries = pdSeries.str.replace(f'\( ?{strSearch} ?\)', '', case=False, regex=True)
        
        # replace rest 
        if noCase:
            caseSensitive = False
        else: #case-sensitive for single words
            caseSensitive = not " " in strSearch
        
        pdSeries = pdSeries.str.replace(f'{SEARCH_PREFIX}{strSearch}{SEARCH_SUFFIX}', strReplace, case=caseSensitive, regex=True)

        #TODO: add s after possesive ', when principal name ends with s, e.g., "Reeves' performance" -> "the actor's performance"

        return pdSeries.str.replace(r'  +', ' ', regex=True)

    def replace_metadata(self, pdSeries, metadata):
        """ replace titles and names of principals """

        replacedStrings = []
     
        # replace titles and names
        # (this is done in a loop on purpose, so longer strings get replaced first)
        for md in metadata.itertuples(index=False):

            replacedStrings.extend(md.strSearch.upper().split())
            
            if md.conflicts or md.ambiguous:
                # delay replacement until NER, if e.g., title equals character name
                continue
                
            pdSeries = self._replace_searchStr(pdSeries, md.strSearchEscaped, md.strReplace, True, md.category == 'PERSON')

        # remove double words and confusing word combinations
        for rep in REPETITIONS_1:
            pdSeries = self._replace_searchStr(pdSeries, rep[0], rep[1], False, False, True)
            
        # replace/remove common abbreviations, unless they are part of a title or name
        for abb in ABBREVIATIONS:
            if abb[0] not in replacedStrings:
                pdSeries = self._replace_searchStr(pdSeries, abb[0], abb[1], True, True, True)
        
        return pdSeries
        

    def _split_sentences(self, text):

        sentences = []
        open_parentesis = False
        
        for sentence in self._sent_detector.tokenize(text):
        
            #clean up split sentences
            if sentence in NONE_SENTENCES:
                continue
            #numbered list 1.
            if re.fullmatch(r'#?(\d){1,2}[\.\)]+', sentence):
                continue
                
            sentence = re.sub(r'^(\d){1,2}[\.\)]+ ', '', sentence)
            
            #inline numbered list 2.
            if re.search(r'(?<! is)(?<! the)(?<! of)(?<! Volume) (\d)\. ', sentence):
                nr_list = re.split(r'(?<! is)(?<! the)(?<! of)(?<! Volume) (\d)\. ', sentence)
                for sentence in nr_list:
                    if sentence == '':
                        continue
                        
                    if (not sentences) or (re.match(r'[A-Z]', sentence)):
                        sentences.append(sentence)
                    else:
                        sentences[-1] = f"{sentences[-1]}{'' if (sentences[-1][-1] in '.,:;!?') else ','} {sentence}"
                
                open_parentesis = False
            #inline numbered list 3)
            elif (not open_parentesis) and re.search(r' \d\) ', sentence) and not re.search(r'\(.* \d\) ', sentence):
                nr_list = re.split(r' \d\) ', sentence)
                for sentence in nr_list:
                    if sentence == '':
                        continue
                        
                    if (not sentences) or (re.match(r'[A-Z]', sentence)):
                        sentences.append(sentence)
                    else:
                        sentences[-1] = f"{sentences[-1]}{'' if (sentences[-1][-1] in '.,:;!?') else ','} {sentence}"
                
            #fix for nltk splitting sentences too freely, whenever there is a punctuation
            elif sentences and (sentence.startswith("'s ") or re.match(r'[\'"]?[\.,:;!\?\)]', sentence)):
                sentences[-1] = f'{sentences[-1]}{sentence}'
                open_parentesis = False
            #join short sentences, e.g., 'Yes.', 'Why?'
            elif (not ' ' in sentence) and any(char.isalpha() for char in sentence):
                if not sentences: #one-word review beginning
                    sentences.append(sentence)
                    open_parentesis = True
                    continue
                
                sentences[-1] = f'{sentences[-1]} {sentence}'
                open_parentesis = False
            #greedily split sentences with screwed punctuation
            elif len(sentence) > SENTENCE_MAXLENGTH:
                sentences.extend([f'{s.strip()}.' for s in re.split(r'[\.;](?=[ a-dfh-zA-Z])', sentence)])
                open_parentesis = False
            elif open_parentesis:
                #TODO: consider more than two sentences in parentheses 
                sentences[-1] = f'{sentences[-1]} {sentence}'
                open_parentesis = False
            else:
                sentences.append(sentence)
        
            #keep parentheses together
            if ('(' in sentence) and (not ')' in sentence):
                open_parentesis = True
        
        return sentences

    def split_sentences(self, pdSeries):
        """ split review into sentences """

        return pdSeries.apply(self._split_sentences)
        
        
    def get_tokens_from_sentences(self, pdSeries):
        """ tokenize sentences for reviews with unknown title
           (otherwise this is done implicitly in _replace_propernames_corefs)
        """
        
        return pdSeries.apply(lambda sentences: [[(token.text, token.whitespace_, token.pos_) for token in sentence] for sentence in self._nlp.pipe(sentences, disable=['ner', 'ner_split_fix'])])

    def _get_corefs(self, sentences):
        """ performs coreference resolution
        
            Returns: dictionary with token_index: coref_text
        """
        
        tokens = self._maverick.predict([[token.text for token in sentence] for sentence in sentences])

        refs = {}

        clusters_offsets = tokens['clusters_token_offsets']
        
        if clusters_offsets:    
        
            clusters_text = [[text.lower() for text in cluster] for cluster in tokens['clusters_token_text']]
        
            direct_reference = False
            for cluster in clusters_text:
                for text in cluster:
                    if text in ('the movie', 'this movie', 'the film', 'this film', 'this flick'):
                        direct_reference = True
                        break
                else:
                    # assume that 'it' and 'this' refers to the reviewed movie, if the movie itself is never mentioned
                    continue
                break
        
            for texts, offsets in zip(clusters_text, clusters_offsets):
                if (('this movie' in texts) or ('this film' in texts) or ('this flick' in texts) or
                    ((not direct_reference) and all(text in ['it', 'this', "it 's", 'its', 'this one'] for text in texts)) or
                    ((('this' in texts) or ('this one' in texts)) and any(text.split(' ')[-1] in ['movie','film','flick'] for text in texts))):
                
                    for text, offset in zip(texts, offsets):
                        if text.startswith('the '):
                            refs[offset[0]] = 'this'
                        elif text in ['this', 'it', "it 's", 'movie', 'film', 'flick']:
                            refs[offset[0]] = 'this movie'
                        elif text == 'its': # (irrelevant if possesive or a typo)
                            refs[offset[0]] = "this movie's"
                        elif text in ['this one', 'this film', 'this flick']:
                            refs[offset[1]] = "movie"

                else:
                    for principal, pronouns in COREF_SUBS.items():
                        if (principal in texts):
                            for text, offset in zip(texts, offsets):
                                if text in pronouns:
                                    refs[offset[0]] = principal + "'s" if text in ('his', 'her', 'their') else principal
                            
                            break
                
        return refs


    def _replace_propername_coref(self, token, ref):

        # replace with coref text
        if ref is not None:    
            return ref

        # replace proper name (in a way that keeps the whitespace of the last token)
        token_last = 1 if (token.is_sent_end or token.nbor().ent_iob != 1) else 10
        token_flag = token.ent_iob * token.ent_type * token_last
        if (token_flag in TOKEN_MAPPING) and ((token_last != 1) or (token.ent_iob != 3) or (token.text not in NER_EXCLUSIONS)):
        
            # keep . at the end of a sentence
            # (for names with suffix, as well as mistakes in spacy's tokenizer)
            if token.is_sent_end and token.text.endswith('.'):
                return TOKEN_MAPPING[token_flag] + '.'
            else:
                return TOKEN_MAPPING[token_flag]
            
        # keep original
        return token.text


    def _replace_propernames_corefs(self, pdRow):
        
        def getRefText(index):
            if (not pd.isna(pdRow['refs'])) and (index in pdRow['refs']):
                return pdRow['refs'][index]
        
        # propernames & corefs are replaced at the same time in this matter, because:
        # 1) both need the tokenized sentences
        # 2) maverick only uses en_core_web_sm internally, when passing untokenized text
        # 3) the whole review at once is needed for coreference resolution
        # 4) spacy's token elements are read only
        
        counter = itertools.count(0)
        return [[(self._replace_propername_coref(token, getRefText(next(counter))), token.whitespace_, token.pos_) for token in sentence] for sentence in pdRow['tokens']]
        

    def replace_propernames_corefs(self, pdSeries, metadata):
        """ replacement of proper names and coreferences with spacy & maverick
            (should be done after replacing conflict-free metadata)
            
            returns new pdSeries with individual tokens
        """

        conflicts = metadata[metadata['ambiguous'] | metadata['conflicts']]

        if len(conflicts.index) == 0:
            tokens = pdSeries.apply(lambda sentences: [sentence for sentence in self._nlp.pipe(sentences)])
        else:
            tokens = pdSeries.apply(self._handle_name_conflicts, conflicts=conflicts)

        refs = {}
        if self._coref:
            refs = tokens.apply(self._get_corefs)
           
        df = pd.DataFrame({'tokens': tokens, 'refs': refs})
        
        return df.apply(self._replace_propernames_corefs, axis=1)
      
      
    def _handle_name_conflicts(self, sentences, conflicts):

        docs = []
        for sentence in self._nlp.pipe(sentences):
            
            new_sentence = sentence.text
            
            searchStrings = conflicts['strSearch'].unique()
            for strSearch in searchStrings:
            
                repl = conflicts[conflicts['strSearch'] == strSearch]
                
                strSearch_nocase = strSearch.casefold()
                
                if strSearch_nocase not in sentence.text.casefold():
                    continue
                
                char_offset = 0 #handles offset in spacy spans, when a name has to be replaced more than once in a sentence
                
                for ent in sentence.ents:
                    if strSearch_nocase == ent.text.casefold():
                        replacement = repl[repl['category'] == ent.label_]
                        if len(replacement.index) != 0:
                            #TODO: currently the first entry from the matching category is taken, i.e., it doesn't handle one person having multiple jobs
                            strReplace = replacement.iloc[0]['strReplace']
                            new_sentence = new_sentence[:ent.start_char + char_offset] + strReplace + new_sentence[ent.end_char + char_offset:]
                            char_offset = len(strReplace) - (ent.end_char - ent.start_char)
                
                
                # if NER failed for title/name conflict, use title as fallback #TODO: make config
                caseSensitive = True
                if " " in strSearch:
                    caseSensitive = False
                    strSearch = strSearch_nocase
                    
                if not repl.iloc[0]['ambiguous'] and (strSearch in (new_sentence if caseSensitive else new_sentence.casefold())):
                    flags = re.U if caseSensitive else re.I
                    new_sentence = re.sub(f'{SEARCH_PREFIX}{repl.iloc[0]["strSearchEscaped"]}{SEARCH_SUFFIX}', repl.iloc[0]["strReplace"], new_sentence, flags=flags)
            
            if new_sentence == sentence.text:
                docs.append(sentence)
            else:
                new_sentence = re.sub(r'[\'\"]this movie[\'\"]', 'this movie', new_sentence)
                
                for rep in REPETITIONS_1:
                    new_sentence = re.sub(f'{SEARCH_PREFIX}{rep[0]}{SEARCH_SUFFIX}', rep[1], new_sentence, flags=re.I)
            
                new_sentence = re.sub(r'  +', ' ', new_sentence)
                
                docs.append(self._nlp(new_sentence))
        
        return docs


    def get_sentence_from_tokens(self, pdSeries, metadata):
        """ concatenates tokens (incl. whitespaces)
            and then cleans up repetitions from propername & coref replacement
        """

        pdSeries = pdSeries.apply(lambda tokens: ''.join([token[0] + token[1] for token in tokens if token[0] != ''])).str.replace(r'  +', ' ', regex=True)
     
        # remove double words and unintended word combinations
        for rep in REPETITIONS_2:
            pdSeries = self._replace_searchStr(pdSeries, rep[0], rep[1], False, False, True)
        
        pdSeries = pdSeries.str.replace(r'  +', ' ', regex=True)
        
        pdSeries = pdSeries[~pdSeries.isin(['','.','(', 'another person'])]      
        
        # apply sentence case  #TODO: consider sentences starting with, e.g., "
        return pdSeries.apply(lambda sentence: sentence[0].upper() + sentence[1:])


    def add_aspect_term(self, pdSeries):
        """ add aspect term to special phrases """
        
        # overall rating
        ratings = pdSeries.str.fullmatch(r'[\(\[]?((my )?(final )?(rating ?(: ?|- |is )?))?((\d[\d\.,]{0,2}\+?)|([\*]+))(/| out of )(5\*?|10\*?|100|([\*]+))[\)\]]? ?(stars|for me|from me)?[\.!]?', case=False)
        pdSeries.loc[ratings] = 'Overall: ' +  pdSeries[ratings].str.replace(r'((my )?(final )?(rating ?(: ?|- |is )?))',''
                                        ,regex=True, case=False).str.replace('stars','', case=False).astype(str)

        return pdSeries


    def estimate_polarity(self, pdSeries):
        """ estimate sentence polarity with vader
        
            Returns:
                new pdSeries with pos, neg, neutral & compound scores
                (though only compound score is used in the app)
        """

        polarity = pdSeries.apply(self._sia.polarity_scores)
        return pd.DataFrame(polarity.tolist(), index=pdSeries.index)
    

    def preprocess_text(self, text, metadata, sent_polarity = True):
        """ go through all preprocess steps
            to turn an original review into inferable sentences
            
        Arguments:
            text: review text as str
            metadata: imdb metadata - pass None to skip metadata & propername replacement
            sent_polarity: pass False to skip estimating sentence polarities
        
        Returns:
            pandas Series with sentences
        """
        
        df = pd.DataFrame({'text': [text]})
        
        df['text'] = self.normalize_reviews(df['text'])

        if (metadata is not None) and (len(metadata.index) != 0):
            df['text'] = self.replace_metadata(df['text'], metadata)
        
            splits = self.split_sentences(df['text']) 

            df['tokens'] = self.replace_propernames_corefs(splits, metadata)
           
            df = df.explode('tokens', ignore_index=True)
            
            df['sentence'] = self.get_sentence_from_tokens(df['tokens'], metadata)

            df = df[~pd.isna(df['sentence'])]

        else:    
            df['sentence'] = self.split_sentences(df['text'])
            
            df = df.explode('sentence', ignore_index=True)
            
        df = df[['sentence']]

        df['sentence'] = self.add_aspect_term(df['sentence'])
        
        if sent_polarity:
            polarity = self.estimate_polarity(df['sentence'])

            df = pd.concat([df, polarity], axis=1)
        
        return df
        

    def predict_absa(self, pdSeries, aspect_terms):
        """ predict aspect based sentiments for series of sentences
        
        Arguments:
            pdSeries: series of sentences
            aspect_terms: predefined aspect terms mapped to categories
            
        Returns:
            List of spacy docs
            List of aspects incl. polarity label and aspect categories
        """
        
        if not hasattr(self, '_setfit'):
            logging.warning(f"Sentiment prediction impossible, as NLP was initiated without a setfit model.")


        sentences = pdSeries.values

        docs, aspects = self._setfit.predict_to_docs(sentences)
  
        for doc_aspects in aspects:
            for aspect in doc_aspects:
                aspect.categories = get_aspect_categories(aspect.context.lower(), aspect_terms)
        
        return docs, aspects
  

    def predict_sentiments(self, genres, sentences, aspects, features):
        """ calculates final sentiment scores
        
        Arguments:
            genres: List of genres for reviewed title
            sentences: DataFrame with sentences and estimated polarities
            aspects: List of aspects with polarity label and categories
            features: DataFrame with columns matching classifier inputs
            
        Returns:
            DataFrame with aspect_category: discrete rating (1,2,3,4,5)
           ,Boolean for overall binary classification
        """

        if not hasattr(self, '_clf_2'):
            logging.warning(f"Classification impossible, as NLP was initiated without classifier models.")


        #set review polarity
        features.at[0, 'mean_review_polarity'] = sentences['compound'].mean()

        #set genre 'flags'
        for genre in genres:
            if f'genre_{genre}' in features.columns:
                features.at[0, f'genre_{genre}'] = 1

        #set aspect 'count' features
        sentences['absa'] = [[(aspect.categories, aspect.label) for aspect in doc_aspects] for doc_aspects in aspects]
        
        sentences = sentences.explode('absa', ignore_index=True)
        
        sentences['absa'] = sentences['absa'].apply(lambda a: a if isinstance(a, tuple) else ({'None'},'none'))
            
        sentences[['aspect','polarity']] = pd.DataFrame(sentences['absa'].tolist(), index=sentences.index)

        sentences = sentences.explode('aspect', ignore_index=True)

        sentences['polarity_value'] = sentences['polarity'].map(POLARITY_MAPPING)
        
        for sentence in sentences.itertuples(index=False):
            features.at[0, f"{sentence.aspect}_{sentence.polarity.replace(' ','_')}"] += 1
            
        #set aspect 'mean' features
        aspect_means = sentences.groupby(by=['aspect'])[['compound','polarity_value']].mean().reset_index()
        
        for mean in aspect_means.itertuples(index=False):
            features.at[0, f'{mean.aspect}_mean'] = mean.compound
            
        #compute aspect rating  
        aspect_means['rating'] = pd.cut(aspect_means['polarity_value'], [-1,-0.6,-0.2,0.2,0.6,1.0], labels=[1,2,3,4,5], include_lowest=True).astype('str')

        if 'Overall' not in aspect_means['aspect'].values:
            
            #predict overall rating
            overall = pd.DataFrame({'aspect':'Overall', 'rating':self._clf_5.predict(features.values)})
            aspect_means = pd.concat([aspect_means, overall], ignore_index=True)
  
        #predict binary classification
        recommendation = self._clf_2.predict(features.values)
    
        return aspect_means, recommendation