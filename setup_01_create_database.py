import pandas as pd
import os
import sys
from pathlib import Path

from imdb_absa.config import Config
from imdb_absa.db import DB


def write_titles_to_sql(chunk, db, types, incl_adult, year_min, runtime_min):
    """ writes titles to database
    
      Arguments:
        chunk: chunk from pandas iterable
        db: imdb_absa database
        types: types of imdb titles to include
        incl_adult: include adult titles
        year_min: exclude titles released before this year
        runtime_min: exclude titles with a lower runtime
    """
    
    df = pd.DataFrame(chunk)

    #filter
    df = df.query(f'startYear >= {year_min} and isAdult <= {incl_adult} and titleType == {types} and runtimeMinutes >= {runtime_min}')

    if not df.empty:
        
        #expand first three genres
        df = df.join(df['genres'].str.split(',',n=3,expand=True).add_prefix('genre_'))

        if not 'genre_1' in df.columns:
            df['genre_1'] = None
            
        if not 'genre_2' in df.columns:
            df['genre_2'] = None

        df = df[['tconst','titleType','primaryTitle','originalTitle','isAdult','startYear','endYear','runtimeMinutes','genre_0','genre_1','genre_2']].set_index('tconst')

        #save to sql
        df.to_sql(con=db.connection(), name='import_titles', index_label='tconst', if_exists='append')

def write_ratings_to_sql(chunk, db, votes_min):
    """ writes ratings to database
    
      Arguments:
        chunk: chunk from pandas iterable
        db: imdb_absa database
        votes_min: exclude titles with less votes
    """
    
    df = pd.DataFrame(chunk)
    
    #filter
    df = df.query(f'numVotes.notna() & numVotes >= {votes_min}')

    if not df.empty:    

        df = df.set_index('tconst')

        #save to sql
        df.to_sql(con=db.connection(), name='import_ratings', index_label='tconst', if_exists='append')

def write_principals_to_sql(chunk, db, titles):
    """ writes principals to database
    
      Arguments:
        chunk: chunk from pandas iterable
        db: imdb_absa database
        titles: current imdb titles in database
    """
    
    df = pd.DataFrame(chunk)

    #only import principals for titles in database
    df = df[df['tconst'].isin(titles['id'])]

    if not df.empty:    

        # clean up character name
        df['characters'] = df['characters'].str.strip(']["')
        df.loc[((df['category'] == 'self') | (df['characters'].str.match(r'(segment|\(segment|self)', case=False))), 'characters'] = None
        df['characters'] = df['characters'].replace(regex={r'[\(\{\[].+[\)\}\]]':''}).replace(regex={
                                                         r' - .+':''
                                                        ,r'[,:].+':''
                                                        ,'_':' '
                                                        ,'¨':'"'
                                                        ,r'[’´]':"'"}).replace(regex={
                                                         r'[\\\)\(\{\}\[\]\"*]':''}).str.strip("' ")

        df = df[['tconst','nconst','category','job','characters']].set_index('tconst')

        #save to sql
        df.to_sql(con=db.connection(), name='import_principals', index_label=['tconst'], if_exists='append')

def write_names_to_sql(chunk, db, names):
    """ writes names to database
    
      Arguments:
        chunk: chunk from pandas iterable
        db: imdb_absa database
        names: name_ids from current principals to import
    """
    
    df = pd.DataFrame(chunk)

    #only import names for new principals
    df = df[df['nconst'].isin(names['nconst'])]
    
    if not df.empty:    

        # clean up names
        df['primaryName'] = df['primaryName'].replace(regex={r'[*\(\)\"’´¨]':"'"})
        
        df = df[['nconst','primaryName','birthYear','deathYear']].set_index('nconst')

        #save to sql
        df.to_sql(con=db.connection(), name='import_names', index_label='nconst', if_exists='append')

def preprocess_names(db, query, extract, nulColumns):

    # get unprocessed names
    df = db.get_names_for_preprocess()
    
    # get names matching query
    df = df.query(query)
    
    if not df.empty:   
        # extract nameParts using regex
        df = df['primaryName'].str.extract(extract)

        # filter 2 character last names
        df = df[~((df['lastName'].str.len() < 3) | df['lastName'].str.lower().str.endswith('iii'))]

        # fill other nameParts with NULL
        for col in nulColumns:
            df[col] = None
       
        # update imbd_name table and set processed flag
        db.preprocess_names(df)
    
    
if __name__ == "__main__":
    """ Read in imdb datasets and save to sqlite database """

    config = Config()    
    
    print('Assuring database')
    db_path = Path(config.database)
    if len(db_path.parents) > 0:
        db_path.parents[0].mkdir(parents=True, exist_ok=True)
    
    db = DB(config.database)
    
    print('Removing current names & principals')
    db.clear_imdb_names()
    
    db.assure_database()

    print('Reading in titles & ratings', end='')
    sys.stdout.flush()
    
    with pd.read_table(config.import_titles, quoting=3, dtype={'startYear':'Int64','endYear':'Int64','runtimeMinutes':'Int64','genres':'string'}, na_values=['\\N'], chunksize=config.import_chunk_size) as jr:

        for chunk in jr:
            write_titles_to_sql(chunk, db, config.imdb_types, config.imdb_incl_adult, config.imdb_year_min, config.imdb_runtime_min)

            print('.', end='')
            sys.stdout.flush()    
            
    with pd.read_table(config.import_ratings, chunksize=config.import_chunk_size) as jr:

        for chunk in jr:
            write_ratings_to_sql(chunk, db, config.imdb_votes_min)

            print('.', end='')
            sys.stdout.flush()   

    
    print()
    db.import_titles()
    
    titles = db.get_titles()
    count_titles = len(titles.index)

    print('Reading in names & principals', end='')
    sys.stdout.flush()
    
    with pd.read_table(config.import_principals, na_values=['\\N'], chunksize=config.import_chunk_size) as jr:

        for chunk in jr:
            write_principals_to_sql(chunk, db, titles)

            print('.', end='')
            sys.stdout.flush() 
    
    del titles
    
    names = db.get_names_to_import()
    count_names = len(names.index)
    
    with pd.read_table(config.import_names, dtype={'birthYear':'Int64', 'deathYear':'Int64'}, na_values=['\\N'], chunksize=config.import_chunk_size) as jr:

        for chunk in jr:
            write_names_to_sql(chunk, db, names)

            print('.', end='')
            sys.stdout.flush()              
   
    print()
    db.import_names()


    print('Preprocessing names', end='')
    sys.stdout.flush()

    # process aliases
    preprocess_names(db, 'primaryName.str.match(".* \'.*\' .*")'
                       , r'(?P<firstName>.+) \'(?P<aliasName>.+)\' (?P<lastName>.+)'
                       , ['middleName'])
    
    print('.', end='')
    sys.stdout.flush() 
            
    # process long last names
    preprocess_names(db, 'primaryName.str.match(".+ ([dD][eai] |[dD]el |[vV][ao]n )")'
                       , r'(?P<firstName>.+) (?P<lastName>([dD][eai] |[dD]el |[vV][ao]n ).*)'
                       , ['middleName','aliasName'])
    
    print('.', end='')
    sys.stdout.flush() 
               
    # process names with suffix
    preprocess_names(db, 'primaryName.str.match(".* .* .*\.")'
                       , r'(?P<firstName>.+) (?P<lastName>.+ .+\.)'
                       , ['middleName','aliasName'])
    
    print('.', end='')
    sys.stdout.flush() 
                                   
    # process middle names
    preprocess_names(db, 'primaryName.str.count(" ") == 2'
                       , r'(?P<firstName>.+) (?P<middleName>.+) (?P<lastName>.+)'
                       , ['aliasName'])
     
    print('.', end='')
    sys.stdout.flush() 
              
    # process other names
    preprocess_names(db, 'primaryName.str.count(" ") == 1'
                       , r'(?P<firstName>.+) (?P<lastName>.+)'
                       , ['middleName','aliasName'])


    print()
    print('Flagging names for potential NER conflicts')

    ambigous_names = pd.read_csv(config.import_ambiguous)
    
    db.update_names_ambiguous(ambigous_names)
    
    
    print('Reading in aspect terms')
    aspects = pd.read_csv(config.import_aspects)

    db.import_aspects(aspects)


    print('cleaning up database')
    db.vacuum()

    print(f'Imported {count_titles} imdb titles and {count_names} names.')