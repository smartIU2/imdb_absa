import os
import sqlite3
import time
import itertools
import pandas as pd
import logging

from contextlib import closing

class DB:
    """ encapsulates all database requests """

    def __init__(self, connection : str):
        """ instanciate database encapsulation
        
          Arguments:
            connection: path to sqlite3 database file
        """

        self._connection = connection
        self._journal = f'{connection}-journal'

    
    def connection(self):
        """ returns a database connection """       
        return sqlite3.connect(self._connection)        

    def await_access(self):
        """ check sqlite3 journal file to handle concurrent database requests """
        counter = 0
        while os.path.isfile(self._journal) and counter < 25:
            time.sleep(0.5)
            counter += 1     
            
    def vacuum(self):
        """ helper function to clean and compress the database """

        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                cmd.execute('''DROP TABLE IF EXISTS import_ratings''')
                cmd.execute('''DROP TABLE IF EXISTS import_titles''')
                cmd.execute('''DROP TABLE IF EXISTS import_principals''')
                cmd.execute('''DROP TABLE IF EXISTS import_names''')
                cmd.execute('''DROP TABLE IF EXISTS import_aspects''')
                cmd.execute('''DROP TABLE IF EXISTS import_reviews''')
                cmd.execute('''DROP TABLE IF EXISTS import_sentences''')
                cmd.execute('''DROP TABLE IF EXISTS import_words''')
                
                conn.commit()
                
                cmd.execute('''VACUUM;''')   


    def assure_database(self):
        """ creates database file and schema objects if neccessary
        
        Tables:
           imdb_title(id, titleType, primaryTitle, originalTitle, isAdult, startYear, runtimeMinutes, averageRating, numVotes
                    , subtitle, ambiguous_title, ambiguous_subtitle)
           imdb_title_genres(title_id, genre_id)
           imdb_name(id, primaryName, firstName, middleName, lastName, aliasName, noAliasName, birthYear, deathYear, ambiguous)
           imdb_principals(title_id, name_id, category, job, character, ambiguous)
           
           genre(id, displayName)
           #franchise(id, displayName, alternativeName, titlePart)
           aspect(id, category)
           aspect_words(id, aspect_id, POS, term)
      
           review(id, originalText, normalizedText, title_id, genre_flag, rating, tokenized, usage)
           review_sentence(id, review_id, sentence, polNeu, polNeg, polPos, polComp)
           sentence_word(id, sentence_id, POS, word, sentencePart)
           sentence_aspect(id, sentence_id, aspect_id, aspect_term, ordinal, sentiment_term, polarity, verified)

        """

        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                cmd.execute('''PRAGMA foreign_keys = ON;''')

                cmd.execute('''CREATE TABLE IF NOT EXISTS imdb_title(
                               id TEXT PRIMARY KEY                              
                              ,titleType TEXT NOT NULL    
                              ,primaryTitle TEXT NOT NULL
                              ,originalTitle TEXT NOT NULL                              
                              ,isAdult INTEGER NOT NULL DEFAULT(0)
                              ,startYear INTEGER NOT NULL                              
                              ,endYear INTEGER
                              ,runtimeMinutes INTEGER
                              ,averageRating REAL
                              ,numVotes INTEGER
                              ,subtitle TEXT
                              ,ambiguous_title INTEGER NOT NULL DEFAULT(0)
                              ,ambiguous_subtitle INTEGER NOT NULL DEFAULT(0)
                              )''')
                              
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_title_year ON imdb_title(startYear)''')
                
                
                cmd.execute('''CREATE TABLE IF NOT EXISTS genre(
                               id INTEGER PRIMARY KEY
                              ,displayName TEXT NOT NULL
                              )''')
                              
                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_genre ON genre(displayName)''')
                
                cmd.execute('''INSERT OR IGNORE INTO genre(id, displayName)
                               VALUES(1, 'Action')
                                    ,(2, 'Adventure')
                                    ,(3, 'Animation')
                                    ,(4, 'Biography')
                                    ,(5, 'Comedy')
                                    ,(6, 'Crime')
                                    ,(7, 'Documentary')
                                    ,(8, 'Drama')
                                    ,(9, 'Family')
                                    ,(10, 'Fantasy')
                                    ,(11, 'History')
                                    ,(12, 'Horror')
                                    ,(13, 'Musical')
                                    ,(14, 'Mystery')
                                    ,(15, 'Romance')
                                    ,(16, 'Sci-Fi')
                                    ,(17, 'Sport')
                                    ,(18, 'Thriller')
                                    ,(19, 'War')
                                    ,(20, 'Western')
                            ''')
                
                
                cmd.execute('''CREATE TABLE IF NOT EXISTS imdb_title_genres(
                               title_id TEXT NOT NULL
                              ,genre_id INTEGER NOT NULL    
                              ,FOREIGN KEY(title_id) REFERENCES imdb_title(id) ON DELETE CASCADE
                              ,FOREIGN KEY(genre_id) REFERENCES genre(id) ON DELETE CASCADE
                              )''')

                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_genre_title ON imdb_title_genres(title_id,genre_id)''')
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_title_genres ON imdb_title_genres(title_id)''')
                
                
                cmd.execute('''CREATE TABLE IF NOT EXISTS imdb_name (
                               id TEXT PRIMARY KEY
                              ,primaryName TEXT NOT NULL
                              ,firstName TEXT
                              ,middleName TEXT
                              ,lastName TEXT
                              ,aliasName TEXT
                              ,noAliasName TEXT
                              ,birthYear INTEGER
                              ,deathYear INTEGER
                              ,processed INTEGER DEFAULT (0) NOT NULL
                              ,ambiguous INTEGER DEFAULT (0) NOT NULL
                              )''')

                cmd.execute('''CREATE INDEX IF NOT EXISTS index_primaryname ON imdb_name(primaryName)''')
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_lastname ON imdb_name(lastName)''')

                cmd.execute('''CREATE TABLE IF NOT EXISTS imdb_principals(
                               title_id TEXT NOT NULL
                              ,name_id TEXT NOT NULL                              
                              ,category TEXT
                              ,job TEXT
                              ,character TEXT
                              ,ambiguous INTEGER DEFAULT (0) NOT NULL
                              ,FOREIGN KEY(title_id) REFERENCES imdb_title(id) ON DELETE CASCADE
                              ,FOREIGN KEY(name_id) REFERENCES imdb_name(id) ON DELETE CASCADE
                              )''')

                cmd.execute('''CREATE INDEX IF NOT EXISTS index_principals_title ON imdb_principals(title_id)''')
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_principals_name ON imdb_principals(name_id)''')

                # -- not implemented --
                # cmd.execute('''CREATE TABLE IF NOT EXISTS franchise(
                               # id INTEGER PRIMARY KEY
                              # ,displayName TEXT NOT NULL
                              # ,alternativeName TEXT
                              # ,titlePart TEXT
                              # )''')
                              
                # cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_franchise ON franchise(displayName)''')
                

                cmd.execute('''CREATE TABLE IF NOT EXISTS aspect(
                               id INTEGER PRIMARY KEY
                              ,category TEXT NOT NULL
                              )''')
                              
                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_aspect ON aspect(category)''')
                
                cmd.execute('''INSERT OR IGNORE INTO aspect(id, category)
                               VALUES(0, 'Other')
                                    ,(1, 'Audio')
                                    ,(2, 'Effects')
                                    ,(3, 'Scene')
                                    ,(4, 'Story')
                                    ,(5, 'Direction')
                                    ,(6, 'Cast')
                                    ,(7, 'Message')
                                    ,(8, 'Emotion')
                                    ,(9, 'Comparison')
                                    ,(10, 'General')
                                    ,(11, 'Overall')
                                    ,(12, 'Audience')
                            ''')
                
                
                cmd.execute('''CREATE TABLE IF NOT EXISTS aspect_words(
                               id INTEGER PRIMARY KEY
                              ,aspect_id INTEGER NOT NULL
                              ,POS TEXT NOT NULL
                              ,term TEXT NOT NULL
                              ,FOREIGN KEY(aspect_id) REFERENCES aspect(id) ON DELETE CASCADE
                              )''')
                
                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_aspect_term ON aspect_words(aspect_id,term)''')                
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_aspect_words ON aspect_words(aspect_id)''')
                

                cmd.execute('''CREATE TABLE IF NOT EXISTS review(
                               id INTEGER PRIMARY KEY                              
                              ,originalText TEXT NOT NULL    
                              ,normalizedText TEXT
                              ,title_id TEXT
                              ,genre_flag INTEGER DEFAULT (0) NOT NULL
                              ,rating INTEGER
                              ,tokenized INTEGER DEFAULT (0) NOT NULL
                              ,usage TEXT
                              ,FOREIGN KEY(title_id) REFERENCES imdb_title(id) ON DELETE CASCADE
                              )''')
                              
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_review_title ON review(title_id)''')
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_review_genre ON review(genre_flag)''')
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_review_usage ON review(usage)''')


                cmd.execute('''CREATE TABLE IF NOT EXISTS review_sentence(
                               id INTEGER PRIMARY KEY                              
                              ,review_id INTEGER NOT NULL    
                              ,sentence TEXT NOT NULL
                              ,polNeu REAL
                              ,polNeg REAL
                              ,polPos REAL
                              ,polComp REAL
                              ,analyzed INTEGER DEFAULT (0) NOT NULL
                              ,FOREIGN KEY(review_id) REFERENCES review(id) ON DELETE CASCADE
                              )''')
                              
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_sentence_review ON review_sentence(review_id)''')


                cmd.execute('''CREATE TABLE IF NOT EXISTS sentence_word(
                               id INTEGER PRIMARY KEY
                              ,sentence_id INTEGER NOT NULL
                              ,word TEXT NOT NULL
                              ,POS TEXT NOT NULL
                              ,sentencePart INTEGER DEFAULT(0) NOT NULL
                              ,FOREIGN KEY(sentence_id) REFERENCES review_sentence(id) ON DELETE CASCADE
                              )''')
                              
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_word_sentence ON sentence_word(sentence_id)''')
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_word_pos ON sentence_word(pos, word)''')
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_word_sentencePart ON sentence_word(word, sentencePart)''')
             

                cmd.execute('''CREATE TABLE IF NOT EXISTS sentence_aspect(
                               id INTEGER PRIMARY KEY                              
                              ,sentence_id INTEGER NOT NULL  
                              ,aspect_id INTEGER NOT NULL
                              ,aspect_term TEXT NOT NULL
                              ,ordinal INTEGER DEFAULT (0) NOT NULL
                              ,sentiment_term TEXT
                              ,polarity TEXT NOT NULL
                              ,verified INTEGER DEFAULT (0) NOT NULL
                              ,FOREIGN KEY(sentence_id) REFERENCES review_sentence(id) ON DELETE CASCADE
                              )''')
                             
                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_sentence_aspect ON sentence_aspect(sentence_id,aspect_id,aspect_term,ordinal,verified)''') 
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_aspect_verified ON sentence_aspect(sentence_id, verified)''')                
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_aspect_sentence ON sentence_aspect(sentence_id)''')


                conn.commit()
                
       
    def import_titles(self):
        """ import new imdb titles """
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:            

                cmd.execute('''PRAGMA foreign_keys = ON;''')
                
                cmd.execute('''INSERT OR IGNORE INTO imdb_title(id, titleType, primaryTitle, originalTitle, isAdult, startYear
                                                              , runtimeMinutes, averageRating, numVotes)
                            SELECT import_titles.tconst, titleType, primaryTitle, originalTitle, isAdult, startYear, runtimeMinutes, averageRating, numVotes
                            FROM import_titles
                            INNER JOIN import_ratings ON import_titles.tconst = import_ratings.tconst
                            ''')

                cmd.execute('''INSERT OR IGNORE INTO imdb_title_genres(title_id, genre_id)
                            SELECT import_titles.tconst, genre.id
                            FROM import_titles
                            INNER JOIN genre
                            ON genre.displayName IN (genre_0, genre_1, genre_2)
                            WHERE import_titles.tconst IN (SELECT id FROM imdb_title)''')
                
                conn.commit()
               
                cmd.execute('''DELETE FROM imdb_title
                            WHERE NOT EXISTS (SELECT * FROM imdb_title_genres WHERE imdb_title.id = imdb_title_genres.title_id)
                            ''')
               
                cmd.execute('''UPDATE imdb_title
                            SET subtitle = TRIM(SUBSTR([primaryTitle], INSTR([primaryTitle], ' - ') + 3), '-''/#')
                            WHERE INSTR([primaryTitle], ' - ') > 0
                            ''')
                cmd.execute('''UPDATE imdb_title
                            SET subtitle = TRIM(SUBSTR([primaryTitle], INSTR([primaryTitle], ': ') + 2), '-''/#')
                            WHERE INSTR([primaryTitle], ' - ') = 0 AND INSTR([primaryTitle], ': ') > 0
                            ''')
                cmd.execute('''UPDATE imdb_title
                            SET subtitle = NULL
                            WHERE length(subtitle) < 4
                                 OR NOT subtitle GLOB '*[a-zA-Z]*'
                                 OR SUBSTR(subtitle, 1, 1) GLOB '[0-9a-z]'
                                 OR lower(subtitle) = 'the movie'
                                 OR lower(subtitle) = 'movie'
                                 OR subtitle LIKE 'Part %'
                                 OR subtitle LIKE 'Vol. %'
                            ''')
               
                conn.commit()

    def get_titles(self):
        """ get all imdb titles """
        
        return pd.read_sql_query('''SELECT id, primaryTitle, startYear
                                     FROM imdb_title
                                 ''', self.connection())
                                 
    def get_titles_for_selection(self, genre_id = None):
        """ get imdb titles for selection, optionally filtered by genre """
    
        genre_filter = '' if genre_id is None or genre_id < 1 else f"WHERE g.genre_id = '{genre_id}'"
    
        return pd.read_sql_query(f'''SELECT DISTINCT t.id, t.primaryTitle || ' (' || t.startYear || ')' as title
                                     FROM imdb_title t
                                     INNER JOIN imdb_title_genres g
                                     ON g.title_id = t.id
                                     {genre_filter}
                                     ORDER BY t.primaryTitle || ' (' || t.startYear || ')'
                                 ''', self.connection())
                                  
    def clear_imdb_names(self):
        """ drop imdb_principals and imdb_name for clean import """
        
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:            

                cmd.execute('''DROP TABLE IF EXISTS imdb_principals''')
                cmd.execute('''DROP TABLE IF EXISTS imdb_name''')
                
                cmd.execute('''DROP TABLE IF EXISTS import_names''')
                cmd.execute('''DROP TABLE IF EXISTS import_titles''')
                cmd.execute('''DROP TABLE IF EXISTS import_principals''')
                cmd.execute('''DROP TABLE IF EXISTS import_ratings''')
                
                conn.commit()

    def get_names_to_import(self):
        """ get name_ids from temp table """
        
        return pd.read_sql_query('''SELECT DISTINCT nconst
                                     FROM import_principals
                                 ''', self.connection())

    def import_names(self):
        """ import names / principals for new imdb titles """
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:            

                cmd.execute('''PRAGMA foreign_keys = ON;''')
                
                
                cmd.execute('''INSERT OR IGNORE INTO imdb_name(id, primaryName, birthYear, deathYear)
                            SELECT nconst, primaryName, birthYear, deathYear
                            FROM import_names
                            ''')
    
                cmd.execute('''INSERT OR IGNORE INTO imdb_principals(title_id, name_id, category, job, character)
                            SELECT import_principals.tconst, import_principals.nconst, category, job, 
                                   CASE WHEN characters = '' THEN NULL ELSE characters END
                            FROM import_principals
                            WHERE import_principals.nconst IN (SELECT id FROM imdb_name)
                            ''')

                conn.commit()

    def import_aspects(self, aspects):
        """ import aspect terms """

        aspects.to_sql(con=self.connection(), name='import_aspects', if_exists='replace')

        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:
   
                cmd.execute('''INSERT OR IGNORE INTO aspect_words(aspect_id, POS, term)
                            SELECT aspect_id, POS, term
                            FROM import_aspects
                            ''')
                
                conn.commit()
                
                cmd.execute('''DROP TABLE import_aspects''')
                    
                conn.commit()
                                
    def get_names_for_preprocess(self):
        """ get unprocessed imdb primary names"""
        
        return pd.read_sql_query(f'''SELECT id, primaryName
                                 FROM imdb_name
                                 WHERE processed = 0
                                 ''', self.connection(), index_col='id')     
    
    def preprocess_names(self, names):
        """ preprocess names """
        
        names.to_sql(con=self.connection(), name='import_names', index_label='id', if_exists='replace')
        
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:            

                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_import_names ON import_names(id)''')

                cmd.execute('''UPDATE imdb_name
                            SET firstName = import.firstName
                               ,middleName = import.middleName
                               ,lastName = CASE WHEN import.lastName IS NULL OR import.lastName GLOB '*[0-9]*'
                                                THEN NULL
                                                ELSE trim(import.lastName, '-''/') END
                               ,aliasName = import.aliasName
                               ,noAliasName = CASE WHEN import.aliasName IS NOT NULL
                                                   THEN import.firstName || " " || import.lastName
                                                   ELSE NULL END
                               ,processed = 1
                            FROM (SELECT id, firstName, middleName, lastName, aliasName FROM import_names) AS import
                            WHERE import.id = imdb_name.id
                            ''')
                
                conn.commit()    

    def update_names_ambiguous(self, ambiguous_names):
        """ flag potentially conflicting names """
        
        ambiguous_names.to_sql(con=self.connection(), name='import_names', if_exists='replace')
        
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:            
    
                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_ambiguous ON import_names(name)''')
    
                cmd.execute('''UPDATE imdb_name
                               SET lastName = upper(substr(lastName, 1, 1)) || substr(lastName, 2)
                               WHERE lastName IS NOT NULL
                               AND lastName NOT LIKE 'd%'
                               AND NOT lastName GLOB '*[- '']*'
                               AND substr(lastName, 1, 1) GLOB '[a-z]'
                            ''')
                                
                cmd.execute('''UPDATE imdb_name
                               SET ambiguous = 1
                               WHERE aliasName IS NOT NULL
                                AND (length(REPLACE(REPLACE(aliasName, ' ',''), '.','')) < 3
                                 OR (aliasName NOT LIKE '% %'
                                 AND NOT aliasName GLOB '*[a-zA-z]*'))
                            ''')  
                            
                cmd.execute('''UPDATE imdb_name
                               SET ambiguous = 1
                               FROM (SELECT name FROM import_names) AS import
                               WHERE import.name = imdb_name.lastName
                                  OR import.name = imdb_name.aliasName
                                  OR (imdb_name.lastName IS NULL
                                  AND import.name = imdb_name.primaryName)
                            ''')
                
                cmd.execute('''UPDATE imdb_principals
                               SET ambiguous = 1
                               WHERE character IS NOT NULL
                                AND (length(REPLACE(REPLACE(character, ' ',''), '.','')) < 3
                                 OR (character NOT LIKE '% %'
                                 AND NOT character GLOB '*[a-zA-z]*'))
                            ''')
                            
                cmd.execute('''UPDATE imdb_principals
                               SET ambiguous = 1
                               FROM (SELECT name FROM import_names) AS import
                               WHERE import.name = imdb_principals.character
                            ''')
                
                
                cmd.execute('''UPDATE [imdb_title]
                                  SET ambiguous_title = 0
                                     ,ambiguous_subtitle = 0
                            ''')
                
                cmd.execute('''UPDATE [imdb_title]
                               SET ambiguous_title = 1
                               WHERE primaryTitle NOT LIKE '% %'
                                AND (length(REPLACE(REPLACE(primaryTitle, ' ',''), '.','')) < 3
                                 OR (primaryTitle NOT LIKE '% %'
                                 AND NOT primaryTitle GLOB '*[a-zA-z]*'))
                            ''')
                
                cmd.execute('''UPDATE [imdb_title]
                               SET ambiguous_subtitle = 1
                               WHERE subtitle IS NOT NULL
                                AND (length(REPLACE(REPLACE(subtitle, ' ',''), '.','')) < 3
                                 OR (subtitle NOT LIKE '% %'
                                 AND NOT subtitle GLOB '*[a-zA-z]*'))
                            ''')
                
                cmd.execute('''UPDATE imdb_title
                               SET ambiguous_title = 1
                               FROM (SELECT name FROM import_names) AS import
                               WHERE import.name = imdb_title.primaryTitle
                            ''')
                
                cmd.execute('''UPDATE imdb_title
                               SET ambiguous_subtitle = 1
                               FROM (SELECT name FROM import_names) AS import
                               WHERE import.name = imdb_title.subtitle
                            ''')
                
                conn.commit()  
 
                                 
    def get_reviews_for_preprocess(self):
        """ get reviews for preprocessing """

        return pd.read_sql_query('''SELECT id as review_id, title_id, originalText
                                    FROM review
                                    WHERE normalizedText IS NULL
                                 ''', self.connection())    
       
    def import_reviews(self, reviews):
        """ import reviews """
        
        reviews.to_sql(con=self.connection(), name='import_reviews', if_exists='replace')
    
        col_text = 'text'
        for column in reviews.columns.values:
            if 'text' in column.lower():
                col_text = column
                break
    
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:            

                count = cmd.execute(f'''SELECT COUNT(*)
                                        FROM import_reviews
                                        WHERE EXISTS (SELECT id FROM imdb_title WHERE imdb_title.id = import_reviews.title_id)
                                    ''').fetchone()[0]

                cmd.execute(f'''INSERT OR IGNORE INTO review(originalText, title_id, rating, usage)
                               SELECT {col_text}, title_id, rating, usage
                               FROM import_reviews
                               WHERE EXISTS (SELECT id FROM imdb_title WHERE imdb_title.id = import_reviews.title_id)
                            ''')
                
                cmd.execute('''UPDATE review
                              SET genre_flag = genres.flag
                              FROM (SELECT g.[title_id], SUM(power(2, g.genre_id)) AS flag
                                    FROM imdb_title_genres g
                                    GROUP BY g.[title_id]) AS genres
                              WHERE genres.title_id = review.title_id
                            ''')
                
                conn.commit()
                
                cmd.execute('''DROP TABLE import_reviews''')
                
                conn.commit()
                
        return count

    def update_reviews(self, reviews):
        """ update reviews with normalized text
            (before metadata replacement)
        """
        
        reviews.to_sql(con=self.connection(), name='import_reviews', if_exists='replace')
        
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:            

                cmd.execute('''CREATE INDEX IF NOT EXISTS index_import_reviews ON import_reviews(review_id)''')

                cmd.execute('''UPDATE review
                            SET normalizedText = import.normalizedText
                            FROM (SELECT review_id, normalizedText FROM import_reviews) AS import
                            WHERE import.review_id = review.id''')
                
                cmd.execute('''DROP TABLE import_reviews''')
                
                conn.commit()    
    
    
    def get_new_sentence_id(self):
        """ gets new sentence id for import
           (used to facilitate performant insert of sentences + words)
        """
        
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:  
                
                result = cmd.execute('''SELECT MAX(id) FROM review_sentence
                                     ''').fetchone()
        
        if result[0] is None:
            return 1
            
        return result[0] + 1
        
    
    def import_sentences(self, sentences):
        """ import review sentences """
        
        sentences.to_sql(con=self.connection(), name='import_sentences', index_label='id', if_exists='replace')
        
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:            

                cmd.execute('''PRAGMA foreign_keys = ON;''')
                
                cmd.execute('''INSERT OR IGNORE INTO [review_sentence]
                               ([id]
                               ,[review_id]
                               ,[sentence]
                               ,[polNeu]
                               ,[polNeg]
                               ,[polPos]
                               ,[polComp])
                            SELECT
                               [id]
                              ,[review_id]
                              ,[sentence]
                              ,[neu]
                              ,[neg]
                              ,[pos]
                              ,[compound]
                            FROM [import_sentences]
                            ''')

                conn.commit() 
    
    def import_words(self, words):
        """ import words """
        
        words.to_sql(con=self.connection(), name='import_words', index_label='sentence_id', if_exists='replace')
        
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:      
            
                cmd.execute('''PRAGMA foreign_keys = ON;''')
                
                cmd.execute('''INSERT OR IGNORE INTO [sentence_word]
                                   ([sentence_id]
                                   ,[POS]
                                   ,[word]
                                   ,[sentencePart])
                              SELECT [sentence_id]
                                  ,[POS]
                                  ,[word]
                                  ,[sentencePart]
                              FROM [import_words]
                            ''')

                cmd.execute('''UPDATE review
                               SET tokenized = 1
                               FROM (SELECT DISTINCT review_id FROM import_sentences) AS import
                               WHERE import.review_id = review.id
                            ''')
                            
                conn.commit()   
                
                cmd.execute('''DROP TABLE import_sentences''')
                cmd.execute('''DROP TABLE import_words''')
    
                conn.commit() 
    
    def get_genres_for_title(self, title_id):
        """ returns list of genre names for a given title """
        
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:      
           
                genres = cmd.execute(f'''SELECT genre.[displayName]
                                        FROM imdb_title_genres g
                                        INNER JOIN genre
                                        ON g.genre_id = genre.id
                                        WHERE g.title_id = '{title_id}'
                                     ''').fetchall()
            
        
        genres = [item for l in genres for item in l]
        
        return genres
            
    def get_reviews_for_title(self, title_id):
        """ get reviews for a title"""

        return pd.read_sql_query(f'''SELECT id, originalText
                                 FROM review
                                 WHERE title_id = '{title_id}'
                                  AND IFNULL(usage, '') != 'exclude'
                                 ''', self.connection())
        
    def get_metadata_replacements(self, title_id, include_firstNames=False):
        """ get metadata replacements for an imdb title"""
        
        # get title / subtitle, principal names and character names
        # incl. ambiguous flag to identify names unfit for regex replacement
        metadata = pd.read_sql_query(f'''WITH cteTitle AS
                                        (SELECT id, REPLACE(primaryTitle, '¡','') AS primaryTitle, subtitle
                                                ,ambiguous_title, ambiguous_subtitle
                                         FROM imdb_title WHERE id = '{title_id}')
                                    ,ctePrincipals AS
                                        (SELECT CASE WHEN p.category = 'self' THEN 'the person' ELSE 'the ' || category END AS category
                                               ,p.character
                                               ,primaryName, firstName, lastName, aliasName, noAliasName
                                               ,n.ambiguous AS ambiguous_name
                                               ,p.ambiguous AS ambiguous_character
                                          FROM [imdb_principals] p
                                          INNER JOIN imdb_name n
                                           ON p.name_id = n.id
                                          WHERE p.title_id = '{title_id}'
                                            AND p.category NOT LIKE '%\_%' ESCAPE '\\')
                                            
                                    SELECT 'WORK_OF_ART' AS category
                                         , mdType
                                         , strSearch AS strSearch
                                         , 'this movie' AS strReplace
                                         , ambiguous
                                    FROM
                                    (  
                                        SELECT 'title' AS mdType, primaryTitle AS strSearch, ambiguous_title AS ambiguous FROM cteTitle
                                        UNION
                                        SELECT 'title' AS mdType, REPLACE(primaryTitle, '!', '') AS strSearch, ambiguous_title AS ambiguous FROM cteTitle
                                        UNION
                                        SELECT 'title' AS mdType, REPLACE(primaryTitle, '?', '') AS strSearch, ambiguous_title AS ambiguous FROM cteTitle
                                        UNION
                                        SELECT 'title' AS mdType, REPLACE(primaryTitle, ' - ', '- ') AS strSearch, ambiguous_title AS ambiguous FROM cteTitle
                                        UNION
                                        SELECT 'title' AS mdType, REPLACE(primaryTitle, ': The Movie', '') AS strSearch, ambiguous_title AS ambiguous FROM cteTitle
                                        WHERE primaryTitle LIKE '%: The Movie'
                                        UNION
                                        SELECT 'title' AS mdType, REPLACE([primaryTitle], ':', '') AS strSearch, ambiguous_title AS ambiguous FROM cteTitle
                                        UNION
                                        SELECT 'title' AS mdType, REPLACE(REPLACE([primaryTitle], ':', ''), ' - ', ' ') AS strSearch, ambiguous_title AS ambiguous FROM cteTitle
                                        UNION
                                        SELECT 'title' AS mdType, REPLACE(REPLACE([primaryTitle], ':', ''), ' - ', '- ') AS strSearch, ambiguous_title AS ambiguous FROM cteTitle
                                        UNION
                                        
                                        SELECT 'subtitle' AS mdType,  subtitle AS strSearch, ambiguous_subtitle AS ambiguous
                                        FROM cteTitle WHERE subtitle IS NOT NULL
                                    ) AS titles

                                    UNION ALL
                                    
                                    SELECT 'PERSON' AS category
                                         , mdType
                                         , strSearch
                                         , strReplace
                                         , ambiguous
                                    FROM
                                    (
                                        SELECT 'firstName' AS mdType, firstName AS strSearch, 'firstName' AS strReplace, 0 AS ambiguous
                                        FROM ctePrincipals WHERE firstName IS NOT NULL
                                        UNION
                                        
                                        SELECT 'name' AS mdType, primaryName AS strSearch, category AS strReplace
                                              ,CASE WHEN lastName IS NULL THEN ambiguous_name ELSE 0 END AS ambiguous
                                        FROM ctePrincipals WHERE noAliasName IS NULL
                                        UNION
                                        SELECT 'name' AS mdType, aliasName AS strSearch, category AS strReplace
                                              ,ambiguous_name AS ambiguous
                                        FROM ctePrincipals WHERE aliasName IS NOT NULL
                                        UNION
                                        SELECT 'name' AS mdType, noAliasName AS strSearch, category AS strReplace, 0 AS ambiguous
                                        FROM ctePrincipals WHERE noAliasName IS NOT NULL
                                        UNION
                                        SELECT 'name' AS mdType, lastName AS strSearch, category AS strReplace
                                              ,ambiguous_name AS ambiguous
                                        FROM ctePrincipals WHERE lastName IS NOT NULL
                                        UNION
                                        
                                        SELECT 'character' AS mdType, character AS strSearch, 'the character' AS strReplace
                                              ,ambiguous_character AS ambiguous
                                        FROM ctePrincipals
                                        WHERE character IS NOT NULL
                                    ) AS names 
                                    ''', self.connection(), dtype={'ambiguous':bool})
        
        if len(metadata.index) == 0:
            logging.warning(f"No metadata found for '{title_id}'. Please make sure you typed the correct id, you imported the current imdb dataset, and the movie is not filtered out by the 'imdb_' entries in the config.")
        
        # replace abbreviations etc., to match nlp.normalize_reviews output
        metadata['strSearch'] = metadata['strSearch'].replace(regex={
                                r'([a-z])\.([A-Z])':r'\1. \2'
                               ,r'[Mm]ake-[Uu]p':'makeup'
                               ,r'[Cc]a?pt\.':'Captain'
                               ,r'Dr\.':'Doctor'
                               ,r'Mr\.':'Mister'
                               ,r'Mr?s\.':'Lady'
                               ,r'([ \(\'":])[Vv]ol\. ?':r'\1Volume '
                               ,r' [Vv]/?[Ss]\.? ':' versus '
                               ,r'[Nn][Oor]\.':'number'
                               ,r'([a-zA-Z])\(([a-zA-Z])\)':r'\1\2'
                               ,r'\(([d-zD-Z])\)([a-zA-Z])':r'\1\2'
                               ,r'\(([a-zA-Z]{2})\)([a-zA-Z])':r'\1\2'
                               })
        
        # split titles with parentheses
        st_md = metadata[metadata['strSearch'].str.endswith(')') & (metadata['mdType'] == 'title')].copy()
        if len(st_md.index) != 0:
            
            st_md['strSearch'] = st_md['strSearch'].str.split('(')
            st_md = st_md[st_md['strSearch'].str.len() == 2]
            
            if len(st_md.index) != 0:
                st_md = st_md.explode('strSearch')
                st_md['strSearch'] = st_md['strSearch'].str.strip(' )').str.removeprefix('or').str.removesuffix(' or').str.strip(', ')
                st_md['ambiguous'] = (st_md['strSearch'].str.count(' ') == 0)
                metadata = pd.concat([metadata, st_md], ignore_index=True)
        
        # add search entries without "The " prefix
        the_md = metadata[metadata['strSearch'].str.startswith('The ')].copy()
        if len(the_md.index) != 0:
            the_md['strSearch'] = the_md['strSearch'].str[4:]
            the_md['ambiguous'] = (the_md['strSearch'].str.count(' ') == 0)
            metadata = pd.concat([metadata, the_md], ignore_index=True)
        
        # add main character first or last name, if equal to title
        # TODO: properly parse all character descriptions, names, aliases etc. instead
        filter_title = (metadata['category'] == 'WORK_OF_ART')
        
        ch_md = metadata[(metadata['mdType'] == 'character') & metadata['strSearch'].str.contains(' ')].copy()
        if len(ch_md.index) != 0:
            ch_md['strSearch'] = ch_md['strSearch'].str.split(' ')
            ch_md = ch_md.explode('strSearch')
            ch_md = ch_md[ch_md['strSearch'].isin(metadata[filter_title]['strSearch'])]
            
            if len(ch_md.index) != 0:
                metadata = pd.concat([metadata, ch_md], ignore_index=True)
                filter_title = (metadata['category'] == 'WORK_OF_ART')
 
        # flag if title of movie equals character or principal name
        conflicts = metadata[~filter_title][metadata[~filter_title]['strSearch'].isin(metadata[filter_title]['strSearch'])]       
        metadata['conflicts'] = metadata['strSearch'].isin(conflicts['strSearch'])
        
        # filter out first names
        if not include_firstNames:
            metadata = metadata[~(metadata['mdType'] == 'firstName')]
     
        # add regex escaped string
        metadata['strSearchEscaped'] = metadata['strSearch'].str.replace(r'([\]\[\(\)\?\*\+\.\$])', r'\\\1', regex=True)     
  
        # start replacement with longest strings, to avoid, e.g., replacing only the last name
        metadata['lenSearch'] = metadata['strSearch'].str.len()
        metadata = metadata.sort_values(by=['lenSearch', 'category'], ascending=[False, False])
        
        return metadata

    def get_sample_sentences(self, aspect_limit = None, genre_id = None):
        """ get samples of sentences for annotation
            optionally filtered by genre_id
          , a more or less even mix of sentences covering all aspect categories is chosen
        """
        
        if aspect_limit is None:
            aspect_limit = 42
            
        genre_filter = '' if genre_id is None or genre_id < 1 else f'WHERE r.genre_flag & (1 << {genre_id}) != 0'
        
        return pd.read_sql_query(f'''WITH all_terms AS
                                (
                                  SELECT s.id, w.aspect_id,
                                    CASE WHEN s.polComp > 0.6 THEN 5
                                         WHEN s.polComp BETWEEN 0.2 AND 0.6 THEN 4
                                         WHEN s.polComp BETWEEN -0.2 AND 0.2 THEN 3
                                         WHEN s.polComp BETWEEN -0.6 AND -0.2 THEN 2
                                         ELSE 1 END AS polarity
                                  FROM [review_sentence] s
                                  INNER JOIN review r
                                  ON s.review_id = r.id
                                  INNER JOIN aspect_words w
                                  ON s.sentence LIKE '% ' || w.term || ' %'
                                  {genre_filter}
                                ),
                                sentences AS
                                (
                                  SELECT DISTINCT id
                                    FROM (SELECT id, DENSE_RANK() OVER (
                                                     PARTITION BY aspect_id, polarity
                                                     ORDER BY id ASC) AS TermNum
                                          FROM all_terms)
                                    WHERE TermNum < {aspect_limit}
                                )
                                SELECT s.id, s.sentence as text, r.rating
                                FROM sentences
                                INNER JOIN review_sentence s
                                ON sentences.id = s.id
                                INNER JOIN review r
                                ON s.review_id = r.id
                                ''', self.connection())
                                
    def import_sentence_aspects(self, annotations):
        """ import aspect labels
          , make sure to set 'verified' = 1 for gold aspects only
        """
        
        annotations.to_sql(con=self.connection(), name='import_annotations', if_exists='replace')
    
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:            

                cmd.execute('''PRAGMA foreign_keys = ON;''')
                
                cmd.execute('''INSERT OR IGNORE INTO [sentence_aspect]
                                   ([sentence_id]
                                   ,[aspect_id]
                                   ,[aspect_term]
                                   ,[ordinal]
                                   ,[sentiment_term]
                                   ,[polarity]
                                   ,[verified])
                            SELECT i.[id]
                                  ,a.[id]
                                  ,i.[aspect_term]
                                  ,i.[ordinal]
                                  ,i.[sentiment_term]
                                  ,i.[polarity]
                                  ,i.[verified]
                              FROM [import_annotations] i
                            INNER JOIN aspect a
                             ON a.category = i.category
                            WHERE NOT EXISTS (SELECT * FROM sentence_aspect sa
                                               WHERE sa.sentence_id = i.[id]
                                                 AND sa.aspect_id = a.[id]
                                                 AND sa.aspect_term = i.aspect_term
                                                 AND sa.ordinal = i.ordinal
                                                 AND sa.verified = i.verified)
                            ''')
                
                conn.commit()   
                
                cmd.execute('''DROP TABLE import_annotations''')
                
                conn.commit()
    
    
    def get_absa_dataset(self, genre_id, target_amount_per_aspect=150, train_split=0.85):
        """ get absa dataset for a given genre
        
        Arguments:
            target_amount_per_aspect: desired number of samples per aspect category
            train_split: desired ratio of samples in train vs eval, per aspect
            
                - Note that these are not guaranteed, because the final selection is made per sentence -
        
        Returns data in SetFit format:
        
            "text": The full sentence or text containing the aspects.
            "span": An aspect from the full sentence. Can be multiple words. For example: "script".
            "label": The (polarity) label corresponding to the aspect span. For example: "negative".
            "ordinal": If the aspect span occurs multiple times in the text, then this ordinal represents the index of those occurrences. Mostly: 0.
           +
            "dataset": 'train' or 'eval'
        """
    
        # get aspect categories, start with the underrepresented
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:
            
                aspect_ids = cmd.execute('''SELECT [aspect_id]
                                            FROM [sentence_aspect]
                                            WHERE verified = 1
                                              AND aspect_id IS NOT NULL
                                            GROUP BY [aspect_id]
                                            ORDER BY COUNT(*)
                                         ''').fetchall()
   
        
        dataset = None
        for aspect_id in aspect_ids:
        
            current_aspect_count = 0
            sentences = []
            
            if dataset is not None:
                current_aspect_count = len(dataset[dataset['aspect_id'] == aspect_id[0]])
                sentences = dataset['id'].to_list()
                
            while len(sentences) < 2:
                # hack to conform python tuple() representation to sql
                sentences.append(0)
                
            # get iid samples per aspect & polarity, split by ratio
            # ---
            # the selection is rather complicated, because
            # a)
            # even though, we are filtering by aspect
            # all aspects for a given sentence have to be selected
            # to properly set the negative aspect list of the AbsaTrainer
            # b)
            # similarly, the split between 'train' and 'eval' dataset
            # has to be made per sentence, not aspect
            df = pd.read_sql_query(f'''WITH all_sentences AS
                                      (
                                          SELECT s.id, sa.polarity
                                          FROM review_sentence s
                                          INNER JOIN sentence_aspect sa
                                           ON s.id = sa.sentence_id
                                          INNER JOIN review r
                                           ON s.review_id = r.id
                                          WHERE sa.aspect_id = {aspect_id[0]}
                                            AND sa.verified = 1
                                            AND s.id NOT IN {tuple(sentences)}
                                            AND r.genre_flag & (1 << {genre_id}) != 0
                                      )
                                      , polarities AS
                                      (
                                       SELECT {target_amount_per_aspect - current_aspect_count} / COUNT(DISTINCT polarity) as max_count
                                       FROM all_sentences
                                      )
                                      , sentences AS
                                      (
                                        SELECT id, DENSE_RANK() OVER (ORDER BY id) AS nr
                                        FROM (SELECT id, DENSE_RANK() OVER (
                                                         PARTITION BY polarity 
                                                         ORDER BY id ASC) AS AspectNum
                                              FROM all_sentences)
                                        WHERE AspectNum < (SELECT max_count FROM polarities)
                                      )
                                      
                                      SELECT DISTINCT s.id
                                          ,s.sentence AS text
                                          ,sa.aspect_term AS span
                                          ,sa.polarity AS label
                                          ,sa.ordinal
                                          ,CASE WHEN sentences.nr < (SELECT MAX(nr) * {train_split} FROM sentences)
                                                THEN 'train' ELSE 'eval' END as dataset
                                      FROM sentences
                                      INNER JOIN review_sentence s
                                       ON sentences.id = s.id
                                      INNER JOIN sentence_aspect sa
                                       ON sa.sentence_id = s.id
                                      WHERE sa.verified = 1
                                    ''', self.connection())
                       
            if dataset is None:
                dataset = df
            else:
                dataset = pd.concat([dataset,df], ignore_index=True)
        
        return dataset

    def get_aspect_terms(self):
        """ get aspect terms """

        return pd.read_sql_query('''SELECT aspect_id
                                          ,category
                                          ,term
                                      FROM aspect_words w
                                      INNER JOIN aspect a
                                      ON a.id = w.aspect_id
                                    ORDER BY length(term) DESC
                                 ''', self.connection())
  
  
    def reset_predictions(self, genre_id = None):
        """ remove non-gold aspects for sentences """
    
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:
            
                if genre_id is None or genre_id < 1:
                    
                    cmd.execute('''UPDATE [review_sentence]
                                   SET analyzed = 0
                                ''')
                
                    cmd.execute('''DELETE FROM [sentence_aspect]
                                   WHERE verified = 0
                                ''')
                
                else:
                
                    cmd.execute(f'''UPDATE [review_sentence]
                                   SET analyzed = 0
                                   FROM (SELECT r.id FROM review r WHERE r.genre_flag & (1 << {genre_id}) != 0) AS reviews
                                   WHERE review_sentence.review_id = reviews.id
                                ''')
                
                    cmd.execute(f'''DELETE FROM [sentence_aspect]
                                   WHERE verified = 0
                                   AND sentence_id IN (SELECT review_sentence.id
                                                     FROM [review]
                                                     INNER JOIN review_sentence
                                                     ON review.id = review_sentence.review_id
                                                     WHERE genre_flag & (1 << {genre_id}) != 0)
                                ''')
                
                conn.commit()
                             
  
    def get_sentences_for_prediction(self, genre_id = None, chunk_size = None):
        """ get sentences for reviews of titles (belonging to a given genre)
          , that have not been analyzed yet
        """
        
        if chunk_size is None:
            chunk_size = 10000
        
        genre_filter = '' if genre_id is None or genre_id < 1 else f'AND r.genre_flag & (1 << {genre_id}) != 0'
        
        return pd.read_sql_query(f'''SELECT s.id, s.[sentence] as text
                                      FROM [review_sentence] s
                                      INNER JOIN review r
                                      ON s.review_id = r.id
                                      WHERE analyzed = 0
                                      {genre_filter}
                                      ORDER BY s.id
                                      LIMIT {chunk_size}
                                 ''', self.connection()) 
  
    def update_sentences_analyzed(self, ids):
        """ mark sentences as analyzed / aspects predicted """
 
        while len(ids) < 2:
                # hack to conform python tuple() representation to sql
                ids.append(0)
 
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:            

                cmd.execute(f'''UPDATE review_sentence
                                SET analyzed = 1
                                WHERE id IN {tuple(ids)}''')
                
                conn.commit()    
    
    def get_review_polarities_sparse(self, review_id = None, genre_id = None, usage = None, ratings = None):
        """ get review polarities from analyzed sentences
            with separate discrete valued columns for each aspect category / polarity combination
            
        Filter:
            review_id: int - get polarities for a specific review
            genre_id: int - get polarities for all reviews pertaining to a movie of the given genre
            usage: str - get polarities for tagged reviews, e.g., 'train' or 'test'
            ratings: List[int] - get polarities for reviews with one of the given ratings only
        """
        
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:  
            
                genres = cmd.execute('SELECT [id], [displayName] FROM genre ORDER BY [id]').fetchall()
            
                aspects = cmd.execute('SELECT [category] FROM aspect ORDER BY [category]').fetchall()
        
        aspects = [item for l in aspects for item in l]
        polarities = ['very negative', 'negative', 'neutral', 'positive', 'very positive'] #TODO: make config

        aspect_polarity_combinations = list(itertools.product(aspects, polarities))
        
        aspects.append('None')
        aspect_polarity_combinations.append(('None', 'none'))

        # 1 or -1 for each genre of title
        # reduced to 'main' genres to appear in conjunction with setfit filtered genre
        # TODO: make config / will be different set for different setfit model
        flags = [f"""CASE WHEN r.genre_flag & (1 << {genre[0]}) != 0 THEN 1 ELSE -1 END
                      AS genre_{genre[1].replace('-','_')}
                   """
                  for genre in genres if genre[1] in ('Adventure', 'Comedy', 'Drama', 'Fantasy', 'Sci-Fi')]

        # count of appearance of aspect / polarity combination
        # i.e., 0, 1, .. , 9, 10, ..
        # note that these features are purposely _not_ normalized, as they should have a heigher weight then the rest
        counts = [f"""SUM(CASE WHEN IFNULL(a.category, 'None') = '{combo[0]}' AND IFNULL(sa.polarity, 'none') = '{combo[1]}' THEN 1 ELSE 0 END)
                      AS {combo[0]}_{combo[1].replace(' ','_')}
                   """
                  for combo in aspect_polarity_combinations]

        # averages of vader sentiment scores, for sentences containing a given aspect
        # continuous value between -1 and 1
        means = [f"""IFNULL(AVG(CASE WHEN IFNULL(a.category, 'None') = '{aspect}' THEN s.polComp END), 0)
                      AS {aspect}_mean
                   """
                  for aspect in aspects]

        # filter
        review_filter = '' if review_id is None else f'AND r.id = {review_id}'
        genre_filter = '' if genre_id is None else f'AND r.genre_flag & (1 << {genre_id}) != 0'
        usage_filter = '' if usage is None else f"AND r.usage = '{usage}'"
        
        ratings_filter = ''
        if ratings is not None:
        
            while len(ratings) < 2:
                # hack to conform python tuple() representation to sql
                ratings.append(0)
                
            ratings_filter = f'AND r.rating IN {tuple(ratings)}'


        query = f"""SELECT r.id,
                   CAST((r.rating + 1) / 2 AS INTEGER) AS class,
                   CASE WHEN r.rating > 6 THEN 1 ELSE 0 END AS binary_class,
                  {', '.join(flags)},
                   AVG(s.polComp) AS mean_review_polarity,
                  {', '.join(counts)},
                  {', '.join(means)}
                  FROM review r
                  INNER JOIN [review_sentence] s
                   ON r.id = s.review_id
                  LEFT OUTER JOIN sentence_aspect sa
                   ON sa.sentence_id = s.id
                  AND sa.verified = 0
                  LEFT OUTER JOIN aspect a
                   ON a.id = sa.aspect_id
                  WHERE s.analyzed = 1
                  {review_filter}
                  {genre_filter}
                  {usage_filter}
                  {ratings_filter}
                  GROUP BY r.id, r.rating
                """
        
        #set dtypes (mainly for get_review_polarities_input)
        dtypes = {'mean_review_polarity':float}
        for aspect in aspects:
            dtypes[f'{aspect}_mean'] = float
  
        return pd.read_sql_query(query, self.connection(), dtype=dtypes)
    
    def get_review_polarities_input(self):
        """ returns DataFrame with one empty row, in same format as get_review_polarities_sparse
            to create input features for classifier
        """
        
        features = self.get_review_polarities_sparse(review_id=-1)

        features = features.iloc[:,features.columns.get_loc('binary_class') + 1:]

        # set defaults
        for column in features.columns.tolist():
            if column.startswith('genre_'):
                features.at[0, column] = -1
            elif 'mean' in column:
                features.at[0, column] = 0.0
            else:
                features.at[0, column] = 0
                
        return features