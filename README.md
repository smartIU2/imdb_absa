![gui](https://github.com/user-attachments/assets/6c92c077-5c19-4ffc-93dd-1f0443b4e32c)

## Features

- Aspect-based sentiment analysis for movie reviews, using
  - custom setfit-absa models, based on distilroberta sentence transformer, for aspect extraction and polarity estimation
  - SVM classifiers for predicting multi-class rating (1 to 5 stars) and overall binary sentiment
- Includes leightweight dash interface for visualization


## Environment Setup

Python version 3.10 or newer needs to be installed on your machine.

Torch is used by several machine learning models. Please visit the [PyTorch website](https://pytorch.org/get-started/locally/) for setup instructions.
Make sure to install an appropriate version of CUDA beforehand, if you intend to employ a GPU.
Development was done with CUDA version 12.8 under Windows, but other environments should work as well.

Maverick-coref is used for coreference resolution in the preprocessing pipeline.
It is recommended to get the fork from https://github.com/smartIU2/maverick-coref to faciliate setup. Especially under Windows.

A customized setfit version is used for abstract extraction and sentiment prediction.
Please get the fork from https://github.com/smartIU2/setfit.
If you're using git clone and you're running into an error, make sure to disable "Recursively clone submodules too".

To install both maverick and setfit, navigate to the downloaded repo folders, and in each execute:

```commandline
pip install -e .
```
(don't miss the dot at the end)

These two setups will also install all other dependencies for preprocessing and inference.
But you still need to download a spaCy model of your choice.
If you want to employ a GPU, it is recommended to run:

```commandline
spacy download en_core_web_trf
```

otherwise, run:

```commandline
spacy download en_core_web_lg
```

and then change the model name under "model_spacy" in the config.json accordingly.


If you want to use the web interface, you need to additionally install dash and beautifulsoup:

```commandline
pip install dash[diskcache] beautifulsoup4
```

And, finally, if you want to generate confusion matrices for evaluation, matplotlib is needed:

```commandline
pip install matplotlib
```


## Database Setup

During preprocessing, movie metadata in the reviews are replaced with fixed aspect terms like "this movie" or "the director".
To this end, a knowledge base needs to be setup.

First navigate to [IMDb Non-Commercial Datasets](https://datasets.imdbws.com/) and download "name.basics.tsv.gz", "title.basics.tsv.gz", "title.principals.tsv.gz" and "title.ratings.tsv.gz".
Extract all archives into the project's subfolder "/import/imdb".
Then run:

```commandline
setup_01_create_database.py
```

This will take around 5 minutes, and create an SQLite database at "/database/imdb.db".
By default, only imdb titles of type "movie", released in the current century, with a runtime of more than 45 minutes, and at least 50 votes will be imported.
You can change any of these settings with the "imdb_" entries in the config.json.
For example, if your favourite direct-to-video horror flick is missing, you'll need to add "video" to "imdb_types". 

The database will also be used to store all other data, like the movie reviews.

NOTE: It is safe to execute this file multiple times, e.g., with newer imdb data.
The database will not be recreated from scratch. Neither existing titles nor their reviews will be removed.
But, this also means, you cannot remove titles from the database with a more restrictive filter.


## SVM classifier recreation

One final step, before being able to use the system, is to recreate the multi-class and binary classifiers, via sklearn.
This is necessary, because sharing sklearn classifiers via pickle is unsafe and environment specific.

Simply run:

```commandline
setup_02_recreate_classifier.py
```

This should take only a couple of seconds, and will create two pickle files under "/models/classifier-imdb-absa-action".


## Web Interface Usage

After finishing the setup steps, the dash server can be started on localhost:8087 via: 

```commandline
start_gui.py
```

You first need to select a movie from the dropdown.
By default it is limited to Action movies, since that's the only setfit model trained at the moment.
But you can change this in config.json with "dash_filter_genre". 

Then you can either enter your own review for the movie, or click on "Get random movie review".
The latter will select a review from the database if available, or fetch some reviews for the selected movie from imdb first.
(Until imdb changes their website layout...)

Lastly, click on "Analyze".
This will:
- normalize the review using regex
- replace metadata with spaCy and maverick-coref
- split the review text into sentences with NLTK
- estimate sentence sentiment polarities with VADER
- predict aspect sentiment polarities with the setfit models
- predict overall rating and binary classification with the SVM classifiers

The individual aspect term sentiments will be displayed with displaCy's entity visualizer,
alongside the star ratings, and a thumbs up or down for the overall polarity. 


## Troubleshooting

If you're encountering a pandas error during usage, you'll likely need another version.
Try the latest one before the major upgrade in June 2025:

```commandline
pip install pandas==2.2.3
```


## Training your own models

If you want to train your own models, for example fine-tuned to another genre, you can follow along the training steps.

### Import and preprocess reviews

You first need to assemble an appropriate review dataset.
Create a csv, with a line for each review, and the following columns:
- title_id: the imdb id, e.g., "tt4154796"
- text: the review text, without any html
- rating: the imdb rating, from 1 to 10

If you already know, which review will be used for what purpose, you can include an additional column
- usage: a user defined string to save along the review, for example, "train" or "test"

The reviews can then be imported to the database with a call to the first training step along with the path to the .csv:

```commandline
train_01_import_reviews.py reviews.csv
```

Make sure all reviewed movies are in the database, otherwise they will be skipped.

Afterwards, the reviews need to be preprocessed:

```commandline
train_02_preprocess_reviews.py
```

This will run through the preprocessing pipeline for all reviews
and save normalized texts, sentences and sentence polarities to the database.

You can safely import another .csv afterwards and execute the command again, to only preprocess the new reviews.

### Annotate

The setfit trainer needs samples of sentences alongside contained aspects labeled with the appropriate sentiment polarity, e.g., "positive" or "negative".
For this, you are of course free to use whatever annotation tool you prefer.
But it is highly recommended to employ doccano, as the export and import are designed around it.

Go to [doccano](https://github.com/doccano/doccano) and follow the instructions under "Usage - pip" to install, and start the server as well as the task queue.

Create a new project of type "Sequence Labeling".
In the newly created doccano project, navigate to "Labels" and from the Actions menu select "Import Labels".
Click on "File input" and select "/doccano/label_config.json" from this project's repository.
This should create six labels for you, from "very negative" to "very positive", plus a "candidate".

The next step will select a number of sentences per aspect term (see "/import/aspects/aspects.csv")
and export them as a .json file in the format used by doccano.
You can optionally filter by movie genre, e.g., 

```commandline
train_03_export_sentences.py --genre_id 5
```

Then navigate to "Dataset" in the doccano project, select "Import Dataset" from the Actions menu, and JSONL as the file format.
The rest of the settings can be left as default.

And with this you're ready for annotation.
Simply hit "Start Annotation" and make sure to set the filter to "Undone".
All you've left to do, is click on correctly identified aspects and select a polarity from the dropdown.

![doccano](https://github.com/user-attachments/assets/10962b92-dd83-43b5-9a19-f10f538e0c5c)

IMPORTANT: It is useless to define your own aspects by clicking on words that are not marked "candidate".
These will be ignored by the setfit trainer, as they do not match the aspect extraction based on spaCy's determination of noun chunks.

When you're done annotating, navigate back to "Dataset", and select "Export Dataset" from the Actions menu.
Be sure to set the checkbox to "Export only approved documents" and
export your annotations to "./doccano/annotations.jsonl" in this project's repository.

Finally, import the gold aspect labels into the database with:

```commandline
train_04_import_annotations.py
```

### Train setfit models

To train the setfit models, you first need to define the training arguments.
Open config.json, and create a new set of arguments, by copying an existing entry ("action" or "test") under "train_models".
The most important setting is "model_base", which defines the pre-trained embedding model to fine-tune.
"sentence-transformers/all-distilroberta-v1" has been a great choice for training the action models with an 8GB laptop GPU.
For different options, you should check out the [comparison by Jayakody et al.](https://doi.org/10.1109/MERCon63886.2024.10688631).
Most of the other settings directly translate to [setfit trainer arguments](https://github.com/huggingface/setfit/blob/main/src/setfit/training_args.py).

After you defined the set of training arguments, run the next python file along with the name you chose, e.g.:

```commandline
train_05_train_setfit.py my_new_model
```

Depending on your number of samples, your training arguments and your available hardware, this could take a while...
Eventually, two models, one for aspect detection, and one for aspect sentiment prediction, will be created.

To use your newly created models thoughout the project, you need to update the config.json again.
Simply copy the value from "save_as" of your training arguments, to "model_setfit".


### Predict aspect polarities

Running the next file will predict aspect polarities for the sentences in the database, using your new setfit models.
You may filter by genre, e.g., 

```commandline
train_06_predict_aspect_polarities.py --genre_id 5
```

IMPORTANT: This will first remove all (non-gold) aspect predictions (for the given genre) from the database.
Evaluating two models (for the same genre) at once is currently not supported.

Again, the duration of this process is depending on your dataset size and your available hardware, but it should be somewhat faster than the preprocessing.


### Train SVM classifiers

To turn the aspect polarities into overall review ratings / classifications, the final step is to train the SVM classifiers.

For this you can filter the database for reviews by
- "genre_id", e.g., 
  ```commandline
  train_07_train_classifier.py --genre_id 5
  ```
  for comedy reviews
- "ratings", e.g.,
  ```commandline
  train_07_train_classifier.py --ratings 1 2 3 4 7 8 9 10
  ```
  to exclude reviews with neutral ratings, or
- the "usage" tag set when importing the reviews, e.g., 
  ```commandline
  train_07_train_classifier.py --usage train
  ```

This should take no more than a minute or two, and will create both the multi-class and the binary classifiers.
It will also save the inputs, as used by setup_02_recreate_classifier.py


## Evaluation

To evaluate your trained models, use the 8th training step, with similar parameters as for training the classifiers.
This will yield two confusion matrices for the multi-class and binary classification, including balanced accuracy and MCC scores.

![evaluation](https://github.com/user-attachments/assets/061f5e74-de0a-44d2-95f3-eff2b063dd84)


If you don't want to train your own models, but simply evaluate the existing models on a new dataset, you can skip some of the training steps.
For Stanford's Large Movie Review Dataset, a custom conversion to csv is included, to facilitate evaluation.

First download their dataset from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz and
extract it to "/import/aclImdb/" in the project's repository.

Then execute the following commands in order:

```commandline
train_00_convert_aclImdb.py
train_01_import_reviews.py aclImdb_reviews.csv
train_02_preprocess_reviews.py
train_06_predict_aspect_polarities.py
train_07_train_classifier.py --usage train
train_08_evaluate.py --usage test
```

IMPORTANT: Be aware, that you can only evaluate reviews for movies contained in the database.
So, for this example, you would have to first relax/remove the filters on the imdb titles, before calling setup_01_create_database.py

## FAQ

To find the appropriate genre_id for your settings / model training, consult the following table:

|genre_id|displayName|
|---|---|
|1| Action |
|2| Adventure |
|3| Animation |
|4| Biography |
|5| Comedy |
|6| Crime |
|7| Documentary |
|8| Drama |
|9| Family |
|10| Fantasy |
|11| History |
|12| Horror |
|13| Musical |
|14| Mystery |
|15| Romance |
|16| Sci-Fi |
|17| Sport |
|18| Thriller |
|19| War |
|20| Western |

## Disclaimer

**No** part of the source code in this repository or this documentation was created by or with the help of artificial intelligence.
