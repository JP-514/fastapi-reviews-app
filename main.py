from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from DataModel import DataModel
from Model import Model
import nltk
import spacy
import re, unicodedata
import contractions
import inflect
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('stopwords-es')
nltk.download('wordnet')

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        if word is not None:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    return [word.lower() for word in words]

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        if word is not None:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:   
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    palabras_vacias = set(stopwords.words('spanish'))
    return [word for word in words if word.lower() not in palabras_vacias]

def fix_contractions(words):
    return contractions.fix(words).split()

def preprocessing(words):
    words = fix_contractions(words)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def full_preprocessing(reviews_list):
    final_reviews = [preprocessing(review) for review in reviews_list]
    return final_reviews

nlp_spacy_es = spacy.load("es_core_news_sm")

def spacy_tokenizer(words):
    doc_es = nlp_spacy_es(" ".join(words))
    doc_process = [lemma for lemma in [token.lemma_ for token in doc_es]]
    words_lemma = " ".join(doc_process)
    return words_lemma

def spacy_full(reviews_list):
    final_reviews = [spacy_tokenizer(review) for review in reviews_list]
    return final_reviews

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.post("/predict")
def make_predictions(dataModel: DataModel):
    df = pd.DataFrame(dataModel.model_dump(), columns=dataModel.model_dump().keys(), index=[0])
    df.columns = dataModel.columns()
    df["Review"] = spacy_full(full_preprocessing(df['Review']))
    model = Model()
    result = model.make_predictions(df["Review"])
    return {"prediction": int(result)}

@app.post("/words")
def get_important_words(dataModel: DataModel):
    df = pd.DataFrame(dataModel.model_dump(), columns=dataModel.model_dump().keys(), index=[0])
    df.columns = dataModel.columns()
    df["Review"] = spacy_full(full_preprocessing(df['Review']))
    model = Model()
    features = model.model['tfidf'].get_feature_names_out()
    important_features = model.model['svm_model'].coef_[0]
    if hasattr(important_features, 'toarray'):
        important_features = important_features.toarray()
        important_features = important_features.ravel()
    word_indices = np.argsort(important_features)[-300:]
    review_important = []
    user_review = df['Review'].values[0]
    for i in user_review.split():
        for j in range(len(word_indices)):
            if i == features[word_indices[j]]:
                review_important.append(word_indices[j])
    word_list = [features[i] for i in review_important]
    return {"words": word_list}

