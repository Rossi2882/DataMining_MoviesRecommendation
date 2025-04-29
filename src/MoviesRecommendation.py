import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re

    #New comment
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / 'data' / 'movies.csv'

df1 = pd.read_csv(data_path)
print(df1.head())

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

df1['genres_list'] = df1['genres'].str.replace('|', ' ')
df1['clean_title'] = df1['title'].apply(clean_title)

movies_data = df1[['movieId', 'clean_title', 'genres_list']]
print(movies_data.head())

BASE_DIR = Path(__file__).resolve().parent.parent
data_path2 = BASE_DIR / 'data' / 'ratings.csv'

df2 = pd.read_csv(data_path2)
print(df2.head())

ratings_data = df2.drop(['timestamp'], axis=1)
print(ratings_data.head())

combined_data = ratings_data.merge(movies_data, on='movieId')
print(combined_data.head())