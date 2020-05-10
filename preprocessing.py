from rake_nltk import Rake   # ensure this is installed
import nltk
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('data/moviedata.csv')   # same data 250 rows × 38 columns
df = df[['Title','Director','Actors','Plot','Genre']]
df['Plot'] = df['Plot'].str.replace('[^\w\s]','')
df['Key_words'] = ''   # initializing a new column
r = Rake()   # use Rake to discard stop words (based on english stopwords from NLTK)
for index, row in df.iterrows():
    r.extract_keywords_from_text(row['Plot'])   # to extract key words from Plot, default in lower case
    key_words_dict_scores = r.get_word_degrees()    # to get dictionary with key words and their scores
    row['Key_words'] = list(key_words_dict_scores.keys())   # to assign list of key words to new column
df['Director'].map(lambda x:x)
for index, row in df.iterrows():
    row['Genre'] = [x.lower().replace(' ','') for x in row['Genre']]
    row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
    row['Director'] = [x.lower().replace(' ','') for x in row['Director']]
df['Bag_of_words'] = ''
columns = ['Genre', 'Director', 'Actors', 'Key_words']
for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    row['Bag_of_words'] = words
    
df['Bag_of_words'] = df['Bag_of_words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')
df = df[['Title','Bag_of_words']]

df['Bag_of_words'][0]
count = CountVectorizer()
count_matrix = count.fit_transform(df['Bag_of_words'])
count_matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(df['Title'])
cosine_sim = pd.DataFrame(cosine_sim )
cosine_sim.to_json("data/cosine_sim.json")
indices.to_json("data/indices.json")
indices.to_csv("data/indices.csv")