from preprocessing import df
import pandas as pd
import numpy
def recommend(title):
    cosine_sim = pd.read_json("data/cosine_sim.json")
    recommended_movies = []
    indices = pd.Series(df['Title'])
    idx = indices[indices == title].index[0]   # to get the index of the movie title matching the input movie
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)   # similarity scores in descending order
    top_10_indices = list(score_series.iloc[1:11].index)   # to get the indices of top 10 most similar movies
    # [1:11] to exclude 0 (index 0 is the input movie itself)
    for i in top_10_indices:   # to append the titles of top 10 similar movies to the recommended_movies list
        recommended_movies.append(list(df['Title'])[i])
    return recommended_movies
print(recommend("The Shawshank Redemption"))