from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from movie_recommender import recommend


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/movielist')
def ml():
    movies=pd.read_csv('data/indices.csv')
    return render_template('movielist.html',movies=movies['Title'].tolist())

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
    return render_template('result.html',prediction = recommend(message))

@app.route('/<name>')
def predict2(name):
    return render_template('result.html',prediction = recommend(name))

if __name__ == '__main__':
    app.run(debug=True)
