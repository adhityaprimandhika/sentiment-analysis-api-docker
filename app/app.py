# Flask
from flask import Flask, jsonify, request, abort
from flask_cors import CORS, cross_origin
from flask_restx import fields, Resource, Api, reqparse

# MySQL Connector
import mysql.connector

# For data manipulation
import pandas as pd

# Encode URL-encode sentence
from urllib.parse import quote, unquote

# JSON
import json

# Datetime
import datetime

# Text pre-processing
import preprocessor as p
p.set_options(p.OPT.MENTION, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.URL)
import re

# File path
import os

# Sentiment Analysis
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Flask RESTX configurations
api = Api(app=app,
          version='1.0',
          title='Flask MySQL Docker',
          description='Test Run Flask and MYSQL in Docker')

# CORS configurations
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Home route (default)
@api.route('/')
class Index(Resource):
    @cross_origin()
    def get(self):
        return jsonify({'message': 'Hello world!!!'})
    
history_namespace = api.namespace(
    'history', decsription='To get historical data from table'
)

# Retrieve all data from table sentiment_data
@history_namespace.route('/sentiment-data')
class GetHistoryData(Resource):
    @history_namespace.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Internal Server Error'})
    @cross_origin()
    def get(self):
        connection = mysql.connector.connect(
            user="root", password="root", host="db", port="3306", database="my_database")
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM history_data")
        data = cursor.fetchall()
        """
        results = [{'id': id,
                    'created_at': created_at,
                    'sentences': sentences,
                    'clean_sentences': clean_sentences,
                    'sentiment': sentiment,
                    'sentiment_score': sentiment_score} for (id, created_at, sentences, clean_sentences, sentiment, sentiment_score) in data]
        """
        cursor.close()
        connection.close()
        return jsonify(data)

sentiment_namespace = api.namespace(
    'sentiment-analysis', decsription='To get sentiment from inputs'
)

# Function for data cleaning
def cleaning_sentence(text):
        text = p.clean(text) # Remove mention, emoji, URL, etc
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text) # Only alphanumerical regex
        return text

# Function for sentiment analysis using pre-trained model
def sentiment_analysis_sentence(text):
    saved_model_path = "saved-model"
    model_name = "mdhugol/indonesia-bert-sentiment-classification"
    
    # Define sentiment analysis function
    def sentiment(text):
        result = sentiment_analysis(text)
        status = label_index[result[0]["label"]]
        return status
        
    # Define sentiment analysis scoring function
    def sentiment_score(text):
        result = sentiment_analysis(text)
        score = result[0]["score"]
        return score

    # Check if model files exist in the specified directory
    if os.path.exists(saved_model_path):
        print("Loading pretrained model from local storage...")
        tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)
        sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        label_index = {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"}
        
        sentiment_data = {}
        sentiment_data['sentiment'] = sentiment(text)
        sentiment_data['sentiment_score'] = sentiment_score(text)
    else:
        print("Downloading and saving pretrained model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        label_index = {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"}
        
        sentiment_data = {}
        sentiment_data['sentiment'] = sentiment(text)
        sentiment_data['sentiment_score'] = sentiment_score(text)
        
        tokenizer.save_pretrained(saved_model_path)
        model.save_pretrained(saved_model_path)
    return sentiment_data

# Do sentiment analysis using clean sentences and insert it all data into history_data
@sentiment_namespace.route('')
class SentimentAnalysis(Resource):
    @sentiment_namespace.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Internal Server Error'}, 
                             params={'sentences': {'description': 'Sentences that want to get sentiment analysis', 'type': 'String', 'required': False}})
    @cross_origin()
    
    def get(self):
        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Format the timestamp as a string
        timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        parser = reqparse.RequestParser()
        parser.add_argument('sentences', required=False, default=None)
        args = parser.parse_args()

        sentences = args['sentences'] or None
        sentences = unquote(sentences)
        
        clean_sentences = cleaning_sentence(sentences)
        sentiment_data = sentiment_analysis_sentence(clean_sentences)
        
        connection = mysql.connector.connect(
            user="root", password="root", host="db", port="3306", database="my_database")
        cursor = connection.cursor()
        
        # Insert new data into history_data
        cursor.execute("INSERT INTO history_data (created_at, sentences, clean_sentences, sentiment, sentiment_score) VALUES ('{}', '{}', '{}', '{}', {});".format(
            timestamp, sentences, clean_sentences, sentiment_data['sentiment'], sentiment_data['sentiment_score']))
            
        connection.commit()
        cursor.close()
        connection.close()
        
        results = {
        'sentences': sentences,
        'sentiment': sentiment_data['sentiment'],
        'sentiment_score': sentiment_data['sentiment_score']
        }
        return jsonify(results)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0')