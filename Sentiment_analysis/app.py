from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from data_preprocessing import analyze_sentiments, analyze_custom_text
from preprocess import preprocess_tweet 
from transformers import TFBertForSequenceClassification, BertTokenizer
from file import load_tweets, analyze_twitter
import os

app = Flask(__name__)

# Load BERT model and tokenizer
model_path = './models/fine_tuned_bert'  
model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'index.html')

# Serving static files like styles.css and script.js from the same folder
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.getcwd(), filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    if not request.json:
        return jsonify({'error': 'No JSON data found'}), 400

    data_source = request.json.get('data_source')
    tweet_limit = int(request.json.get('tweet_limit', 100))
    keyword = request.json.get('keyword')
    language = request.json.get('language', None)
    custom_text = request.json.get('custom_text', '')

    if not data_source:
        return jsonify({'error': 'Data source is required'}), 400

    if data_source == 'twitter':
        try:
            # Load tweets from the CSV file
            print(f"DEBUG: Starting Twitter analysis for keyword '{keyword}'")
            tweets_df = load_tweets('standardized_dataset.csv')  # Load your tweets using file.py
            if tweets_df is None:
                return jsonify({'error': 'Failed to load tweets.'}), 400

            # Filter and analyze tweets
            tweet_texts,sentiments = analyze_twitter(tweets_df, keyword, tweet_limit, language)

            if not sentiments:
                return jsonify({'error': 'No tweets found for the given criteria.'}), 404

            response = {
                'message': f"Analyzed {len(sentiments)} tweets for keyword '{keyword}'.",
                'sentiments': [{'label': sentiment, 'tweet': tweet} for sentiment, tweet in zip(sentiments, tweet_texts)],
                'distribution': {
                    'positive': sum(1 for s in sentiments if s['label'] == 'Positive'),
                    'negative': sum(1 for s in sentiments if s['label'] == 'Negative'),
                    'neutral': sum(1 for s in sentiments if s['label'] == 'Neutral')
                }
            }
        except Exception as e:
            response = {'error': f"Failed to process tweets: {str(e)}"}
    elif data_source == 'custom_text':
        if not custom_text:
            return jsonify({'error': 'Custom text cannot be empty.'}), 400

        try:
            custom_sentiment = analyze_custom_text(custom_text)

            response = {
                'message': f"Analyzing custom text: {custom_text}",
                'sentiments': [{'label': custom_sentiment, 'tweet': custom_text}],
                'distribution': {
                    'positive': 1 if custom_sentiment == 'Positive' else 0,
                    'negative': 1 if custom_sentiment == 'Negative' else 0,
                    'neutral': 1 if custom_sentiment == 'Neutral' else 0
                }
            }
        except Exception as e:
            response = {'error': f"Failed to analyze custom text: {str(e)}"}
    else:
        response = {'message': 'Invalid data source'}

    return jsonify(response)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.get_json()  
    tweet_text = data.get('tweet', '')

    if tweet_text:
        try:
            preprocessed_data = preprocess_tweet(tweet_text)
            return jsonify({
                'input_ids': preprocessed_data['input_ids'].tolist(),
                'attention_mask': preprocessed_data['attention_mask'].tolist()
            })
        except Exception as e:
            return jsonify({'error': f"Preprocessing failed: {str(e)}"}), 500
    else:
        return jsonify({'error': 'No tweet provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
