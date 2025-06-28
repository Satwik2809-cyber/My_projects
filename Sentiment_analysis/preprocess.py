import re
from transformers import BertTokenizer
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean tweet text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z\s]", '', text)  # Remove special characters and numbers
    text = text.lower().strip()  # Convert to lowercase and trim whitespace
    return text

# Function to remove stopwords and perform lemmatization
def preprocess_words(text):
    words = text.split()
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)

# Function to preprocess a tweet for BERT
def preprocess_tweet(tweet_text):
    # Step 1: Clean the text
    cleaned_text = clean_text(tweet_text)
    
    # Step 2: Preprocess words (lemmatization and stopword removal)
    processed_text = preprocess_words(cleaned_text)

    # Step 3: Tokenize using BERT tokenizer
    tokens = tokenizer.encode_plus(
        processed_text,
        max_length=128,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='tf'  # Return as TensorFlow tensors
    )
    return {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask']
    }

if __name__ == '__main__':
    tweet = "Check out my website at https://example.com! #excited @user"
    preprocessed_data = preprocess_tweet(tweet)
    print("Input IDs:", preprocessed_data['input_ids'])
    print("Attention Mask:", preprocessed_data['attention_mask'])
