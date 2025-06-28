from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np
import tensorflow as tf

# Load the fine-tuned model and tokenizer
model_path = './models/fine_tuned_bert'  # Path to the saved model
try:
    model = TFBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
except OSError as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

# Preprocess the input text for the model
def preprocess_text(text):
    try:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        return {
            'input_ids': encoded_dict['input_ids'],
            'attention_mask': encoded_dict['attention_mask']
        }
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return None

# Predict the sentiment of a list of texts
def analyze_sentiments(texts):
    sentiments = []
    for text in texts:
        preprocessed_data = preprocess_text(text)
        if preprocessed_data is None:
            sentiments.append("Error")
            continue
        input_ids = preprocessed_data['input_ids']
        attention_mask = preprocessed_data['attention_mask']

        try:
            logits = model(input_ids, attention_mask=attention_mask).logits
            predicted_label = np.argmax(logits, axis=-1)[0]
            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            sentiment = sentiment_map[predicted_label]
            sentiments.append(sentiment)
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            sentiments.append("Error")
    return sentiments
# Predict the sentiment of a single custom text input
#def analyze_sentiments(preprocessed_data):
    # Assuming your model requires input_ids and attention_mask
    input_ids = [data['input_ids'] for data in preprocessed_data]
    attention_masks = [data['attention_mask'] for data in preprocessed_data]

    # Convert to tensors
    input_ids_tensor = tf.convert_to_tensor(input_ids)
    attention_masks_tensor = tf.convert_to_tensor(attention_masks)

    # Get predictions
    predictions = model.predict([input_ids_tensor, attention_masks_tensor])
    predicted_labels = tf.argmax(predictions.logits, axis=1)

    # Map predictions to sentiment labels
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return [{'label': label_map[label.numpy()], 'tweet': tweet} for label, tweet in zip(predicted_labels, text)]

def analyze_custom_text(text):
    preprocessed_data = preprocess_text(text)
    if preprocessed_data is None:
        return "Error"
    input_ids = preprocessed_data['input_ids']
    attention_mask = preprocessed_data['attention_mask']

    try:
        logits = model(input_ids, attention_mask=attention_mask).logits
        predicted_label = np.argmax(logits, axis=-1)[0]
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = sentiment_map[predicted_label]
        return sentiment
    except Exception as e:
        print(f"Error analyzing custom text: {e}")
        return "Error"
