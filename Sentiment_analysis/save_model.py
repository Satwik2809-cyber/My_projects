import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

# Load your fine-tuned model (assuming it's already trained and saved in the specified directory)
model = TFBertForSequenceClassification.from_pretrained(r'C:\Users\Sanjana\AppData\Project 42\models\fine_tuned_bert')
tokenizer = BertTokenizer.from_pretrained(r'C:\Users\Sanjana\AppData\Project 42\models\fine_tuned_bert')

# Specify where to save the model
save_path = r'C:\Users\Sanjana\AppData\Project 42\models\fine_tuned_bert'  # Update this path to your desired location

# Save the model and tokenizer
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Model saved successfully!")
