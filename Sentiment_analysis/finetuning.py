import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the preprocessed Kaggle dataset (replace 'Twitter_Data.csv' with actual file)
data = pd.read_csv('Twitter_Data.csv').head(5000)

# Drop rows with missing values in 'clean_text' or 'category' columns
data = data.dropna(subset=['clean_text', 'category'])

# Assuming 'clean_text' column contains the tweets and 'category' contains -1, 0, 1 for sentiment
texts = data['clean_text'].tolist()
labels = data['category'].tolist()

# Map the labels from -1, 0, 1 to 0, 1, 2
label_mapping = {-1: 0, 0: 1, 1: 2}
labels = [label_mapping[label] for label in labels]

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
input_ids = []
attention_masks = []

for text in texts:
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=32,            # Pad & truncate all sentences to max length
        padding='max_length',
        truncation=True,
        return_attention_mask=True,  # Construct attention masks
        return_tensors='tf'           # Return tf.Tensor objects
    )
    
    # Flatten the tensors to a list before appending
    input_ids.append(encoded_dict['input_ids'].numpy().flatten())
    attention_masks.append(encoded_dict['attention_mask'].numpy().flatten())

# Convert lists to NumPy arrays
input_ids = np.array(input_ids)
attention_masks = np.array(attention_masks)
labels = np.array(labels)

# Train-test split
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
train_masks, val_masks = train_test_split(attention_masks, test_size=0.2, random_state=42)

# Load pre-trained BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Compile the model with Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)  # Lower learning rate for fine-tuning
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Implement class weights to handle class imbalance
class_weights = {0: 2.0, 1: 1.0, 2: 1.5}  # Adjust these based on your data distribution

# Optionally freeze some layers (example: freeze all layers except the last two)
for layer in model.layers[:-2]:
    layer.trainable = False

# Train the model
history = model.fit(
    [train_inputs, train_masks],
    train_labels,
    validation_data=([val_inputs, val_masks], val_labels),
    epochs=5,  # Increase number of epochs for better training
    batch_size=16,  # Smaller batch size can help with limited GPU memory
    class_weight=class_weights  # Include class weights to handle class imbalance
)

# Evaluate the model
val_predictions = model.predict([val_inputs, val_masks])
preds = np.argmax(val_predictions.logits, axis=-1)
accuracy = accuracy_score(val_labels, preds)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(classification_report(val_labels, preds, target_names=['Negative', 'Neutral', 'Positive']))
# Save the model
save_path = './models/fine_tuned_bert'  # You can change this path as needed
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Model saved successfully!")