import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# sample data
texts = ['Bonjour,comment ca va?','hello,how are you?','hola, cómo estás','Ciao, come stai','नमस्ते, आप कैसे हैं']
labels = ['French','English','Spanish','Italian','Hindi']

#encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

#pad sequences to ensure uniform length
max_sequences_length = max(len(seq)for seq in sequences)
padded_sequences = pad_sequences(sequences,maxlen=max_sequences_length,padding='post')

#split data into train 
X_train, X_test, y_train, y_test = train_test_split(padded_sequences,encoded_labels, test_size=0.2,random_state=42)

#model
model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, 
input_length=max_sequences_length),
tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dense(64,activation='relu'), 
tf.keras.layers.Dense(len(labels),activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

loss,accuracy = model.evaluate(X_test, y_test) 
print(f'Test Accuracy: {accuracy}')

sample_text = "hola, cómo estás"
sample_sequences = tokenizer.texts_to_sequences([sample_text])
sample_padded = pad_sequences(sample_sequences, maxlen=max_sequences_length,padding='post')

predictions = model.predict(sample_padded)
predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])

print(f"input text:{sample_text}")
print(f"predicted language:{predicted_label[0]}")