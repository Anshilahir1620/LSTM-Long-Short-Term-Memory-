import re
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping


with open("dataset/t8.shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Original text length:", len(text))

# convert all text into lowercase and remove special characters
text = text.lower()
text = re.sub(r"[^a-z\s]", "", text)
text = re.sub(r"\s+", " ", text)

print("After cleaning:", len(text))

# tokenizer convert every word into a unique number
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
words = len(tokenizer.word_index) + 1
print("Total words:", words)

# convert complete text into number sequence using tokenizer
tokens = tokenizer.texts_to_sequences([text])[0]

# 20 previous words will be used to predict the next word
sequence_length = 20
X = []
y = []


# create training sequences from the token list
for i in range(sequence_length, len(tokens)):
    X.append(tokens[i - sequence_length:i])
    y.append(tokens[i])


# convert python lists into numpy arrays
# numpy format is easier for tensorflow to process
X = np.array(X)
y = np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)


# use 90 percent data for training and remaining for validation
split = int(len(X) * 0.9)

# training data contain first 90 percent of the sequences
X_train = X[:split]
y_train = y[:split]
X_val = X[split:]
y_val = y[split:]


# create the LSTM model using sequential layers

model = Sequential()
# embedding convert word numbers into useful word vectors
model.add(Embedding(input_dim=words, output_dim=128))
model.add(LSTM(128))

# dense layer predict the next word from all vocabulary words
model.add(Dense(words, activation="softmax"))


# prepare model for training
model.compile(
optimizer="adam",                       # update model weights
loss="sparse_categorical_crossentropy", # compare predicted and actual word
metrics=["accuracy"]                    # show prediction accuracy
)

# check model layers and parameters
model.summary()

# stop if validation loss stops improving
early_stop = EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)

# maximum 20 epochs and 128 samples are used in one batch
model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs=1,batch_size=128,validation_split=0.1,callbacks=[early_stop])

model.save("lstm_text_generator.keras")
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)
print("Model and tokenizer saved.")