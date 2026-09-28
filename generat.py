import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("lstm_text_generator.keras")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)
sequence_length = 20
index_word = {}

# store each word with there number
for word, number in tokenizer.word_index.items():
    index_word[number] = word







# generate next words from given seed text
# number of words tell how many new words we want
def generate_text(seed, number_of_words=30):

    text = seed.lower()
    for i in range(number_of_words):
        # tokenizer use same word mapping from training
        tokens = tokenizer.texts_to_sequences([text])[0]
        tokens = tokens[-sequence_length:]
        tokens = pad_sequences([tokens],maxlen=sequence_length,padding="pre")


        # predict the next word from trained model
        prediction = model.predict(tokens, verbose=0)
        next_word_id = np.argmax(prediction[0])
        # convert predicted number into actual word
        next_word = index_word.get(next_word_id)
        if next_word is None:
            break
        text = text + " " + next_word
    return text


# some seed text are given for testing the model
seeds = [
    "to be or not to be",
    "the king is",
    "love is",
    "my lord"
]
for seed in seeds:
    print("\nSeed:", seed)
    # generate 30 new words from the given seed
    result = generate_text(seed, 30)
    print("Result:", result)