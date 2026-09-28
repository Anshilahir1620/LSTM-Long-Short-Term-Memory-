# LSTM Text Generator

A simple next-word prediction and text generation project built using **Python, TensorFlow, Keras, NumPy, and an LSTM neural network**.

The model learns word patterns from Shakespeare's text dataset and generates new text from a given seed sentence.

> **Note:** This project does **not** use ChatGPT, Gemini, or any external LLM/API. The LSTM model is trained directly on the dataset.

---

## Project Objective

The main objective of this project is to build a basic next-word prediction system using an LSTM neural network.

1. The user provides a starting sentence (seed text).
2. The model predicts the most probable next word based on patterns learned from the training data.
3. The predicted word is appended to the sentence.
4. The process repeats to generate longer text.

---

## Problem Statement

Given a sequence of words, predict a probable next word.

---

## Tech Stack

| Category            | Tools                         |
| ------------------- | ----------------------------- |
| Language            | Python                        |
| Deep Learning       | TensorFlow, Keras             |
| Model               | LSTM (Long Short-Term Memory) |
| Numerical Computing | NumPy                         |
| Dataset             | Shakespeare text              |

---

## How It Works

1. **Load the dataset** – read Shakespeare's text.
2. **Preprocess** – clean the text and tokenize it into words.
3. **Create sequences** – build input sequences of words paired with the next word as the label.
4. **Pad sequences** – make input sequences the same length.
5. **Build the model** – Embedding layer → LSTM layer → Dense layer with softmax.
6. **Train** – train the model on the prepared sequences.
7. **Generate text** – predict the next word and append it to the input.
8. **Repeat** – continue the process to generate more words.

---

## Model Architecture

The model uses the following layers:

```text
Input Text
    ↓
Tokenization
    ↓
Padding
    ↓
Embedding Layer
    ↓
LSTM Layer
    ↓
Dense Layer
    ↓
Softmax
    ↓
Next Word Prediction
```

### Embedding Layer

The Embedding layer converts word IDs into numerical vectors that the model can understand.

### LSTM Layer

The LSTM layer learns patterns and relationships between words in the sequence.

### Dense + Softmax Layer

The Dense layer produces a probability for each word in the vocabulary. The word with the highest probability is selected as the predicted next word.

---

## Training

The model is compiled using:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

- **Adam** updates the model weights during training.
- **Sparse categorical crossentropy** measures the difference between the predicted word and the actual next word.
- **Accuracy** shows how often the predicted word matches the actual word.

The model was trained for **1 epoch** because training on CPU takes a significant amount of time.

---

## Example Input and Output

The model takes a short text as input and predicts the next words one by one.

**Input 1**

```text
to be or not to be
```

**Output 1**

```text
to be or not to be a man and the king of the world and the king of the king and the king of york and the king enter the
```

**Input 2**

```text
the king is
```

**Output 2**

```text
the king is the king of the king and the king of the king and the king is the king and the king of york and the king enter the king
```

**Input 3**

```text
love is
```

**Output 3**

```text
love is the king of the king and the king of the king and the king is the king and the king of york and the king enter the king
```

**Input 4**

```text
my lord
```

**Output 4**

```text
my lord and the king of the king and the king of the king and the king of york and the king enter the king
```

---

## Output Note

The current output is repetitive because the model was trained for only **1 epoch**.

The model successfully performs next-word prediction, but the generated text is not always grammatically correct or meaningful.

More training and better text-sampling techniques can improve the generated output.

---

## Dataset

The project uses Shakespeare's text dataset:

```text
dataset/
└── t8.shakespeare.txt
```

The text is used to train the model to learn word patterns and relationships.

---

## Text Preprocessing

The text goes through the following preprocessing steps:

```text
Raw Shakespeare Text
        ↓
Convert Text to Lowercase
        ↓
Tokenization
        ↓
Create Word Sequences
        ↓
Create Input and Target Words
        ↓
Padding
        ↓
Training Data
```

---

## Padding

Padding is used to make all input sequences the same length.

For example:

```text
[1, 2]
```

can become:

```text
[0, 0, 1, 2]
```

The `0` values are padding tokens.

The project uses Keras `pad_sequences` for this process.

---

## Prediction

During text generation, the model predicts the next word using:

```python
prediction = model.predict(padded, verbose=0)
```

The prediction contains probabilities for the words in the vocabulary.

The word with the highest probability is selected and added to the generated text.

The process continues until the required number of words has been generated.

---

## Training Configuration

The current training configuration is:

```text
Epochs     : 1
Batch Size : 128
Optimizer  : Adam
Loss       : Sparse Categorical Crossentropy
Metric     : Accuracy
```

---

## Saved Model

After training, the model and tokenizer are saved so they can be used later without training again.

```text
lstm_text_generator.keras
tokenizer.pkl
```

### Model File

`lstm_text_generator.keras`

Contains the trained LSTM model.

### Tokenizer File

`tokenizer.pkl`

Contains the tokenizer used during training to convert words into numerical IDs.

---

## Project Structure

```text
LSTM-Text-Generator/
│
├── dataset/
│   └── t8.shakespeare.txt
│
├── train.py
├── generat.py
├── tokenizer.pkl
├── lstm_text_generator.keras
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git

# Go into the project folder
cd <your-repo-name>

# Create virtual environment
python -m venv venv

# Activate virtual environment on Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Requirements

The main libraries used in this project are:

```text
Python
TensorFlow
Keras
NumPy
```

You can install the required packages using:

```bash
pip install tensorflow numpy
```

---

## How to Run

### 1. Train the Model

Run the training script:

```bash
python train.py
```

This will train the LSTM model and save:

```text
lstm_text_generator.keras
tokenizer.pkl
```

### 2. Generate Text

Run the text generation script:

```bash
python generat.py
```

Enter a seed sentence such as:

```text
to be or not to be
```

The model will generate the next words based on patterns learned during training.

---

## Limitations

- The generated text can be repetitive.
- The model was trained for only 1 epoch.
- The model has limited context compared with modern language models.
- The generated sentences are not always grammatically correct.
- The model always selects the highest-probability word, which can increase repetition.
- The model is a learning project and is not intended to compete with modern large language models.

---

## Future Improvements

- Train the model for more epochs.
- Use temperature sampling to make the output more diverse.
- Use top-k sampling to reduce repetitive predictions.
- Train using a larger text dataset.
- Experiment with multiple LSTM layers.
- Try GRU or Transformer-based models.
- Add a simple Streamlit interface for text generation.
- Improve text preprocessing and sequence length.

---

## Key Learnings

Through this project, I learned:

- How text data is prepared for machine learning.
- How tokenization converts words into numerical values.
- How input sequences are created for next-word prediction.
- How padding works.
- How Embedding layers represent words as vectors.
- How LSTM networks learn sequential patterns.
- How Dense and Softmax layers are used for word prediction.
- How a trained model can generate text one word at a time.
- How to save and load a trained TensorFlow/Keras model.

---

## Conclusion

This project demonstrates how an LSTM neural network can be used for basic next-word prediction and text generation.

The model learns patterns from Shakespeare's text and generates additional words based on a given seed sentence.

Although the current output is repetitive due to limited training, the project provides a practical understanding of how a simple language model works from dataset preprocessing to model training and text generation.

---

## Author

**Anshil Chotara**

Computer Science Engineering Student

```
