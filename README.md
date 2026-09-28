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

## Problem Statement

Given a sequence of words, predict a probable next word.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model | LSTM (Long Short-Term Memory) |
| Numerical Computing | NumPy |
| Dataset | Shakespeare text |

---

## How It Works

1. **Load the dataset** – read Shakespeare's text.
2. **Preprocess** – clean the text and tokenize it into words.
3. **Create sequences** – build input sequences of words paired with the next word as the label.
4. **Pad and encode** – pad sequences to equal length and one-hot encode the labels.
5. **Build the model** – Embedding layer → LSTM layer(s) → Dense layer with softmax.
6. **Train** – fit the model on the sequences to learn word patterns.
7. **Generate text** – repeatedly predict the next word from the seed text and append it.

---

## Example Input and Output

The model takes a short text as input and predicts the next words one by one.

**Input 1**
```
to be or not to be
```
**Output 1**
```
to be or not to be a man and the king of the world and the king of the king and the king of york and the king enter the
```

**Input 2**
```
the king is
```
**Output 2**
```
the king is the king of the king and the king of the king and the king is the king and the king of york and the king enter the king
```

**Input 3**
```
love is
```
**Output 3**
```
love is the king of the king and the king of the king and the king is the king and the king of york and the king enter the king
```

**Input 4**
```
my lord
```
**Output 4**
```
my lord and the king of the king and the king of the king and the king of york and the king enter the king
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow numpy
```

## Usage

1. Make sure the Shakespeare dataset file is in the project folder.
2. Run the training script / notebook to train the model.
3. Enter a seed sentence and the number of words to generate.

```python
seed_text = "to be or not to be"
next_words = 20
# generated text is printed after running the generation step
```

---

## Limitations

- The output is often **repetitive** (for example, "the king of the king"). This is common for small LSTM models that always pick the single most probable word.
- The model has a limited vocabulary and context, so the text is not always grammatically or logically correct.
- It is a learning project and not meant to compete with modern large language models.

## Future Improvements

- Use **temperature sampling** or top-k sampling instead of always picking the top word, to reduce repetition.
- Train for more epochs on a larger dataset.
- Stack more LSTM layers or try GRU / Transformer-based models.
- Add a simple web interface (for example with Streamlit) for entering seed text.

---

## Key Learnings

- How text is tokenized and converted into training sequences.
- How LSTM networks capture sequential patterns in language.
- How next-word prediction can be looped to generate text.

---

## License

This project is for educational purposes. Add a license of your choice (for example, MIT) if you plan to share it publicly.
