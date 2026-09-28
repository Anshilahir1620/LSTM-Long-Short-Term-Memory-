# LSTM Text Generator

A simple next-word prediction and text generation project built using Python, TensorFlow, Keras, NumPy, and an LSTM neural network.

The model learns word patterns from Shakespeare's text dataset and generates text from a given seed sentence.

This project does not use ChatGPT, Gemini, or any external LLM/API. The LSTM model is trained directly on the dataset.

## Project Objective

The main objective of this project is to build a basic next-word prediction system using an LSTM neural network.

The user provides a starting sentence, and the model predicts the next word based on patterns learned from the training data.

The predicted word is added to the sentence, and the model predicts another word. This process continues to generate text.

## Problem Statement

Given a sequence of words, predict a probable next word.

## Input and Output

### Input 1

```text
Output 1
to be or not to be a man and the king of the world and the king of
the king and the king of york and the king enter the
Input 2
the king is
Output 2
the king is the king of the king and the king of the king and the king
is the king and the king of york and the king enter the king
Input 3
love is
Output 3
love is the king of the king and the king of the king and the king is
the king and the king of york and the king enter the king
Input 4
my lord
Output 4
my lord and the king of the king and the king of the king and the king
of york and the king enter the king```
