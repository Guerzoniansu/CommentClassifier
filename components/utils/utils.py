
# functions.py

# This file contains utility functions for the application.

import os
from pathlib import Path
import numpy as np
import pandas as pd
import re
import nltk
import string

nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.utils import shuffle

stop_words = set(stopwords.words('english'))

def load_text_from_folder(folder_path: str) -> str:
    """
    Load text files from a specified folder and return their contents as a string.
    Args:
        folder_path (str): The path to the folder containing text files.
    Returns:
        str: A string containing the contents of all text files in the folder, separated by newlines.
    """
    all_text = ""
    for file_path in Path(folder_path).rglob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            all_text += f.read() + "<|endoftext|>"
    return all_text

def load_corpus_from_folder(folder_path: str) -> list:
    """
    Load text files from a specified folder and return their contents as a list of strings.
    Args:
        folder_path (str): The path to the folder containing text files.
    Returns:
        list: A list of strings, each containing the contents of a text file.
    """
    corpus = []
    for file_path in Path(folder_path).rglob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            corpus.append(f.read())
    return corpus

def load_labeled_corpus(pos_path, neg_path):
    """
    Load and label text data from two folders (positive and negative samples), shuffle the combined dataset,
    and return it as a pandas DataFrame.

    Args:
        pos_path (str or Path): Path to the folder containing positive text files.
        neg_path (str or Path): Path to the folder containing negative text files.

    Returns:
        pandas.DataFrame: A shuffled DataFrame with two columns:
                          - 'text': the raw text content of each file.
                          - 'label': sentiment label (1 for positive, 0 for negative).
    """
    data = []
    for file_path in Path(pos_path).rglob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            data.append((f.read(), 1))  # positive = 1
    for file_path in Path(neg_path).rglob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            data.append((f.read(), 0))  # negative = 0
    return shuffle(pd.DataFrame(data, columns=['text', 'label']), random_state=42).reset_index(drop=True)


def clean_text(text: str) -> str:
    """
    Clean string from useless words, tokens and return the cleaned string
    Args:
        text (str): string to be cleaned
    Returns:
        str: cleaned text
    """
    text = text.lower()

    # Remove HTML tags like <br/> and others
    text = re.sub(r'<.*?>', ' ', text)
    # Remove URLs and mentions
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Tokenize and remove stopwords and non-alphabetic tokens
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and w.isalpha()]

    return " ".join(tokens)


