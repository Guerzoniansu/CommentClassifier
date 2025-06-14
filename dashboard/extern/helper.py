import re
import numpy as np
import pandas as pd
from collections import Counter
import json
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from io import BytesIO
from flask import session

nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizers import Tokenizer
import tiktoken
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

stop_words = set(stopwords.words('english'))
# 'movie', 'movies', 'film', 'films', 'cinema', 'screen', 'scene', 'actor', 'actress', 'actors', 'character', 'characters', 'director', 
custom_stopwords = {
    'hollywood'
}


MODEL_MAPPING = {
    'lstm_word': {
        'tokenizer': 'word',
        'model': 'classifier_sentiment_word_tokenized_lstm_model.h5'
    },
    'lstm_subword': {
        'tokenizer': 'bpe',
        'model': 'classifier_sentiment_subword_tokenized_lstm_model.h5'
    },
    'lstm_gpt2': {
        'tokenizer': 'tiktoken',
        'model': 'classifier_sentiment_subword_tokenized_pretrained_gpt2_lstm_model.h5'
    },
    'gru_word': {
        'tokenizer': 'word',
        'model': 'classifier_sentiment_word_tokenized_gru_model.h5'
    },
    'gru_subword': {
        'tokenizer': 'bpe',
        'model': 'classifier_sentiment_subword_tokenized_gru_model.h5'
    },
    'gru_gpt2': {
        'tokenizer': 'tiktoken',
        'model': 'classifier_sentiment_subword_tokenized_pretrained_gpt2_gru_model.h5'
    },
    'transformer_subword': {
        'tokenizer': 'bpe',
        'model': 'classifier_sentiment_subword_tokenized_transformer_model.h5'
    },
    'transformer_gpt2': {
        'tokenizer': 'tiktoken',
        'model': 'classifier_sentiment_subword_tokenized_pretrained_gpt2_transformer_model.h5'
    }
}

def clean_text(text: str) -> str:
    """
    Clean string from useless words, tokens and return the cleaned string
    Args:
        text (str): string to be cleaned
    Returns:
        str: cleaned text
    """
    text = str(text)
    text = text.lower()

    # Remove HTML tags like <br/> and others
    text = re.sub(r'<.*?>', ' ', text)
    # Remove URLs and mentions
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Tokenize and remove stopwords and non-alphabetic tokens
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in (stop_words | custom_stopwords) and w.isalpha()]

    return " ".join(tokens)

def clean_dataframe(df, column_name):
    df[f"clean_{column_name}"] = df[column_name].apply(clean_text)
    return df

def generate_text_statistics(df, column_name):
    """Generate comprehensive text statistics for the specified column"""
    text_series = df[column_name].dropna().astype(str)
    
    # Basic text metrics
    text_lengths = text_series.str.len()
    word_counts = text_series.str.split().str.len()
    
    stats = {
        # Basic info
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'target_column': column_name,
        
        # Text length statistics
        'avg_text_length': round(text_lengths.mean(), 2),
        'min_text_length': int(text_lengths.min()),
        'max_text_length': int(text_lengths.max()),
        
        # Word statistics
        'avg_word_count': round(word_counts.mean(), 2),
        'min_word_count': int(word_counts.min()),
        'max_word_count': int(word_counts.max())
    }
    
    return stats

def get_word_frequency_data(df, column_name, top_n=20):
    """Get word frequency data for charts"""
    
    text_series = df[f"clean_{column_name}"].dropna().astype(str)
    
    # Combine all text and split into words
    all_text = ' '.join(text_series.str.lower())
    words = re.findall(r'\b[a-zA-Z]+\b', all_text)
    
    # Get word frequencies
    word_freq = Counter(words)
    top_words = word_freq.most_common(top_n)
    
    return top_words

def get_top_bigrams(df, column_name, top_n=20):
    """
    Extract the top N bigrams from a specified text column in a DataFrame.
    """
    # Clean and concatenate all text
    text_series = df[f"clean_{column_name}"].dropna().astype(str)
    all_text = text_series.str.lower().str.cat(sep=' ')

    vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b[a-zA-Z]{2,}\b')
    X = vectorizer.fit_transform([all_text])

    # Sum frequencies
    bigram_counts = X.sum(axis=0).A1
    bigrams = vectorizer.get_feature_names_out()
    bigram_freq = list(zip(bigrams, bigram_counts))

    # Sort and return top N
    top_bigrams = sorted(bigram_freq, key=lambda x: x[1], reverse=True)[:top_n]
    return top_bigrams

def generate_wordcloud_html(df: pd.DataFrame, column_name: str) -> str:
    """
    Generate a word cloud from a specified text column in a DataFrame
    and return the image as an HTML <img> tag with base64-encoded data.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Combine all text entries into one string
    text_data = df[f"clean_{column_name}"].dropna().astype(str).str.cat(sep=' ')

    # Generate the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=200,
        colormap='Reds'
    ).generate(text_data)

    # Save to a BytesIO buffer
    buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()

    # Encode image as base64
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    # Return as HTML <img> tag
    html_img = f'<img src="data:image/png;base64,{img_base64}" alt="WordCloud"/>'
    return html_img



def get_length_distribution_data(df, column_name):
    """Get text length distribution data"""
    text_series = df[column_name].dropna().astype(str)
    text_lengths = text_series.str.len()
    
    # Create length bins
    bins = [0, 50, 100, 200, 500, 1000, 2000, float('inf')]
    labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1001-2000', '2000+']
    
    length_dist = pd.cut(text_lengths, bins=bins, labels=labels, right=True)
    counts = length_dist.value_counts().sort_index()
    
    return [(label, count) for label, count in counts.items()]

def get_character_analysis_data(df, column_name):
    """Get character type analysis data"""
    text_series = df[column_name].dropna().astype(str)
    
    total_chars = text_series.str.len().sum()
    digit_chars = text_series.str.count(r'\d').sum()
    upper_chars = text_series.str.count(r'[A-Z]').sum()
    lower_chars = text_series.str.count(r'[a-z]').sum()
    punct_chars = text_series.str.count(r'[^\w\s]').sum()
    space_chars = text_series.str.count(r'\s').sum()
    
    char_data = [
        ('Lowercase', lower_chars),
        ('Uppercase', upper_chars),
        ('Digits', digit_chars),
        ('Punctuation', punct_chars),
        ('Spaces', space_chars)
    ]
    
    return char_data

def generate_sample_table_data(df, column_name, n_samples=10):
    """Generate sample data for table display"""
    text_series = df[column_name].dropna().astype(str)
    
    # Get samples of different lengths
    samples = []
    
    # Shortest texts
    shortest_indices = text_series.str.len().nsmallest(n_samples//2).index
    for idx in shortest_indices:
        samples.append({
            'type': 'Shortest',
            'word_count': len(text_series[idx].split()),
            'preview': text_series[idx][:100] + '...' if len(text_series[idx]) > 100 else text_series[idx]
        })
    
    # Longest texts
    longest_indices = text_series.str.len().nlargest(n_samples//2).index
    for idx in longest_indices:
        samples.append({
            'type': 'Longest',
            'word_count': len(text_series[idx].split()),
            'preview': text_series[idx][:100] + '...' if len(text_series[idx]) > 100 else text_series[idx]
        })
    
    return samples

def predict(df, column_name, model_id):
    """
    Corrected prediction function with proper error handling and tokenizer loading
    """
    model_id = int(model_id) - 1

    model_keys = list(MODEL_MAPPING.keys())
    if model_id < 0 or model_id >= len(model_keys):
        raise ValueError(f"Model ID {model_id + 1} is not valid.")
    
    selected_model = MODEL_MAPPING[model_keys[model_id]]
    
    tokenizer_name = selected_model['tokenizer']
    model_path = f"../components/models/classifiers/{selected_model['model']}"

    # Load the tokenizer with corrected logic
    if tokenizer_name == 'tiktoken':
        tokenizer = tiktoken.get_encoding("gpt2")
        BPE_MAX_LEN = 500
        def tokenize(text):
            ids = tokenizer.encode(str(text))
            if len(ids) < BPE_MAX_LEN:
                return [0] * (BPE_MAX_LEN - len(ids)) + ids
            else:
                return ids[:BPE_MAX_LEN]
                
    elif tokenizer_name == 'word':
        with open('../components/models/tokenizers/word.json', 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
        MAX_LEN = 500
        def tokenize(text):
            seq = tokenizer.texts_to_sequences([str(text)])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            return padded[0]
            
    else:  # BPE tokenizer - fixed the file path issue
        tokenizer = Tokenizer.from_file("../components/models/tokenizers/bpe.json")
        BPE_MAX_LEN = 500
        def tokenize(text):
            encoded = tokenizer.encode(str(text)).ids
            if len(encoded) < BPE_MAX_LEN:
                return [0] * (BPE_MAX_LEN - len(encoded)) + encoded
            else:
                return encoded[:BPE_MAX_LEN]

    # Load the model
    model = load_model(model_path)

    # Prepare the input
    texts = df[column_name].astype(str).tolist()
    X = np.array([tokenize(text) for text in texts])

    # Predict
    probs = model.predict(X)
    preds = (probs > 0.5).astype(int)

    result_df = pd.DataFrame({
        "text": texts,
        "predicted": preds.flatten()
    })

    return result_df

def generate_prediction_html(df, column_name, model_id, cache_key):
    """
    Generate a comprehensive HTML dashboard with prediction statistics and visualizations
    """
    from wordcloud import WordCloud
    from collections import Counter
    import re
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Get predictions
    predict_df = predict(df, column_name, model_id)
    
    # Use the cleaned text column instead of the original
    cleaned_column_name = f"clean_{column_name}"
    if cleaned_column_name in df.columns:
        # Merge with cleaned text
        predict_df = predict_df.merge(
            df[[column_name, cleaned_column_name]], 
            left_on='text', 
            right_on=column_name, 
            how='left'
        )
        # Use cleaned text for analysis, handle NaN values
        predict_df['clean_text'] = predict_df[cleaned_column_name].fillna('').astype(str)
    else:
        # Fallback to original text if cleaned column doesn't exist
        predict_df['clean_text'] = predict_df['text'].fillna('').astype(str)
    
    # Calculate statistics
    total_reviews = len(predict_df)
    positive_count = (predict_df['predicted'] == 1).sum()
    negative_count = (predict_df['predicted'] == 0).sum()
    positive_percentage = (positive_count / total_reviews) * 100
    negative_percentage = (negative_count / total_reviews) * 100
    
    # Separate cleaned texts by prediction, filter out empty strings
    positive_texts = [text for text in predict_df[predict_df['predicted'] == 1]['clean_text'].tolist() 
                     if text and text.strip()]
    negative_texts = [text for text in predict_df[predict_df['predicted'] == 0]['clean_text'].tolist() 
                     if text and text.strip()]
    
    # Function to get bigrams from already cleaned text
    def get_bigrams(texts):
        all_words = []
        for text in texts:
            if text and text.strip():  # Skip empty texts
                words = str(text).split()  # Simple split since text is already cleaned
                all_words.extend(words)
        
        # Create bigrams
        bigrams = []
        for i in range(len(all_words) - 1):
            bigrams.append(f"{all_words[i]} {all_words[i+1]}")
        
        # Get top 10 bigrams
        bigram_counts = Counter(bigrams)
        return bigram_counts.most_common(10)
    
    # Generate word clouds
    def generate_wordcloud(texts, title):
        if not texts:
            return None
        
        # Filter out empty strings and convert to string, then combine
        valid_texts = [str(text) for text in texts if text and str(text).strip()]
        
        if not valid_texts:
            return None
        
        combined_text = ' '.join(valid_texts)
        
        if not combined_text.strip():
            return None
            
        # Create word cloud
        wordcloud = WordCloud(
            width=400, 
            height=300, 
            background_color='white',
            colormap='viridis' if 'Positive' in title else 'Reds',
            max_words=100
        ).generate(combined_text)
        
        # Convert to base64 image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    # Generate word clouds
    positive_wordcloud = generate_wordcloud(positive_texts, "Positive Reviews Word Cloud")
    negative_wordcloud = generate_wordcloud(negative_texts, "Negative Reviews Word Cloud")
    
    # Get top bigrams
    positive_bigrams = get_bigrams(positive_texts) if positive_texts else []
    negative_bigrams = get_bigrams(negative_texts) if negative_texts else []
    
    # Generate HTML dashboard
    html_template = f"""
    <div class="dashboard-container">
        <!-- Statistics Cards Grid -->
        <div class="stats-grid">
            <div class="review-card-2">
                <div class="card-icon">📊</div>
                <div class="card-content">
                    <div class="stat-number">{total_reviews:,}</div>
                    <div class="stat-label">Total Reviews</div>
                </div>
            </div>
            <div class="review-card-2 positive-card">
                <div class="card-icon">👍</div>
                <div class="card-content">
                    <div class="stat-number">{positive_count:,}</div>
                    <div class="stat-label">Positive Reviews</div>
                    <div class="stat-percentage">{positive_percentage:.1f}%</div>
                </div>
            </div>
            <div class="review-card-2 negative-card">
                <div class="card-icon">👎</div>
                <div class="card-content">
                    <div class="stat-number">{negative_count:,}</div>
                    <div class="stat-label">Negative Reviews</div>
                    <div class="stat-percentage">{negative_percentage:.1f}%</div>
                </div>
            </div>
        </div>
        
        <!-- Word Clouds Grid -->
        <div class="wordcloud-grid">
            <div class="wordcloud-container">
                <h3>Positive Reviews Word Cloud</h3>
                {f'<img src="{positive_wordcloud}" alt="Positive Word Cloud" class="wordcloud-image">' if positive_wordcloud else '<div class="no-data">No positive reviews to display</div>'}
            </div>
            <div class="wordcloud-container">
                <h3>Negative Reviews Word Cloud</h3>
                {f'<img src="{negative_wordcloud}" alt="Negative Word Cloud" class="wordcloud-image">' if negative_wordcloud else '<div class="no-data">No negative reviews to display</div>'}
            </div>
        </div>
        
        <!-- Top Bigrams Section -->
        <div class="bigrams-section">
            <div class="bigrams-container">
                <h3>🔝 Top 10 Positive Bigrams</h3>
                <div class="bigrams-list">
                    {' '.join([f'<div class="bigram-item positive-bigram"><span class="bigram-text">{bigram}</span><span class="bigram-count">{count}</span></div>' for bigram, count in positive_bigrams]) if positive_bigrams else '<div class="no-data">No positive bigrams found</div>'}
                </div>
            </div>
            <div class="bigrams-container">
                <h3>🔻 Top 10 Negative Bigrams</h3>
                <div class="bigrams-list">
                    {' '.join([f'<div class="bigram-item negative-bigram"><span class="bigram-text">{bigram}</span><span class="bigram-count">{count}</span></div>' for bigram, count in negative_bigrams]) if negative_bigrams else '<div class="no-data">No negative bigrams found</div>'}
                </div>
            </div>
        </div>
        <div class="download-section">
            <button id="downloadBtn" class="download-button" onclick="downloadClassifiedData()">
                <span class="download-icon">📥</span>
                Download Classified Dataset (CSV)
            </button>
        </div>
        
        <style>
            .dashboard-container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            
            /* Statistics Cards Grid */
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin-bottom: 40px;
            }}
            
            .review-card-2 {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                padding: 25px;
                color: white;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                display: flex;
                align-items: center;
                gap: 20px;
            }}
            
            .review-card-2:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.3);
            }}
            
            .positive-card {{
                background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
            }}
            
            .negative-card {{
                background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
            }}
            
            .confidence-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            
            .card-icon {{
                font-size: 2.5em;
                opacity: 0.8;
            }}
            
            .card-content {{
                flex: 1;
            }}
            
            .stat-number {{
                font-size: 2.2em;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.9;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .stat-percentage {{
                font-size: 0.8em;
                opacity: 0.8;
                margin-top: 5px;
            }}
            
            /* Word Clouds Grid */
            .wordcloud-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 30px;
                margin-bottom: 40px;
            }}
            
            .wordcloud-container {{
                background: white;
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                text-align: center;
            }}
            
            .wordcloud-container h3 {{
                margin-bottom: 20px;
                color: #2c3e50;
                font-size: 1.4em;
            }}
            
            .wordcloud-image {{
                max-width: 100%;
                height: auto;
                border-radius: 10px;
            }}
            
            /* Bigrams Section */
            .bigrams-section {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 30px;
            }}
            
            .bigrams-container {{
                background: white;
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            
            .bigrams-container h3 {{
                margin-bottom: 20px;
                color: #2c3e50;
                font-size: 1.4em;
                text-align: center;
            }}
            
            .bigrams-list {{
                display: flex;
                flex-direction: column;
                gap: 10px;
            }}
            
            .bigram-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 16px;
                border-radius: 8px;
                transition: transform 0.2s ease;
            }}
            
            .bigram-item:hover {{
                transform: translateX(5px);
            }}
            
            .positive-bigram {{
                background: linear-gradient(90deg, #a8e6cf, #dcedc8);
                border-left: 4px solid #4caf50;
            }}
            
            .negative-bigram {{
                background: linear-gradient(90deg, #ffcdd2, #ffebee);
                border-left: 4px solid #f44336;
            }}
            
            .bigram-text {{
                font-weight: 500;
                color: #2c3e50;
            }}
            
            .bigram-count {{
                background: rgba(0,0,0,0.1);
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.9em;
                font-weight: bold;
            }}
            
            .no-data {{
                text-align: center;
                color: #666;
                font-style: italic;
                padding: 20px;
            }}
            
            /* Responsive Design */
            @media (max-width: 768px) {{
                .stats-grid {{
                    grid-template-columns: repeat(2, 1fr);
                }}
                
                .wordcloud-grid,
                .bigrams-section {{
                    grid-template-columns: 1fr;
                }}
                
                .review-card-2 {{
                    flex-direction: column;
                    text-align: center;
                }}
            }}
            /* Download Section */
            .download-section {{
                margin-top: 40px;
                text-align: center;
                padding: 30px;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}

            .download-button {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 50px;
                font-size: 1.1em;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                display: inline-flex;
                align-items: center;
                gap: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}

            .download-button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }}

            .download-icon {{
                font-size: 1.2em;
            }}
        </style>
        <script>
            function downloadClassifiedData() {{
                // Use the actual cache key value
                window.location.href = '/download_classified_data/{cache_key}';
            }}
        </script>
    </div>
    """
    
    return html_template, predict_df.to_dict('records')

def generate_charts_html_v2(df, column_name, model_id, cache_key):
    """Generate comprehensive HTML for multiple charts and tables"""

    stats = generate_text_statistics(df, column_name)
    word_freq_data = get_word_frequency_data(df, column_name, top_n=15)
    bigram_data = get_top_bigrams(df, column_name, top_n=10)
    length_dist_data = get_length_distribution_data(df, column_name)
    char_analysis_data = get_character_analysis_data(df, column_name)
    sample_data = generate_sample_table_data(df, column_name, n_samples=10)
    wordcloud_html = generate_wordcloud_html(df, column_name)
    predict_html, rec = generate_prediction_html(df, column_name, model_id, cache_key)
    
    # Prepare data for JavaScript
    word_labels = [word for word, count in word_freq_data]
    word_counts = [count for word, count in word_freq_data]
    
    bigram_labels = [bigram for bigram, count in bigram_data]
    bigram_counts = [count for bigram, count in bigram_data]
    
    length_labels = [label for label, count in length_dist_data]
    length_counts = [count for label, count in length_dist_data]
    
    char_labels = [char_type for char_type, count in char_analysis_data]
    char_counts = [count for char_type, count in char_analysis_data]

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Analytics Dashboard</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
        <style>
            :root {{
                /*Sidebar background*/
                --sidebar-bg: black;
                --lighter-sidebar-bg: hsl(0, 0%, 6%);

                /*Backgrounds*/
                --h1-colors: hsl(200, 88%, 53%);

                /*Text colors*/
                --black: hsl(270, 3%, 11%);
                --white: hsl(0, 0%, 100%);
                --dark-grey: hsl(264, 5%, 20%);
                --light-grey: hsl(210, 17%, 95%);

                /*linecolor*/
                --bg: hsl(0, 0%, 84%);

                /*Hovers*/
                --row-hover: rgb(222, 226, 230);
                --li-hover: hsl(20, 15%, 92%);

                /*Borders*/
                --border: hsl(0, 5%, 85%);

                /*Highlights*/
                --high-light-1: hsl(0, 0%, 100%);
                --high-light-2: hsl(0, 8%, 95%);

                /*Colors*/
                --red: hsl(348, 80%, 50%);
                --green: hsl(168, 100%, 36%);
                --orange: hsl(48, 98%, 52%);

                --color1: #57efc4;
                --color2: rgb(131, 236, 236);
                --color3: rgb(117, 186, 255);
                --color4: rgb(161, 154, 254);
                --color5: #dee5e8;
                --color6: hsl(168, 100%, 36%);
                --color7: rgb(0, 204, 201);
                --color8: rgb(9, 132, 225);
                --color9: rgb(108, 92, 230);
                --color10: hsl(198, 12%, 73%);
                --color11: hsl(46, 100%, 83%);
                --color12: rgb(250, 175, 158);
                --color13: rgb(255, 117, 117);
                --color14: hsl(339, 96%, 73%);
                --color15: hsl(196, 7%, 42%);
                --color16: rgb(253, 203, 109);
                --color17: rgb(225, 114, 86);
                --color18: #d62e2e;
                --color19: hsl(331, 78%, 59%);
                --color20: hsl(193, 9%, 19%);
            }}
            .review {{
                width: 100%;
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-bottom: 16px;
            }}

            .review-2 {{
                width: 100%;
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin-bottom: 16px;
            }}

            .review .review-card {{
                width: 100%;
                padding: 15px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 15px 15px;
                border: 1px solid var(--border);
                border-radius: 10px;
                color: var(--light-text-color-1);
            }}

            .review-2 .review-card {{
                width: 100%;
                padding: 15px 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px solid var(--border);
                border-radius: 10px;
                color: var(--light-text-color-1);
            }}

            .review-2 .review-card#chart-most-freq-bigram {{
                width: 100%;
                padding: 15px 15px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                border: 1px solid var(--border);
                border-radius: 10px;
                color: var(--light-text-color-1);
            }}

            .review .review-card#dataset-info .head .svg {{
                background: var(--color18);
            }}

            .review .review-card#text-statistics .head .svg {{
                background: var(--color9);
            }}

            .review .review-card#most-freq-word .head .svg {{
                background: var(--color17);
            }}

            .review .review-card#most-freq-bigram .head .svg {{
                background: var(--color6);
            }}

            .review .review-card .head {{
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: flex-start;
            }}

            .review .review-card .head .svg {{
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
                border-radius: 5px;
                color: var(--white);
            }}

            .review .review-card .head .text {{
                font-size: 14px;
                text-align: left;
                padding: 2px 8px;
            }}

            .review .review-card .cont {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                width: 100%;
            }}

            .review .review-card .cont .val {{
                font-size: 3.5rem;
            }}

            .review .review-card .cont .text {{
                font-size: 12px;
            }}

            .review .review-card .other {{
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
            }}

            .review .review-card .other .date {{
                font-size: 15px;
            }}
            .wordcloud-section {{
                grid-column: 1 / -1;
                text-align: center;
                padding: 20px;
                background: white;
                border: 1px solid var(--border);
            }}
            
            .wordcloud-section img {{
                max-width: 100%;
            }}

            .stats-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 3px 0;
                font-size: 12px;
                border-radius: 8px;
                overflow: hidden;
            }}

            .stats-table th,
            .stats-table td {{
                padding: 2px 4px;
                text-align: center;
            }}

            .stats-table th {{
                color: var(--dark-grey);
                font-weight: 500;
                font-size: 13px;
            }}

            .stats-table tr:nth-child(even) {{
                background-color: #fafafa;
            }}

            .stats-table tr:nth-child(odd) {{
                background-color: #fff;
            }}

            .stats-table td {{
                color: #555;
            }}

            .horizontal-bar-chart {{
                max-height: 400px;
                overflow-y: auto;
                width: 100%;
            }}

            .horizontal-bar-chart .head {{
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: flex-start;
                margin-bottom: 10px;
            }}

            .horizontal-bar-chart .head .svg {{
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
                border-radius: 5px;
                color: var(--white);
                background: var(--color6);
            }}

            .horizontal-bar-chart .head .text {{
                font-size: 14px;
                text-align: left;
                padding: 2px 8px;
            }}
            
            .h-bar-item {{
                display: flex;
                align-items: center;
                padding: 1px 0;
                margin-bottom: 5px
            }}
            
            .h-bar-label {{
                min-width: 80px;
                font-size: 12px;
                color: #495057;
                text-align: right;
                margin-right: 10px;
            }}
            
            .h-bar-container {{
                flex: 1;
                display: flex;
                align-items: center;
            }}
            
            .h-bar {{
                background: linear-gradient(to right, #28a745, #20c997);
                height: 20px;
                margin-right: 8px;
                min-width: 2px;
            }}
            
            .h-bar-value {{
                font-size: 11px;
                color: #6c757d;
                min-width: 30px;
            }}

            .bar-chart {{
                display: flex;
                align-items: end;
                justify-content: space-around;
                height: 100%;
                padding: 10px 0;
            }}

            .review-2 .review-card#chart-most-freq-bigram .head {{
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: flex-start;
                margin-bottom: 10px;
            }}

            .review-2 .review-card#chart-most-freq-bigram .head .svg {{
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
                border-radius: 5px;
                color: var(--white);
                background: var(--color18);
            }}

            .review-2 .review-card#chart-most-freq-bigram .head .text {{
                font-size: 14px;
                text-align: left;
                padding: 2px 8px;
            }}
            
            .bar-item {{
                display: flex;
                flex-direction: column;
                align-items: center;
                flex: 1;
                margin: 0 2px;
            }}
            
            .bar {{
                background: linear-gradient(to top, #ff4d4d, #b30000);
                width: 100%;
                min-height: 2px;
                border-radius: 2px 2px 0 0;
                margin-bottom: 8px;
                transition: opacity 0.3s;
            }}
            
            .bar:hover {{
                opacity: 0.8;
            }}
            
            .bar-label {{
                font-size: 11px;
                text-align: center;
                color: #6c757d;
                margin-bottom: 4px;
            }}
            
            .bar-value {{
                font-weight: bold;
                font-size: 12px;
                color: #495057;
            }}

        </style>
    </head>
    <body>
        <div class="review">
            <div class="review-card" id="dataset-info">
                <div class="head">
                    <div class="svg">
                        <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="M320-240h320v-80H320v80Zm0-160h320v-80H320v80ZM240-80q-33 0-56.5-23.5T160-160v-640q0-33 23.5-56.5T240-880h320l240 240v480q0 33-23.5 56.5T720-80H240Zm280-520v-200H240v640h480v-440H520ZM240-800v200-200 640-640Z"/></svg>
                    </div>
                    <p class="text">Dataset Info</p>
                </div>
                <div class="cont">
                    <p class="val">{stats['total_rows']:,}</p>
                    <p class="text">Rows</p>
                </div>
                <div class="other">
                    <table class="stats-table">
                        <tr>
                            <th>Statistic</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Columns</td>
                            <td>{stats['total_columns']:,}</td>
                        </tr>
                        <tr>
                            <td>Target Column</td>
                            <td>{stats['target_column']}</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div class="review-card" id="text-statistics">
                <div class="head">
                    <div class="svg">
                        <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="M680-360q-17 0-28.5-11.5T640-400v-160q0-17 11.5-28.5T680-600h120q17 0 28.5 11.5T840-560v40h-60v-20h-80v120h80v-20h60v40q0 17-11.5 28.5T800-360H680Zm-300 0v-240h160q17 0 28.5 11.5T580-560v40q0 17-11.5 28.5T540-480q17 0 28.5 11.5T580-440v40q0 17-11.5 28.5T540-360H380Zm60-150h80v-30h-80v30Zm0 90h80v-30h-80v30Zm-320 60v-200q0-17 11.5-28.5T160-600h120q17 0 28.5 11.5T320-560v200h-60v-60h-80v60h-60Zm60-120h80v-60h-80v60Z"/></svg>
                    </div>
                    <p class="text">Text Statistics</p>
                </div>
                <div class="cont">
                    <p class="val">{stats['avg_text_length']:,}</p>
                    <p class="text">Avg Length</p>
                </div>
                <div class="other">
                    <table class="stats-table">
                        <tr>
                            <th>Statistic</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Min Length</td>
                            <td>{stats['min_text_length']:,}</td>
                        </tr>
                        <tr>
                            <td>Max Length</td>
                            <td>{stats['max_text_length']:,}</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div class="review-card" id="most-freq-word">
                <div class="head">
                    <div class="svg">
                        <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="M200-200v-80h560v80H200Zm76-160 164-440h80l164 440h-76l-38-112H392l-40 112h-76Zm138-176h132l-64-182h-4l-64 182Z"/></svg>
                    </div>
                    <p class="text">Most Frequent Word</p>
                </div>
                <div class="cont">
                    <p class="val">{word_labels[0]}</p>
                </div>
                <div class="other">
                    <table class="stats-table">
                        <tr>
                            <th>Statistic</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Count</td>
                            <td>{word_counts[0]}</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div class="review-card" id="most-freq-bigram">
                <div class="head">
                    <div class="svg">
                        <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="M680-80v-60h100v-30h-60v-60h60v-30H680v-60h120q17 0 28.5 11.5T840-280v40q0 17-11.5 28.5T800-200q17 0 28.5 11.5T840-160v40q0 17-11.5 28.5T800-80H680Zm0-280v-110q0-17 11.5-28.5T720-510h60v-30H680v-60h120q17 0 28.5 11.5T840-560v70q0 17-11.5 28.5T800-450h-60v30h100v60H680Zm60-280v-180h-60v-60h120v240h-60ZM120-200v-80h480v80H120Zm0-240v-80h480v80H120Zm0-240v-80h480v80H120Z"/></svg>
                    </div>
                    <p class="text">Most Frequent Bigram</p>
                </div>
                <div class="cont">
                    <p class="val" style="font-size: 2.5rem;">{bigram_labels[0]}</p>
                </div>
                <div class="other">
                    <table class="stats-table">
                        <tr>
                            <th>Statistic</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Count</td>
                            <td>{bigram_counts[0]}</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        <div class="review-2">
            <div class="review-card" id="chart-most-freq-word">
                <div class="horizontal-bar-chart">
                    <div class="head">
                        <div class="svg">
                            <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="M120-240v-80h240v80H120Zm0-200v-80h480v80H120Zm0-200v-80h720v80H120Z"/></svg>
                        </div>
                        <p class="text">Most Frequent Words</p>
                    </div>
    """

    max_word_freq = max([freq for _, freq in word_freq_data]) if word_freq_data else 1
    for word, freq in word_freq_data:
        width = (freq / max_word_freq) * 100 if max_word_freq > 0 else 0
        html += f"""
                    <div class="h-bar-item">
                        <div class="h-bar-label">{word}</div>
                        <div class="h-bar-container">
                            <div class="h-bar" style="width: {width}%; height: 15px" title="{freq} occurrences"></div>
                            <div class="h-bar-value">{freq}</div>
                        </div>
                    </div>
        """

    html += f"""
                </div>
            </div>
            <div class="review-card" id="chart-most-freq-bigram">
                <div class="head">
                    <div class="svg">
                        <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="M640-160v-280h160v280H640Zm-240 0v-640h160v640H400Zm-240 0v-440h160v440H160Z"/></svg>
                    </div>
                    <p class="text">Most Frequent Bigrams</p>
                </div>
                <div class="bar-chart">
    """

    max_bigram_freq = max([freq for _, freq in bigram_data]) if bigram_data else 1
    for bigram, freq in bigram_data:
        height = (freq / max_bigram_freq) * 150 if max_bigram_freq > 0 else 0
        percentage = (freq / stats['total_rows']) * 100 if stats['total_rows'] > 0 else 0
        html += f"""
                    <div class="bar-item">
                        <div class="bar" style="height: {height}px;" title="{freq} texts ({percentage:.1f}%)"></div>
                        <div class="bar-label">{bigram}</div>
                        <div class="bar-value">{freq}</div>
                    </div>
        """

    html += f"""
                </div>
            </div>
        </div>
        <div class="wordcloud-section">
            {wordcloud_html}
        </div>
        {predict_html}
    </body>
    """

    return html, rec


    


















