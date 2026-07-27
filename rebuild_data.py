import os
import ast
import re
import pickle
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import nltk
# pyrefly: ignore [missing-import]
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

print("Starting data rebuild process...")

# Download NLTK datasets
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def safe_parse_genres(genre_str):
    if pd.isna(genre_str) or not isinstance(genre_str, str):
        return ""
    try:
        parsed = ast.literal_eval(genre_str)
        if isinstance(parsed, list):
            return " ".join([i['name'] for i in parsed if isinstance(i, dict) and 'name' in i])
    except Exception:
        pass
    return ""

# 1. Load CSV
csv_path = 'movies_metadata.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path}")

print("Reading movies_metadata.csv...")
df = pd.read_csv(csv_path, low_memory=False)
print(f"Loaded {len(df)} rows.")

# 2. Drop duplicates
df = df.drop_duplicates().reset_index(drop=True)

# 3. Retain relevant columns
cols_to_keep = [
    'title', 'overview', 'genres', 'tagline', 'vote_average', 
    'popularity', 'imdb_id', 'poster_path', 'vote_count', 'release_date'
]
# Clean columns list to only keep existing ones
cols_to_keep = [c for c in cols_to_keep if c in df.columns]
df = df[cols_to_keep]

# 4. Filter missing titles
df = df.dropna(subset=['title'])
df['title'] = df['title'].astype(str)

# 5. Fill missing values
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')
df['genres_raw'] = df['genres']  # keep raw copy if needed
df['genres'] = df['genres'].apply(safe_parse_genres)

# Ensure popularity is numeric
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0.0)
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0.0)
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0.0)

# Reset index to match matrix rows
df = df.reset_index(drop=True)

print("Preprocessing tags...")
# 6. Build tags exactly like notebook
df['tags'] = df['overview'] + " " + df['genres'] + " " + df['tagline']
df['tags'] = df['tags'].apply(preprocess_text)

# 7. Create indices series mapping title -> index
print("Creating indices...")
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# 8. Compute TF-IDF matrix
print("Computing TF-IDF Matrix...")
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# 9. Save pickles
print("Saving pickle files...")
pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))
pickle.dump(indices, open('indices.pkl', 'wb'))
df.to_pickle('df.pkl')
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

print("Data rebuild completed successfully!")

