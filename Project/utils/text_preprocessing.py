import numpy as np
import pandas as pd
import re
import json
import fasttext
import fasttext.util
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import unicodedata
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from config
from config import (
    JSON_FILE_PATH,
    HINDI_STOPWORDS_FILE,
    FASTTEXT_INPUT_FILE,
    FASTTEXT_MODEL_FILE,
    EMOTION_DIST_PLOT,
    Y_LABELS_FILE,
    PROCESSED_DATA_FILE,
    ADV_FEATURES_FILE,
    EMO_FREQUENCY,
    GRAPH_CLUSTER_OF_EMOTIONS
)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Access English stopwords
stop_words = set(stopwords.words('english'))


emotion_mapping = {
    'Anger': 0,
    'Surprise': 1,
    'Ridicule': 2,
    'Sad': 3
}

reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

# Step 1: Data Loading and Initial Exploration
def load_data(json_file_path):
    """Load and format the dataset from a JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded dataset with {len(data)} entries")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_dataframe(data, limit=None):
    """Create a dataframe from the loaded JSON data."""
    processed_examples = []
    count = 0
    for id, instance in data.items():
        if limit is not None and count >= limit:
            break
        text = instance["target_utterance"]
        context = " ".join(instance["context_utterances"]) if "context_utterances" in instance else ""
        emotion = instance["emotion"]
        full_text = context + " " + text
        if emotion not in emotion_mapping:
            print(f"Skipping instance {id} due to unknown emotion: {emotion}")
            continue
        processed_examples.append({
            "text": full_text,
            "emotion": emotion_mapping[emotion],
            "target_utterance": text,
            "context_utterances": context,
            "full_text": full_text
        })
        count += 1
    
    train_df = pd.DataFrame(processed_examples)
    print(f"Created DataFrame with {len(train_df)} entries and {train_df.shape[1]} columns")
    print(train_df.head())
    
    return train_df

def explore_dataset(train_df):
    """Basic exploration of the dataset."""
    print("\nDataset Information:")
    print(f"Number of entries: {len(train_df)}")
    print("\nColumns in dataset:")
    for col in train_df.columns:
        print(f"- {col}")
    
    if 'emotion' in train_df.columns:
        print("\nEmotion distribution:")
        emotion_counts = train_df['emotion'].value_counts()
        print(emotion_counts)
        
        # Visualize emotion distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
        plt.title('Distribution of Emotions in Dataset')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(EMOTION_DIST_PLOT)
        print("Emotion distribution chart saved as 'emotion_distribution.png'")
    
    return emotion_counts if 'emotion' in train_df.columns else None

# Step 2: Text Preprocessing Functions
def load_hindi_stopwords(file_path):
    """Load Hindi stopwords from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            stopwords = [line.strip() for line in file if line.strip()]
        return set(stopwords)
    except FileNotFoundError:
        print(f"Warning: Stopwords file not found at {file_path}. Using empty stopword list.")
        return set()

def normalize_hindi_spelling(text):
    """Normalize different spellings of Hindi words based on common interchangeable letters."""
    # Replace interchangeable letters
    replacements = {
        'q': 'k', 
        'z': 'j',
        'o': 'u',
        'w': 'v'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Handle repeated sequential letters (keep only one occurrence)
    text = re.sub(r'(.)\1+', r'\1', text)
    
    return text

def create_hindi_normalization_dict(texts):
    """Create a dictionary for normalizing Hindi words based on frequency."""
    # Group similar words
    word_groups = defaultdict(list)
    
    # Process each text
    for text in texts:
        words = text.split()
        for word in words:
            # Skip English words
            if re.match(r'^[a-zA-Z]+$', word):
                continue
            
            # Normalize the word to create a key
            key = normalize_hindi_spelling(word)
            word_groups[key].append(word)
    
    # For each group, find the most frequent form
    normalization_dict = {}
    for key, variants in word_groups.items():
        if len(variants) > 1:  # Only process words with variants
            # Count occurrences of each variant
            variant_counts = Counter(variants)
            # Select the most frequent variant
            most_common = variant_counts.most_common(1)[0][0]
            
            # Create mappings for all variants
            for variant in variants:
                if variant != most_common:
                    normalization_dict[variant] = most_common
    
    return normalization_dict

def clean_text_advanced(text, hindi_stopwords_file=None, remove_stopwords=True):

    """
    Advanced cleaning for code-mixed text (Hindi and English).
    Includes Unicode normalization, stopword removal, and stemming for English words.
    """
    if not isinstance(text, str):
        return ""
    
    # Unicode normalization (combines characters and their diacritics)
    text = unicodedata.normalize('NFC', text)
    text = normalize_hindi_spelling(text)
    
    # Convert English text to lowercase (Hindi is not affected)
    english_parts = re.findall(r'[a-zA-Z]+', text)
    for part in english_parts:
        text = text.replace(part, part.lower())
    
    # Remove special characters, keeping Hindi characters (Devanagari Unicode range), English characters, and spaces
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    
    # Remove stopwords
    if remove_stopwords:
        english_stopwords = set(stopwords.words('english'))

        # Load Hindi stopwords if file is provided
        hindi_stopwords = load_hindi_stopwords(hindi_stopwords_file) if hindi_stopwords_file else set()
        
        # Combine stopwords
        all_stopwords = english_stopwords.union(hindi_stopwords)
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in all_stopwords]
        text = ' '.join(words)
    
    # Stem English words
    stemmer = PorterStemmer()
    words = text.split()
    
    # Use regex to identify English words (assuming Hindi words use Devanagari script)
    stemmed_words = []
    for word in words:
        if re.match(r'^[a-zA-Z]+$', word):  # English word
            stemmed_words.append(stemmer.stem(word))
        else:  # Hindi word or mixed
            stemmed_words.append(word)
    
    text = ' '.join(stemmed_words)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def simple_clean_text(text):
    """Simple cleaning function (similar to the one in the second snippet)."""
    if not isinstance(text, str):
        return ""
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text

def print_text_comparison(original_text, cleaned_text):
    """Print comparison between original and cleaned text."""
    # Split the original and cleaned texts into words for comparison
    original_words = set(original_text.split())
    cleaned_words = set(cleaned_text.split())
    
    # Find removed words (present inˀ original but not in cleaned)
    removed_words = original_words - cleaned_words
    
    print("Original Text:")
    print(original_text)
    print("\nCleaned Text:")
    print(cleaned_text)
    print("\nRemoved Words:")
    print(removed_words)

# Step 3: FastText Model Training and Feature Extraction
def create_fasttext_file(dataframe, filename, text_column="text_cleaned", label_column="emotion"):
    """Create a text file in the format required by FastText."""
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in dataframe.iterrows():
            f.write(f"__label__{row[label_column]} {row[text_column]}\n")
    print(f"Created FastText input file: {filename}")

def train_fasttext_model(input_file, output_model_name=FASTTEXT_MODEL_FILE, params=None):
    """Train a FastText model with the given parameters."""
    if params is None:
        params = {
            'lr': 0.5,
            'epoch': 25,
            'wordNgrams': 2,
            'dim': 100,
            'loss': 'softmax'
        }
    
    # Convert Path object to string if necessary
    input_file_str = str(input_file)
    
    print(f"Training FastText model with parameters: {params}")
    model = fasttext.train_supervised(
        input=input_file_str,
        lr=params['lr'],
        epoch=params['epoch'],
        wordNgrams=params['wordNgrams'],
        dim=params['dim'],
        loss=params['loss']
    )
    
    # Convert output_model_name to string if it's a Path object
    output_model_name_str = str(output_model_name)
    
    # Save the model
    model.save_model(output_model_name_str)
    print(f"FastText model saved as: {output_model_name_str}")
    return model

def extract_features(model, dataframe, text_column="text_cleaned"):
    """Extract features using the trained FastText model."""
    X_features = []
    y_labels = []
    
    print("Extracting features...")
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        # Get text embedding (sentence vector)
        embedding = model.get_sentence_vector(row[text_column])
        X_features.append(embedding)
        y_labels.append(row['emotion'])
    
    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    
    print(f"Features extracted: X shape = {X_features.shape}, y shape = {y_labels.shape}")
    return X_features, y_labels

# Step 4: Main Execution Pipeline
def main(json_file_path, hindi_stopwords_file=None, limit=None, use_advanced_cleaning=True):
    """Main execution pipeline."""
    # 1. Load data
    print("\n=== Loading Data ===")
    data = load_data(json_file_path)
    if data is None:
        return
    
    # 2. Create DataFrame
    print("\n=== Creating DataFrame ===")
    train_df = create_dataframe(data, limit=limit)
    
    
    # 3. Explore dataset
    print("\n=== Exploring Dataset ===")
    explore_dataset(train_df)
    
    # 4. Preprocess text
    print("\n=== Preprocessing Text ===")
    if use_advanced_cleaning:
        train_df['text_cleaned'] = train_df['text'].apply(
            lambda x: clean_text_advanced(x, hindi_stopwords_file=hindi_stopwords_file)
        )
        cleaning_method = "advanced"
    else:
        train_df['text_cleaned'] = train_df['text'].apply(simple_clean_text)
        cleaning_method = "simple"
    
    print(f"Applied {cleaning_method} cleaning to text")
    
    # 5. Print sample comparison
    if len(train_df) > 0:
        print("\n=== Sample Text Cleaning Comparison ===")
        print_text_comparison(train_df['text'].iloc[0], train_df['text_cleaned'].iloc[0])
    
    # 6. Create FastText input file
    print("\n=== Creating FastText Input File ===")
    print("\n=== Creating FastText Input File ===")
    fasttext_file = f"fasttext_input_{cleaning_method}.txt"
    # Ensure the Path object is converted to string
    create_fasttext_file(train_df, str(FASTTEXT_INPUT_FILE))
    # 7. Train FastText model
    print("\n=== Training FastText Model ===")
    model = train_fasttext_model(FASTTEXT_INPUT_FILE, output_model_name=f"emotion_model_{cleaning_method}.bin")
    
    # 8. Extract features
    print("\n=== Extracting Features ===")
    X_features, y_labels = extract_features(model, train_df)
    # Save y_labels
    pd.DataFrame(y_labels, columns=['emotion']).to_csv(Y_LABELS_FILE)
    print("y_labels saved to 'y_labels.csv'")
    # 9. Print some feature statistics
    print("\n=== Feature Statistics ===")
    print(f"Feature dimensionality: {X_features.shape[1]}")
    print(f"Feature mean: {np.mean(X_features)}")
    print(f"Feature std: {np.std(X_features)}")
    
    # 10. Save processed data
    print("\n=== Saving Processed Data ===")
    train_df.to_csv(PROCESSED_DATA_FILE)
    
    # 11. Save feature data
    print("\n=== Saving Feature Data ===")
    feature_df = pd.DataFrame(X_features)
    feature_df['emotion'] = y_labels
    feature_df.to_csv(ADV_FEATURES_FILE)
    
    print("\n=== Pipeline Completed Successfully ===")
    return train_df, model, X_features, y_labels

# Example usage
if __name__ == "__main__":
    # Set parameters
    json_file_path = JSON_FILE_PATH
    hindi_stopwords_file = None  # Set to None if not available
    limit = None # Set to None to process all data
    use_advanced_cleaning = True  # Set to False to use simple cleaning
    
    # Run the pipeline
    train_df, model, X_features, y_labels = main(
        json_file_path=json_file_path, 
        hindi_stopwords_file=hindi_stopwords_file,
        limit=limit,
        use_advanced_cleaning=use_advanced_cleaning
    )

# If i use Unsupervised fasttext model for sentence vector embedding extraction and not training a classification model.
# def train_unsupervised_fasttext(input_file, output_model_name="unsupervised_model.bin"):
#     """Train an unsupervised FastText model to get sentence embeddings."""
#     model = fasttext.train_unsupervised(input=input_file, model="skipgram", dim=100)
    
#     # Save the model
#     model.save_model(output_model_name)
#     print(f"FastText unsupervised model saved as: {output_model_name}")
#     return model

# def extract_sentence_embeddings(model, dataframe, text_column="text_cleaned"):
#     """Extract sentence embeddings using an unsupervised FastText model."""
#     X_features = []
    
#     print("Extracting sentence embeddings...")
#     for _, row in dataframe.iterrows():
#         embedding = model.get_sentence_vector(row[text_column])
#         X_features.append(embedding)
    
#     X_features = np.array(X_features)
#     print(f"Extracted embeddings shape: {X_features.shape}")
#     return X_features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter

def plot_embeddings(X_features, y_labels, method="tsne", perplexity=30, n_components=2, random_state=42):
    """
    Reduce the dimensionality of FastText sentence embeddings and visualize the clustering by emotion.

    Parameters:
        X_features (numpy.ndarray): Sentence embeddings.
        y_labels (numpy.ndarray): Corresponding emotion labels.
        method (str): Dimensionality reduction method ('tsne' or 'pca').
        perplexity (int): Perplexity parameter for t-SNE (ignored if using PCA).
        n_components (int): Number of dimensions to reduce to (default: 2).
        random_state (int): Random seed for reproducibility.
    """
    print(f"Reducing dimensionality using {method.upper()}...")
    
    if method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    elif method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("Invalid method! Choose 'tsne' or 'pca'.")
    
    X_reduced = reducer.fit_transform(X_features)

    # Convert to DataFrame for visualization
    df_plot = pd.DataFrame(X_reduced, columns=['x', 'y'])
    df_plot['emotion'] = y_labels

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="x", y="y", hue="emotion", data=df_plot, palette="tab10", alpha=0.7, s=50, edgecolor="k"
    )
    plt.savefig(EMO_FREQUENCY)
    
    # Add emotion cluster labels if possible
    most_common_emotions = [e[0] for e in Counter(y_labels).most_common()]
    for emotion in most_common_emotions:
        subset = df_plot[df_plot['emotion'] == emotion]
        centroid_x = subset["x"].mean()
        centroid_y = subset["y"].mean()
        plt.text(centroid_x, centroid_y, emotion, fontsize=12, fontweight='bold', ha='center', va='center')

    plt.title(f"Sentence Embeddings Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Emotions", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(GRAPH_CLUSTER_OF_EMOTIONS)
    plt.show()

# Example usage
plot_embeddings(X_features, y_labels, method="tsne")  # Use "pca" for PCA visualization