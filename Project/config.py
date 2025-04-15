from pathlib import Path

# Root directory (the directory containing this config file)
ROOT_DIR = Path(__file__).resolve().parent

# Data directory paths
DATA_DIR = ROOT_DIR / "data"
TEXT_DIR = DATA_DIR / "text"
VIDEO_DIR = DATA_DIR / "video"
PLOT_DIR = DATA_DIR / "plots"

# Files to be used or created
# JSON_FILE_PATH = TEXT_DIR / "your_data.json"  # change the filename accordingly
JSON_FILE_PATH = "/Users/anjelica/Downloads/labeled_train_data.json"
HINDI_STOPWORDS_FILE = TEXT_DIR / "hindi_stopwords.txt"
FASTTEXT_INPUT_FILE = TEXT_DIR / "fasttext_input.txt"
FASTTEXT_MODEL_FILE = TEXT_DIR / "emotion_model.bin"
EMOTION_DIST_PLOT = PLOT_DIR / "emotion_distribution.png"
GRAPH_CLUSTER_OF_EMOTIONS = PLOT_DIR / "graph_cluster_of_emotions.png"
EMO_FREQUENCY = PLOT_DIR / "emotions_frequency.png"
Y_LABELS_FILE = TEXT_DIR / "y_labels.csv"
PROCESSED_DATA_FILE = TEXT_DIR / "processed_data.csv"
ADV_FEATURES_FILE = TEXT_DIR / "features_advanced.csv"

# Ensure directories exist
for folder in [TEXT_DIR, VIDEO_DIR, PLOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
