import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources (only run first time)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("/Users/dulyanethmini/Desktop/Y 4 S 2/NLP/final project/cyber_bullying/data/cyberbullying_tweets.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns)


# Rename them for consistency if needed
df = df.rename(columns={"tweet_text": "text", "cyberbullying_type": "label"})

# Drop missing rows
df = df.dropna(subset=["text", "label"])

# Initialize tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove links
    text = re.sub(r"[^a-z\s]", "", text)         # keep only letters
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)

print("\nSample cleaned tweets:")
print(df[["text", "clean_text", "label"]].head())


# Save cleaned dataset
df.to_csv("data/cleaned_cyberbullying_tweets.csv", index=False)
print("✅ Cleaned dataset saved to data/cleaned_cyberbullying_tweets.csv")
