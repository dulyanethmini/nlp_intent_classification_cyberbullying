
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib 

# 1. Load cleaned dataset
df = pd.read_csv("data/cleaned_cyberbullying_tweets.csv")
print("✅ Cleaned dataset loaded successfully!")

# 2. Drop rows with NaN in important columns
df = df.dropna(subset=["clean_text", "label"])

# 3. Features (X) and Labels (y)
X = df["clean_text"].astype(str)   # convert to string to avoid np.nan issues
y = df["label"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)   # <- fit here
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

# 7. Predictions
y_pred = model.predict(X_test_tfidf)

# 8. Evaluation
print("\n✅ Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Save model and vectorizer
joblib.dump(model, "models/logistic_regression_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("💾 Model and vectorizer saved to 'models/' folder")