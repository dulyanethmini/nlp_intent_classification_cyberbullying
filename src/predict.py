import joblib

# Load saved model & vectorizer using joblib
model = joblib.load("models/logistic_regression_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_intent(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return prediction

# --- Quick Test ---
if __name__ == "__main__":
    sample_text = "You are so stupid and annoying!, your age is too much to be acting like this"
    result = predict_intent(sample_text)
    print(f"Text: {sample_text}")
    print(f"Predicted intent: {result}")
