import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from src.data_prep import preprocess

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Cyberbullying Detection", layout="wide")
st.title("📊 Cyberbullying Detection App")

# -----------------------------
# Load trained pipeline
# -----------------------------
vectorizer, model = joblib.load("models/cyberbullying_pipeline.pkl")

# -----------------------------
# Single Tweet Prediction
# -----------------------------
st.header("🔮 Predict a Tweet")
tweet = st.text_area("Enter a tweet here:")

if st.button("Check Prediction"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        try:
            tweet_vector = vectorizer.transform([tweet])
            prediction = model.predict(tweet_vector)
            st.success(f"Prediction: **{prediction[0]}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(tweet_vector)[0]
                prob_df = pd.DataFrame({"Class": model.classes_, "Probability": proba})
                st.bar_chart(prob_df.set_index("Class"))
        except Exception as e:
            st.error(f"Error predicting tweet: {e}")

# -----------------------------
# Model Performance
# -----------------------------
st.header("📈 Model Performance")
try:
    df = preprocess()
    X = vectorizer.transform(df["clean_text"])
    y_true = df["cyberbullying_type"]
    y_pred = model.predict(X)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)

    # Classification Report
    st.subheader("📋 Classification Report")
    report = classification_report(y_true, y_pred, target_names=model.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="Blues"))

except Exception as e:
    st.error(f"Error generating model performance: {e}")

# -----------------------------
# t-SNE Embedding Visualization
# -----------------------------
st.header("🌀 t-SNE Visualization of TF-IDF Features")
try:
    sample_df = df.sample(min(500, len(df)), random_state=42)
    X_sample = vectorizer.transform(sample_df["clean_text"])
    y_sample = sample_df["cyberbullying_type"]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    X_embedded = tsne.fit_transform(X_sample.toarray())

    tsne_df = pd.DataFrame(X_embedded, columns=["Dim1", "Dim2"])
    tsne_df["label"] = y_sample.values

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=tsne_df,
        x="Dim1",
        y="Dim2",
        hue="label",
        palette="tab10",
        alpha=0.7,
        s=50
    )
    plt.title("t-SNE Visualization of TF-IDF Features")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc="best", fontsize="small")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error generating t-SNE visualization: {e}")



