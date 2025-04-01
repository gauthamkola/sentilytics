import pandas as pd
import streamlit as st
import plotly.express as px
from transformers import pipeline
from bertopic import BERTopic

# Title
st.title("Sentilytics: AI-Powered Competitor Insights")

# Sample data (increased to 10 reviews for better clustering)
st.subheader("Sample Reviews")
sample_reviews = [
    {"review_text": "The battery life is amazing, but the design is bulky.", "product": "Product A", "rating": 4},
    {"review_text": "Terrible UI, crashes all the time.", "product": "Product B", "rating": 2},
    {"review_text": "Great product, super fast delivery!", "product": "Product A", "rating": 5},
    {"review_text": "Poor quality, broke after a week.", "product": "Product B", "rating": 1},
    {"review_text": "Love the sleek design, but the price is high.", "product": "Product A", "rating": 3},
    {"review_text": "Fast processor, but overheats quickly.", "product": "Product B", "rating": 3},
    {"review_text": "Amazing camera quality!", "product": "Product A", "rating": 5},
    {"review_text": "Cheap materials, feels flimsy.", "product": "Product B", "rating": 2},
    {"review_text": "Battery lasts forever, highly recommend.", "product": "Product A", "rating": 4},
    {"review_text": "Software is buggy, needs updates.", "product": "Product B", "rating": 3},
]
df = pd.DataFrame(sample_reviews)
st.write(df)  # Display sample data

# Sentiment Analysis
st.subheader("Sentiment Distribution")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
df["sentiment"] = df["review_text"].apply(lambda x: sentiment_pipeline(x)[0]["label"])
fig_sentiment = px.pie(df, names="sentiment", title="Sentiment Breakdown", hole=0.3)
st.plotly_chart(fig_sentiment)

# Topic Clustering
st.subheader("Topic Clusters")
try:
    topic_model = BERTopic(
        min_topic_size=2,  # Minimum documents per topic
        n_gram_range=(1, 2),  # Allow unigrams and bigrams
        nr_topics="auto",  # Automatically reduce topics
        umap_model=None  # Use default UMAP with adjusted settings internally
    )
    topics, _ = topic_model.fit_transform(df["review_text"].tolist())
    df["topic"] = topics
    fig_topics = px.histogram(df, x="topic", color="product", title="Review Topics by Product", barmode="group")
    st.plotly_chart(fig_topics)
except ValueError as e:
    st.write("Topic clustering failed due to small dataset. Please add more reviews for better results.")
    st.write(f"Error: {e}")

# Footer
st.write("Powered by DistilBERT and BERTopic | Demo by xAI")