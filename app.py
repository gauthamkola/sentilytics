import pandas as pd
import streamlit as st
import plotly.express as px
from transformers import pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Title
st.title("Sentilytics: AI-Powered Competitor Insights")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("amazon.csv")
        st.write("Loaded amazon.csv")
    except FileNotFoundError:
        st.error("amazon.csv not found. Using sample data.")
        return pd.DataFrame([
            {"product_name": "Sony Speaker", "review_content": "Great sound!", "rating": 5},
            {"product_name": "Bose Speaker", "review_content": "Battery died fast.", "rating": 2}
        ])

    reviews = []
    for _, row in df.iterrows():
        review_texts = row["review_content"].split(",") if pd.notna(row["review_content"]) else [""]
        rating_str = str(row["rating"]).replace("|", "").strip()
        try:
            rating = float(rating_str) if rating_str else 0.0
        except ValueError:
            rating = 0.0
        for review in review_texts:
            reviews.append({
                "review_text": review.strip(),
                "product": row["product_name"],
                "rating": rating
            })
    df_reviews = pd.DataFrame(reviews)
    return df_reviews[["review_text", "product", "rating"]].dropna(subset=["review_text"])

# Load data first
df_full = load_data()
if df_full.empty:
    st.stop()

# Sidebar for interactivity
st.sidebar.header("Filters")
product_filter = st.sidebar.selectbox("Select Product", ["All"] + sorted(df_full["product"].unique().tolist()), index=0)

# Apply filter
df = df_full if product_filter == "All" else df_full[df_full["product"] == product_filter]
df = df.head(1000)  # Limit for demo

# Display data
st.subheader("Review Data")
st.write(f"Loaded {len(df)} reviews")
st.dataframe(df.head())

# Sentiment Analysis
st.subheader("Sentiment Distribution")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
df["sentiment"] = df["review_text"].apply(lambda x: sentiment_pipeline(x)[0]["label"])
fig_sentiment = px.pie(df, names="sentiment", title="Sentiment Breakdown", hole=0.3, color_discrete_map={"POSITIVE": "#00CC96", "NEGATIVE": "#EF553B"})
st.plotly_chart(fig_sentiment)

# Sentiment by Product
st.subheader("Sentiment by Product")
sentiment_by_product = df.groupby("product")["sentiment"].value_counts(normalize=True).unstack().fillna(0) * 100
fig_sentiment_product = px.bar(sentiment_by_product, barmode="stack", title="Sentiment by Product", color_discrete_map={"POSITIVE": "#00CC96", "NEGATIVE": "#EF553B"})
st.plotly_chart(fig_sentiment_product)

# Topic Clustering
st.subheader("Topic Clusters")
if len(df) > 5:  # Minimum threshold for clustering
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Dynamically adjust UMAP n_components and min_topic_size
        n_components = min(2, max(1, len(df) - 1))  # Ensure k < N
        min_topic_size = max(2, min(10, len(df) // 5))
        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=min_topic_size,
            n_gram_range=(1, 3),
            nr_topics=10,
            top_n_words=5,
            umap_model=UMAP(n_components=n_components)  # Explicitly set UMAP
        )
        topics, _ = topic_model.fit_transform(df["review_text"].tolist())
        df["topic"] = topics
        topic_info = topic_model.get_topic_info()
        topic_names = topic_info.set_index("Topic")["Name"].to_dict()
        df["topic_name"] = df["topic"].map(lambda x: topic_names.get(x, "Miscellaneous"))
        fig_topics = px.histogram(df, x="topic_name", color="product", title="Topics by Product", barmode="group")
        st.plotly_chart(fig_topics)
    except Exception as e:
        st.write(f"Topic clustering failed: {e}. Dataset may be too small.")
        df["topic_name"] = "N/A"
else:
    st.write("Not enough reviews for topic clustering (<5).")
    df["topic_name"] = "N/A"

# Business Insights
st.subheader("Business Insights")
st.write("### Key Findings")
pos_pct = df["sentiment"].value_counts(normalize=True).get("POSITIVE", 0) * 100
st.write(f"- **Overall Sentiment**: {pos_pct:.1f}% positive ({'Strong' if pos_pct > 70 else 'Moderate' if pos_pct > 50 else 'Weak'} satisfaction).")

top_products = df.groupby("product").agg({"sentiment": lambda x: (x == "POSITIVE").mean() * 100, "rating": "mean"}).sort_values("sentiment", ascending=False).head(3)
st.write("- **Top Products**:")
for product, row in top_products.iterrows():
    st.write(f"  - **{product}**: {row['sentiment']:.1f}% positive, Avg. Rating: {row['rating']:.1f}")

if "topic_name" in df.columns and df["topic_name"].nunique() > 1:
    topic_sentiment = df.groupby("topic_name")["sentiment"].value_counts(normalize=True).unstack().fillna(0) * 100
    st.write("- **Key Topics**:")
    for topic in topic_sentiment.index:
        if topic != "-1_miscellaneous" and topic != "N/A":
            pos = topic_sentiment.loc[topic, "POSITIVE"]
            neg = topic_sentiment.loc[topic, "NEGATIVE"]
            st.write(f"  - **{topic}**: {pos:.1f}% positive, {neg:.1f}% negative")
else:
    st.write("- **Key Topics**: Not enough data for meaningful topics.")

st.write("### Recommendations")
if pos_pct < 60:
    st.write("- **Boost Satisfaction**: Focus on addressing negative feedback in key areas.")
if "topic_name" in df.columns and df["topic_name"].nunique() > 1:
    for topic in topic_sentiment.index:
        if topic_sentiment.loc[topic, "NEGATIVE"] > 50 and topic != "-1_miscellaneous" and topic != "N/A":
            st.write(f"- **Improve '{topic}'**: High negative sentiment ({topic_sentiment.loc[topic, 'NEGATIVE']:.1f}%) suggests a pain point.")
for product, row in top_products.iterrows():
    if row["sentiment"] > 80:
        st.write(f"- **Promote '{product}'**: Leverage its strong {row['sentiment']:.1f}% positive sentiment in marketing.")

# Footer
st.write("Powered by DistilBERT and BERTopic | Demo by xAI")