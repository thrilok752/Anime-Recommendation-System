import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/anime_master_ready.csv")

    # clean numeric columns
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["members"] = pd.to_numeric(df["members"], errors="coerce")
    df["favorites"] = pd.to_numeric(df["favorites"], errors="coerce")

    df["title"] = df["title"].str.lower()

    return df


df = load_data()

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():

    with open("models/svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)

    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return svd_model, vectorizer


model, vectorizer = load_models()

# -----------------------------
# BUILD TF-IDF MATRIX
# -----------------------------
@st.cache_resource
def build_tfidf():

    tfidf_matrix = vectorizer.transform(df["content"])

    return tfidf_matrix


tfidf_matrix = build_tfidf()

# -----------------------------
# INDEX MAPPING
# -----------------------------
indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

# -----------------------------
# CONTENT CANDIDATE GENERATION
# -----------------------------
def get_content_candidates(title, top_n=50):

    idx = indices.get(title)

    if idx is None:
        return None

    query_vector = tfidf_matrix[idx]

    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top = scores.argsort()[::-1][1:top_n+1]

    return df.iloc[top], scores


# -----------------------------
# HYBRID RECOMMENDER
# -----------------------------
def hybrid_recommend(title, user_id, top_n=10):

    result = get_content_candidates(title)

    if result is None:
        return None

    candidates, scores = result

    results = []

    for i, row in candidates.iterrows():

        anime_id = row["anime_id"]

        # collaborative prediction
        collab_score = model.predict(user_id, anime_id).est

        # popularity signal
        popularity = ((row["members"] or 0) + (row["favorites"] or 0)) / 1e6

        # hybrid score
        final_score = (
            0.6 * collab_score +
            0.3 * (row["score"] if pd.notna(row["score"]) else 0) +
            0.1 * popularity * 10
        )

        results.append((row["title"], anime_id, round(final_score, 2)))

    # remove input anime
    results = [r for r in results if r[0] != title]

    results.sort(key=lambda x: x[2], reverse=True)

    return results[:top_n]


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎌 Anime Recommendation System")

st.write(
    "Hybrid recommender using TF-IDF content similarity and collaborative filtering (SVD)."
)

# dropdown avoids spelling errors
anime_input = st.selectbox(
    "Choose an Anime",
    sorted(df["title"].unique())
)

user_id = st.number_input(
    "Enter User ID",
    min_value=0,
    step=1
)

if st.button("Recommend"):

    results = hybrid_recommend(anime_input.lower(), user_id)

    if results is None:
        st.error("Anime not found in database.")

    else:

        st.subheader("Recommended Anime")

        for i, (title, anime_id, score) in enumerate(results, 1):

            st.write(f"{i}. **{title.title()}**  —  Score: {score}")