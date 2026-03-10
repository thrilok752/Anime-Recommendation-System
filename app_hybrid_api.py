import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# MAL API CONFIG
# -----------------------------
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("MAL_CLIENT_ID")


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("dataset/anime_master_ready.csv")

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
# BUILD TFIDF MATRIX
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
# MAL API FUNCTION
# -----------------------------
@st.cache_data
def fetch_anime_info(anime_id):

    url = f"https://api.myanimelist.net/v2/anime/{anime_id}"

    headers = {
        "X-MAL-CLIENT-ID": CLIENT_ID
    }

    params = {
        "fields": "title,main_picture,synopsis,mean"
    }

    try:

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:

            data = response.json()

            return {
                "title": data.get("title"),
                "image": data.get("main_picture", {}).get("large"),
                "synopsis": data.get("synopsis"),
                "mal_score": data.get("mean")
            }

    except:
        return None

    return None


# -----------------------------
# CONTENT CANDIDATES
# -----------------------------
def get_content_candidates(title, top_n=50):

    idx = indices.get(title)

    if idx is None:
        return None

    query_vector = tfidf_matrix[idx]

    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top = scores.argsort()[::-1][1:top_n+1]

    return df.iloc[top]


# -----------------------------
# HYBRID RECOMMENDER
# -----------------------------
def hybrid_recommend(title, user_id, top_n=10):

    candidates = get_content_candidates(title)

    if candidates is None:
        return None

    results = []

    for _, row in candidates.iterrows():

        anime_id = row["anime_id"]

        collab_score = model.predict(user_id, anime_id).est

        popularity = ((row["members"] or 0) + (row["favorites"] or 0)) / 1e6

        final_score = (
            0.6 * collab_score +
            0.3 * (row["score"] if pd.notna(row["score"]) else 0) +
            0.1 * popularity * 10
        )

        results.append((row["title"], anime_id, round(final_score, 2)))

    results = [r for r in results if r[0] != title]

    results.sort(key=lambda x: x[2], reverse=True)

    return results[:top_n]


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎌 Anime Recommendation System")

st.write(
    "Hybrid recommender using TF-IDF similarity + collaborative filtering (SVD)."
)


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

        st.error("Anime not found.")

    else:

        st.subheader("Recommended Anime")

        for i, (title, anime_id, score) in enumerate(results, 1):

            info = fetch_anime_info(anime_id)

            if info:

                col1, col2 = st.columns([1, 3])

                with col1:
                    if info["image"]:
                        st.image(info["image"], width=120)

                with col2:
                    st.subheader(info["title"])
                    st.write(f"⭐ Hybrid Score: {score}")

                    if info["mal_score"]:
                        st.write(f"MAL Rating: {info['mal_score']}")

                    if info["synopsis"]:
                        st.write(info["synopsis"][:250] + "...")