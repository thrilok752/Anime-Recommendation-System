# Anime Recommendation System

A machine learning project that recommends anime using content-based filtering and hybrid recommendation techniques.

The system uses anime metadata, TF-IDF feature engineering, cosine similarity, collaborative filtering (SVD), and the official MyAnimeList API.

---

## Features

- Content-based anime recommendation using TF-IDF
- Hybrid recommendation using collaborative filtering (SVD)
- MyAnimeList API integration
- Interactive Streamlit web application
- Genre, type, and rating filters
- Poster and synopsis display

---

## Project Architecture

User selects anime  
↓  
TF-IDF feature extraction  
↓  
Cosine similarity search  
↓  
Candidate anime selection  
↓  
Hybrid ranking (similarity + rating + popularity)  
↓  
MyAnimeList API  
↓  
Poster + synopsis displayed in Streamlit  

---

## Application Versions

The repository contains multiple application versions demonstrating the evolution of the recommender system.

app_userid.py — Hybrid recommender using TF-IDF + collaborative filtering  
app_hybrid_api.py — Hybrid recommender with MyAnimeList API integration  
app_content_api.py — Content-based recommender using TF-IDF  
app_content_api_filters.py — Content recommender with genre/type/rating filters  

Recommended version to run:

streamlit run app_content_api_filters.py

---

## Dataset

Multiple anime datasets from Kaggle were merged and cleaned to create a unified dataset containing approximately **25,000 anime records**.

Key fields used in the final dataset:

- title
- anime_id
- genres
- synopsis
- score
- members
- favorites
- type
- tags

These fields are combined into a **content feature** used for TF-IDF vectorization.

---

## Models Used

### Content-Based Recommendation

Uses:

- TF-IDF Vectorizer
- Cosine Similarity

Text features include:

- genres
- tags
- synopsis

Similarity is computed dynamically instead of storing a full similarity matrix to reduce memory usage.

---

### Hybrid Recommendation

Combines:

- similarity score
- anime rating
- popularity metrics
- collaborative filtering (SVD)

Ranking formula:

Final Score =  
0.6 × collaborative score  
+ 0.3 × rating  
+ 0.1 × popularity  

---
## Model Files

Due to GitHub file size limits, trained models are hosted externally.

Download models from:
https://drive.google.com/drive/folders/1duP2j3rVbuFTxKOD_gW9fVW2knJKt0At?usp=sharing

Place them inside the `models/` folder before running the application.
## Installation

Clone the repository:

git clone https://github.com/thrilok752/Anime-Recommendation-System.git  
cd Anime-Recommendation-System  

Install dependencies:

pip install -r requirements.txt  

Create a `.env` file:

MAL_CLIENT_ID=your_myanimelist_client_id  

---

## Run the Application

Run the content-based version:

streamlit run app_content_api_filters.py  

Run the hybrid recommender:

streamlit run app_hybrid_api.py  

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Scikit-surprise
- Streamlit
- MyAnimeList API

