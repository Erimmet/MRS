import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
import zipfile
import os

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# App title and description
st.title("🎬 Movie Recommendation System")
st.markdown("Discover movies you'll love based on their storylines and themes!")

@st.cache_data
def load_data():
    """Load local movie dataset."""
    try:
        movies = pd.read_csv("movies.csv")   # 👈 LOAD FROM LOCAL FILE
    except FileNotFoundError:
        st.error("❌ Local dataset 'movies.csv' not found in project folder.")
        return None
    
    # Fix missing fields
    if "genres" not in movies.columns:
        movies["genres"] = ""
    if "overview" not in movies.columns:
        movies["overview"] = ""

    movies["genres"] = movies["genres"].fillna("")
    movies["overview"] = movies["overview"].fillna("")

    return movies


@st.cache_resource
def build_recommendation_engine(movies):
    """Build the recommendation system using TF-IDF on movie overviews"""
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Combine genres + overview for richer representation
    movies['metadata'] = movies['genres'] + " " + movies['overview']
    tfidf_matrix = tfidf.fit_transform(movies['metadata'])
    
    # Compute cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    return cosine_sim, indices

def get_recommendations(title, movies, cosine_sim, indices, num_recommendations=10):
    """Get movie recommendations"""
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        return movies.iloc[movie_indices][['title', 'genres', 'overview']]
    except:
        return None

def main():
    # Load data
    with st.spinner('Loading movie data...'):
        movies = load_data()
        if movies is None:
            st.stop()
        cosine_sim, indices = build_recommendation_engine(movies)
    
    st.success(f'✅ Loaded {len(movies)} movies!')

    # Sidebar
    st.sidebar.header("Controls")
    selected_movie = st.sidebar.selectbox("🎥 Select a movie you like:", movies['title'].values)
    num_recommendations = st.sidebar.slider("Number of recommendations:", 5, 20, 10)
    
    if st.sidebar.button("Get Recommendations 🍿", type="primary"):
        with st.spinner('Finding similar movies...'):
            recommendations = get_recommendations(selected_movie, movies, cosine_sim, indices, num_recommendations)
        
        if recommendations is not None and not recommendations.empty:
            st.header(f"Because you watched: **{selected_movie}**")
            st.subheader("You might also like:")
            
            for i, row in enumerate(recommendations.itertuples(), 1):
                st.markdown(f"**{i}. {row.title}**  \n*Genres:* {row.genres}  \n*Overview:* {row.overview[:300]}...")
            st.info(f"🎉 Found {len(recommendations)} recommendations!")
        else:
            st.error("Sorry, couldn't find recommendations for that movie. Try another one!")

    with st.expander("📋 Browse All Movies"):
        st.dataframe(movies[['title', 'genres', 'overview']].head(100))

if __name__ == "__main__":
    main()
