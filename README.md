# 🎥 Movie Recommendation System (MRS)

A web-based movie recommendation system that suggests similar movies based on genre similarity using machine learning. Built with Streamlit and scikit-learn.
## 🔧 Features

### 🎓 For Users:
- Browse and search through thousands of movies
- Select favorite movies from an intuitive dropdown interface
- Get personalized movie recommendations based on genre similarity
- Choose how many recommendations to receive (5-20)
- View all movies in an expandable browser with genres
- Clean, responsive web interface

### ⚙️ Technical Features:
- Automatic dataset download and setup
- Content-based filtering using TF-IDF and Cosine Similarity
- Error handling for robust user experience
- Professional sidebar layout with controls

## 🏗️ Tech Stack
- **Web Framework:** Streamlit (Python)
- **Machine Learning:** scikit-learn (TF-IDF, Cosine Similarity)
- **Data Processing:** pandas , numpy
- **File Handling:** requests, Zipfile , os

## 📂 Project Structure
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── movies.csv              # movie dataset
└── (Optional deployment files)
