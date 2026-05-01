# Bollywood Movie Recommendation System

A simple content-based recommender for Bollywood movies using NLP, TF-IDF, and cosine similarity.

## Project Structure

- `data/`
  - `dataset_template.csv` (required schema example)
  - `bollywood_movies_raw.csv` (you place your raw 500+ movies dataset here)
  - `bollywood_movies_cleaned.csv` (generated)
- `src/preprocess.py` (cleaning + feature engineering)
- `src/recommender.py` (TF-IDF + cosine similarity + recommendation function)
- `app.py` (Streamlit UI)
- `requirements.txt`

## Dataset Requirements

Your raw dataset CSV must include these columns:

- `title`
- `genres`
- `keywords`
- `plot`

Minimum requirement: **500+ Bollywood movies after cleaning**.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1: Add Dataset

Place your dataset at:

`data/bollywood_movies_raw.csv`

## Step 2: Preprocess and Feature Engineering

```bash
python src/preprocess.py
```

What it does:

- normalizes columns
- handles missing values
- combines `genres + keywords + plot` into `content`
- lowercases, removes special characters
- applies stemming
- validates 500+ movies
- saves `data/bollywood_movies_cleaned.csv`

## Step 3: Recommendation Logic Test

```bash
python src/recommender.py
```

This performs a quick test by selecting one title and printing top recommendations.

## Step 4: Run Web App

```bash
streamlit run app.py
```

Features:

- select a Bollywood movie from dropdown
- choose number of recommendations
- get top-N similar movies
- graceful error handling when data/model is missing or invalid

## Recommendation Function

Implemented in `src/recommender.py`:

`get_recommendations(movie_title, top_n=10)`

Behavior:

- finds movie index
- computes similarity ranking
- excludes the selected movie
- returns top N movie titles

## Suggested Next Enhancements

- add fuzzy title search for typo tolerance
- include poster URLs and release year in UI
- persist TF-IDF model using `joblib`
- evaluate recommendations with user feedback
