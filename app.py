from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.recommender import RecommendationEngine, canonicalize_title

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("Smart Movie Recommender")
st.caption("Content-based filtering using NLP + TF-IDF + Cosine Similarity")

cleaned_path = Path("data/bollywood_movies_cleaned.csv")

engine = None

if not cleaned_path.exists():
    st.error(
        "Cleaned dataset not found at data/bollywood_movies_cleaned.csv. "
        "Run: python src/preprocess.py"
    )
    st.stop()

try:
    engine = RecommendationEngine.from_cleaned_csv(cleaned_path)
except Exception as exc:
    st.error(f"Failed to load recommendation engine: {exc}")
    st.stop()

if engine is None:
    st.stop()

title_map: dict[str, str] = {}
for raw_title in engine.movies_df["title"].dropna().astype(str):
    clean_title = canonicalize_title(raw_title)
    if clean_title:
        title_map.setdefault(clean_title, raw_title)

movie_titles = sorted(title_map.keys())

selected_title = st.selectbox(
    "Pick a Bollywood movie you like",
    options=movie_titles,
    index=0,
)

top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if st.button("Recommend"):
    try:
        source_title = title_map[selected_title]
        recommendations = engine.get_recommendations(source_title, top_n=top_n)
        if not recommendations:
            st.warning("No recommendations found for this title.")
        else:
            st.subheader("Recommended Movies")
            for rank, title in enumerate(recommendations, start=1):
                st.write(f"{rank}. {title}")
    except KeyError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

st.divider()
st.write(f"Movies loaded: {len(engine.movies_df)}")
st.write(f"Unique movie names: {len(movie_titles)}")
