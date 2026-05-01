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

selected_raw_title = title_map[selected_title]
selected_row = engine.movies_df[engine.movies_df["title"].astype(str) == selected_raw_title].head(1)
if not selected_row.empty:
    selected_row = selected_row.iloc[0]
    st.subheader("Selected Movie Details")
    st.write(f"**Title:** {canonicalize_title(selected_row['title'])}")
    st.write(f"**Genres:** {selected_row.get('genres', '')}")
    st.write(f"**Keywords:** {selected_row.get('keywords', '')}")
    st.write(f"**Plot:** {selected_row.get('plot', '')}")

top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if st.button("Recommend"):
    try:
        source_title = title_map[selected_title]
        recommendations = engine.get_recommendations_with_scores(source_title, top_n=top_n)
        if not recommendations:
            st.warning("No recommendations found for this title.")
        else:
            st.subheader("Recommended Movies")
            for rank, (title, score) in enumerate(recommendations, start=1):
                st.markdown(f"**{rank}. {title}**  (match: {score * 100:.1f}%)")
                rec_row = engine.movies_df[
                    engine.movies_df["title"].astype(str).apply(canonicalize_title).str.lower()
                    == title.lower()
                ].head(1)
                if not rec_row.empty:
                    rec_row = rec_row.iloc[0]
                    st.write(f"Genres: {rec_row.get('genres', '')}")
                    st.write(f"Keywords: {rec_row.get('keywords', '')}")
                    st.write(f"Plot: {rec_row.get('plot', '')}")
                st.write("---")
    except KeyError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

st.divider()
st.write(f"Movies loaded: {len(engine.movies_df)}")
st.write(f"Unique movie names: {len(movie_titles)}")
