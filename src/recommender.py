from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_TITLE_SUFFIX_RE = re.compile(r"\s*\[\d+\]\s*$")


def canonicalize_title(title: str) -> str:
    """Normalize title text so duplicate variants collapse to one movie name."""
    text = str(title).strip()
    return _TITLE_SUFFIX_RE.sub("", text)


@dataclass
class RecommendationEngine:
    movies_df: pd.DataFrame
    cosine_sim_matrix: np.ndarray

    @classmethod
    def from_cleaned_csv(cls, csv_path: str | Path) -> "RecommendationEngine":
        df = pd.read_csv(csv_path)
        if "content" not in df.columns:
            raise ValueError(
                "Cleaned dataset must include a 'content' column. "
                "Run preprocessing first."
            )

        # Main semantic signal from combined content.
        content_tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        content_matrix = content_tfidf.fit_transform(df["content"].fillna(""))
        content_sim = cosine_similarity(content_matrix, content_matrix)

        # Explicit genre overlap signal.
        genres_text = (
            df["genres"]
            .fillna("")
            .astype(str)
            .str.replace("|", " ", regex=False)
        )
        genre_vectorizer = CountVectorizer(token_pattern=r"(?u)\b[\w-]+\b", lowercase=True)
        genre_matrix = genre_vectorizer.fit_transform(genres_text)
        genre_sim = cosine_similarity(genre_matrix, genre_matrix)

        # Title character n-grams help same-franchise/sequel matching.
        title_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        title_matrix = title_tfidf.fit_transform(df["title"].fillna("").astype(str))
        title_sim = cosine_similarity(title_matrix, title_matrix)

        cosine_sim = (0.55 * content_sim) + (0.30 * genre_sim) + (0.15 * title_sim)
        return cls(movies_df=df, cosine_sim_matrix=cosine_sim)

    def get_recommendations(self, movie_title: str, top_n: int = 10) -> list[str]:
        ranked = self.get_recommendations_with_scores(movie_title, top_n=top_n)
        return [title for title, _ in ranked]

    def get_recommendations_with_scores(self, movie_title: str, top_n: int = 10) -> list[tuple[str, float]]:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")

        title_lookup: dict[str, int] = {}
        for idx, raw_title in enumerate(self.movies_df["title"].astype(str)):
            full_key = raw_title.strip().lower()
            title_lookup.setdefault(full_key, idx)

            canonical_key = canonicalize_title(raw_title).lower()
            title_lookup.setdefault(canonical_key, idx)

        key = movie_title.strip().lower()
        if key not in title_lookup:
            raise KeyError(
                f"Movie '{movie_title}' not found in dataset. "
                "Check spelling or try another title."
            )

        movie_idx = title_lookup[key]
        scores = list(enumerate(self.cosine_sim_matrix[movie_idx]))
        scores_sorted = sorted(scores, key=lambda item: item[1], reverse=True)

        recommendations: list[tuple[str, float]] = []
        seen_titles: set[str] = {canonicalize_title(self.movies_df.iloc[movie_idx]["title"]).lower()}

        for idx, score in scores_sorted:
            if idx == movie_idx:
                continue

            title = canonicalize_title(self.movies_df.iloc[idx]["title"])
            title_key = title.lower()
            if title_key in seen_titles:
                continue

            seen_titles.add(title_key)
            recommendations.append((title, float(score)))
            if len(recommendations) >= top_n:
                break

        return recommendations


def quick_test() -> None:
    engine = RecommendationEngine.from_cleaned_csv("data/bollywood_movies_cleaned.csv")
    sample_title = engine.movies_df.iloc[0]["title"]
    recs = engine.get_recommendations(sample_title, top_n=5)
    print(f"Input: {canonicalize_title(sample_title)}")
    print("Recommendations:")
    for title in recs:
        print(f"- {title}")


if __name__ == "__main__":
    quick_test()
