from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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

        tfidf = TfidfVectorizer(stop_words="english", max_features=15000)
        matrix = tfidf.fit_transform(df["content"].fillna(""))
        cosine_sim = cosine_similarity(matrix, matrix)
        return cls(movies_df=df, cosine_sim_matrix=cosine_sim)

    def get_recommendations(self, movie_title: str, top_n: int = 10) -> list[str]:
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

        recommendations: list[str] = []
        seen_titles: set[str] = {canonicalize_title(self.movies_df.iloc[movie_idx]["title"]).lower()}

        for idx, _ in scores_sorted:
            if idx == movie_idx:
                continue

            title = canonicalize_title(self.movies_df.iloc[idx]["title"])
            title_key = title.lower()
            if title_key in seen_titles:
                continue

            seen_titles.add(title_key)
            recommendations.append(title)
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
