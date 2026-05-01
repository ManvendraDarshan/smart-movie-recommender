from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from nltk.stem import PorterStemmer

REQUIRED_COLUMNS = ["title", "genres", "keywords", "plot"]


def _normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _tokenize_and_stem(value: str, stemmer: PorterStemmer) -> str:
    tokens = value.split()
    return " ".join(stemmer.stem(token) for token in tokens)


def load_and_clean_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    rename_map = {column: column.strip().lower() for column in df.columns}
    df = df.rename(columns=rename_map)

    missing_cols = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    for column in ["genres", "keywords", "plot"]:
        df[column] = df[column].fillna("")

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df = df[df["title"] != ""].reset_index(drop=True)

    if len(df) < 500:
        raise ValueError(
            f"Dataset contains {len(df)} rows after cleaning. "
            "At least 500 Bollywood movies are required."
        )

    stemmer = PorterStemmer()

    combined = (
        df["genres"].astype(str)
        + " "
        + df["keywords"].astype(str)
        + " "
        + df["plot"].astype(str)
    )

    df["content"] = combined.apply(_normalize_text).apply(
        lambda text: _tokenize_and_stem(text, stemmer)
    )

    return df


def save_clean_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def summarize_dataset(df: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": len(df),
        "missing_genres": int((df["genres"] == "").sum()),
        "missing_keywords": int((df["keywords"] == "").sum()),
        "missing_plot": int((df["plot"] == "").sum()),
    }


if __name__ == "__main__":
    source = Path("data/bollywood_movies_raw.csv")
    target = Path("data/bollywood_movies_cleaned.csv")

    cleaned_df = load_and_clean_dataset(source)
    save_clean_dataset(cleaned_df, target)

    stats = summarize_dataset(cleaned_df)
    print("Dataset cleaned successfully")
    print(stats)
