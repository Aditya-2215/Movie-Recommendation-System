from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.tfidf_matrix = None
        self.indices = None
        self.df = None
        self.tfidf = None
        self.is_loaded = False

    def load_models(self) -> None:
        required = {
            "tfidf_matrix": self.models_dir / "tfidf_matrix.pkl",
            "indices": self.models_dir / "indices.pkl",
            "df": self.models_dir / "df.pkl",
            "tfidf": self.models_dir / "tfidf.pkl",
        }

        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing model files: " + ", ".join(missing) +
                ". Put the four .pkl files inside backend/models/."
            )

        with open(required["tfidf_matrix"], "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(required["indices"], "rb") as f:
            self.indices = pickle.load(f)
        with open(required["df"], "rb") as f:
            self.df = pickle.load(f)
        with open(required["tfidf"], "rb") as f:
            self.tfidf = pickle.load(f)

        if "title" not in self.df.columns:
            raise ValueError("df.pkl must contain a 'title' column.")

        self.df = self.df.reset_index(drop=True)
        self.df["title"] = self.df["title"].fillna("").astype(str).str.strip()

        # Rebuild the title index from the dataframe so it always matches
        # the row positions of tfidf_matrix and df.
        self.indices = pd.Series(self.df.index, index=self.df["title"])
        self.is_loaded = True

    @property
    def movie_count(self) -> int:
        return 0 if self.df is None else len(self.df)

    def _check_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("Recommendation models are not loaded.")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
            return number if np.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    def _row_to_movie(self, row: pd.Series) -> dict:
        result = {
            "title": str(row.get("title", "")),
            "rating": self._safe_float(row.get("vote_average", 0)),
            "popularity": self._safe_float(row.get("popularity", 0)),
        }

        # Preserve useful local dataset fields when they exist.
        for key in ["id", "release_date", "overview", "poster_path"]:
            if key in row.index:
                value = row.get(key)
                if pd.notna(value):
                    result[key] = str(value)

        return result

    def _find_index(self, title: str) -> int | None:
        title = title.strip()
        if not title:
            return None

        # Exact match first, case-insensitive.
        titles = self.df["title"].astype(str)
        exact = titles.str.casefold() == title.casefold()
        matches = self.df.index[exact].tolist()
        if matches:
            return int(matches[0])

        # Then try the original Series index if available.
        try:
            value = self.indices.get(title)
            if value is not None:
                if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                    return int(value[0])
                return int(value)
        except Exception:
            pass

        return None

    def get_movie(self, title: str) -> dict:
        self._check_loaded()
        idx = self._find_index(title)
        if idx is None:
            raise ValueError(f"Movie '{title}' was not found in the local dataset.")
        return self._row_to_movie(self.df.iloc[idx])

    def search_titles(self, query: str, limit: int = 10) -> list[dict]:
        self._check_loaded()
        query = query.strip().casefold()
        if not query:
            return []

        titles = self.df["title"].astype(str)
        contains = titles.str.casefold().str.contains(query, regex=False, na=False)
        indexes = self.df.index[contains].tolist()[:limit]
        return [self._row_to_movie(self.df.iloc[i]) for i in indexes]

    def recommend(self, title: str, n: int = 10) -> list[dict]:
        self._check_loaded()
        idx = self._find_index(title)
        if idx is None:
            raise ValueError(f"Movie '{title}' was not found in the local dataset.")

        similarity = cosine_similarity(
            self.tfidf_matrix[idx], self.tfidf_matrix
        ).flatten()

        # Remove the input movie itself.
        similarity[idx] = -1.0

        # Get more candidates than needed so we can safely skip bad rows.
        candidate_count = min(len(similarity), max(n * 3, n + 10))
        candidate_indexes = np.argpartition(-similarity, candidate_count - 1)[:candidate_count]
        candidate_indexes = candidate_indexes[np.argsort(-similarity[candidate_indexes])]

        results = []
        for movie_idx in candidate_indexes:
            if len(results) >= n:
                break
            if similarity[movie_idx] < 0:
                continue

            row = self.df.iloc[int(movie_idx)]
            movie = self._row_to_movie(row)
            score = float(similarity[movie_idx])
            movie["similarity"] = round(score, 4)
            movie["match_percent"] = round(max(0.0, min(1.0, score)) * 100, 2)
            results.append(movie)

        return results
