from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PopularityRecommender:
    def fit(self, ratings: pd.DataFrame):
        stats = (
            ratings.groupby("movieId")
            .agg(
                mean_rating=("rating", "mean"),
                rating_count=("rating", "count"),
            )
            .reset_index()
        )

        global_mean = ratings["rating"].mean()
        min_votes = 20

        stats["score"] = (
            stats["rating_count"]
            / (stats["rating_count"] + min_votes)
            * stats["mean_rating"]
            + min_votes
            / (stats["rating_count"] + min_votes)
            * global_mean
        )

        self.ranking = stats.sort_values(
            ["score", "rating_count"],
            ascending=False,
        )

        return self

    def recommend(self, user_id, k, seen=None):
        seen = set(seen or [])

        return [
            movie_id
            for movie_id in self.ranking["movieId"]
            if movie_id not in seen
        ][:k]


class ContentBasedRecommender:
    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.movies = movies.copy().reset_index(drop=True)
        self.ratings = ratings

        text = (
            self.movies["title"].fillna("")
            + " "
            + self.movies["genres"]
            .fillna("")
            .str.replace("|", " ", regex=False)
        )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
        )

        matrix = self.vectorizer.fit_transform(text)

        self.similarity = cosine_similarity(matrix)

        self.movie_ids = self.movies["movieId"].to_numpy()

        self.movie_to_index = {
            movie_id: index
            for index, movie_id in enumerate(self.movie_ids)
        }

        return self

    def recommend(self, user_id, k, seen=None):
        seen = set(seen or [])

        user_ratings = self.ratings[
            self.ratings["userId"] == user_id
        ]

        liked = user_ratings[
            user_ratings["rating"] >= 4.0
        ]

        if liked.empty:
            return []

        liked_indices = []
        weights = []

        for row in liked.itertuples():
            index = self.movie_to_index.get(row.movieId)

            if index is not None:
                liked_indices.append(index)
                weights.append(row.rating)

        if not liked_indices:
            return []

        scores = np.average(
            self.similarity[liked_indices],
            axis=0,
            weights=weights,
        )

        for movie_id in seen:
            index = self.movie_to_index.get(movie_id)

            if index is not None:
                scores[index] = -np.inf

        count = min(k, len(scores))

        top_indices = np.argpartition(
            scores,
            -count,
        )[-count:]

        top_indices = top_indices[
            np.argsort(scores[top_indices])[::-1]
        ]

        return self.movie_ids[top_indices].tolist()


class ItemBasedCollaborativeRecommender:
    def fit(self, ratings: pd.DataFrame):
        self.ratings = ratings.copy()

        matrix = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0,
        )

        self.movie_ids = matrix.columns.to_numpy()

        self.movie_to_index = {
            movie_id: index
            for index, movie_id in enumerate(self.movie_ids)
        }

        self.similarity = cosine_similarity(
            matrix.T.to_numpy()
        )

        np.fill_diagonal(self.similarity, 0)

        return self

    def recommend(self, user_id, k, seen=None):
        seen = set(seen or [])

        user_ratings = self.ratings[
            self.ratings["userId"] == user_id
        ]

        positive = user_ratings[
            user_ratings["rating"] >= 3
        ]

        if positive.empty:
            return []

        indices = []
        weights = []

        for row in positive.itertuples():
            index = self.movie_to_index.get(row.movieId)

            if index is not None:
                indices.append(index)
                weights.append(row.rating)

        if not indices:
            return []

        scores = np.average(
            self.similarity[indices],
            axis=0,
            weights=weights,
        )

        for movie_id in seen:
            index = self.movie_to_index.get(movie_id)

            if index is not None:
                scores[index] = -np.inf

        count = min(k, len(scores))

        top_indices = np.argpartition(
            scores,
            -count,
        )[-count:]

        top_indices = top_indices[
            np.argsort(scores[top_indices])[::-1]
        ]

        return self.movie_ids[top_indices].tolist()


class HybridRecommender:
    def __init__(
        self,
        content,
        collaborative,
        alpha=0.5,
    ):
        self.content = content
        self.collaborative = collaborative
        self.alpha = alpha

    def recommend(self, user_id, k, seen=None):
        content_ids = self.content.recommend(
            user_id,
            k * 5,
            seen,
        )

        collaborative_ids = self.collaborative.recommend(
            user_id,
            k * 5,
            seen,
        )

        content_rank = {
            movie_id: 1 / (index + 1)
            for index, movie_id in enumerate(content_ids)
        }

        collaborative_rank = {
            movie_id: 1 / (index + 1)
            for index, movie_id in enumerate(collaborative_ids)
        }

        candidates = set(content_ids) | set(collaborative_ids)

        scores = {}

        for movie_id in candidates:
            scores[movie_id] = (
                self.alpha
                * content_rank.get(movie_id, 0)
                + (1 - self.alpha)
                * collaborative_rank.get(movie_id, 0)
            )

        return sorted(
            scores,
            key=scores.get,
            reverse=True,
        )[:k]