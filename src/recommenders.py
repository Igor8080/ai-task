
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PopularityRecommender:
    """Recommends globally popular movies."""

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
        if k <= 0:
            return []

        seen = set(seen or [])

        return [
            movie_id
            for movie_id in self.ranking["movieId"]
            if movie_id not in seen
        ][:k]


class ContentBasedRecommender:
    """Recommends movies similar to those liked by the user."""

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
        if k <= 0:
            return []

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

        scores = scores.copy()

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
    """Recommends movies based on similarity between movies."""

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
        if k <= 0:
            return []

        seen = set(seen or [])

        user_ratings = self.ratings[
            self.ratings["userId"] == user_id
        ]

        positive = user_ratings[
            user_ratings["rating"] >= 3.0
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

        scores = scores.copy()

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


class UserBasedCollaborativeRecommender:
    """
    Recommends movies based on users with similar rating patterns.
    """

    def fit(self, ratings: pd.DataFrame):
        self.ratings = ratings.copy()

        matrix = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0,
        )

        self.user_ids = matrix.index.to_numpy()
        self.movie_ids = matrix.columns.to_numpy()

        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(self.user_ids)
        }

        self.movie_to_index = {
            movie_id: index
            for index, movie_id in enumerate(self.movie_ids)
        }

        self.user_item_matrix = matrix.to_numpy()

        self.similarity = cosine_similarity(
            self.user_item_matrix
        )

        np.fill_diagonal(self.similarity, 0)

        return self

    def recommend(self, user_id, k, seen=None):
        if k <= 0:
            return []

        seen = set(seen or [])

        user_index = self.user_to_index.get(user_id)

        if user_index is None:
            return []

        similarities = self.similarity[user_index]

        user_ratings = self.user_item_matrix[user_index]

        # Select users with positive similarity.
        similar_users = np.where(similarities > 0)[0]

        if len(similar_users) == 0:
            return []

        # Sort similar users by similarity.
        similar_users = similar_users[
            np.argsort(similarities[similar_users])[::-1]
        ]

        scores = np.zeros(len(self.movie_ids), dtype=float)

        for similar_user_index in similar_users:
            similarity_score = similarities[similar_user_index]

            ratings = self.user_item_matrix[
                similar_user_index
            ]

            positive_ratings = ratings >= 3.0

            scores += (
                similarity_score
                * positive_ratings
                * ratings
            )

        # Exclude movies already rated by the target user.
        scores[user_ratings > 0] = -np.inf

        # Exclude explicitly provided seen movies.
        for movie_id in seen:
            movie_index = self.movie_to_index.get(movie_id)

            if movie_index is not None:
                scores[movie_index] = -np.inf

        # Do not recommend movies with no positive support.
        scores[scores == 0] = -np.inf

        count = min(k, np.isfinite(scores).sum())

        if count == 0:
            return []

        top_indices = np.argpartition(
            scores,
            -count,
        )[-count:]

        top_indices = top_indices[
            np.argsort(scores[top_indices])[::-1]
        ]

        return self.movie_ids[top_indices].tolist()


class GenrePopularityRecommender:
    """
    Recommends popular movies from genres preferred by the user.
    """

    def fit(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
    ):
        self.movies = movies.copy()
        self.ratings = ratings.copy()

        self.movie_stats = (
            ratings.groupby("movieId")
            .agg(
                mean_rating=("rating", "mean"),
                rating_count=("rating", "count"),
            )
            .reset_index()
        )

        global_mean = ratings["rating"].mean()
        min_votes = 20

        self.movie_stats["score"] = (
            self.movie_stats["rating_count"]
            / (self.movie_stats["rating_count"] + min_votes)
            * self.movie_stats["mean_rating"]
            + min_votes
            / (self.movie_stats["rating_count"] + min_votes)
            * global_mean
        )

        self.movie_stats = self.movie_stats.merge(
            movies[["movieId", "genres"]],
            on="movieId",
            how="left",
        )

        return self

    def recommend(self, user_id, k, seen=None):
        if k <= 0:
            return []

        seen = set(seen or [])

        user_ratings = self.ratings[
            self.ratings["userId"] == user_id
        ]

        liked = user_ratings[
            user_ratings["rating"] >= 4.0
        ]

        if liked.empty:
            # Fallback to global popularity.
            candidates = self.movie_stats.sort_values(
                "score",
                ascending=False,
            )

            return [
                movie_id
                for movie_id in candidates["movieId"]
                if movie_id not in seen
            ][:k]

        liked_movies = liked.merge(
            self.movies[["movieId", "genres"]],
            on="movieId",
            how="left",
        )

        genre_weights = {}

        for row in liked_movies.itertuples():
            genres = str(row.genres).split("|")

            for genre in genres:
                if genre and genre != "(no genres listed)":
                    genre_weights[genre] = (
                        genre_weights.get(genre, 0)
                        + row.rating
                    )

        if not genre_weights:
            return []

        candidates = self.movie_stats[
            ~self.movie_stats["movieId"].isin(seen)
        ].copy()

        def calculate_genre_score(genres):
            genres = str(genres).split("|")

            return sum(
                genre_weights.get(genre, 0)
                for genre in genres
            )

        candidates["genre_score"] = candidates[
            "genres"
        ].apply(calculate_genre_score)

        candidates["final_score"] = (
            candidates["score"]
            * candidates["genre_score"]
        )

        candidates = candidates.sort_values(
            ["final_score", "score"],
            ascending=False,
        )

        return candidates["movieId"].head(k).tolist()


class HybridRecommender:
    """Combines Content-Based and Collaborative recommendations."""

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
        if k <= 0:
            return []

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
