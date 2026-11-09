
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TwoTowerNetwork(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim,
        )

        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim,
        )

        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.movie_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, user_ids, movie_ids):
        user_vectors = self.user_tower(
            self.user_embedding(user_ids)
        )

        movie_vectors = self.movie_tower(
            self.movie_embedding(movie_ids)
        )

        scores = (
            user_vectors * movie_vectors
        ).sum(dim=1)

        scores = scores / np.sqrt(user_vectors.shape[1])

        scores += self.user_bias(user_ids).squeeze(1)
        scores += self.movie_bias(movie_ids).squeeze(1)

        return scores


class TwoTowerRecommender:
    def __init__(
        self,
        embedding_dim: int = 64,
        epochs: int = 15,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        negative_ratio: int = 4,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.seed = seed

        self.device = torch.device("cpu")

    def fit(self, ratings: pd.DataFrame):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.ratings = ratings.copy()

        self.user_ids = np.sort(
            ratings["userId"].unique()
        )

        self.movie_ids = np.sort(
            ratings["movieId"].unique()
        )

        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(self.user_ids)
        }

        self.movie_to_index = {
            movie_id: index
            for index, movie_id in enumerate(self.movie_ids)
        }

        self.user_positive_movies = {
            user_id: set(
                group.loc[
                    group["rating"] >= 4.0,
                    "movieId",
                ]
            )
            for user_id, group in ratings.groupby("userId")
        }

        positives = ratings[
            ratings["rating"] >= 4.0
        ][["userId", "movieId"]].drop_duplicates()

        if positives.empty:
            raise ValueError(
                "No positive interactions found. "
                "Check the rating threshold."
            )

        all_movie_ids = set(self.movie_ids)

        rng = np.random.default_rng(self.seed)

        negative_pairs = []

        for row in positives.itertuples(index=False):
            user_id = row.userId

            positive_movies = self.user_positive_movies.get(
                user_id,
                set(),
            )

            candidates = list(
                all_movie_ids - positive_movies
            )

            if not candidates:
                continue

            sampled = rng.choice(
                candidates,
                size=min(
                    self.negative_ratio,
                    len(candidates),
                ),
                replace=False,
            )

            negative_pairs.extend(
                (user_id, movie_id)
                for movie_id in sampled
            )

        positive_pairs = list(
            positives.itertuples(index=False, name=None)
        )

        pairs = positive_pairs + negative_pairs

        labels = (
            [1.0] * len(positive_pairs)
            + [0.0] * len(negative_pairs)
        )

        user_indices = [
            self.user_to_index[user_id]
            for user_id, _ in pairs
        ]

        movie_indices = [
            self.movie_to_index[movie_id]
            for _, movie_id in pairs
        ]

        dataset = TensorDataset(
            torch.tensor(user_indices, dtype=torch.long),
            torch.tensor(movie_indices, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.model = TwoTowerNetwork(
            num_users=len(self.user_ids),
            num_movies=len(self.movie_ids),
            embedding_dim=self.embedding_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        criterion = nn.BCEWithLogitsLoss()

        self.training_losses = []

        for epoch in range(self.epochs):
            self.model.train()

            total_loss = 0.0
            total_samples = 0

            for batch_users, batch_movies, batch_labels in loader:
                batch_users = batch_users.to(self.device)
                batch_movies = batch_movies.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()

                predictions = self.model(
                    batch_users,
                    batch_movies,
                )

                loss = criterion(
                    predictions,
                    batch_labels,
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_labels)
                total_samples += len(batch_labels)

            epoch_loss = total_loss / total_samples
            self.training_losses.append(epoch_loss)

            print(
                f"Two-Tower Epoch {epoch + 1}/{self.epochs} "
                f"- Loss: {epoch_loss:.4f}"
            )

        self.model.eval()

        with torch.no_grad():
            movie_indices = torch.arange(
                len(self.movie_ids),
                dtype=torch.long,
                device=self.device,
            )

            self.movie_vectors = self.model.movie_tower(
                self.model.movie_embedding(movie_indices)
            )

            self.movie_biases = self.model.movie_bias(
                movie_indices
            ).squeeze(1)

        return self

    def recommend(self, user_id, k, seen=None):
        if k <= 0:
            return []

        user_index = self.user_to_index.get(user_id)

        if user_index is None:
            return []

        seen = set(seen or [])

        self.model.eval()

        with torch.no_grad():
            user_tensor = torch.tensor(
                [user_index],
                dtype=torch.long,
                device=self.device,
            )

            user_vector = self.model.user_tower(
                self.model.user_embedding(user_tensor)
            )

            scores = (
                    user_vector @ self.movie_vectors.T
            ).squeeze(0)

            scores = scores / np.sqrt(self.embedding_dim)

            scores += self.model.user_bias(
                user_tensor
            ).squeeze()

            scores += self.movie_biases

            scores = scores.cpu().numpy()

        scores = scores.copy()

        for movie_id in seen:
            movie_index = self.movie_to_index.get(movie_id)

            if movie_index is not None:
                scores[movie_index] = -np.inf

        count = min(
            k,
            int(np.isfinite(scores).sum()),
        )

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
