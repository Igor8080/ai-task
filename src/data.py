
from pathlib import Path
import zipfile
import urllib.request

import pandas as pd


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def load_data(data_dir: str = "data"):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    zip_path = data_path / "ml-latest-small.zip"
    extract_dir = data_path / "ml-latest-small"

    if not (extract_dir / "movies.csv").exists() or not (extract_dir / "ratings.csv").exists():
        if not zip_path.exists():
            print("Downloading MovieLens...")
            urllib.request.urlretrieve(MOVIELENS_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(data_path)

    movies = pd.read_csv(extract_dir / "movies.csv")
    ratings = pd.read_csv(extract_dir / "ratings.csv")

    return movies, ratings


def leave_one_out_split(
    ratings: pd.DataFrame,
    positive_rating: float = 4.0,
):
    """
    Для каждого пользователя выбираем последнее положительное
    взаимодействие для тестирования.

    Все взаимодействия пользователя с timestamp >= timestamp
    тестового взаимодействия исключаются из обучения.
    """

    positive = ratings[
        ratings["rating"] >= positive_rating
    ].copy()

    if positive.empty:
        return ratings.copy(), ratings.iloc[0:0].copy()

    test = (
        positive
        .sort_values(["userId", "timestamp", "movieId"])
        .groupby("userId")
        .tail(1)
    )

    cutoffs = test.set_index("userId")["timestamp"].to_dict()

    keep_mask = ratings.apply(
        lambda row: (
            row["userId"] not in cutoffs
            or row["timestamp"] < cutoffs[row["userId"]]
        ),
        axis=1,
    )

    train = ratings.loc[keep_mask].copy()

    return train, test.copy()
