
import argparse

from src.data import load_data, leave_one_out_split
from src.evaluation import evaluate
from src.recommenders import (
    PopularityRecommender,
    ContentBasedRecommender,
    ItemBasedCollaborativeRecommender,
    UserBasedCollaborativeRecommender,
    GenrePopularityRecommender,
    HybridRecommender,
)
from src.two_tower import TwoTowerRecommender

def main():
    parser = argparse.ArgumentParser(
        description="MovieLens Recommendation System"
    )

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--positive-rating", type=float, default=4.0)

    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be greater than zero")

    movies, ratings = load_data()

    train, test = leave_one_out_split(
        ratings,
        args.positive_rating,
    )

    print("Dataset statistics")
    print("-" * 60)
    print(f"Movies: {len(movies)}")
    print(f"Ratings: {len(ratings)}")
    print(f"Train: {len(train)}")
    print(f"Test: {len(test)}")
    print()

    if test.empty:
        print("Test dataset is empty. Cannot evaluate recommenders.")
        return

    # Initialize recommenders.
    popularity = PopularityRecommender().fit(train)

    content = ContentBasedRecommender().fit(
        movies,
        train,
    )

    item_based = ItemBasedCollaborativeRecommender().fit(
        train,
    )

    user_based = UserBasedCollaborativeRecommender().fit(
        train,
    )

    genre_popularity = GenrePopularityRecommender().fit(
        movies,
        train,
    )

    two_tower = TwoTowerRecommender(
        embedding_dim=64,
        epochs=15,
        batch_size=1024,
        learning_rate=0.001,
        negative_ratio=4,
    ).fit(train)

    hybrid = HybridRecommender(
        content,
        item_based,
    )

    recommenders = {
        "Popularity": popularity,
        "Content-Based": content,
        "Item-Based Collaborative": item_based,
        "User-Based Collaborative": user_based,
        "Genre Popularity": genre_popularity,
        "Two-Tower": two_tower,
        "Hybrid": hybrid,
    }

    # Evaluate all recommenders.
    print("Evaluation")
    print("-" * 60)

    for name, recommender in recommenders.items():
        metrics = evaluate(
            recommender,
            train,
            test,
            args.k,
        )

        print(name)

        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        print()

    # Show example recommendations.
    user_id = int(test.iloc[0]["userId"])

    seen = set(
        train.loc[
            train["userId"] == user_id,
            "movieId",
        ]
    )

    titles = movies.set_index("movieId")["title"].to_dict()

    print(f"Example recommendations for user {user_id}")
    print("-" * 60)

    for name, recommender in recommenders.items():
        print(f"\n{name}:")

        recommendations = recommender.recommend(
            user_id,
            args.k,
            seen,
        )

        if not recommendations:
            print("  No recommendations available")
            continue

        for movie_id in recommendations:
            print(f"  - {titles.get(movie_id, movie_id)}")


if __name__ == "__main__":
    main()
