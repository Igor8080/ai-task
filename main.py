import argparse

from src.data import load_data, leave_one_out_split
from src.evaluation import evaluate
from src.recommenders import (
    ContentBasedRecommender,
    HybridRecommender,
    ItemBasedCollaborativeRecommender,
    PopularityRecommender,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--positive-rating", type=float, default=4.0)
    args = parser.parse_args()

    movies, ratings = load_data()
    train, test = leave_one_out_split(ratings, args.positive_rating)

    print(f"Movies: {len(movies)}")
    print(f"Ratings: {len(ratings)}")
    print(f"Train: {len(train)}")
    print(f"Test: {len(test)}\n")

    popularity = PopularityRecommender().fit(train)
    content = ContentBasedRecommender().fit(movies, train)
    collaborative = ItemBasedCollaborativeRecommender().fit(train)
    hybrid = HybridRecommender(content, collaborative)

    recommenders = {
        "Popularity": popularity,
        "Content-Based": content,
        "Item-Based Collaborative": collaborative,
        "Hybrid": hybrid,
    }

    print("Evaluation")
    print("-" * 60)

    for name, recommender in recommenders.items():
        metrics = evaluate(recommender, train, test, args.k)
        print(name)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()

    user_id = int(test.iloc[0]["userId"])
    seen = set(train[train["userId"] == user_id]["movieId"])
    titles = movies.set_index("movieId")["title"].to_dict()

    print(f"Example recommendations for user {user_id}")
    for name, recommender in recommenders.items():
        print(f"\n{name}:")
        for movie_id in recommender.recommend(user_id, args.k, seen):
            print(f"  - {titles.get(movie_id, movie_id)}")


if __name__ == "__main__":
    main()
