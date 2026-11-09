
import math


def precision_at_k(recommended, relevant, k):
    if k <= 0:
        return 0.0

    recommended = recommended[:k]

    return len(set(recommended) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    return (
        len(set(recommended[:k]) & set(relevant))
        / len(set(relevant))
    )


def hit_rate_at_k(recommended, relevant, k):
    return float(
        bool(set(recommended[:k]) & set(relevant))
    )


def ndcg_at_k(recommended, relevant, k):
    relevant = set(relevant)

    dcg = sum(
        1 / math.log2(i + 2)
        for i, movie_id in enumerate(recommended[:k])
        if movie_id in relevant
    )

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1 / math.log2(i + 2)
        for i in range(ideal_hits)
    )

    return dcg / idcg


def evaluate(recommender, train, test, k=10):
    train_by_user = (
        train.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    test_by_user = (
        test.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    values = {
        "Precision@K": [],
        "Recall@K": [],
        "HitRate@K": [],
        "NDCG@K": [],
    }

    for user_id, relevant in test_by_user.items():
        seen = train_by_user.get(user_id, set())

        recommendations = recommender.recommend(
            user_id,
            k,
            seen,
        )

        values["Precision@K"].append(
            precision_at_k(recommendations, relevant, k)
        )

        values["Recall@K"].append(
            recall_at_k(recommendations, relevant, k)
        )

        values["HitRate@K"].append(
            hit_rate_at_k(recommendations, relevant, k)
        )

        values["NDCG@K"].append(
            ndcg_at_k(recommendations, relevant, k)
        )

    if not test_by_user:
        return {name: 0.0 for name in values}

    return {
        name: sum(items) / len(items)
        for name, items in values.items()
    }
