import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy, dump
from surprise.model_selection import train_test_split, cross_validate


def build_dataset(user_df: pd.DataFrame, rating_scale: tuple[int, int] = (0, 5)) -> Dataset:
    reader = Reader(rating_scale=rating_scale)
    return Dataset.load_from_df(user_df[['user_id', 'recipe_id', 'rating']], reader)


def run_cross_validation(data: Dataset, n_factors: int = 50, cv: int = 3, random_state: int = 42):
    return cross_validate(
        SVD(n_factors=n_factors, random_state=random_state),
        data,
        measures=['RMSE', 'MAE'],
        cv=cv,
        verbose=True,
    )


def train_svd(
    data: Dataset,
    n_factors: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[SVD, list]:
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    algo = SVD(n_factors=n_factors, random_state=random_state)
    algo.fit(trainset)

    predictions = algo.test(testset)
    print(f"\nHoldout evaluation ({int(test_size * 100)}%)")
    accuracy.rmse(predictions)
    accuracy.mae(predictions)

    return algo, predictions


def save_model(algo: SVD, path: str = 'svd.pkl') -> None:
    dump.dump(path, algo=algo)
    print(f"\nModel saved to {path}")


def load_model(path: str = 'svd.pkl') -> SVD:
    _, algo = dump.load(path)
    return algo