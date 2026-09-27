import pandas as pd


def load_data(interactions_path: str, recipes_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_df = pd.read_csv(interactions_path)
    recipe_df = pd.read_csv(recipes_path)
    return user_df, recipe_df


def clean_data(user_df: pd.DataFrame, recipe_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_df = user_df.dropna(subset=['user_id', 'recipe_id', 'rating'])
    recipe_df = recipe_df.dropna(subset=['id', 'name'])
    recipe_df = recipe_df.copy()
    recipe_df['description'] = recipe_df['description'].fillna('')

    user_df = user_df.drop_duplicates(subset=['user_id', 'recipe_id'])
    recipe_df = recipe_df.drop_duplicates(subset=['id'])

    valid_recipe_ids = set(recipe_df['id'])
    user_df = user_df[user_df['recipe_id'].isin(valid_recipe_ids)]

    return user_df, recipe_df


# não sei se a gnt vai precisar usar
def filter_by_min_ratings(
    user_df: pd.DataFrame,
    min_ratings_per_user: int = 0,
    min_ratings_per_recipe: int = 0,
) -> pd.DataFrame:
    recipe_counts = user_df['recipe_id'].value_counts()
    user_df = user_df[
        user_df['recipe_id'].isin(
            recipe_counts[recipe_counts >= min_ratings_per_recipe].index
        )
    ]

    user_counts = user_df['user_id'].value_counts()
    user_df = user_df[
        user_df['user_id'].isin(
            user_counts[user_counts >= min_ratings_per_user].index
        )
    ]

    return user_df


def prepare_data(
    interactions_path: str,
    recipes_path: str,
    min_ratings_per_user: int = 0,
    min_ratings_per_recipe: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_df, recipe_df = load_data(interactions_path, recipes_path)
    user_df, recipe_df = clean_data(user_df, recipe_df)
    user_df = filter_by_min_ratings(user_df, min_ratings_per_user, min_ratings_per_recipe)
    return user_df, recipe_df

def load_recipe_data(recipes_path: str) -> pd.DataFrame:
    recipe_df = pd.read_csv(recipes_path)
    return recipe_df


def clean_recipe_data(recipe_df: pd.DataFrame) -> pd.DataFrame:
    recipe_df = recipe_df.dropna(subset=['id', 'name'])
    recipe_df = recipe_df.copy()
    recipe_df['description'] = recipe_df['description'].fillna('')
    recipe_df = recipe_df.drop_duplicates(subset=['id'])
    return recipe_df