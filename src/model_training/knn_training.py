import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import parse_list  # aceita string "[...]" OU lista ja convertida


def build_content_text(recipe_df: pd.DataFrame) -> pd.DataFrame:
    recipe_df = recipe_df.dropna(subset=['ingredients']).reset_index(drop=True)
    recipe_df = recipe_df.copy()

    recipe_df['ingredients_list'] = recipe_df['ingredients'].apply(parse_list)
    if 'tags' in recipe_df.columns:
        recipe_df['tags_list'] = recipe_df['tags'].apply(parse_list)
    else:
        recipe_df['tags_list'] = [[] for _ in range(len(recipe_df))]

    recipe_df['content_text'] = (
        recipe_df['ingredients_list'].apply(lambda lst: ' '.join(lst))
        + ' '
        + recipe_df['tags_list'].apply(lambda lst: ' '.join(lst))
    )
    return recipe_df


def train_tfidf(recipe_df: pd.DataFrame, max_features: int = 5000, stop_words: str = 'english'):
    recipe_df = build_content_text(recipe_df)

    tfidf = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    content_matrix = tfidf.fit_transform(recipe_df['content_text'])

    id_to_row = pd.Series(recipe_df.index.values, index=recipe_df['id'])
    recipe_lookup = recipe_df.set_index('id')[['name', 'description']]
    recipe_ids = recipe_df['id']

    return content_matrix, tfidf, id_to_row, recipe_lookup, recipe_ids


def save_artifacts(path, content_matrix, id_to_row, recipe_ids, recipe_lookup) -> None:
    with open(path, 'wb') as f:
        pickle.dump({
            'content_matrix': content_matrix,
            'id_to_row': id_to_row,
            'recipe_ids': recipe_ids,
            'recipe_lookup': recipe_lookup,
        }, f)


def load_artifacts(path) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)