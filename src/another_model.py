import ast
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


recipe_df = pd.read_csv('../data/RAW_recipes.csv')
recipe_df = recipe_df.dropna(subset=['id', 'name', 'ingredients'])
recipe_df['description'] = recipe_df['description'].fillna('')
recipe_df = recipe_df.drop_duplicates(subset=['id']).reset_index(drop=True)

recipe_lookup = recipe_df.set_index('id')[['name', 'description']]


def parse_list_str(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


recipe_df['ingredients_list'] = recipe_df['ingredients'].apply(parse_list_str)
recipe_df['tags_list'] = recipe_df['tags'].apply(parse_list_str) if 'tags' in recipe_df.columns else [[]] * len(recipe_df)

recipe_df['content_text'] = (
    recipe_df['ingredients_list'].apply(lambda lst: ' '.join(lst))
    + ' '
    + recipe_df['tags_list'].apply(lambda lst: ' '.join(lst))
)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
content_matrix = tfidf.fit_transform(recipe_df['content_text'])

id_to_row = pd.Series(recipe_df.index.values, index=recipe_df['id'])


def get_similar_recipes_content(recipe_id, n=10, exclude=None):
    exclude = exclude or set()
    if recipe_id not in id_to_row.index:
        return pd.DataFrame(columns=['recipe_id', 'name', 'description', 'similarity'])

    row = id_to_row[recipe_id]
    sims = cosine_similarity(content_matrix[row], content_matrix).flatten()
    ranked_idx = sims.argsort()[::-1]

    rows = []
    for candidate_row in ranked_idx:
        candidate_id = recipe_df['id'].iloc[candidate_row]
        if candidate_id == recipe_id or candidate_id in exclude:
            continue
        meta = recipe_lookup.loc[candidate_id]
        rows.append({
            'recipe_id': candidate_id,
            'name': meta['name'],
            'description': meta['description'],
            'similarity': round(float(sims[candidate_row]), 3)
        })
        if len(rows) == n:
            break
    return pd.DataFrame(rows)


def print_similar_recipes_content(recipe_id, n=5, exclude=None):
    """Prints the query recipe (name + description) followed by its top-N
    most similar recipes by ingredients/tags, each with a similarity score."""
    if recipe_id not in recipe_lookup.index:
        print(f"Recipe {recipe_id} not found in recipe_df.")
        return

    query = recipe_lookup.loc[recipe_id]
    print(f"\nReceita base: {query['name']}")
    print(f"Descrição: {query['description'][:200]}")

    similar = get_similar_recipes_content(recipe_id, n=n, exclude=exclude)
    if similar.empty:
        print("-> Nenhuma receita similar encontrada.")
        return

    print("\nReceitas similares (content-based, por ingredientes/tags):")
    for _, row in similar.iterrows():
        print(f"\n  {row['name']}  (similaridade: {row['similarity']})")
        print(f"  {row['description'][:200]}")


with open('content_similarity_artifacts.pkl', 'wb') as f:
    pickle.dump({
        'content_matrix': content_matrix,
        'id_to_row': id_to_row,
        'recipe_ids': recipe_df['id'],
        'recipe_lookup': recipe_lookup,
    }, f)
print("Artifacts saved to content_similarity_artifacts.pkl")

chocolate_recipes = recipe_df[recipe_df['name'].str.contains('chocolate chip', case=False, na=False)]
if not chocolate_recipes.empty:
    test_id = chocolate_recipes['id'].iloc[0]
    print_similar_recipes_content(test_id, n=5)
else:
    print("No 'chocolate chip' recipe found to test with -- pick any recipe id manually.")