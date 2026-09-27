
from preprocess import load_recipe_data, clean_recipe_data
from models import ContentRecommender

RECIPES_PATH = '../data/RAW_recipes.csv'
ARTIFACTS_PATH = '../models/knn.pkl'
MAX_FEATURES = 5000
FORCE_RETRAIN = False


def main():
    recipe_df = load_recipe_data(RECIPES_PATH)
    recipe_df = clean_recipe_data(recipe_df)

    recommender = ContentRecommender.get_or_train(
        ARTIFACTS_PATH,
        recipe_df,
        max_features=MAX_FEATURES,
        force_retrain=FORCE_RETRAIN,
    )

    search_term = 'pasta'
    matches = recommender.recipe_lookup[
        recommender.recipe_lookup['name'].str.contains(search_term, case=False, na=False)
    ]
    if not matches.empty:
        test_id = matches.index[0]
        recommender.print_similar_recipes(test_id, n=5)
    else:
        print(f"No '{search_term}' recipe found.")


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()