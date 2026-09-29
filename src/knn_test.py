
from preprocess import load_recipe_data, clean_recipe_data
from models import ContentRecommender, build_recipe


RECIPES_PATH = "../data/RAW_recipes.csv"
ARTIFACTS_PATH = "../models/knn.pkl"

MAX_FEATURES = 5000
FORCE_RETRAIN = False
N_SIMILAR = 5


def print_recipe(recipe, index=None):
    """Imprime todos os campos disponíveis de um Recipe."""

    prefix = f"{index}. " if index is not None else ""

    print(f"\n{prefix}ID: {recipe.id}")
    print(f"Name: {recipe.name}")
    print(f"Description: {recipe.description}")
    print(f"Tags: {recipe.tags}")
    print(f"Nutrition: {recipe.nutrition}")
    print(f"Steps: {recipe.steps}")

    if recipe.predicted_rating is not None:
        print(f"Predicted rating: {recipe.predicted_rating}")

    if recipe.user_rating is not None:
        print(f"User rating: {recipe.user_rating}")

    if recipe.source is not None:
        print(f"Source: {recipe.source}")


def main():

    # ------------------------------------------------------
    # 1. Pré-processamento
    # ------------------------------------------------------
    print("Preparing recipe data...")

    recipe_df = load_recipe_data(RECIPES_PATH)
    recipe_df = clean_recipe_data(recipe_df)

    print(f"Recipes: {len(recipe_df)}")

    # ------------------------------------------------------
    # 2. ContentRecommender
    # ------------------------------------------------------
    print("\nLoading ContentRecommender...")

    recommender = ContentRecommender.get_or_train(
        ARTIFACTS_PATH,
        recipe_df,
        max_features=MAX_FEATURES,
        force_retrain=FORCE_RETRAIN,
    )

    # ------------------------------------------------------
    # 3. Procura uma receita para testar
    # ------------------------------------------------------
    search_term = "pasta"

    matches = recommender.recipe_lookup[
        recommender.recipe_lookup["name"].str.contains(
            search_term,
            case=False,
            na=False,
        )
    ]

    if matches.empty:
        print(
            f"No '{search_term}' recipe found."
        )
        return

    test_id = matches.index[0]

    # ------------------------------------------------------
    # 4. Receita base
    # ------------------------------------------------------
    base = build_recipe(
        recommender.recipe_lookup,
        test_id,
    )

    print("\n" + "=" * 80)
    print("BASE RECIPE")
    print("=" * 80)

    print_recipe(base)

    # ------------------------------------------------------
    # 5. Receitas similares
    # ------------------------------------------------------
    similar = recommender.get_similar_recipes(
        test_id,
        n=N_SIMILAR,
    )

    if not similar:
        print(
            "\nNenhuma receita similar encontrada."
        )
        return

    print("\n" + "=" * 80)
    print("SIMILAR RECIPES")
    print("=" * 80)

    print(f"Source: {similar[0].source}")

    for index, recipe in enumerate(
        similar,
        start=1,
    ):
        print_recipe(
            recipe,
            index,
        )


if __name__ == "__main__":
    main()
