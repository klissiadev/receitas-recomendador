
import time

from preprocess import prepare_data
from models import RecipeRecommender


INTERACTIONS_PATH = "../data/RAW_interactions.csv"
RECIPES_PATH = "../data/RAW_recipes.csv"

MIN_RATINGS_PER_USER = 0
MIN_RATINGS_PER_RECIPE = 0

N_FACTORS = 50
MODEL_PATH = "../models/svd.pkl"

NEW_USER_ID = -1
FORCE_RETRAIN = False


def print_recipe(recipe, index=None):

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


def run_demo(
    recommender: RecipeRecommender,
    user_ids: list,
) -> None:

    for uid in user_ids:
        print("\n" + "=" * 80)
        print(f"USER {uid}")
        print("=" * 80)
        history = recommender.get_user_history(uid)

        print("\n--- RATING HISTORY ---")

        if not history:
            print("None (new user / cold start)")
        else:
            for i, recipe in enumerate(history, start=1):
                print_recipe(recipe, i)

        recommendations = recommender.get_recommendations(
            uid,
            n=5,
        )

        print("\n--- RECOMMENDATIONS ---")

        if not recommendations:
            print("No recommendations.")
            continue

        source = recommendations[0].source

        print(f"Source: {source}")
        print(f"Number of recommendations: {len(recommendations)}")

        for i, recipe in enumerate(
            recommendations,
            start=1,
        ):
            print_recipe(recipe, i)


def main():

    print("Preparing data...")

    user_df, recipe_df = prepare_data(
        INTERACTIONS_PATH,
        RECIPES_PATH,
        min_ratings_per_user=MIN_RATINGS_PER_USER,
        min_ratings_per_recipe=MIN_RATINGS_PER_RECIPE,
    )

    print(f"Users: {user_df['user_id'].nunique()}")
    print(f"Interactions: {len(user_df)}")
    print(f"Recipes: {recipe_df['id'].nunique()}")

    print("\nLoading RecipeRecommender...")

    recommender = RecipeRecommender.get_or_train(
        MODEL_PATH,
        user_df,
        recipe_df,
        n_factors=N_FACTORS,
        force_retrain=FORCE_RETRAIN,
    )

    assert NEW_USER_ID not in set(
        user_df["user_id"]
    )

    existing_users = (
        user_df["user_id"]
        .drop_duplicates()
        .sample(
            n=4,
            random_state=int(time.time()),
        )
        .tolist()
    )

    demo_users = existing_users + [NEW_USER_ID]

    print("\nUsers selected for demo:")
    print(demo_users)

    run_demo(
        recommender,
        demo_users,
    )


if __name__ == "__main__":
    main()

