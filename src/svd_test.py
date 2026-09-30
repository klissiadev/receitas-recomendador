import time

from preprocess import prepare_data
from model_training.svd_training import build_dataset, run_cross_validation
from models import RecipeRecommender

INTERACTIONS_PATH = '../data/RAW_interactions.csv'
RECIPES_PATH = '../data/RAW_recipes.csv'
MIN_RATINGS_PER_USER = 0
MIN_RATINGS_PER_RECIPE = 0
N_FACTORS = 50
MODEL_PATH = '../models/svd.pkl'
NEW_USER_ID = -1
FORCE_RETRAIN = False 
RUN_CV = True


def run_demo(recommender: RecipeRecommender, user_ids: list) -> None:
    for uid in user_ids:
        print(f"\n User {uid} ")

        history = recommender.get_user_history(uid)
        if history.empty:
            print("Rating history: none (new user / cold start)")
        else:
            print("Rating history:")
            for _, row in history.iterrows():
                print(f"  - {row['name']} (rated {row['rating']})")

        print("Recommendations:")
        recs = recommender.get_recommendations(uid, n=5)
        for _, row in recs.iterrows():
            print(f"  - {row['name']} (predicted rating: {row['predicted_rating']})")

def print_cv(results: dict) -> None:
    print("\nCross-validation (3 partições):")
    for metric in ('test_rmse', 'test_mae'):
        s = results[metric]
        print(f"  {metric[5:].upper()}: {s.mean():.4f} ± {s.std():.4f}  "
              f"(partições: {[round(float(x), 4) for x in s]})")
    print(f"  Tempo médio de treino: {results['fit_time'].mean():.0f}s")


def main():
    # 1. Pré-processamento
    user_df, recipe_df = prepare_data(
        INTERACTIONS_PATH,
        RECIPES_PATH,
        min_ratings_per_user=MIN_RATINGS_PER_USER,
        min_ratings_per_recipe=MIN_RATINGS_PER_RECIPE,
    )

    if RUN_CV:
        results = run_cross_validation(build_dataset(user_df), n_factors=N_FACTORS)
        print_cv(results)

    # 2. Recommender
    recommender = RecipeRecommender.get_or_train(
        MODEL_PATH,
        user_df,
        recipe_df,
        n_factors=N_FACTORS,
        force_retrain=FORCE_RETRAIN,
    )
    assert NEW_USER_ID not in set(user_df['user_id'])
    existing_users = (
        user_df['user_id']
        .drop_duplicates()
        .sample(4, random_state=int(time.time()))
        .tolist()
    )
    demo_users = existing_users + [NEW_USER_ID]
    run_demo(recommender, demo_users)


if __name__ == '__main__':
    main()