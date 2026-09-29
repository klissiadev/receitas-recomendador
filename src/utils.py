import ast
import math

import pandas as pd

NUTRITION_FIELDS = [
    "calories_kcal", "fat_pdv", "sugar_pdv", "sodium_pdv",
    "protein_pdv", "sat_fat_pdv", "carbs_pdv",
]


def parse_list(value) -> list:
    """Converte "['a', 'b']" em ['a', 'b']. Aceita listas já convertidas e NaN."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        parsed = ast.literal_eval(value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def parse_nutrition(value) -> dict:
    vals = [float(v) for v in parse_list(value)][:len(NUTRITION_FIELDS)]
    vals += [0.0] * (len(NUTRITION_FIELDS) - len(vals))
    return dict(zip(NUTRITION_FIELDS, vals))


def _clean(value, default=None):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if hasattr(value, "item"):  # numpy -> tipo nativo (necessário p/ JSON)
        return value.item()
    return value


def compute_rating_stats(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """avg_rating e n_ratings por receita. Calcule UMA vez na inicialização."""
    stats = interactions_df.groupby("recipe_id")["rating"].agg(["mean", "count"])
    stats = stats.rename(columns={"mean": "avg_rating", "count": "n_ratings"})
    stats["avg_rating"] = stats["avg_rating"].round(2)
    return stats


def enrich_recipes(recipe_ids, recipes_df, rating_stats=None, detail=False) -> list[dict]:
    """ids -> dicts completos, na mesma ordem dos ids.
    detail=False -> RecipeSummary (+ description)
    detail=True  -> RecipeDetail (steps, nutrition, ingredients, ...)
    """
    ids = list(recipe_ids)
    subset = (recipes_df[recipes_df["id"].isin(ids)]
              .drop_duplicates(subset="id").set_index("id"))

    records = []
    for rid in ids:
        if rid not in subset.index:
            continue
        r = subset.loc[rid]

        avg, n = 0.0, 0
        if rating_stats is not None and rid in rating_stats.index:
            avg = float(rating_stats.at[rid, "avg_rating"])
            n = int(rating_stats.at[rid, "n_ratings"])

        rec = {
            "id": int(rid),
            "name": _clean(r.get("name"), ""),
            "minutes": _clean(r.get("minutes"), 0),
            "tags": parse_list(r.get("tags")),
            "avg_rating": avg,
            "n_ratings": n,
            "description": _clean(r.get("description"), ""),
        }
        if detail:
            steps = parse_list(r.get("steps"))
            ingredients = parse_list(r.get("ingredients"))
            rec.update({
                "n_ingredients": _clean(r.get("n_ingredients"), len(ingredients)),
                "ingredients": ingredients,
                "n_steps": _clean(r.get("n_steps"), len(steps)),
                "steps": steps,
                "nutrition": parse_nutrition(r.get("nutrition")),
                "contributor_id": _clean(r.get("contributor_id")),
                "submitted": _clean(r.get("submitted")),
                "user_rating": None,
            })
        records.append(rec)
    return records


def get_recipe_detail(recipe_id, recipes_df, interactions_df,
                      rating_stats=None, user_id=None):
    found = enrich_recipes([recipe_id], recipes_df, rating_stats, detail=True)
    if not found:
        return None
    rec = found[0]
    if user_id is not None:
        own = interactions_df[(interactions_df["user_id"] == user_id)
                              & (interactions_df["recipe_id"] == recipe_id)]
        if not own.empty:
            row = own.iloc[-1]
            rec["user_rating"] = {"rating": int(row["rating"]), "date": str(row["date"])}
    return rec


def get_user_history(user_id, interactions_df, recipes_df, rating_stats=None) -> list[dict]:
    """Formato HistoryItem: {recipe: RecipeSummary, rating, date}, mais recente primeiro."""
    historico = interactions_df[interactions_df["user_id"] == user_id]
    if historico.empty:
        return []
    historico = historico.sort_values("date", ascending=False)
    summaries = {r["id"]: r for r in enrich_recipes(
        historico["recipe_id"].tolist(), recipes_df, rating_stats)}
    return [
        {"recipe": summaries[int(row["recipe_id"])],
         "rating": int(row["rating"]), "date": str(row["date"])}
        for _, row in historico.iterrows() if int(row["recipe_id"]) in summaries
    ]


def get_seen_recipe_ids(user_id, interactions_df) -> set:
    return set(interactions_df.loc[interactions_df["user_id"] == user_id, "recipe_id"])


def filter_seen_items(candidate_ids, user_id, interactions_df) -> list:
    vistos = get_seen_recipe_ids(user_id, interactions_df)
    return [rid for rid in candidate_ids if rid not in vistos]


def get_recipe_metadata(recipe_ids, recipes_df, rating_stats=None, detail=False) -> list[dict]:
    return enrich_recipes(recipe_ids, recipes_df, rating_stats, detail=detail)


def is_new_user(user_id, interactions_df) -> bool:
    return user_id not in interactions_df["user_id"].values