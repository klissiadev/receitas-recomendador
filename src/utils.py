"""
Funções utilitárias usadas tanto pelos modelos quanto pela interface Streamlit.

Contrato de dados esperado:

interactions_df: colunas [user_id, recipe_id, date, rating, review]
recipes_df:      colunas [id, name, minutes, tags, n_ingredients, ...]
"""

import pandas as pd

def get_user_history(user_id: int, interactions_df: pd.DataFrame,
                      recipes_df: pd.DataFrame) -> pd.DataFrame:
    """Retorna o histórico de avaliações de um usuário, já com o nome da receita, ordenado da mais recente 
    para a mais antiga.

    Se o usuário não tiver nenhuma interação, retorna um DataFrame vazio com as colunas esperadas.
    """
    historico = interactions_df[interactions_df["user_id"] == user_id]

    colunas_vazias = ["recipe_id", "name", "rating", "date"]
    if historico.empty:
        return pd.DataFrame(columns=colunas_vazias)

    historico = historico.merge(
        recipes_df[["id", "name", "minutes", "tags"]],
        left_on="recipe_id", right_on="id", how="left"
    )
    historico = historico.sort_values("date", ascending=False)
    return historico[["recipe_id", "name", "rating", "date", "minutes", "tags"]].reset_index(drop=True)


def get_seen_recipe_ids(user_id: int, interactions_df: pd.DataFrame) -> set:
    """Retorna o conjunto de recipe_id que o usuário já avaliou (já viu)."""
    return set(interactions_df.loc[interactions_df["user_id"] == user_id, "recipe_id"])


def filter_seen_items(candidate_ids, user_id: int, interactions_df: pd.DataFrame) -> list:
    """Recebe uma lista de recipe_id candidatos e remove os que o usuário já avaliou."""
    vistos = get_seen_recipe_ids(user_id, interactions_df)
    return [rid for rid in candidate_ids if rid not in vistos]


def get_recipe_metadata(recipe_ids, recipes_df: pd.DataFrame) -> pd.DataFrame:
    """Retorna nome, tempo de preparo e tags para uma lista de recipe_id, na mesma ordem em que os ids foram passados."""
    metadados = recipes_df[recipes_df["id"].isin(recipe_ids)].copy()
    ordem = {rid: pos for pos, rid in enumerate(recipe_ids)}
    metadados["_ordem"] = metadados["id"].map(ordem)
    metadados = metadados.sort_values("_ordem").drop(columns="_ordem")
    return metadados.reset_index(drop=True)


def is_new_user(user_id: int, interactions_df: pd.DataFrame) -> bool:
    """Indica se o usuário é 'novo', usado para decidir se entra no fluxo normal ou no fluxo de cold start."""
    return user_id not in interactions_df["user_id"].values
