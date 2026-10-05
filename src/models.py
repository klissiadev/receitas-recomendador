"""
models.py
Modelos de recomendacao.

    - PopularityRecommender: baseline por popularidade (media bayesiana).
    - cold_start_recommendations: recomendacoes para usuario sem historico.
    - onboarding_recipes: receitas variadas para o usuario novo avaliar.
    - RecipeRecommender: SVD (com fallback de popularidade).
    - ContentRecommender: similaridade por conteudo (TF-IDF).

Todas as saidas voltadas ao front passam por `enrich_recipes`, entao trazem
os campos completos do dataset (minutes, tags, avg_rating, n_ratings,
description, ...).
"""

import os

import pandas as pd
from surprise import SVD
from sklearn.metrics.pairwise import cosine_similarity

from utils import filter_seen_items, enrich_recipes, compute_rating_stats

# ---------------------------------------------------------------------------
# Popularidade
# ---------------------------------------------------------------------------

class PopularityRecommender:
    """Recomenda as receitas mais bem avaliadas, usando media bayesiana para
    nao deixar receitas com 1 ou 2 notas altas dominarem o ranking.

    score(i) = (v / (v + m)) * R_i + (m / (v + m)) * C

    onde:
        v = numero de avaliacoes da receita i
        R_i = nota media da receita i
        C = nota media global (de todas as receitas)
        m = numero minimo de avaliacoes para a receita ser "confiavel"
    """

    def __init__(self, min_avaliacoes: int = 5):
        self.min_avaliacoes = min_avaliacoes
        self.ranking_ = None  # DataFrame ordenado por score, preenchido em fit()

    def fit(self, interactions_df: pd.DataFrame):
        """Calcula o ranking de popularidade das receitas com base nas interacoes."""
        stats = interactions_df.groupby("recipe_id")["rating"].agg(["mean", "count"])
        stats = stats.rename(columns={"mean": "nota_media", "count": "n_avaliacoes"})

        media_global = interactions_df["rating"].mean()
        m = self.min_avaliacoes

        stats["score"] = (
            (stats["n_avaliacoes"] / (stats["n_avaliacoes"] + m)) * stats["nota_media"]
            + (m / (stats["n_avaliacoes"] + m)) * media_global
        )

        self.ranking_ = stats.sort_values("score", ascending=False).reset_index()
        return self

    def recommend(self, user_id: int, interactions_df: pd.DataFrame,
                  recipes_df: pd.DataFrame | None = None,
                  rating_stats: pd.DataFrame | None = None,
                  n: int = 10, exclude_seen: bool = True) -> list:
        """Recomenda por popularidade, excluindo o que o usuario ja avaliou.

        Se `recipes_df` for passado, retorna lista de dicts completos
        (RecipeSummary). Sem ele, retorna apenas a lista de recipe_id.
        """
        if self.ranking_ is None:
            raise RuntimeError("Chame fit() antes de recommend().")

        candidatos = self.ranking_["recipe_id"].tolist()

        if exclude_seen:
            candidatos = filter_seen_items(candidatos, user_id, interactions_df)

        candidatos = candidatos[:n]
        if recipes_df is None:
            return candidatos
        return enrich_recipes(candidatos, recipes_df, rating_stats)


def cold_start_recommendations(recipes_df: pd.DataFrame, interactions_df: pd.DataFrame,
                               popularity_model: PopularityRecommender,
                               n: int = 10, tag_filter: str | None = None,
                               tags: list[str] | None = None,
                               rating_stats: pd.DataFrame | None = None) -> list[dict]:
    """Recomendacoes para um usuario sem NENHUM historico (cold start).

    Usa o ranking de popularidade ja treinado. Filtros opcionais:
        tag_filter: uma unica tag (compatibilidade com o codigo antigo)
        tags: varias tags; a receita precisa ter TODAS (igual ao front)
    Requer `recipes_df["tags"]` ja convertido em lista (ver preprocess).
    """
    ranking = popularity_model.ranking_["recipe_id"].tolist()

    required = set(tags or [])
    if tag_filter:
        required.add(tag_filter)

    if required:
        ids_com_tags = set(
            recipes_df.loc[
                recipes_df["tags"].apply(lambda t: required.issubset(set(t))), "id"
            ]
        )
        ranking = [rid for rid in ranking if rid in ids_com_tags]

    return enrich_recipes(ranking[:n], recipes_df, rating_stats)


def onboarding_recipes(recipes_df: pd.DataFrame, popularity_model: PopularityRecommender,
                       n: int = 8,
                       rating_stats: pd.DataFrame | None = None) -> list[dict]:
    """Seleciona receitas variadas e populares para o usuario novo avaliar.

    Estrategia: pega o topo do ranking e tenta variar as tags, para nao
    sugerir n receitas todas muito parecidas.
    """
    ranking = popularity_model.ranking_["recipe_id"].tolist()
    tags_by_id = dict(zip(recipes_df["id"], recipes_df["tags"]))  # evita .loc por receita

    selecionadas, tags_usadas = [], set()
    for rid in ranking:
        tags = set(tags_by_id.get(rid, []))
        if not tags & tags_usadas or len(selecionadas) < n // 2:
            selecionadas.append(rid)
            tags_usadas |= tags
        if len(selecionadas) == n:
            break

    # completa com o restante do ranking se ainda faltar (bases pequenas)
    for rid in ranking:
        if len(selecionadas) >= n:
            break
        if rid not in selecionadas:
            selecionadas.append(rid)

    return enrich_recipes(selecionadas[:n], recipes_df, rating_stats)