"""
Modelos de recomendacao.

Neste arquivo (Integrante B):
    - PopularityRecommender: baseline por popularidade (media bayesiana).
    - cold_start_recommendations: recomendacoes para usuario sem historico.
    - onboarding_recipes: receitas variadas para o usuario novo avaliar.

Ficam para a Integrante A (nao mexer aqui):
    - Item-based KNN
    - SVD
"""

import pandas as pd

from utils import filter_seen_items, get_recipe_metadata


class PopularityRecommender:
    """Recomenda as receitas mais bem avaliadas, usando media bayesiana para nao deixar receitas com 1 ou 2 
    notas altas dominarem o ranking.

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

        # Calcula a média e o número de avaliações por receita
        stats = interactions_df.groupby("recipe_id")["rating"].agg(["mean", "count"])
        stats = stats.rename(columns={"mean": "nota_media", "count": "n_avaliacoes"})

        # Calcula o score bayesiano para cada receita
        media_global = interactions_df["rating"].mean()
        m = self.min_avaliacoes

        stats["score"] = (
            (stats["n_avaliacoes"] / (stats["n_avaliacoes"] + m)) * stats["nota_media"]
            + (m / (stats["n_avaliacoes"] + m)) * media_global
        )

        self.ranking_ = stats.sort_values("score", ascending=False).reset_index()
        return self

    def recommend(self, user_id: int, interactions_df: pd.DataFrame,
                n: int = 10, exclude_seen: bool = True) -> list:
        """Retorna uma lista de recipe_id recomendados por popularidade, excluindo os que o usuario ja avaliou"""

        if self.ranking_ is None:
            raise RuntimeError("Chame fit() antes de recommend().")

        candidatos = self.ranking_["recipe_id"].tolist()

        if exclude_seen:
            candidatos = filter_seen_items(candidatos, user_id, interactions_df)

        return candidatos[:n]


def cold_start_recommendations(recipes_df: pd.DataFrame, interactions_df: pd.DataFrame,
                                popularity_model: PopularityRecommender,
                                n: int = 10, tag_filter: str = None) -> list:
    """Recomendacoes para um usuario sem NENHUM historico (cold start).

    Usa o ranking de popularidade ja treinado e, opcionalmente, filtra por uma tag (ex.: 'vegetariano', 'sobremesa') 
    caso o usuario informe uma preferencia na tela de boas-vindas.
    """
    ranking = popularity_model.ranking_["recipe_id"].tolist()

    if tag_filter:
        ids_com_tag = set(
            recipes_df.loc[recipes_df["tags"].apply(lambda tags: tag_filter in tags), "id"] 
        )
        ranking = [rid for rid in ranking if rid in ids_com_tag]

    return ranking[:n]


def onboarding_recipes(recipes_df: pd.DataFrame, popularity_model: PopularityRecommender,
                        n: int = 8) -> list:
    """Seleciona receitas variadas e populares para o usuario novo avaliar na tela de onboarding
    (passo que da inicio ao historico dele).

    Estrategia simples: pega o top popular e tenta variar as tags, para nao sugerir 8 receitas todas muito parecidas.
    """
    ranking = popularity_model.ranking_["recipe_id"].tolist()

    selecionadas = []
    tags_usadas = set()

    for recipe_id in ranking:
        tags_receita = set(recipes_df.loc[recipes_df["id"] == recipe_id, "tags"].iloc[0])
        if not tags_receita & tags_usadas or len(selecionadas) < n // 2:
            selecionadas.append(recipe_id)
            tags_usadas |= tags_receita
        if len(selecionadas) == n:
            break

    # completa com o restante do ranking se ainda faltar (bases pequenas)
    if len(selecionadas) < n:
        for recipe_id in ranking:
            if recipe_id not in selecionadas:
                selecionadas.append(recipe_id)
            if len(selecionadas) == n:
                break

    return selecionadas[:n]
