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


import os

import pandas as pd
from surprise import SVD

from model_training.svd_training import build_dataset, run_cross_validation, train_svd, save_model, load_model
from model_training.knn_training import train_tfidf, save_artifacts, load_artifacts
from sklearn.metrics.pairwise import cosine_similarity

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

class RecipeRecommender:
    def __init__(self, algo: SVD, user_df: pd.DataFrame, recipe_df: pd.DataFrame):
        self.algo = algo
        self.user_df = user_df.copy()
        self.recipe_df = recipe_df.copy()

        self.recipe_lookup = self.recipe_df.set_index('id')[['name', 'description']]
        self.all_recipe_ids = self.recipe_df['id'].unique()

        self._stats = self._compute_weighted_scores()
        self.popular_recipe_ids = (
            self._stats.sort_values('weighted_score', ascending=False).index.tolist()
        )

    def _compute_weighted_scores(self) -> pd.DataFrame:
        stats = self.user_df.groupby('recipe_id')['rating'].agg(['mean', 'count'])
        c = stats['count'].mean()
        m = stats['mean'].mean()
        stats['weighted_score'] = (
            (stats['count'] / (stats['count'] + c)) * stats['mean']
            + (c / (stats['count'] + c)) * m
        )
        return stats

    @classmethod
    def from_pretrained(cls, model_path: str, user_df: pd.DataFrame, recipe_df: pd.DataFrame) -> "RecipeRecommender":
        algo = load_model(model_path)
        return cls(algo, user_df, recipe_df)

    @classmethod
    def train_new(
        cls,
        user_df: pd.DataFrame,
        recipe_df: pd.DataFrame,
        n_factors: int = 50,
        model_path: str | None = None,
        run_cv: bool = True,
    ) -> "RecipeRecommender":
        data = build_dataset(user_df)
        if run_cv:
            run_cross_validation(data, n_factors=n_factors)
        algo, _ = train_svd(data, n_factors=n_factors)
        if model_path:
            save_model(algo, model_path)
        return cls(algo, user_df, recipe_df)

    @classmethod
    def get_or_train(
        cls,
        model_path: str,
        user_df: pd.DataFrame,
        recipe_df: pd.DataFrame,
        n_factors: int = 50,
        force_retrain: bool = False,
    ) -> "RecipeRecommender":
        if not force_retrain and os.path.exists(model_path):
            return cls.from_pretrained(model_path, user_df, recipe_df)
        return cls.train_new(user_df, recipe_df, n_factors=n_factors, model_path=model_path)

    def has_history(self, user_id) -> bool:
        return user_id in set(self.user_df['user_id'])

    def get_user_history(self, user_id) -> pd.DataFrame:
        rated = self.user_df.loc[self.user_df['user_id'] == user_id, ['recipe_id', 'rating']]
        if rated.empty:
            return pd.DataFrame(columns=['recipe_id', 'name', 'description', 'rating'])

        rated = rated.merge(self.recipe_lookup, left_on='recipe_id', right_index=True)
        return rated[['recipe_id', 'name', 'description', 'rating']].sort_values(
            'rating', ascending=False
        )

    def get_popular_recommendations(self, n: int = 10, exclude: set | None = None) -> pd.DataFrame:
        exclude = exclude or set()
        top_ids = [rid for rid in self.popular_recipe_ids if rid not in exclude][:n]

        rows = []
        for recipe_id in top_ids:
            meta = self.recipe_lookup.loc[recipe_id]
            rows.append({
                'recipe_id': recipe_id,
                'name': meta['name'],
                'description': meta['description'],
                'predicted_rating': round(self._stats.loc[recipe_id, 'mean'], 2),
            })
        return pd.DataFrame(rows)

    def get_recommendations(self, user_id, n: int = 10) -> pd.DataFrame:
        already_rated = set(self.user_df.loc[self.user_df['user_id'] == user_id, 'recipe_id'])

        if not self.has_history(user_id):
            return self.get_popular_recommendations(n=n, exclude=already_rated)

        candidates = [rid for rid in self.all_recipe_ids if rid not in already_rated]
        scored = [(rid, self.algo.predict(user_id, rid).est) for rid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_n = scored[:n]

        rows = []
        for recipe_id, predicted_rating in top_n:
            meta = self.recipe_lookup.loc[recipe_id]
            rows.append({
                'recipe_id': recipe_id,
                'name': meta['name'],
                'description': meta['description'],
                'predicted_rating': round(predicted_rating, 2),
            })
        return pd.DataFrame(rows)

    def add_rating(self, user_id, recipe_id, rating) -> None:
        new_row = pd.DataFrame([{'user_id': user_id, 'recipe_id': recipe_id, 'rating': rating}])
        self.user_df = pd.concat([self.user_df, new_row], ignore_index=True)
        # adicionar no cvs de interations dps..


class ContentRecommender:
    def __init__(self, content_matrix, id_to_row: pd.Series, recipe_ids: pd.Series, recipe_lookup: pd.DataFrame):
        self.content_matrix = content_matrix
        self.id_to_row = id_to_row
        self.recipe_ids = recipe_ids 
        self.recipe_lookup = recipe_lookup

    @classmethod
    def from_pretrained(cls, artifacts_path: str) -> "ContentRecommender":
        artifacts = load_artifacts(artifacts_path)
        return cls(
            artifacts['content_matrix'],
            artifacts['id_to_row'],
            artifacts['recipe_ids'],
            artifacts['recipe_lookup'],
        )

    @classmethod
    def train_new(
        cls,
        recipe_df: pd.DataFrame,
        max_features: int = 5000,
        stop_words: str = 'english',
        artifacts_path: str | None = None,
    ) -> "ContentRecommender":
        content_matrix, _tfidf, id_to_row, recipe_lookup, recipe_ids = train_tfidf(
            recipe_df, max_features=max_features, stop_words=stop_words
        )
        if artifacts_path:
            save_artifacts(artifacts_path, content_matrix, id_to_row, recipe_ids, recipe_lookup)
        return cls(content_matrix, id_to_row, recipe_ids, recipe_lookup)

    @classmethod
    def get_or_train(
        cls,
        artifacts_path: str,
        recipe_df: pd.DataFrame,
        max_features: int = 5000,
        stop_words: str = 'english',
        force_retrain: bool = False,
    ) -> "ContentRecommender":
        if not force_retrain and os.path.exists(artifacts_path):
            return cls.from_pretrained(artifacts_path)
        return cls.train_new(
            recipe_df, max_features=max_features, stop_words=stop_words, artifacts_path=artifacts_path
        )

    def get_similar_recipes(self, recipe_id, n: int = 10, exclude: set | None = None) -> pd.DataFrame:
        exclude = exclude or set()
        if recipe_id not in self.id_to_row.index:
            return pd.DataFrame(columns=['recipe_id', 'name', 'description', 'similarity'])

        row = self.id_to_row[recipe_id]
        sims = cosine_similarity(self.content_matrix[row], self.content_matrix).flatten()
        ranked_idx = sims.argsort()[::-1]

        rows = []
        for candidate_row in ranked_idx:
            candidate_id = self.recipe_ids.iloc[candidate_row]
            if candidate_id == recipe_id or candidate_id in exclude:
                continue
            meta = self.recipe_lookup.loc[candidate_id]
            rows.append({
                'recipe_id': candidate_id,
                'name': meta['name'],
                'description': meta['description'],
                'similarity': round(float(sims[candidate_row]), 3),
            })
            if len(rows) == n:
                break
        return pd.DataFrame(rows)

    
    def print_similar_recipes(self, recipe_id, n: int = 5, exclude: set | None = None) -> None:
        if recipe_id not in self.recipe_lookup.index:
            print(f"Recipe {recipe_id} not found in recipe_df.")
            return

        query = self.recipe_lookup.loc[recipe_id]
        print(f"\nReceita base: {query['name']}")
        similar = self.get_similar_recipes(recipe_id, n=n, exclude=exclude)
        if similar.empty:
            print("Nenhuma receita similar encontrada.")
            return

        print("\nReceitas similares:")
        for _, row in similar.iterrows():
            print(f"\n  {row['name']}  (similaridade: {row['similarity']})")