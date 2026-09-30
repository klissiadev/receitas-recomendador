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
from model_training.svd_training import build_dataset, run_cross_validation, train_svd, save_model, load_model
from model_training.knn_training import train_tfidf, save_artifacts, load_artifacts


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


# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

class RecipeRecommender:
    def __init__(self, algo: SVD, user_df: pd.DataFrame, recipe_df: pd.DataFrame):
        self.algo = algo
        self.user_df = user_df.copy()
        self.recipe_df = recipe_df.copy()
        self.all_recipe_ids = self.recipe_df['id'].unique()
        self.rating_stats = compute_rating_stats(self.user_df)

        self._stats = self._compute_weighted_scores()
        self.popular_recipe_ids = (
            self._stats.sort_values('weighted_score', ascending=False).index.tolist()
        )

    # -- helpers ----------------------------------------------------------

    def _to_frame(self, ids, predicted: dict | None = None) -> pd.DataFrame:
        """ids -> DataFrame com todos os campos do RecipeSummary.
        Mantem `recipe_id` por compatibilidade com o codigo antigo."""
        records = enrich_recipes(ids, self.recipe_df, self.rating_stats)
        for rec in records:
            rec['recipe_id'] = rec['id']
            if predicted is not None:
                rec['predicted_rating'] = round(float(predicted[rec['id']]), 2)
        return pd.DataFrame(records)

    def _compute_weighted_scores(self) -> pd.DataFrame:
        stats = self.user_df.groupby('recipe_id')['rating'].agg(['mean', 'count'])
        c = stats['count'].mean()
        m = stats['mean'].mean()
        stats['weighted_score'] = (
            (stats['count'] / (stats['count'] + c)) * stats['mean']
            + (c / (stats['count'] + c)) * m
        )
        return stats

    # -- construcao -------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path: str, user_df: pd.DataFrame,
                        recipe_df: pd.DataFrame) -> "RecipeRecommender":
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

    # -- consultas --------------------------------------------------------

    def has_history(self, user_id) -> bool:
        return user_id in set(self.user_df['user_id'])

    def get_user_history(self, user_id) -> pd.DataFrame:
        """Receitas avaliadas pelo usuario (campos completos + rating + date),
        da mais recente para a mais antiga."""
        cols = ['recipe_id', 'rating'] + (['date'] if 'date' in self.user_df.columns else [])
        rated = self.user_df.loc[self.user_df['user_id'] == user_id, cols]
        if rated.empty:
            return pd.DataFrame()

        sort_col = 'date' if 'date' in cols else 'rating'
        rated = rated.sort_values(sort_col, ascending=False)
        frame = self._to_frame(rated['recipe_id'].tolist())
        return rated.merge(frame, on='recipe_id').reset_index(drop=True)

    def get_popular_recommendations(self, n: int = 10, exclude: set | None = None) -> pd.DataFrame:
        exclude = exclude or set()
        top_ids = [rid for rid in self.popular_recipe_ids if rid not in exclude][:n]
        means = self._stats['mean'].to_dict()
        return self._to_frame(top_ids, predicted=means)

    def get_recommendations(self, user_id, n: int = 10) -> pd.DataFrame:
        already_rated = set(self.user_df.loc[self.user_df['user_id'] == user_id, 'recipe_id'])

        if not self.has_history(user_id):
            return self.get_popular_recommendations(n=n, exclude=already_rated)

        candidates = [rid for rid in self.all_recipe_ids if rid not in already_rated]
        scored = sorted(
            ((rid, self.algo.predict(user_id, rid).est) for rid in candidates),
            key=lambda x: x[1], reverse=True,
        )[:n]
        return self._to_frame([rid for rid, _ in scored], predicted=dict(scored))

    def add_rating(self, user_id, recipe_id, rating, date=None) -> None:
        """Adiciona (ou substitui) a nota do usuario para a receita, em memoria."""
        row = {'user_id': user_id, 'recipe_id': recipe_id, 'rating': rating}
        if 'date' in self.user_df.columns:
            row['date'] = date or pd.Timestamp.today().strftime('%Y-%m-%d')

        mask = (self.user_df['user_id'] == user_id) & (self.user_df['recipe_id'] == recipe_id)
        self.user_df = pd.concat([self.user_df[~mask], pd.DataFrame([row])], ignore_index=True)
        self.rating_stats = compute_rating_stats(self.user_df)
        # TODO: persistir tambem no CSV de interacoes


# ---------------------------------------------------------------------------
# Conteudo (TF-IDF)
# ---------------------------------------------------------------------------

class ContentRecommender:
    def __init__(self, content_matrix, id_to_row: pd.Series, recipe_ids: pd.Series,
                 recipe_lookup: pd.DataFrame,
                 recipe_df: pd.DataFrame | None = None,
                 rating_stats: pd.DataFrame | None = None):
        self.content_matrix = content_matrix
        self.id_to_row = id_to_row
        self.recipe_ids = recipe_ids
        self.recipe_lookup = recipe_lookup
        # opcionais: quando presentes, as saidas trazem todos os campos do dataset
        self.recipe_df = recipe_df
        self.rating_stats = rating_stats

    @classmethod
    def from_pretrained(cls, artifacts_path: str,
                        recipe_df: pd.DataFrame | None = None,
                        rating_stats: pd.DataFrame | None = None) -> "ContentRecommender":
        artifacts = load_artifacts(artifacts_path)
        return cls(
            artifacts['content_matrix'],
            artifacts['id_to_row'],
            artifacts['recipe_ids'],
            artifacts['recipe_lookup'],
            recipe_df=recipe_df,
            rating_stats=rating_stats,
        )

    @classmethod
    def train_new(
        cls,
        recipe_df: pd.DataFrame,
        max_features: int = 5000,
        stop_words: str = 'english',
        artifacts_path: str | None = None,
        rating_stats: pd.DataFrame | None = None,
    ) -> "ContentRecommender":
        content_matrix, _tfidf, id_to_row, recipe_lookup, recipe_ids = train_tfidf(
            recipe_df, max_features=max_features, stop_words=stop_words
        )
        if artifacts_path:
            save_artifacts(artifacts_path, content_matrix, id_to_row, recipe_ids, recipe_lookup)
        return cls(content_matrix, id_to_row, recipe_ids, recipe_lookup,
                   recipe_df=recipe_df, rating_stats=rating_stats)

    @classmethod
    def get_or_train(
        cls,
        artifacts_path: str,
        recipe_df: pd.DataFrame,
        max_features: int = 5000,
        stop_words: str = 'english',
        force_retrain: bool = False,
        rating_stats: pd.DataFrame | None = None,
    ) -> "ContentRecommender":
        if not force_retrain and os.path.exists(artifacts_path):
            return cls.from_pretrained(artifacts_path, recipe_df=recipe_df,
                                       rating_stats=rating_stats)
        return cls.train_new(
            recipe_df, max_features=max_features, stop_words=stop_words,
            artifacts_path=artifacts_path, rating_stats=rating_stats,
        )

    def get_similar_recipes(self, recipe_id, n: int = 10, exclude: set | None = None) -> pd.DataFrame:
        exclude = exclude or set()
        if recipe_id not in self.id_to_row.index:
            return pd.DataFrame(columns=['recipe_id', 'name', 'description', 'similarity'])

        row = self.id_to_row[recipe_id]
        sims = cosine_similarity(self.content_matrix[row], self.content_matrix).flatten()
        ranked_idx = sims.argsort()[::-1]

        ids, sim_by_id = [], {}
        for candidate_row in ranked_idx:
            candidate_id = self.recipe_ids.iloc[candidate_row]
            if candidate_id == recipe_id or candidate_id in exclude:
                continue
            ids.append(candidate_id)
            sim_by_id[candidate_id] = round(float(sims[candidate_row]), 3)
            if len(ids) == n:
                break

        # Sem recipe_df completo: fallback so com name/description
        if self.recipe_df is None:
            rows = [{
                'recipe_id': rid,
                'name': self.recipe_lookup.loc[rid, 'name'],
                'description': self.recipe_lookup.loc[rid, 'description'],
                'similarity': sim_by_id[rid],
            } for rid in ids]
            return pd.DataFrame(rows)

        records = enrich_recipes(ids, self.recipe_df, self.rating_stats)
        for rec in records:
            rec['recipe_id'] = rec['id']
            rec['similarity'] = sim_by_id[rec['id']]
        return pd.DataFrame(records)

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