
"""
Modelos de recomendacao.

Neste arquivo:
    - PopularityRecommender: baseline por popularidade (media bayesiana).
    - cold_start_recommendations: recomendacoes para usuario sem historico.
    - onboarding_recipes: receitas variadas para o usuario novo avaliar.
    - RecipeRecommender: SVD (filtragem colaborativa) + fold-in + fallback por popularidade.
    - ContentRecommender: receitas similares por conteudo (TF-IDF).

Padrao de saida: toda receita devolvida ao front e um objeto `Recipe`
(id, name, description, tags, nutrition, steps + predicted_rating /
user_rating / similarity / source, conforme o contexto).
"""

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD

from model_training.knn_training import (
    train_tfidf,
    save_artifacts,
    load_artifacts,
)
from model_training.svd_training import (
    build_dataset,
    run_cross_validation,
    train_svd,
    save_model,
    load_model,
)
from recipe import Recipe
from utils import filter_seen_items


DEFAULT_INTERACTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "RAW_interactions.csv"
)

RECIPE_COLUMNS = [
    "name",
    "description",
    "tags",
    "nutrition",
    "steps",
]


def build_recipe(lookup: pd.DataFrame, recipe_id, **extra) -> Recipe:
    """
    Constroi um Recipe padronizado a partir de uma tabela indexada por recipe_id.

    Todos os recommenders devem usar esta funcao para garantir que a receita
    retornada contenha os mesmos campos de metadata.
    """

    if recipe_id not in lookup.index:
        raise KeyError(f"Recipe {recipe_id} not found in recipe lookup.")

    meta = lookup.loc[recipe_id]

    def as_list(column):
        value = meta.get(column)

        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)

        if pd.isna(value) if not isinstance(value, (dict, set)) else False:
            return []

        return []

    def as_text(column):
        value = meta.get(column)

        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if pd.isna(value):
            return ""

        return str(value)

    recipe_data = {
        "id": int(recipe_id),
        "name": as_text("name"),
        "description": as_text("description"),
        "tags": as_list("tags"),
        "nutrition": as_list("nutrition"),
        "steps": as_list("steps"),
    }

    # Campos adicionais, como predicted_rating, similarity,
    # user_rating e source.
    recipe_data.update(extra)

    return Recipe(**recipe_data)


def build_recipes(
    lookup: pd.DataFrame,
    recipe_ids,
    **extra_columns,
) -> list[Recipe]:
    """
    Constrói varias receitas usando o mesmo padrao.

    `extra_columns` pode conter listas alinhadas a recipe_ids.
    Exemplo:
        build_recipes(
            lookup,
            [1, 2, 3],
            predicted_rating=[4.5, 4.2, 4.0],
            source="popular",
        )
    """

    recipe_ids = list(recipe_ids)
    results = []

    for index, recipe_id in enumerate(recipe_ids):
        extra = {}

        for column, values in extra_columns.items():
            if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
                extra[column] = values[index]
            else:
                extra[column] = values

        results.append(
            build_recipe(
                lookup,
                recipe_id,
                **extra,
            )
        )

    return results


class PopularityRecommender:
    """
    Recomenda as receitas mais bem avaliadas usando media bayesiana.

    score(i) = (v / (v + m)) * R_i + (m / (v + m)) * C

    onde:
        v = numero de avaliacoes da receita i
        R_i = nota media da receita i
        C = nota media global
        m = numero minimo de avaliacoes para a receita ser "confiavel"
    """

    def __init__(
        self,
        min_avaliacoes: int = 5,
        recipe_df: pd.DataFrame | None = None,
    ):
        self.min_avaliacoes = min_avaliacoes
        self.ranking_ = None

        self.recipe_lookup = None

        if recipe_df is not None:
            self.recipe_lookup = recipe_df.set_index("id")[RECIPE_COLUMNS]

    def fit(
        self,
        interactions_df: pd.DataFrame,
        recipe_df: pd.DataFrame | None = None,
    ):
        """Calcula o ranking de popularidade das receitas."""

        if recipe_df is not None:
            self.recipe_lookup = recipe_df.set_index("id")[RECIPE_COLUMNS]

        if self.recipe_lookup is None:
            raise ValueError(
                "recipe_df e necessario para que o recommender "
                "retorne objetos Recipe completos."
            )

        stats = (
            interactions_df
            .groupby("recipe_id")["rating"]
            .agg(["mean", "count"])
            .rename(
                columns={
                    "mean": "nota_media",
                    "count": "n_avaliacoes",
                }
            )
        )

        media_global = interactions_df["rating"].mean()
        m = self.min_avaliacoes

        stats["score"] = (
            (
                stats["n_avaliacoes"]
                / (stats["n_avaliacoes"] + m)
            )
            * stats["nota_media"]
            + (
                m
                / (stats["n_avaliacoes"] + m)
            )
            * media_global
        )

        self.ranking_ = (
            stats
            .sort_values("score", ascending=False)
            .reset_index()
        )

        return self

    def recommend(
        self,
        user_id: int,
        interactions_df: pd.DataFrame,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[Recipe]:
        """Retorna receitas completas recomendadas por popularidade."""

        if self.ranking_ is None:
            raise RuntimeError("Chame fit() antes de recommend().")

        recipe_ids = self.ranking_["recipe_id"].tolist()

        if exclude_seen:
            recipe_ids = filter_seen_items(
                recipe_ids,
                user_id,
                interactions_df,
            )

        recipe_ids = [
            rid
            for rid in recipe_ids
            if rid in self.recipe_lookup.index
        ][:n]

        predicted_ratings = [
            round(
                float(
                    self.ranking_.loc[
                        self.ranking_["recipe_id"] == rid,
                        "nota_media",
                    ].iloc[0]
                ),
                2,
            )
            for rid in recipe_ids
        ]

        return build_recipes(
            self.recipe_lookup,
            recipe_ids,
            predicted_rating=predicted_ratings,
            source="popular",
        )


def cold_start_recommendations(
    recipes_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    popularity_model: PopularityRecommender,
    n: int = 10,
    tag_filter: str = None,
) -> list[Recipe]:
    """Retorna receitas completas para usuarios sem historico."""

    ranking = popularity_model.ranking_["recipe_id"].tolist()

    if tag_filter:
        ids_com_tag = set(
            recipes_df.loc[
                recipes_df["tags"].apply(
                    lambda tags: tag_filter in tags
                    if isinstance(tags, (list, tuple, set))
                    else False
                ),
                "id",
            ]
        )

        ranking = [
            rid
            for rid in ranking
            if rid in ids_com_tag
        ]

    lookup = recipes_df.set_index("id")[RECIPE_COLUMNS]

    ranking = [
        rid
        for rid in ranking
        if rid in lookup.index
    ][:n]

    return build_recipes(
        lookup,
        ranking,
        source="cold_start",
    )


def onboarding_recipes(
    recipes_df: pd.DataFrame,
    popularity_model: PopularityRecommender,
    n: int = 8,
) -> list[Recipe]:
    """Retorna receitas variadas para o usuario avaliar no onboarding."""

    ranking = popularity_model.ranking_["recipe_id"].tolist()

    tags_by_id = recipes_df.set_index("id")["tags"].to_dict()

    selected = []
    tags_used = set()

    for recipe_id in ranking:
        if recipe_id not in tags_by_id:
            continue

        tags = tags_by_id[recipe_id]

        if not isinstance(tags, (list, tuple, set)):
            tags = []

        recipe_tags = set(tags)

        if not recipe_tags & tags_used or len(selected) < n // 2:
            selected.append(recipe_id)
            tags_used.update(recipe_tags)

        if len(selected) == n:
            break

    # Completa com o restante do ranking caso necessario.
    if len(selected) < n:
        for recipe_id in ranking:
            if recipe_id not in selected:
                selected.append(recipe_id)

            if len(selected) == n:
                break

    lookup = recipes_df.set_index("id")[RECIPE_COLUMNS]

    selected = [
        rid
        for rid in selected
        if rid in lookup.index
    ][:n]

    return build_recipes(
        lookup,
        selected,
        source="onboarding",
    )


class RecipeRecommender:
    def __init__(
        self,
        algo: SVD,
        user_df: pd.DataFrame,
        recipe_df: pd.DataFrame,
        interactions_path: Path | str = DEFAULT_INTERACTIONS_PATH,
        retrain_every: int | None = 50,
    ):
        self.algo = algo
        self.user_df = user_df.copy()
        self.recipe_df = recipe_df.copy()

        self.recipe_lookup = (
            self.recipe_df
            .set_index("id")[RECIPE_COLUMNS]
        )

        self.all_recipe_ids = self.recipe_df["id"].unique()

        self._stats = self._compute_weighted_scores()

        self.popular_recipe_ids = (
            self._stats
            .sort_values(
                "weighted_score",
                ascending=False,
            )
            .index
            .tolist()
        )

        self.interactions_path = Path(interactions_path)
        self.retrain_every = retrain_every
        self._pending_ratings = 0

    def _build_recipe(self, recipe_id, **extra) -> Recipe:
        return build_recipe(
            self.recipe_lookup,
            recipe_id,
            **extra,
        )

    def _compute_weighted_scores(self) -> pd.DataFrame:
        stats = (
            self.user_df
            .groupby("recipe_id")["rating"]
            .agg(["mean", "count"])
        )

        m = stats["count"].mean()
        c = stats["mean"].mean()

        stats["weighted_score"] = (
            (
                stats["count"]
                / (stats["count"] + m)
            )
            * stats["mean"]
            + (
                m
                / (stats["count"] + m)
            )
            * c
        )

        return stats

    def _fold_in_scores(self, user_id, reg=1.0):
        """Calcula scores usando as avaliacoes atuais do usuario."""

        ts = self.algo.trainset
        algo = self.algo

        rows = self.user_df[
            self.user_df["user_id"] == user_id
        ]

        X, y = [], []

        for recipe_id, rating in zip(
            rows["recipe_id"],
            rows["rating"],
        ):
            try:
                inner_id = ts.to_inner_iid(recipe_id)
            except ValueError:
                continue

            X.append(
                np.r_[
                    1.0,
                    algo.qi[inner_id],
                ]
            )

            y.append(
                rating
                - ts.global_mean
                - algo.bi[inner_id]
            )

        if not X:
            return None

        X = np.array(X)
        y = np.array(y)

        weights = np.linalg.solve(
            X.T @ X + reg * np.eye(X.shape[1]),
            X.T @ y,
        )

        return (
            ts.global_mean
            + algo.bi
            + weights[0]
            + algo.qi @ weights[1:]
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        user_df: pd.DataFrame,
        recipe_df: pd.DataFrame,
    ) -> "RecipeRecommender":
        algo = load_model(model_path)

        return cls(
            algo,
            user_df,
            recipe_df,
        )

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
            run_cross_validation(
                data,
                n_factors=n_factors,
            )

        algo, _ = train_svd(
            data,
            n_factors=n_factors,
        )

        if model_path:
            save_model(
                algo,
                model_path,
            )

        return cls(
            algo,
            user_df,
            recipe_df,
        )

    @classmethod
    def get_or_train(
        cls,
        model_path: str,
        user_df: pd.DataFrame,
        recipe_df: pd.DataFrame,
        n_factors: int = 50,
        force_retrain: bool = False,
    ) -> "RecipeRecommender":

        if (
            not force_retrain
            and os.path.exists(model_path)
        ):
            return cls.from_pretrained(
                model_path,
                user_df,
                recipe_df,
            )

        return cls.train_new(
            user_df,
            recipe_df,
            n_factors=n_factors,
            model_path=model_path,
        )

    def retrain(self) -> None:
        data = build_dataset(self.user_df)

        self.algo, _ = train_svd(
            data,
            n_factors=self.algo.n_factors,
        )

        self._pending_ratings = 0

    def has_history(self, user_id) -> bool:
        return bool(
            (
                self.user_df["user_id"] == user_id
            ).any()
        )

    def get_user_history(
        self,
        user_id,
    ) -> list[Recipe]:

        rated = self.user_df.loc[
            self.user_df["user_id"] == user_id,
            ["recipe_id", "rating"],
        ]

        rated = rated[
            rated["recipe_id"].isin(
                self.recipe_lookup.index
            )
        ]

        rated = rated.sort_values(
            "rating",
            ascending=False,
        )

        return [
            self._build_recipe(
                recipe_id,
                user_rating=int(rating),
                source="history",
            )
            for recipe_id, rating
            in zip(
                rated["recipe_id"],
                rated["rating"],
            )
        ]

    def get_popular_recommendations(
        self,
        n: int = 10,
        exclude: set | None = None,
    ) -> list[Recipe]:

        exclude = exclude or set()

        recipe_ids = [
            recipe_id
            for recipe_id in self.popular_recipe_ids
            if (
                recipe_id not in exclude
                and recipe_id in self.recipe_lookup.index
            )
        ][:n]

        return [
            self._build_recipe(
                recipe_id,
                predicted_rating=round(
                    float(
                        self._stats.loc[
                            recipe_id,
                            "mean",
                        ]
                    ),
                    2,
                ),
                source="popular",
            )
            for recipe_id in recipe_ids
        ]

    def get_recommendations(
        self,
        user_id,
        n: int = 10,
    ) -> list[Recipe]:

        already_rated = set(
            self.user_df.loc[
                self.user_df["user_id"] == user_id,
                "recipe_id",
            ]
        )

        scores = self._fold_in_scores(user_id)

        if scores is None:
            return self.get_popular_recommendations(
                n=n,
                exclude=already_rated,
            )

        trainset = self.algo.trainset
        top_n = []

        for inner_id in np.argsort(-scores):
            recipe_id = trainset.to_raw_iid(inner_id)

            if (
                recipe_id in already_rated
                or recipe_id not in self.recipe_lookup.index
            ):
                continue

            top_n.append(
                (
                    recipe_id,
                    scores[inner_id],
                )
            )

            if len(top_n) == n:
                break

        return [
            self._build_recipe(
                recipe_id,
                predicted_rating=round(
                    float(
                        np.clip(
                            score,
                            1,
                            5,
                        )
                    ),
                    2,
                ),
                source="personalized",
            )
            for recipe_id, score in top_n
        ]

    def add_rating(
        self,
        user_id,
        recipe_id,
        rating,
        review: str = "",
    ) -> None:

        if not 1 <= rating <= 5:
            raise ValueError(
                "rating must be between 1 and 5"
            )

        if recipe_id not in self.recipe_lookup.index:
            raise ValueError(
                f"unknown recipe_id: {recipe_id}"
            )

        mask = (
            (self.user_df["user_id"] == user_id)
            & (
                self.user_df["recipe_id"]
                == recipe_id
            )
        )

        new_row = pd.DataFrame(
            [
                {
                    "user_id": user_id,
                    "recipe_id": recipe_id,
                    "rating": rating,
                }
            ]
        )

        self.user_df = pd.concat(
            [
                self.user_df[~mask],
                new_row,
            ],
            ignore_index=True,
        )

        pd.DataFrame(
            [
                {
                    "user_id": user_id,
                    "recipe_id": recipe_id,
                    "date": date.today().isoformat(),
                    "rating": rating,
                    "review": review,
                }
            ]
        ).to_csv(
            self.interactions_path,
            mode="a",
            header=False,
            index=False,
        )

        self._pending_ratings += 1

        if (
            self.retrain_every
            and self._pending_ratings
            >= self.retrain_every
        ):
            self.retrain()


class ContentRecommender:
    def __init__(
        self,
        content_matrix,
        id_to_row: pd.Series,
        recipe_ids: pd.Series,
        recipe_lookup: pd.DataFrame,
    ):
        self.content_matrix = content_matrix
        self.id_to_row = id_to_row
        self.recipe_ids = recipe_ids
        self.recipe_lookup = recipe_lookup

        missing = [
            column
            for column in RECIPE_COLUMNS
            if column not in self.recipe_lookup.columns
        ]

        if missing:
            raise ValueError(
                f"recipe_lookup sem as colunas {missing}."
            )

    def complete_recipe_lookup(
        recipe_lookup: pd.DataFrame,
        recipe_df: pd.DataFrame,
    ) -> pd.DataFrame:

        full_lookup = recipe_df.set_index("id")[RECIPE_COLUMNS]

        # Mantém a ordem/IDs usados pelo TF-IDF
        recipe_lookup = full_lookup.loc[
            recipe_lookup.index.intersection(full_lookup.index)
        ].copy()

        return recipe_lookup
    
    @classmethod
    def from_pretrained(
        cls,
        artifacts_path: str,
        recipe_df: pd.DataFrame,
    ) -> "ContentRecommender":

        artifacts = load_artifacts(artifacts_path)

        # Recupera todas as informações da receita
        # diretamente do recipe_df completo.
        recipe_lookup = recipe_df.set_index("id")[RECIPE_COLUMNS]

        return cls(
            artifacts["content_matrix"],
            artifacts["id_to_row"],
            artifacts["recipe_ids"],
            recipe_lookup,
        )

    @classmethod
    def train_new(
        cls,
        recipe_df: pd.DataFrame,
        max_features: int = 5000,
        stop_words: str = "english",
        artifacts_path: str | None = None,
    ) -> "ContentRecommender":

        (
            content_matrix,
            _tfidf,
            id_to_row,
            _recipe_lookup,
            recipe_ids,
        ) = train_tfidf(
            recipe_df,
            max_features=max_features,
            stop_words=stop_words,
        )

        # Usa o DataFrame completo para os dados retornados
        # pelo recommender.
        recipe_lookup = recipe_df.set_index("id")[RECIPE_COLUMNS]

        if artifacts_path:
            save_artifacts(
                artifacts_path,
                content_matrix,
                id_to_row,
                recipe_ids,
                recipe_lookup,
            )

        return cls(
            content_matrix,
            id_to_row,
            recipe_ids,
            recipe_lookup,
        )

        return cls(
            content_matrix,
            id_to_row,
            recipe_ids,
            recipe_lookup,
        )

    @classmethod
    def get_or_train(
        cls,
        artifacts_path: str,
        recipe_df: pd.DataFrame,
        max_features: int = 5000,
        stop_words: str = "english",
        force_retrain: bool = False,
    ) -> "ContentRecommender":

        if (
            not force_retrain
            and os.path.exists(artifacts_path)
        ):
            return cls.from_pretrained(
                artifacts_path,
                recipe_df,
            )

        return cls.train_new(
            recipe_df,
            max_features=max_features,
            stop_words=stop_words,
            artifacts_path=artifacts_path,
        )

    def get_similar_recipes(
        self,
        recipe_id,
        n: int = 10,
        exclude: set | None = None,
    ) -> list[Recipe]:

        exclude = exclude or set()

        if recipe_id not in self.id_to_row.index:
            return []

        row = self.id_to_row[recipe_id]

        similarities = cosine_similarity(
            self.content_matrix[row],
            self.content_matrix,
        ).flatten()

        ranked_indices = similarities.argsort()[::-1]

        results = []

        for candidate_row in ranked_indices:
            candidate_id = self.recipe_ids.iloc[
                candidate_row
            ]

            if (
                candidate_id == recipe_id
                or candidate_id in exclude
            ):
                continue

            if candidate_id not in self.recipe_lookup.index:
                continue

            results.append(
                build_recipe(
                    self.recipe_lookup,
                    candidate_id,
                    source="content",
                )
            )

            if len(results) == n:
                break

        return results

    def print_similar_recipes(
        self,
        recipe_id,
        n: int = 5,
        exclude: set | None = None,
    ) -> None:

        if recipe_id not in self.recipe_lookup.index:
            print(
                f"Recipe {recipe_id} not found in recipe_df."
            )
            return

        query = self.recipe_lookup.loc[
            recipe_id
        ]

        print(
            f"\nReceita base: {query['name']}"
        )

        similar = self.get_similar_recipes(
            recipe_id,
            n=n,
            exclude=exclude,
        )

        if not similar:
            print(
                "Nenhuma receita similar encontrada."
            )
            return

        print("\nReceitas similares:")

        for recipe in similar:
            print(
                f"\n  {recipe.name}"
                f"  (similaridade: {recipe.similarity})"
            )
