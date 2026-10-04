import pandas as pd

from recommender_test import recommend_recipes, center_users
from utils import filter_seen_items, enrich_recipes, compute_rating_stats


class RecipeRecommender:

    def __init__(self, user_df: pd.DataFrame, recipe_df: pd.DataFrame,
                 k: int = 50, min_neighbors: int = 2, min_common: int = 2,
                 pearson: bool = False):
        self.user_df = user_df.copy()
        self.recipe_df = recipe_df.copy()
        self.all_recipe_ids = self.recipe_df['id'].unique()
        self.k = k
        self.min_neighbors = min_neighbors
        self.min_common = min_common
        self.pearson = pearson
 
        self.rating_stats = compute_rating_stats(self.user_df)
        self._refresh_popularity()
        self._build_index()
  
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
 
    def _refresh_popularity(self) -> None:
        self._stats = self._compute_weighted_scores()
        self.popular_recipe_ids = (
            self._stats.sort_values('weighted_score', ascending=False).index.tolist()
        )
 
    def _build_index(self) -> None:
        """{user: {recipe: nota}} e indice invertido {recipe: {users}}."""
        self.users = {}
        self.item_users = {}
        for u, r, s in zip(self.user_df['user_id'], self.user_df['recipe_id'],
                           self.user_df['rating']):
            self.users.setdefault(u, {})[r] = s
            self.item_users.setdefault(r, set()).add(u)
        self.sim_users = center_users(self.users) if self.pearson else None
 
    def _candidates(self, user_id) -> set:
        """Usuarios que compartilham ao menos uma receita com user_id (inclui ele mesmo).
        Evita comparar com usuarios sem nada em comum (similaridade seria 0 de qualquer jeito)."""
        cand = set()
        for recipe_id in self.users[user_id]:
            cand |= self.item_users[recipe_id]
        return cand
  
    def has_history(self, user_id) -> bool:
        return user_id in self.users
 
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
        already_rated = set(self.users.get(user_id, {}))
 
        # cold start: usuario sem historico
        if not self.has_history(user_id):
            return self.get_popular_recommendations(n=n, exclude=already_rated)
 
        cand = self._candidates(user_id)
        users_sub = {u: self.users[u] for u in cand}
        sim_sub = {u: self.sim_users[u] for u in cand} if self.pearson else None
 
        recs = recommend_recipes(
            user_id, users_sub, None, num_recommendations=n,
            k=self.k, min_neighbors=self.min_neighbors,
            min_common=self.min_common, sim_users=sim_sub,
        )
        ids = [rid for rid, _ in recs]
        predicted = dict(recs)
 
        if len(ids) < n:
            means = self._stats['mean'].to_dict()
            fill = [rid for rid in self.popular_recipe_ids
                    if rid not in already_rated and rid not in predicted][: n - len(ids)]
            for rid in fill:
                predicted[rid] = means[rid]
            ids += fill
 
        return self._to_frame(ids, predicted=predicted)
 
    def recommend_ids(self, user_id, n: int = 10, accept=None) -> list:
        if not self.has_history(user_id):
            return []
 
        cand = self._candidates(user_id)
        users_sub = {u: self.users[u] for u in cand}
        sim_sub = {u: self.sim_users[u] for u in cand} if self.pearson else None
 
        recs = recommend_recipes(
            user_id, users_sub, None, num_recommendations=10 ** 9,  # tudo, ja ordenado
            k=self.k, min_neighbors=self.min_neighbors,
            min_common=self.min_common, sim_users=sim_sub,
        )
        ids = [rid for rid, _ in recs if accept is None or accept(rid)]
        return ids[:n]
 
    def add_rating(self, user_id, recipe_id, rating, date=None) -> None:

        row = {'user_id': user_id, 'recipe_id': recipe_id, 'rating': rating}
        if 'date' in self.user_df.columns:
            row['date'] = date or pd.Timestamp.today().strftime('%Y-%m-%d')
 
        mask = (self.user_df['user_id'] == user_id) & (self.user_df['recipe_id'] == recipe_id)
        self.user_df = pd.concat([self.user_df[~mask], pd.DataFrame([row])], ignore_index=True)
 
        self.users.setdefault(user_id, {})[recipe_id] = rating
        self.item_users.setdefault(recipe_id, set()).add(user_id)
        if self.pearson:
            d = self.users[user_id]
            mean = sum(d.values()) / len(d)
            self.sim_users[user_id] = {r: s - mean for r, s in d.items()}
 
        self.rating_stats = compute_rating_stats(self.user_df)
        self._refresh_popularity()
