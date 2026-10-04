import pandas as pd

from cs import cosseno_alg
from utils import enrich_recipes, compute_rating_stats


class ItemRecipeRecommender:

    def __init__(self, user_df: pd.DataFrame, recipe_df: pd.DataFrame,
                 k: int = 100, min_common: int = 3, min_seed_rating: float | None = 4,
                 centered: bool = False):
        self.user_df = user_df.copy()
        self.recipe_df = recipe_df.copy()
        self.all_recipe_ids = self.recipe_df['id'].unique()
        self.k = k
        self.min_common = min_common
        self.min_seed_rating = min_seed_rating
        self.centered = centered

        self.rating_stats = compute_rating_stats(self.user_df)
        self._refresh_popularity()
        self._build_index()

    # -- helpers ----------------------------------------------------------

    def _to_frame(self, ids, predicted: dict | None = None) -> pd.DataFrame:
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
        self.users = {}
        self.items = {}
        for u, r, s in zip(self.user_df['user_id'], self.user_df['recipe_id'],
                           self.user_df['rating']):
            self.users.setdefault(u, {})[r] = s
            self.items.setdefault(r, {})[u] = s

        if self.centered:
            means = {u: sum(d.values()) / len(d) for u, d in self.users.items()}
            self.item_vecs = {
                r: {u: s - means[u] for u, s in d.items()} for r, d in self.items.items()
            }
        else:
            self.item_vecs = self.items  # mesmo objeto: add_rating ja atualiza os dois

        # receitas de cada usuario em ordem cronologica (por data; sem 'date', pela ordem das linhas)
        ordered = (self.user_df.sort_values('date') if 'date' in self.user_df.columns
                   else self.user_df)
        self.user_order = {}
        for u, r in zip(ordered['user_id'], ordered['recipe_id']):
            self.user_order.setdefault(u, []).append(r)

        self._neighbors = {}  # cache: recipe_id -> [(recipe_id, sim), ...]

    def _item_neighbors(self, item_id) -> list:
        if item_id in self._neighbors:
            return self._neighbors[item_id]
        if item_id not in self.item_vecs:
            return []

        vec_i = self.item_vecs[item_id]
        cand = set()
        for u in vec_i:
            cand.update(self.users[u])
        cand.discard(item_id)

        sims = []
        for j in cand:
            s = cosseno_alg(vec_i, self.item_vecs[j], self.min_common)
            if s > 0:
                sims.append((j, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        self._neighbors[item_id] = sims[: self.k]
        return self._neighbors[item_id]

    def _seed(self, user_id):
        seen = self.users[user_id]
        for r in reversed(self.user_order[user_id]):
            if self.min_seed_rating is None or seen[r] >= self.min_seed_rating:
                return r
        return None

    def _score_items(self, user_id) -> list:
        seed = self._seed(user_id)
        if seed is None:
            return []  # nenhuma receita bem avaliada: quem chama completa com populares
        seen = self.users[user_id]
        return [(i, s) for i, s in self._item_neighbors(seed) if i not in seen]

    # -- consultas --------------------------------------------------------

    def has_history(self, user_id) -> bool:
        return user_id in self.users

    def get_user_history(self, user_id) -> pd.DataFrame:
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

    def get_similar_recipes(self, recipe_id, n: int = 10, exclude: set | None = None) -> pd.DataFrame:
        """Receitas mais similares a uma receita especifica (com coluna 'similarity')."""
        exclude = exclude or set()
        sims = [(rid, s) for rid, s in self._item_neighbors(recipe_id) if rid not in exclude][:n]
        if not sims:
            return pd.DataFrame()
        frame = self._to_frame([rid for rid, _ in sims])
        frame['similarity'] = frame['recipe_id'].map(dict(sims)).round(3)
        return frame.sort_values('similarity', ascending=False).reset_index(drop=True)

    def get_recommendations(self, user_id, n: int = 10) -> pd.DataFrame:
        already_rated = set(self.users.get(user_id, {}))

        if not self.has_history(user_id):
            return self.get_popular_recommendations(n=n, exclude=already_rated)

        recs = self._score_items(user_id)[:n]
        ids = [rid for rid, _ in recs]
        sims = dict(recs)

        if len(ids) < n:
            fill = [rid for rid in self.popular_recipe_ids
                    if rid not in already_rated and rid not in sims][: n - len(ids)]
            ids += fill

        # aqui nao ha nota prevista: predicted_rating = media da receita; a ordem vem da similaridade
        means = self._stats['mean'].to_dict()
        frame = self._to_frame(ids, predicted=means)
        frame['similarity'] = frame['recipe_id'].map(sims).round(3)  # NaN nas completadas
        return frame

    def recommend_ids(self, user_id, n: int = 10, accept=None) -> list:
        if not self.has_history(user_id):
            return []
        ids = [rid for rid, _ in self._score_items(user_id) if accept is None or accept(rid)]
        return ids[:n]

    def add_rating(self, user_id, recipe_id, rating, date=None) -> None:
        row = {'user_id': user_id, 'recipe_id': recipe_id, 'rating': rating}
        if 'date' in self.user_df.columns:
            row['date'] = date or pd.Timestamp.today().strftime('%Y-%m-%d')

        mask = (self.user_df['user_id'] == user_id) & (self.user_df['recipe_id'] == recipe_id)
        self.user_df = pd.concat([self.user_df[~mask], pd.DataFrame([row])], ignore_index=True)

        self.users.setdefault(user_id, {})[recipe_id] = rating
        self.items.setdefault(recipe_id, {})[user_id] = rating
        order = self.user_order.setdefault(user_id, [])
        if recipe_id in order:
            order.remove(recipe_id)
        order.append(recipe_id) 

        if self.centered:
            d = self.users[user_id]
            mean = sum(d.values()) / len(d)
            for r, s in d.items():  
                self.item_vecs.setdefault(r, {})[user_id] = s - mean

        for r in self.users[user_id]:
            self._neighbors.pop(r, None)

        self.rating_stats = compute_rating_stats(self.user_df)
        self._refresh_popularity()