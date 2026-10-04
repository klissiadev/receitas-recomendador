import random
import math as m
from preprocess import get_data, split_leave_last_out  

INTERACTIONS_PATH = '../data/RAW_interactions.csv'
RECIPES_PATH = '../data/RAW_recipes.csv'

N_USERS_TEST = 50     
TOP_N = 20
K = 50
MIN_NEIGHBORS = 2
MIN_COMMON = 2


def cosseno(rating1, rating2):
 
    dot_product = sum(a * b for a, b in zip(rating1, rating2))
    norm1 = sum(a ** 2 for a in rating1) ** 0.5
    norm2 = sum(b ** 2 for b in rating2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)
 
 
def cosseno_alg(rating1, rating2, min_common=2): 
    xy = 0
    sum_x2 = 0
    sum_y2 = 0
    n_common = 0
 
    for key in rating1:
        if key in rating2:
            xy += rating1[key] * rating2[key]
            sum_x2 += rating1[key] ** 2
            sum_y2 += rating2[key] ** 2
            n_common += 1
 
    if n_common < min_common:
        return 0.0
 
    x = m.sqrt(sum_x2)
    y = m.sqrt(sum_y2)
 
    if x * y == 0:
        return 0.0
    return xy / (x * y)
 
 
def center_users(users):
    return {
        u: {r: s - sum(d.values()) / len(d) for r, s in d.items()}
        for u, d in users.items()
    }

def similaridade(rating1, rating2, min_common=2, alpha=None):
    return cosseno_alg(rating1, rating2, min_common)
 
 
def compute_nearest_neighbors(user_id, users, k=50, min_common=2):  
    distances = []
    for user in users:
        if user != user_id:
            distance = cosseno_alg(users[user_id], users[user], min_common)
            if distance > 0:  # MUDANCA: so vizinhos com similaridade positiva
                distances.append((user, distance))
 
    distances.sort(key=lambda x: x[1], reverse=True)
    return distances[:k]  
 
 
def recommend_recipes(user_id, users, recipes, num_recommendations=5,
                      k=50, min_neighbors=2, min_common=2, sim_users=None):

    nearest_neighbors = compute_nearest_neighbors(
        user_id, sim_users if sim_users is not None else users, k, min_common
    )
    recommended_recipes = {}
    for neighbor_id, sim in nearest_neighbors:
        for recipe_id, rating in users[neighbor_id].items():
            if recipe_id not in users[user_id]:
                if recipe_id not in recommended_recipes:
                    recommended_recipes[recipe_id] = []
                recommended_recipes[recipe_id].append((sim, rating))  # MUDANCA: guarda a similaridade
 
    averaged_recommendations = {
        recipe_id: sum(s * r for s, r in pares) / sum(s for s, _ in pares)
        for recipe_id, pares in recommended_recipes.items()
        if len(pares) >= min_neighbors
    }
 
    sorted_recommendations = sorted(
        averaged_recommendations.items(), key=lambda x: x[1], reverse=True
    )
 
    return sorted_recommendations[:num_recommendations]

def build_users(df):
    users = {}
    for u, r, s in zip(df['user_id'], df['recipe_id'], df['rating']):
        users.setdefault(u, {})[r] = s
    return users

def build_items(df):
    items = {}
    for u, r, s in zip(df['user_id'], df['recipe_id'], df['rating']):
        items.setdefault(r, {})[u] = s
    return items

 
def build_user_items(items):
    user_items = {}
    for r, d in items.items():
        for u in d:
            user_items.setdefault(u, []).append(r)
    return user_items


def compute_similar_items(item_id, items, user_items, k=50, min_common=3,
                          alpha=1.0, exclude=None):
    if item_id not in items:
        return []
    exclude = exclude or ()
    ratings_i = items[item_id]
 
    cand = set()
    for u in ratings_i:
        cand.update(user_items[u])
    cand.discard(item_id)
 
    sims = []
    for j in cand:
        if j in exclude:
            continue
        s = similaridade(ratings_i, items[j], min_common, alpha)
        if s > 0:
            sims.append((j, s))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

def get_similar_recipes(item_id, items, user_items, nomes, n=10,
                        min_common=3, alpha=1.0, exclude=None):
        
    sims = compute_similar_items(item_id, items, user_items, n, min_common, alpha, exclude)
    return [(rid, nomes.get(rid, rid), s) for rid, s in sims]
 
 
def precompute_item_neighbors(items, k=50, min_common=3, alpha=1.0):
    """Vizinhos de todos os itens (usa compute_similar_items)."""
    user_items = build_user_items(items)
    return {
        i: compute_similar_items(i, items, user_items, k, min_common, alpha)
        for i in items
    }