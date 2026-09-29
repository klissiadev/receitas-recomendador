""" main.py
API FastAPI que expoe os modelos de recomendacao no contrato do api.ts.

Rodar (na raiz do projeto):
    pip install fastapi uvicorn
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /users
    POST /users                      {"name": "..."}
    GET  /onboarding?k=12
    GET  /recommendations?user_id=1&model=popularity&k=10&tags=vegetarian&tags=easy
    GET  /history/{user_id}
    GET  /recipes/{recipe_id}?user_id=1
    POST /ratings                    {"user_id":1,"recipe_id":44061,"rating":5}
    GET  /tags
    GET  /models
"""

import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocess import prepare_data, parse_recipe_columns
from utils import (compute_rating_stats, enrich_recipes, get_recipe_detail,
                   get_user_history, get_seen_recipe_ids)
from models import PopularityRecommender, RecipeRecommender, ContentRecommender

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "..", "data"))
INTERACTIONS_PATH = os.path.join(DATA_DIR, "RAW_interactions.csv")
RECIPES_PATH = os.path.join(DATA_DIR, "RAW_recipes.csv")
SVD_PATH = os.getenv("SVD_PATH", os.path.join(BASE_DIR, "svd.pkl"))
CONTENT_PATH = os.getenv("CONTENT_PATH", os.path.join(BASE_DIR, "content_similarity_artifacts.pkl"))
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "app.db"))

SVD_POOL = 2000     # candidatos avaliados pelo SVD (o dataset inteiro seria lento)
KNN_SEEDS = 3       # quantas receitas curtidas servem de base para o "knn"
LIKED_MIN = 4       # nota minima para considerar que o usuario gostou

MODELS = [
    {"id": "popularity", "label": "Por popularidade"},
    {"id": "knn", "label": "Porque você gostou de uma receita"},
    {"id": "svd", "label": "Personalizado para você"},
]

S: dict = {}  # estado carregado no startup


# ---------------------------------------------------------------------------
# Banco (usuarios do app + notas novas)
# ---------------------------------------------------------------------------

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(user_df: pd.DataFrame) -> pd.DataFrame:
    """Cria tabelas, popula usuarios demo e devolve as notas novas ja gravadas."""
    with db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.execute("""CREATE TABLE IF NOT EXISTS ratings (
            user_id INTEGER, recipe_id INTEGER, rating INTEGER, date TEXT,
            PRIMARY KEY (user_id, recipe_id))""")

        # o dataset nao tem nomes: usa os 10 usuarios mais ativos como demo
        DEMO_USERS = 10

        # o dataset nao tem nomes: usa os usuarios mais ativos como demo
        for uid in user_df["user_id"].value_counts().head(DEMO_USERS).index:
            conn.execute("INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)",
                        (int(uid), f"Usuário {int(uid)}"))

        rows = conn.execute("SELECT user_id, recipe_id, rating, date FROM ratings").fetchall()

    return pd.DataFrame([dict(r) for r in rows],
                        columns=["user_id", "recipe_id", "rating", "date"])


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    user_df, recipe_df = prepare_data(INTERACTIONS_PATH, RECIPES_PATH)
    recipe_df = parse_recipe_columns(recipe_df)

    new_ratings = init_db(user_df)
    if not new_ratings.empty:
        # notas do app substituem as do dataset para o mesmo par usuario/receita
        key = ["user_id", "recipe_id"]
        user_df = pd.concat([user_df, new_ratings], ignore_index=True)
        user_df = user_df.drop_duplicates(subset=key, keep="last")

    popularity = PopularityRecommender().fit(user_df)
    svd = RecipeRecommender.get_or_train(SVD_PATH, user_df, recipe_df)
    content = ContentRecommender.get_or_train(
        CONTENT_PATH, recipe_df, rating_stats=svd.rating_stats)

    S.update(
        recipe_df=recipe_df,
        svd=svd,
        content=content,
        popularity=popularity,
        pop_ids=popularity.ranking_["recipe_id"].tolist(),
        tags_by_id={rid: set(t) for rid, t in zip(recipe_df["id"], recipe_df["tags"])},
        tags=recipe_df["tags"].explode().value_counts().head(40).index.tolist(),
    )
    yield


app = FastAPI(title="Recipe Recommender API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Recomendacao
# ---------------------------------------------------------------------------

def _first_valid(ids, ok, k):
    out = []
    for rid in ids:
        if ok(rid):
            out.append(rid)
            if len(out) == k:
                break
    return out


def rank_ids(model: str, user_id: int, tags: list[str], k: int) -> list:
    svd: RecipeRecommender = S["svd"]
    df = svd.user_df
    seen = get_seen_recipe_ids(user_id, df)
    need = set(tags)
    tags_by_id = S["tags_by_id"]

    def ok(rid):
        return rid not in seen and need <= tags_by_id.get(rid, set())

    has_history = svd.has_history(user_id)

    if model == "svd" and has_history:
        pool = _first_valid(svd.popular_recipe_ids, ok, SVD_POOL)
        scored = sorted(((rid, svd.algo.predict(user_id, rid).est) for rid in pool),
                        key=lambda x: x[1], reverse=True)
        return [rid for rid, _ in scored[:k]]

    if model == "knn" and has_history:
        mine = df[(df["user_id"] == user_id) & (df["rating"] >= LIKED_MIN)]
        if "date" in mine.columns:
            mine = mine.sort_values("date", ascending=False)
        seeds = mine["recipe_id"].head(KNN_SEEDS).tolist()
        if seeds:
            best: dict = {}
            for seed in seeds:
                sims = S["content"].get_similar_recipes(seed, n=100, exclude=seen)
                for rid, sim in zip(sims.get("recipe_id", []), sims.get("similarity", [])):
                    best[rid] = max(best.get(rid, 0), sim)
            ranked = sorted(best, key=best.get, reverse=True)
            return _first_valid(ranked, ok, k)

    # popularity (e fallback para usuario sem historico / sem nota alta)
    return _first_valid(S["pop_ids"], ok, k)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class NewUser(BaseModel):
    name: str


class NewRating(BaseModel):
    user_id: int
    recipe_id: int
    rating: int


@app.get("/users")
def get_users():
    with db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT id, name FROM users ORDER BY id")]
    df = S["svd"].user_df
    counts = df[df["user_id"].isin([r["id"] for r in rows])]["user_id"].value_counts()
    return [{"id": r["id"], "name": r["name"], "n_ratings": int(counts.get(r["id"], 0))}
            for r in rows]


@app.post("/users")
def create_user(body: NewUser):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Nome vazio.")
    with db() as conn:
        db_max = conn.execute("SELECT COALESCE(MAX(id), 0) FROM users").fetchone()[0]
        new_id = int(max(S["svd"].user_df["user_id"].max(), db_max)) + 1
        conn.execute("INSERT INTO users (id, name) VALUES (?, ?)", (new_id, name))
    return {"id": new_id, "name": name, "n_ratings": 0}


@app.get("/onboarding")
def get_onboarding(k: int = 12):
    from models import onboarding_recipes
    return onboarding_recipes(S["recipe_df"], S["popularity"], n=k,
                              rating_stats=S["svd"].rating_stats)


@app.get("/recommendations")
def get_recommendations(user_id: int, model: str = "popularity", k: int = 10,
                        tags: list[str] = Query(default=[])):
    if model not in {m["id"] for m in MODELS}:
        raise HTTPException(400, f"Modelo desconhecido: {model}")
    ids = rank_ids(model, user_id, tags, k)
    return enrich_recipes(ids, S["recipe_df"], S["svd"].rating_stats)


@app.get("/history/{user_id}")
def get_history(user_id: int):
    svd: RecipeRecommender = S["svd"]
    return get_user_history(user_id, svd.user_df, S["recipe_df"], svd.rating_stats)


@app.get("/recipes/{recipe_id}")
def get_recipe(recipe_id: int, user_id: int | None = None):
    svd: RecipeRecommender = S["svd"]
    rec = get_recipe_detail(recipe_id, S["recipe_df"], svd.user_df,
                            svd.rating_stats, user_id)
    if rec is None:
        raise HTTPException(404, "Receita não encontrada.")
    return rec


@app.post("/ratings")
def post_rating(body: NewRating):
    if not 0 <= body.rating <= 5:
        raise HTTPException(400, "A nota deve estar entre 0 e 5.")
    if body.recipe_id not in S["tags_by_id"]:
        raise HTTPException(404, "Receita não encontrada.")

    today = date.today().isoformat()
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO ratings (user_id, recipe_id, rating, date) "
                     "VALUES (?, ?, ?, ?)", (body.user_id, body.recipe_id, body.rating, today))

    svd: RecipeRecommender = S["svd"]
    svd.add_rating(body.user_id, body.recipe_id, body.rating, date=today)
    S["content"].rating_stats = svd.rating_stats
    return {"ok": True}


@app.get("/tags")
def get_tags():
    return S["tags"]


@app.get("/models")
def get_models():
    return MODELS