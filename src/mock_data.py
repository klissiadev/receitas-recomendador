"""
As colunas e tipos seguem exatamente o formato dos CSVs reais do Food.com:

RAW_interactions.csv -> user_id, recipe_id, date, rating, review
RAW_recipes.csv      -> id, name, minutes, tags, n_ingredients, ingredients

"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# para gerar sempre os mesmos dados mockados (reprodutível)
random.seed(42) 
np.random.seed(42)

# Vocabulário simples só para os nomes/tags ficarem plausíveis na demo
ADJETIVOS = ["Fácil", "Rápida", "Caseira", "Especial", "Clássica", "Cremosa", "Leve", "Tradicional"]
PRATOS = ["Bolo de Chocolate", "Salada de Frutas", "Frango Assado", "Lasanha", "Risoto de Cogumelos",
          "Torta de Limão", "Sopa de Legumes", "Panqueca", "Feijoada", "Brownie", "Quiche",
          "Macarrão ao Alho e Óleo", "Pão Caseiro", "Salmão Grelhado", "Curry de Grão-de-Bico"]
TAGS_POOL = ["vegetariano", "vegano", "sobremesa", "sem-gluten", "rapido", "low-carb",
             "sem-lactose", "cafe-da-manha", "jantar", "almoco", "doce", "salgado", "saudavel"]


def gerar_receitas_mock(n_receitas: int = 60) -> pd.DataFrame:
    """Gera um DataFrame de receitas fake, no formato de RAW_recipes.csv."""
    linhas = []
    for i in range(1, n_receitas + 1):
        nome = f"{random.choice(PRATOS)} {random.choice(ADJETIVOS)}"
        n_tags = random.randint(1, 3)
        tags = random.sample(TAGS_POOL, n_tags)
        linhas.append({
            "id": i,
            "name": nome,
            "minutes": random.choice([10, 15, 20, 30, 45, 60, 90]),
            "tags": tags,  # no CSV real vem como string; aqui já como lista para facilitar
            "n_ingredients": random.randint(3, 12),
        })
    return pd.DataFrame(linhas)


def gerar_interacoes_mock(recipes_df: pd.DataFrame, n_usuarios: int = 30,
                           min_interacoes: int = 5, max_interacoes: int = 20) -> pd.DataFrame:
    """Gera um DataFrame de interações fake, no formato de RAW_interactions.csv.

    Cada usuário avalia um número aleatório de receitas distintas, com nota de 1 a 5.
    Algumas receitas recebem propositalmente mais avaliações para simular popularidade.
    """
    receita_ids = recipes_df["id"].tolist()
    # pesos para simular receitas "populares" que aparecem mais nas interações
    pesos = np.random.exponential(scale=1.0, size=len(receita_ids))
    pesos = pesos / pesos.sum()

    linhas = []
    data_base = datetime(2023, 1, 1)
    for user_id in range(1, n_usuarios + 1):
        n_int = random.randint(min_interacoes, max_interacoes)
        receitas_usuario = np.random.choice(receita_ids, size=min(n_int, len(receita_ids)),
                                             replace=False, p=pesos)
        for recipe_id in receitas_usuario:
            data = data_base + timedelta(days=random.randint(0, 500))
            linhas.append({
                "user_id": user_id,
                "recipe_id": int(recipe_id),
                "date": data.strftime("%Y-%m-%d"),
                "rating": random.choices([1, 2, 3, 4, 5], weights=[5, 5, 15, 35, 40])[0],
                "review": "",
            })
    return pd.DataFrame(linhas)


def carregar_dados_mock():
    """Atalho: gera receitas + interações mockadas já prontas para uso.

    Returns:
        (recipes_df, interactions_df)
    """
    recipes_df = gerar_receitas_mock()
    interactions_df = gerar_interacoes_mock(recipes_df)
    return recipes_df, interactions_df


if __name__ == "__main__":
    recipes_df, interactions_df = carregar_dados_mock()
    print("Receitas mockadas:", recipes_df.shape)
    print(recipes_df.head())
    print("\nInterações mockadas:", interactions_df.shape)
    print(interactions_df.head())
