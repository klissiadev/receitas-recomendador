"""
Teste manual rápido

Rodar com:
    cd src
    python test_manual.py
"""

from mock_data import carregar_dados_mock
from models import PopularityRecommender, cold_start_recommendations, onboarding_recipes
from utils import get_user_history, get_recipe_metadata, is_new_user


def linha():
    print("-" * 60)


# 1. Carrega dados mockados (troque por pd.read_csv quando o dataset real chegar)
recipes_df, interactions_df = carregar_dados_mock()
print(f"Receitas: {len(recipes_df)} | Interações: {len(interactions_df)} | "
      f"Usuários: {interactions_df['user_id'].nunique()}")
linha()

# 2. Treina o baseline de popularidade
pop_model = PopularityRecommender(min_avaliacoes=5)
pop_model.fit(interactions_df)
print("Top 5 receitas mais populares (média bayesiana):")
print(pop_model.ranking_.head())
linha()

# 3. Recomendação para um usuário EXISTENTE, com histórico
user_id = 1
print(f"Histórico do usuário {user_id}:")
print(get_user_history(user_id, interactions_df, recipes_df))
linha()

recomendadas = pop_model.recommend(user_id, interactions_df, n=5)
print(f"Recomendações de popularidade para o usuário {user_id} (já sem itens vistos):")
print(get_recipe_metadata(recomendadas, recipes_df))
linha()

# 4. Usuário NOVO (cold start) — id que não existe nas interações
novo_user_id = 9999
print(f"O usuário {novo_user_id} é novo? {is_new_user(novo_user_id, interactions_df)}")

recomendadas_cold = cold_start_recommendations(recipes_df, interactions_df, pop_model, n=5)
print("Recomendações de cold start (populares gerais):")
print(get_recipe_metadata(recomendadas_cold, recipes_df))
linha()

recomendadas_cold_tag = cold_start_recommendations(
    recipes_df, interactions_df, pop_model, n=5, tag_filter="vegetariano"
)
print("Recomendações de cold start filtradas por tag 'vegetariano':")
print(get_recipe_metadata(recomendadas_cold_tag, recipes_df))
linha()

# 5. Receitas de onboarding para o usuário novo avaliar
onboarding = onboarding_recipes(recipes_df, pop_model, n=8)
print("Receitas de onboarding sugeridas (variadas) para o usuário novo avaliar:")
print(get_recipe_metadata(onboarding, recipes_df))
