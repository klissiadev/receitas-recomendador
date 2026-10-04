import random
import math as m
from collections import Counter

from preprocess import get_data, split_leave_last_out
import cs as cs

INTERACTIONS_PATH = '../data/RAW_interactions.csv'
RECIPES_PATH = '../data/RAW_recipes.csv'

N_USERS_TEST = 50
TOP_N = 20
K = 50
MIN_NEIGHBORS = 2
MIN_COMMON = 2
LIKED_MIN = 4   # nota minima para um item de teste contar como "gostou"


def _avg(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def evaluate(train_users, test_users, sim_users, label, baseline=False):
    """
    baseline=True ignora o KNN e recomenda as receitas mais populares (nao vistas),
    para servir de referencia.
    """
    random.seed(42)
    candidatos = [u for u in test_users if u in train_users]
    amostra = random.sample(candidatos, min(N_USERS_TEST, len(candidatos)))

    pop = Counter(r for d in train_users.values() for r in d)  # no de notas por receita no treino
    pop_order = [r for r, _ in pop.most_common()]
    catalogo = len(pop)
    desconto = [1 / m.log2(i + 2) for i in range(TOP_N)]

    hit, prec, rec, rr, ndcg, hit_liked = [], [], [], [], [], []
    recomendados, pops = set(), []
    err, err_base = [], []
    n_teste = n_prevista = sem_rec = 0

    for u in amostra:
        teste = test_users[u]
        n_teste += len(teste)

        if baseline:
            todas = [(r, 0.0) for r in pop_order if r not in train_users[u]][:TOP_N]
        else:
            # ranking completo (nao so o top-N): permite medir a nota prevista dos itens de teste
            todas = cs.recommend_recipes(
                u, train_users, None, num_recommendations=10 ** 9,
                k=K, min_neighbors=MIN_NEIGHBORS, min_common=MIN_COMMON,
                sim_users=sim_users,
            )

        liked = {r for r, s in teste.items() if s >= LIKED_MIN}

        if not todas:
            sem_rec += 1
            for lst in (hit, prec, rec, rr, ndcg):
                lst.append(0.0)  # sem recomendacao conta como erro nas metricas de ranking
            if liked:
                hit_liked.append(0.0)
            continue

        # --- ranking ---
        top = [rid for rid, _ in todas[:TOP_N]]
        acertos = [1 if r in teste else 0 for r in top]
        h = sum(acertos)
        hit.append(1.0 if h else 0.0)
        prec.append(h / TOP_N)
        rec.append(h / len(teste))
        rr.append(next((1 / (i + 1) for i, a in enumerate(acertos) if a), 0.0))
        dcg = sum(d for d, a in zip(desconto, acertos) if a)
        idcg = sum(desconto[: min(len(teste), TOP_N)])
        ndcg.append(dcg / idcg)
        if liked:
            hit_liked.append(1.0 if any(r in liked for r in top) else 0.0)

        # --- cobertura de catalogo e vies de popularidade ---
        recomendados.update(top)
        pops.extend(pop.get(r, 0) for r in top)

        # --- erro na nota prevista (so itens de teste que receberam previsao) ---
        if not baseline:
            pred = dict(todas)
            media_u = _avg(list(train_users[u].values()))
            for rid, nota in teste.items():
                if rid in pred:
                    n_prevista += 1
                    err.append(pred[rid] - nota)
                    err_base.append(media_u - nota)  # baseline: media do usuario

    n = len(amostra)
    print(f'\n[{label}] usuarios: {n} | sem recomendacao: {sem_rec} '
          f'(cobertura de usuarios: {1 - sem_rec / n:.1%})')
    print(f'  ranking@{TOP_N}   hit-rate: {_avg(hit):.2%} | precision: {_avg(prec):.4f} | '
          f'recall: {_avg(rec):.4f} | MRR: {_avg(rr):.4f} | NDCG: {_avg(ndcg):.4f}')
    print(f'  hit-rate@{TOP_N} so com itens de teste nota >= {LIKED_MIN}: '
          f'{_avg(hit_liked):.2%} ({len(hit_liked)} usuarios)')
    print(f'  cobertura de catalogo: {len(recomendados) / catalogo:.2%} '
          f'({len(recomendados)}/{catalogo}) | popularidade media das recs: {_avg(pops):.0f} notas '
          f'(media do catalogo: {_avg(list(pop.values())):.0f})')

    if not baseline:
        rmse = lambda e: m.sqrt(_avg([x * x for x in e]))
        mae = lambda e: _avg([abs(x) for x in e])
        print(f'  nota prevista: cobertura {n_prevista}/{n_teste} itens de teste | '
              f'RMSE {rmse(err):.3f} (baseline media do usuario {rmse(err_base):.3f}) | '
              f'MAE {mae(err):.3f} (baseline {mae(err_base):.3f})')


def main():
    interactions, recipes = get_data(
        INTERACTIONS_PATH, RECIPES_PATH,
        min_ratings_per_user=10, min_ratings_per_recipe=5,
    )
    train, test = split_leave_last_out(interactions)

    nomes = dict(zip(recipes['id'], recipes['name']))
    print(f'Treino: {len(train):,} | Teste: {len(test):,}')

    train_users = cs.build_users(train)
    test_users = cs.build_users(test)

    evaluate(train_users, test_users, None, 'popularidade (baseline)', baseline=True)
    evaluate(train_users, test_users, None, 'cosseno')
    evaluate(train_users, test_users, cs.center_users(train_users), 'pearson')

    u = next(iter(test_users))
    print(f'\nRecomendacoes para o usuario {u}:')
    for rid, score in cs.recommend_recipes(
        u, train_users, nomes, num_recommendations=5,
        k=K, min_neighbors=MIN_NEIGHBORS, min_common=MIN_COMMON,
    ):
        print(f'  {score:.2f}  {nomes.get(rid, rid)}')

    # receitas similares à última avaliada pelo usuário (no treino)
    items = cs.build_items(train)
    user_items = cs.build_user_items(items)

    last_rid = (train[train['user_id'] == u]
                .sort_values('date')
                .iloc[-1]['recipe_id'])

    print(f'\nSimilares à última receita avaliada ({nomes.get(last_rid, last_rid)}):')
    for rid, nome, sim in cs.get_similar_recipes(
        last_rid, items, user_items, nomes, n=5,
        min_common=3, alpha=0.5, exclude=set(train_users[u]),
    ):
        print(f'  {sim:.2f}  {nome}')


if __name__ == '__main__':
    main()