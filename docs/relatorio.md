# Recomendador de Receitas com Filtragem Colaborativa

**Equipe:** [Nome A] e [Nome B]

## 1. Objetivo

Plataformas de culinária reúnem centenas de milhares de receitas, e o usuário tem dificuldade de encontrar as que combinam com seu gosto. Este trabalho constrói um sistema de recomendação de receitas baseado em **filtragem colaborativa**, usando o histórico de avaliações de usuários do Food.com.

**Objetivo geral:** desenvolver e avaliar um sistema que gere recomendações personalizadas de receitas, excluindo as que o usuário já avaliou.

**Objetivos específicos:**

- Preparar a base de interações usuário–receita (limpeza, filtros e divisão treino/teste).
- Implementar e comparar três abordagens: baseline por popularidade, filtragem colaborativa item-based (KNN) e fatoração de matrizes (SVD).
- Tratar o caso de usuário sem histórico (*cold start*).
- Disponibilizar uma interface para consultar históricos, ver recomendações e avaliar receitas.
- Avaliar os resultados com métricas de ranking e com usuários de teste.

## 2. Fundamentação teórica

### 2.1 Sistemas de recomendação

Sistemas de recomendação sugerem itens que provavelmente interessarão a um usuário, reduzindo a sobrecarga de informação. As abordagens mais comuns são a **filtragem baseada em conteúdo**, que recomenda itens parecidos com os que o usuário já gostou usando atributos dos itens (ingredientes, tags), e a **filtragem colaborativa**, que usa apenas o comportamento de muitos usuários. Existem também sistemas **híbridos**, que combinam as duas. Este trabalho foca na filtragem colaborativa.

### 2.2 Filtragem colaborativa

A ideia central é que usuários com gostos parecidos no passado tendem a ter gostos parecidos no futuro. Os dados de entrada são a **matriz usuário–item**, em que cada célula guarda a nota que o usuário deu ao item. Essa matriz é extremamente esparsa, pois cada usuário avalia uma fração mínima do catálogo. Há duas famílias de métodos: os baseados em vizinhança (*memory-based*) e os baseados em modelo (*model-based*).

**User-based.** Recomenda ao usuário u os itens bem avaliados por usuários similares a ele. A similaridade entre usuários é calculada sobre os itens que ambos avaliaram. O custo cresce com o número de usuários, e o perfil de um usuário muda com frequência, o que dificulta manter as similaridades atualizadas.

**Item-based.** Calcula a similaridade entre itens a partir das avaliações que receberam. Um item é recomendado se for similar aos que o usuário já avaliou bem (“quem gostou de X também gostou de Y”). Em geral escala melhor que o user-based quando há mais usuários que itens relevantes, e é mais fácil de explicar ao usuário (Sarwar et al., 2001). Este é o modelo principal do projeto.

A similaridade mais usada é a do **cosseno** entre os vetores de dois itens *i* e *j*:

```
sim(i, j) = (r_i · r_j) / (||r_i|| × ||r_j||)
```

onde *r_i* é o vetor de avaliações do item *i* sobre todos os usuários. A pontuação de um item candidato *j* para o usuário *u* pode ser calculada como a soma das similaridades entre *j* e os itens que *u* avaliou, ponderada pelas notas de *u*.

### 2.3 Fatoração de matrizes (SVD)

Métodos baseados em modelo aprendem representações compactas de usuários e itens. Na fatoração de matrizes, a matriz usuário–item *R* é aproximada por dois fatores de baixa dimensão:

```
R ≈ P · Qᵀ
```

em que cada usuário e cada item é representado por um vetor de *k* **fatores latentes** (por exemplo, tendência a receitas doces, rápidas ou elaboradas), e a afinidade é o produto interno dos dois vetores. Variantes populares, como o SVD popularizado no Netflix Prize, incluem vieses de usuário e de item e regularização (Koren, Bell e Volinsky, 2009). Neste trabalho, o SVD é usado como comparação com o método de vizinhança.

### 2.4 Cold start

O *cold start* ocorre quando não há dados suficientes para recomendar: **usuário novo**, sem histórico, ou **item novo**, sem avaliações. A filtragem colaborativa pura não funciona nesses casos, pois depende das interações. Neste trabalho, o usuário novo recebe inicialmente as receitas mais populares e bem avaliadas, com filtro opcional por tag (por exemplo, vegetariano ou sobremesa). Em seguida, ele avalia de 5 a 10 receitas de *onboarding*, o que gera histórico mínimo para o modelo colaborativo passar a personalizar as sugestões.

### 2.5 Baseline por popularidade

Recomenda as receitas mais bem avaliadas globalmente. Para evitar que itens com poucas notas dominem o ranking, usa-se a **média bayesiana** (média ponderada):

```
score(i) = ( v / (v + m) ) × R_i + ( m / (v + m) ) × C
```

onde *v* é o número de avaliações do item, *R_i* sua nota média, *C* a nota média global e *m* um limiar mínimo de avaliações. Serve como referência de comparação e como solução de cold start.

### 2.6 Métricas de avaliação

Como o objetivo é gerar uma lista de *K* recomendações, usam-se métricas de ranking. Uma receita é considerada **relevante** quando o usuário lhe deu nota ≥ 4 no conjunto de teste.

- **Precision@K:** proporção de itens relevantes entre os K recomendados.
- **Recall@K:** proporção dos itens relevantes do usuário que aparecem entre os K recomendados.
- **NDCG@K:** considera a posição dos itens relevantes, valorizando acertos no topo da lista.
- **MAP@K:** média, entre os usuários, da precisão média nas posições em que há acerto.
- **Cobertura do catálogo:** fração das receitas que o sistema chega a recomendar, indicando diversidade e viés de popularidade.
- **RMSE/MAE:** erro na previsão das notas, quando o modelo prevê valores.

### 2.7 Referências

- RESNICK, P. et al. GroupLens: an open architecture for collaborative filtering of netnews. *CSCW*, 1994.
- SARWAR, B. et al. Item-based collaborative filtering recommendation algorithms. *WWW*, 2001.
- KOREN, Y.; BELL, R.; VOLINSKY, C. Matrix factorization techniques for recommender systems. *IEEE Computer*, 42(8), 2009.
- RICCI, F.; ROKACH, L.; SHAPIRA, B. (eds.). *Recommender Systems Handbook*. Springer.
- MAJUMDER, B. P. et al. Generating personalized recipe from historical user preferences. *EMNLP-IJCNLP*, 2019 (origem do dataset Food.com).

## 3. Dados
*(a preencher)*

## 4. Método
*(a preencher)*

## 5. Resultados
*(a preencher)*

## 6. Limitações
*(a preencher)*

## 7. Conclusão
*(a preencher)*
