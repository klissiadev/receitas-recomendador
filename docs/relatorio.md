# Relatório do Projeto: Sistema de Recomendação de Receitas por Filtragem Colaborativa

**Equipe:** Ana Klissia Furtado Martins e Jéssica Rodrigues de Souza.

\---

## 1\. Objetivo

Plataformas de culinária reúnem centenas de milhares de receitas, o que frequentemente gera sobrecarga de informação para os usuários ao tentarem encontrar opções alinhadas às suas preferências. Este trabalho constrói e avalia um sistema de recomendação de receitas personalizado utilizando algoritmos de **filtragem colaborativa** com base no histórico de interações do portal **Food.com**.

**Objetivo Geral:** Desenvolver e avaliar um sistema interativo que gere recomendações personalizadas de receitas culinárias para usuários cadastrados, excluindo itens já avaliados por eles.

**Objetivos Específicos:**

* Processar e estruturar as bases de dados brutas de interações e receitas do Food.com.
* Implementar e comparar modelos de recomendação: baseline por popularidade (Média Bayesiana), filtragem colaborativa baseada em itens (kNN) e fatoração de matrizes por decomposição em valores singulares (SVD).
* Tratar o problema do início a frio (*cold start*) para novos usuários sem histórico de interações.
* Disponibilizar uma interface funcional para navegação, consulta de histórico, obtenção de recomendações e inserção de avaliações.
* Avaliar o desempenho dos modelos por meio de métricas de acurácia de ranking, cobertura de catálogo e testes de usabilidade com usuários.

\---

## 2\. Fundamentação Teórica

### 2.1 Sistemas de Recomendação e Filtragem Colaborativa

Sistemas de Recomendação (SRs) estimam a utilidade $\\hat{R}(u,i)$ de um item $i$ para um usuário $u$ a partir de dados de **usuários**, **itens** e **interações** (feedback explícito ou implícito).

A **Filtragem Colaborativa (FC)** utiliza exclusivamente o histórico comportamental coletivo contido na matriz usuário-item $R \\in \\mathbb{R}^{M \\times N}$, cujas entradas $R\_{u,i}$ indicam a avaliação atribuída por $u$ a $i$. Devido à alta esparsidade dessa matriz, dividem-se as abordagens em:

* **Métodos baseados em memória (vizinhança):** calculam similaridades diretas entre usuários (*User-Based*) ou itens (*Item-Based*).
* **Métodos baseados em modelo:** utilizam técnicas de redução de dimensionalidade e aprendizado para prever preferências.

### 2.2 Filtragem Colaborativa Baseada em Itens (Item-Based kNN)

A FC *Item-Based* explora a maior estabilidade temporal das relações entre itens em comparação às preferências dos usuários. A similaridade entre os itens $i$ e $j$ é calculada pela **Similaridade do Cosseno**:

$$\\text{sim}(i, j) = \\frac{\\sum\_{u \\in U} R\_{u,i} , R\_{u,j}}{\\sqrt{\\sum\_{u \\in U} R\_{u,i}^2} \\sqrt{\\sum\_{u \\in U} R\_{u,j}^2}}$$

A nota prevista para o item candidato $j$ é obtida pela média das avaliações prévias do usuário $u$ nos $k$-vizinhos mais próximos ($k\\text{NN}$) ponderada pelas similaridades:

$$\\hat{R}*{u,j} = \\frac{\\sum*{i \\in N\_k(j)} \\text{sim}(i,j) \\cdot R\_{u,i}}{\\sum\_{i \\in N\_k(j)} |\\text{sim}(i,j)|}$$

### 2.3 Fatoração de Matrizes (SVD)

A fatoração de matrizes projeta usuários e itens em um espaço latente $k \\ll \\min(M, N)$, decompondo $R \\approx P \\cdot Q^T$. Incorporando vieses globais ($\\mu$), de usuário ($b\_u$) e de item ($b\_i$), a predição é expressa por:

$$\\hat{R}\_{u,i} = \\mu + b\_u + b\_i + p\_u \\cdot q\_i^T$$

A otimização minimiza o erro quadrático com regularização $\\lambda$:

$$\\min\_{P, Q, b} \\sum\_{(u,i) \\in R\_{\\text{treino}}} \\left( R\_{u,i} - (\\mu + b\_u + b\_i + p\_u \\cdot q\_i^T) \\right)^2 + \\lambda \\left( |p\_u|\_2^2 + |q\_i|\_2^2 + b\_u^2 + b\_i^2 \\right)$$

### 2.4 Cold Start e Média Bayesiana

O problema do **início a frio** (*cold start*) ocorre pela falta de histórico para novos usuários ou itens. Para novos usuários, adota-se uma abordagem em duas etapas:

1. **Recomendação por Popularidade:** Exibição inicial de itens bem avaliados (com suporte a filtros por tags).
2. **Onboarding:** Coleta de 5 a 10 avaliações iniciais para inicializar a filtragem colaborativa.

Para mitigar distorções de itens com poucas avaliações no ranking por popularidade, utiliza-se a **Média Bayesiana**:

$$\\text{score}(i) = \\left( \\frac{v}{v + m} \\right) \\cdot R\_i + \\left( \\frac{m}{v + m} \\right) \\cdot C$$

onde $v$ é o número de avaliações do item $i$, $R\_i$ sua nota média simples, $C$ a média global e $m$ a constante de suavização.

### 2.5 Métricas de Avaliação

O desempenho dos modelos é avaliado por três dimensões:

* **Acurácia de Predição:** MAE ($\\frac{1}{|T|} \\sum |R\_{u,i} - \\hat{R}*{u,i}|$) e RMSE ($\\sqrt{\\frac{1}{|T|} \\sum (R*{u,i} - \\hat{R}\_{u,i})^2}$).
* **Ranking Top-$K$:** Precision@K, Recall@K, MAP@K e NDCG@K.
* **Diversidade:** Cobertura do Catálogo (proporção de itens do acervo recomendados a pelo menos um usuário).

\---

## 3\. Dados

O projeto utiliza o dataset público do **Food.com** (Majumder et al., 2019), composto por dois arquivos CSV brutos (\~644 MB no total).

### 3.1 Detalhamento dos Arquivos do Dataset

#### 1\. `RAW\_interactions.csv` (Tamanho: \~349,44 MB)

Armazena o histórico de interações e avaliações dos usuários.

|Coluna|Tipo de Dado|Descrição e Papel no Sistema|
|-|-|-|
|`user\_id`|Inteiro / ID|Identificador único do usuário.|
|`recipe\_id`|Inteiro / ID|Identificador único da receita.|
|`date`|Data (`YYYY-MM-DD`)|Data do registro da interação.|
|`rating`|Inteiro (Escala 0–5)|Nota atribuída pelo usuário (**feedback explícito** utilizado na matriz $R$).|
|`review`|Texto (*String*)|Comentário/crítica textual sobre a receita.|

#### 2\. `RAW\_recipes.csv` (Tamanho: \~294,52 MB)

Contém os metadados e detalhes de preparo de cada receita.

|Coluna|Tipo de Dado|Descrição e Papel no Sistema|
|-|-|-|
|`name`|Texto (*String*)|Nome da receita.|
|`id`|Inteiro / ID|Identificador único da receita (chave de ligação com `recipe\_id`).|
|`minutes`|Inteiro|Tempo estimado de preparo (em minutos).|
|`contributor\_id`|Inteiro / ID|Identificador do usuário autor da receita.|
|`submitted`|Data (`YYYY-MM-DD`)|Data de publicação da receita.|
|`tags`|Lista de *Strings*|Categorias e tags (utilizadas nos filtros de *cold start*).|
|`nutrition`|Lista numérica|Informações nutricionais em % do valor diário (PDV).|
|`n\_steps`|Inteiro|Número de passos do modo de preparo.|
|`steps`|Lista de *Strings*|Instruções detalhadas de preparo.|
|`description`|Texto (*String*)|Descrição geral da receita.|

\---

## 4\. Método

4.1 Pré-processamento

As bases RAW_interactions.csv e RAW_recipes.csv foram carregadas e limpas: remoção de linhas sem user_id, recipe_id, rating, id ou name; remoção de duplicatas (par usuário-receita e id de receita); preenchimento de descrições vazias; e descarte de interações cujas receitas não constam no catálogo. As colunas tags, steps, ingredients e nutrition, armazenadas como texto "[...]", foram convertidas em listas Python.

4.2 Modelos

Popularidade (baseline). As receitas são ranqueadas pela Média Bayesiana (Seção 2.4), com $m = 5$, o que evita que receitas com poucas notas altas dominem o ranking.

"TF-IDF" (content-based filtering). Cada receita é representada por um vetor TF-IDF (5.000 atributos, stop words em inglês) construído a partir do texto de seus ingredientes e tags. A similaridade entre receitas é o cosseno desses vetores. Para um usuário, as sementes são as 3 receitas mais recentes que ele avaliou com nota $\geq 4$, na interface é buscado a última avaliação. Cada candidato recebe como pontuação a maior similaridade com qualquer semente, e as receitas já avaliadas são excluídas.

SVD (collaborative filtering). Usa a implementação da biblioteca Surprise, com $k = 50$ fatores latentes, escala de notas 0–5 e semente fixa. O treino é feito por gradiente descendente sobre as avaliações observadas. Foram executadas validação cruzada de 3 partições e avaliação holdout de 20%. No momento da recomendação, a nota prevista $\hat{R}_{u,i} = \mu + b_u + b_i + p_u \cdot q_i^T$ é calculada de forma vetorizada (NumPy) para todo o catálogo e as $K$ maiores são retornadas. O modelo treinado é salvo em disco (svd.pkl) e reutilizado nas execuções seguintes.

4.3 Cold start e onboarding

Usuários sem histórico, ou sem nota alta (no caso do kNN), recebem o ranking de popularidade como alternativa. Na criação de conta, o sistema apresenta receitas populares e variadas (com diversificação por tags) para o usuário avaliar, gerando os primeiros dados para os modelos personalizados. Todas as listas aceitam filtro por tags (a receita precisa ter todas as selecionadas).

4.4 Arquitetura

O back-end é uma API FastAPI que carrega os dados e os modelos na inicialização e mantém o estado em memória. Usuários e novas avaliações são persistidos em SQLite; ao iniciar, as notas do app substituem as do dataset para o mesmo par usuário-receita. Os endpoints cobrem usuários, onboarding, recomendações (por modelo, com filtro de tags), histórico, detalhe de receita e envio de avaliações.

## 5\. Resultados

| Métrica | Fold 1 | Fold 2 | Fold 3 | Média | Desvio-padrão |
|---|---|---|---|---|---|
| RMSE | 1,2212 | 1,2173 | 1,2208 | 1,2198 | 0,0017 |
| MAE | 0,7406 | 0,7377 | 0,7386 | 0,7390 | 0,0012 |

## 6\. Limitações

O modelo SVD é treinado offline e seus parâmetros ficam congelados depois disso. Novas avaliações inseridas pela aplicação atualizam o histórico do usuário (usado para excluir itens já vistos e escolher sementes do kNN), mas não alteram os vetores latentes $p_u$, $q_i$ nem os vieses. Assim, usuários criados depois do treino não possuem vetor latente e recebem recomendações por popularidade, e usuários existentes só têm suas novas notas refletidas no SVD após um novo treinamento. É possível observar como uma opção content-based filtering trata melhor usuário novatos. 

## 7\. Conclusão

*(A preencher)*

\---

### Referências Bibliográficas

* KOREN, Y.; BELL, R.; VOLINSKY, C. Matrix factorization techniques for recommender systems. *IEEE Computer*, v. 42, n. 8, p. 30-37, 2009.
* MAJUMDER, B. P. et al. Generating personalized recipe from historical user preferences. In: *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)*, 2019.
* RICCI, F.; ROKACH, L.; SHAPIRA, B.; KANTOR, P. B. (Eds.). *Recommender Systems Handbook*. New York: Springer, 2011.
* SARWAR, B. et al. Item-based collaborative filtering recommendation algorithms. In: *Proceedings of the 10th international conference on World Wide Web (WWW)*, 2001. p. 285-295.

