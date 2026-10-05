# Relatório do Projeto: Sistema de Recomendação de Receitas por Filtragem Colaborativa

**Equipe:** Ana Klissia Furtado Martins e Jéssica Rodrigues de Souza.

---

## 1. Objetivo

Plataformas de culinária reúnem centenas de milhares de receitas, o que frequentemente gera sobrecarga de informação para os usuários ao tentarem encontrar opções alinhadas às suas preferências. Este trabalho constrói e avalia um sistema de recomendação de receitas personalizado, utilizando algoritmos de **filtragem colaborativa** sobre o histórico de interações do portal **Food.com**.

**Objetivo Geral:** Desenvolver e avaliar um sistema interativo que gere recomendações personalizadas de receitas culinárias para usuários, excluindo itens já avaliados por eles.

**Objetivos Específicos:**

* Implementar e comparar duas abordagens: A filtragem colaborativa baseada em usuários (*user-based*) e a filtragem colaborativa baseada em itens (*item-based*), ambas com similaridade do cosseno.
* Tratar o problema do início a frio (*cold start*) para novos usuários sem histórico de interações.
* Disponibilizar uma interface funcional para navegação, consulta de histórico, obtenção de recomendações com filtro por tags e inserção de avaliações.
* Avaliar os modelos personalizados por meio de um teste com usuários, usando as métricas *precision* e *hit rate*.

---

## 2. Fundamentação Teórica

### 2.1 Sistemas de Recomendação e Filtragem Colaborativa

Sistemas de Recomendação (SRs) estimam a utilidade $\hat{R}(u,i)$ de um item $i$ para um usuário $u$ a partir de dados de **usuários**, **itens** e **interações** (feedback explícito ou implícito).

A **Filtragem Colaborativa (FC)** utiliza exclusivamente o histórico comportamental coletivo contido na matriz usuário-item $R \in \mathbb{R}^{M \times N}$, cujas entradas $R_{u,i}$ indicam a avaliação atribuída por $u$ a $i$. Devido à alta esparsidade dessa matriz, dividem-se as abordagens em:

* **Métodos baseados em memória (vizinhança):** calculam similaridades diretas entre usuários (*user-based*) ou itens (*item-based*). São os métodos adotados neste trabalho.
* **Métodos baseados em modelo:** utilizam redução de dimensionalidade e aprendizado (por exemplo, fatoração de matrizes) para prever preferências.

### 2.2 Similaridade do Cosseno

Dois vetores esparsos de avaliações $x$ e $y$ são comparados apenas nas posições em que ambos possuem nota (conjunto $C$ de itens ou usuários em comum):

$$\text{sim}(x, y) = \frac{\sum_{c \in C} x_c \, y_c}{\sqrt{\sum_{c \in C} x_c^2} \; \sqrt{\sum_{c \in C} y_c^2}}$$

Como as notas são positivas, a similaridade fica entre 0 e 1. Pares com menos de $m_c$ elementos em comum recebem similaridade 0, pois similaridades calculadas sobre um ou dois pontos são instáveis (por exemplo, um único item em comum sempre resulta em 1).

### 2.3 Filtragem Colaborativa Baseada em Usuários (User-Based)

A similaridade entre os usuários $u$ e $v$ é o cosseno de seus vetores de avaliações, calculado sobre as receitas que ambos avaliaram. Seja $N_k(u)$ o conjunto dos $k$ usuários mais similares a $u$ (com similaridade positiva). A nota prevista de $u$ para uma receita $i$ ainda não avaliada é a média das notas dos vizinhos que avaliaram $i$, ponderada pela similaridade:

$$\hat{R}_{u,i} = \frac{\sum_{v \in N_k(u),\, v \text{ avaliou } i} \text{sim}(u,v) \cdot R_{v,i}}{\sum_{v \in N_k(u),\, v \text{ avaliou } i} \text{sim}(u,v)}$$

Exige-se um número mínimo de vizinhos que tenham avaliado $i$, para que uma receita com um único voto não domine o ranking. As receitas são recomendadas em ordem decrescente de $\hat{R}_{u,i}$.

### 2.4 Filtragem Colaborativa Baseada em Itens (Item-Based)

A abordagem *item-based* explora a maior estabilidade das relações entre itens em comparação às preferências dos usuários (Sarwar et al., 2001). A similaridade entre as receitas $i$ e $j$ é o cosseno de seus vetores de avaliações, calculado sobre os usuários que avaliaram ambas:

$$\text{sim}(i, j) = \frac{\sum_{u \in U_{ij}} R_{u,i} \, R_{u,j}}{\sqrt{\sum_{u \in U_{ij}} R_{u,i}^2} \; \sqrt{\sum_{u \in U_{ij}} R_{u,j}^2}}$$

Na formulação clássica, a nota prevista é a média das notas do usuário nos itens vizinhos, ponderada pela similaridade. Neste trabalho, a recomendação é ancorada em uma única receita: a **semente** $s_u$, definida como a receita mais recente que o usuário avaliou com nota $\geq 4$. As receitas ainda não avaliadas são ranqueadas por $\text{sim}(s_u, j)$, e o sistema mostra as de maior similaridade. Como nenhuma nota é prevista, essa abordagem funciona como "quem gostou desta receita também gostaria destas".

### 2.5 Cold Start e Média Bayesiana

O problema do **início a frio** (*cold start*) ocorre pela falta de histórico para novos usuários ou itens. Para novos usuários, adota-se uma abordagem em duas etapas:

1. **Recomendação por popularidade:** exibição de itens bem avaliados, com suporte a filtros por tags.
2. **Onboarding:** coleta de avaliações iniciais (3 receitas obrigatórias) para inicializar a filtragem colaborativa.

Para mitigar distorções de itens com poucas avaliações no ranking por popularidade, utiliza-se a **Média Bayesiana**:

$$\text{score}(i) = \left( \frac{v}{v + m} \right) \cdot R_i + \left( \frac{m}{v + m} \right) \cdot C$$

onde $v$ é o número de avaliações do item $i$, $R_i$ sua nota média simples, $C$ a média global e $m$ a constante de suavização.

### 2.6 Métricas de Avaliação

A avaliação com usuários utiliza duas métricas de ranking sobre uma lista de $K$ receitas recomendadas, cujas receitas cada usuário marca como relevantes ou não:

* **Precision@K:** fração das receitas recomendadas que o usuário considerou relevantes.

$$\text{Precision@}K = \frac{|\text{relevantes} \cap \text{recomendadas}|}{K}$$

* **Hit rate@K:** fração dos usuários para os quais pelo menos uma receita da lista foi considerada relevante.

$$\text{Hit rate@}K = \frac{|\{u : \text{ao menos uma receita relevante na lista de } u\}|}{|U|}$$

O *recall* e o F1 exigem conhecer o total de receitas relevantes de cada usuário no catálogo, informação que não está disponível: o usuário só julga as receitas que lhe são mostradas. Por isso essas métricas não foram calculadas (ver Seção 6).

---

## 3. Dados

O projeto utiliza o dataset público do **Food.com**, composto por dois arquivos CSV brutos (~644 MB no total).

### 3.1 Detalhamento dos Arquivos do Dataset

#### 1. `RAW_interactions.csv` (~349,44 MB)

Armazena o histórico de interações e avaliações dos usuários.

| Coluna | Tipo de Dado | Descrição e Papel no Sistema |
|---|---|---|
| `user_id` | Inteiro / ID | Identificador único do usuário. |
| `recipe_id` | Inteiro / ID | Identificador único da receita. |
| `date` | Data (`YYYY-MM-DD`) | Data do registro da interação (define a receita mais recente do usuário). |
| `rating` | Inteiro (escala 0–5) | Nota atribuída pelo usuário (**feedback explícito** utilizado na matriz $R$). |
| `review` | Texto (*String*) | Comentário textual sobre a receita. |

#### 2. `RAW_recipes.csv` (~294,52 MB)

Contém os metadados e detalhes de preparo de cada receita.

| Coluna | Tipo de Dado | Descrição e Papel no Sistema |
|---|---|---|
| `name` | Texto (*String*) | Nome da receita. |
| `id` | Inteiro / ID | Identificador único da receita (chave de ligação com `recipe_id`). |
| `minutes` | Inteiro | Tempo estimado de preparo (em minutos). |
| `contributor_id` | Inteiro / ID | Identificador do usuário autor da receita. |
| `submitted` | Data (`YYYY-MM-DD`) | Data de publicação da receita. |
| `tags` | Lista de *Strings* | Categorias e tags (usadas nos filtros da interface). |
| `nutrition` | Lista numérica | Informações nutricionais em % do valor diário (PDV). |
| `n_steps` | Inteiro | Número de passos do modo de preparo. |
| `steps` | Lista de *Strings* | Instruções detalhadas de preparo. |
| `description` | Texto (*String*) | Descrição geral da receita. |

---

## 4. Método

### 4.1 Pré-processamento

As bases `RAW_interactions.csv` e `RAW_recipes.csv` foram carregadas e limpas: remoção de linhas sem `user_id`, `recipe_id`, `rating`, `id` ou `name`; remoção de duplicatas (par usuário-receita e id de receita); preenchimento de descrições vazias; e descarte de interações cujas receitas não constam no catálogo. As colunas `tags`, `steps`, `ingredients` e `nutrition`, armazenadas como texto "[...]", foram convertidas em listas Python.

### 4.2 Modelos

**Popularidade (baseline).** As receitas são ranqueadas pela Média Bayesiana (Seção 2.5), com $m = 5$, o que evita que receitas com poucas notas altas dominem o ranking.

**Filtragem colaborativa user-based.** Para cada usuário, calcula-se o cosseno (Seção 2.2) com todos os usuários que avaliaram ao menos uma receita em comum, usando um índice invertido (receita → usuários) para não comparar usuários sem nada em comum. Parâmetros: $k = 50$ vizinhos, mínimo de 2 receitas em comum e mínimo de 2 vizinhos por receita candidata. A nota prevista é a média ponderada da Seção 2.3; receitas já avaliadas são excluídas. O ranking completo de cada usuário é guardado em cache e invalidado quando o usuário envia uma nova avaliação.

**Filtragem colaborativa item-based.** Os vizinhos de cada receita (até 50) são calculados sob demanda e mantidos em cache, usando o cosseno entre receitas (Seção 2.4) com mínimo de 3 usuários em comum. A semente é a receita mais recente do usuário com nota $\geq 4$, e as receitas já avaliadas são excluídas da lista.

Nos dois modelos personalizados, quando o ranking tem menos receitas que o solicitado (poucos vizinhos ou filtro de tags restritivo), a lista é completada com as receitas mais populares que atendam ao mesmo filtro. Novas avaliações enviadas pela interface atualizam imediatamente as estruturas dos dois modelos, sem necessidade de novo treinamento.

### 4.3 Cold start e onboarding

Usuários sem histórico, ou sem nenhuma nota alta (no caso do *item-based*), recebem o ranking de popularidade. Na criação de conta, o sistema apresenta receitas populares e variadas (com diversificação por tags) e exige a avaliação de 3 receitas antes de liberar as recomendações personalizadas. Todas as listas aceitam filtro por tags (a receita precisa ter todas as selecionadas).

### 4.4 Arquitetura

O back-end é uma API FastAPI que carrega os dados e os modelos na inicialização e mantém o estado em memória. Usuários e novas avaliações são persistidos em SQLite; ao iniciar, as notas registradas no aplicativo substituem as do dataset para o mesmo par usuário-receita. Os endpoints cobrem usuários, onboarding, recomendações (por modelo, com filtro de tags), histórico, detalhe de receita e envio de avaliações. A interface web consome essa API.

### 4.5 Protocolo de avaliação com usuários

A avaliação foi feita presencialmente com 10 usuários, em duas etapas:

1. **Treino do sistema de avaliação.** O usuário avaliou com 5 estrelas as 3 receitas obrigatórias do onboarding, mais uma receita da recomendação personalizada seguinte que considerou interessante ou que gostaria de experimentar. Isso fornece ao sistema o histórico mínimo e uma semente para o *item-based*.
2. **Avaliação das recomendações.** Das dez receitas seguintes recomendadas, o usuário apontou quais eram relevantes (interessantes) para ele e quais não eram.

Para cada abordagem, calcularam-se a *precision@10* e o *hit rate@10* (Seção 2.6). Após o teste, os usuários puderam registrar comentários livres.

---

## 5. Resultados

| Abordagem | Precision@10 | Hit rate@10 |
|---|---|---|
| User-based | 44% | 90% (9 de 10 usuários) |
| Item-based | 49% | 100% (10 de 10 usuários) |

Em média, o *user-based* teve 4,4 receitas relevantes entre as 10 recomendadas, e o *item-based*, 4,9. O *user-based* não encontrou nenhuma receita relevante para um dos usuários, enquanto o *item-based* encontrou ao menos uma para todos.

**Comentários dos usuários.** Dois comentários se destacaram na abordagem *user-based*:

* "Não conseguiu reconhecer que eu sou vegetariana, continuava mandando receitas saudáveis só que com carne."
* "Muitos bolos estavam sendo recomendados sendo que eu nem gosto de bolo."

**Discussão.** A diferença entre as abordagens é pequena: 5 pontos percentuais de precision e um usuário de diferença no hit rate. Com apenas 10 usuários, ela não permite afirmar que uma abordagem seja melhor que a outra. Uma hipótese para o resultado ligeiramente superior do *item-based* é que ele parte de uma receita concreta que o usuário já marcou como interessante, enquanto o *user-based* depende de vizinhos formados a partir de poucas avaliações em comum.

Os comentários mostram uma limitação estrutural da filtragem colaborativa pura: os modelos só enxergam as notas, e não os ingredientes ou o tipo de prato. Restrições alimentares (como o vegetarianismo) e preferências por categorias só são respeitadas quando o usuário seleciona as tags correspondentes. A predominância de bolos pode refletir a popularidade desse tipo de receita entre os vizinhos, mas essa explicação também não foi verificada.

---

## 6. Limitações

* **Poucos usuários.** O teste teve apenas 10 participantes. Cada usuário representa 10 pontos percentuais no hit rate, e as diferenças observadas entre os modelos estão dentro da variação esperada para uma amostra desse tamanho.
* **Fadiga do processo.** A avaliação exigiu julgar muitas receitas, e vários participantes precisaram pesquisar a tradução dos nomes, já que as receitas do Food.com estão em inglês. Isso pode ter reduzido a atenção nas últimas respostas e influenciado o julgamento de relevância.
* **Pouco histórico por usuário.** Cada participante forneceu apenas 4 avaliações antes do teste, e os parâmetros mínimos de itens e usuários em comum limitam o número de vizinhos disponíveis. Não houve ajuste fino dos hiperparâmetros ($k$, mínimos de itens e usuários em comum) por falta de tempo, então os modelos podem não estar bem sintonizados.
* **Viés de seleção da semente.** A quarta avaliação do treino foi escolhida entre as recomendações personalizadas. Dependendo de qual modelo gerou essa lista, a semente do *item-based* pode ter sido influenciada pelo próprio sistema.
* **Ausência de baseline na comparação.** O ranking por popularidade não foi avaliado com os mesmos usuários, então não é possível afirmar quanto os modelos personalizados superam a recomendação não personalizada.
* **Métricas incompletas.** *Recall* e F1 não foram calculados porque o total de receitas relevantes por usuário é desconhecido. Calculá-los exigiria um conjunto comum de receitas (*pooling*) julgado por cada usuário.
* **Filtragem colaborativa pura.** Os modelos não usam ingredientes, tags nem texto das receitas, o que explica a dificuldade com restrições alimentares e preferências por categoria, apontadas nos comentários.
* **Desempenho.** O cálculo de similaridades é feito em tempo de requisição, em Python, e os resultados ficam em cache por usuário. A primeira recomendação de um usuário, ou de uma receita-semente nova, é mais lenta que as seguintes.

---

## 7. Conclusão

Este trabalho implementou um sistema de recomendação de receitas com filtragem colaborativa baseada em vizinhança, nas versões *user-based* e *item-based*, mais um baseline por popularidade e um tratamento de *cold start* por onboarding. No teste com 10 usuários, o *user-based* obteve precision@10 de 44% e hit rate de 90%, e o *item-based*, 49% e 100%. Ambos encontraram receitas relevantes para quase todos os participantes, mas a diferença entre eles é pequena demais para ser conclusiva.

Os comentários dos usuários indicam o principal ponto de melhoria: a falta de informação de conteúdo impede que os modelos respeitem restrições alimentares e preferências por tipo de prato. Como trabalhos futuros, propõem-se (i) uma abordagem híbrida que combine a filtragem colaborativa com atributos das receitas (ingredientes e tags), (ii) a avaliação com mais usuários e com o baseline de popularidade, (iii) o cálculo de *recall* e F1 por meio de *pooling* e (iv) o ajuste dos hiperparâmetros com uma avaliação offline complementar.

