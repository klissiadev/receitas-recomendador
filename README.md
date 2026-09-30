# Recomendador de Receitas (Food.com)

Sistema de recomendação de receitas por **filtragem colaborativa**, desenvolvido como trabalho da disciplina de **Oficina de Desenvolvimento de Sistemas I**.

**Equipe:** Ana Klissia Furtado Martins e Jéssica Rodrigues de Souza.

---

## Sumário

- [Objetivo](#objetivo)
- [Funcionalidades](#funcionalidades)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Tecnologias](#tecnologias)
- [Como executar](#como-executar)
- [Documentação](#documentação)

---

## Objetivo

Recomendar receitas personalizadas a partir do histórico de avaliações dos usuários, **excluindo as receitas já avaliadas**, com uma interface para consultar histórico, ver recomendações e avaliar receitas.

## Funcionalidades

- **Criação de usuário com onboarding:** o sistema apresenta receitas populares e variadas (diversificadas por tags) para o novo usuário avaliar, gerando os primeiros dados para os modelos personalizados.
- **Recomendações personalizadas** por três modelos diferentes (popularidade, conteúdo e SVD).
- **Filtro por tags** em todas as listas (a receita precisa conter todas as tags selecionadas).
- **Histórico de avaliações** do usuário.
- **Detalhe da receita** (ingredientes, passos, informações nutricionais).
- **Envio de novas avaliações**, persistidas em banco e já consideradas para excluir itens vistos.
- **Tratamento de cold start:** usuários sem histórico recebem o ranking de popularidade.

## Estrutura do projeto

```
receitas-recomendador/
├── data/                  # Instruções para baixar os dados
├── docs/
│   └── relatorio.md       # Relatório completo do projeto
├── frontend/              # Interface do sistema
├── src/
│   ├── main.py            # API FastAPI
│   ├── preprocess.py      # Limpeza e estruturação das bases brutas
│   ├── models.py          # Classes de recomendação (popularidade, SVD e conteúdo)
│   ├── utils.py           # Funções auxiliares (filtro de itens vistos, enriquecimento de receitas, estatísticas de notas)
│   └── model_training/
│       ├── svd_training.py    # Dataset, validação cruzada, treino e persistência do SVD
│       └── knn_training.py    # Treino do TF-IDF e persistência dos artefatos
├── .gitignore
├── .prettierignore
├── requirements.txt       # Dependências Python
└── README.md
```
 
| Pasta / arquivo | Responsabilidade |
| --- | --- |
| `data/` | Guia para baixar as bases do Food.com. |
| `src/preprocess.py` | Remove linhas sem campos essenciais, duplicatas e interações de receitas fora do catálogo; preenche descrições vazias; converte `tags`, `steps`, `ingredients` e `nutrition` de texto `"[...]"` para listas Python. |
| `src/models.py` | Classes de recomendação: `PopularityRecommender` (baseline por média bayesiana), `RecipeRecommender` (SVD, com fallback de popularidade), `ContentRecommender` (similaridade por conteúdo com TF-IDF), além das funções de cold start e onboarding. |
| `src/utils.py` | Funções compartilhadas: `filter_seen_items`, `enrich_recipes` e `compute_rating_stats`. |
| `src/model_training/` | Treino e persistência dos modelos. `svd_training.py` inclui a validação cruzada; `knn_training.py` gera os artefatos TF-IDF. Se o modelo já estiver salvo, ele é carregado em vez de treinado de novo. |
| `src/main.py` | API: carrega dados e modelos na inicialização, gerencia o SQLite e expõe os endpoints. |
| `frontend/` | Interface para navegar, receber recomendações e avaliar. |
| `docs/` | Documentação do projeto. |

## Tecnologias

- **Back-end e ciência de dados:** Python 3.10+, FastAPI, Uvicorn, pandas, NumPy (`<2`), SciPy, scikit-learn, scikit-surprise, matplotlib, seaborn, tqdm
- **Persistência:** SQLite
- **Frontend:** Node.js + npm

## Como executar

### Pré-requisitos

- **Python 3.10 ou 3.11** (o código usa a sintaxe `str | None`, que exige 3.10+)
- **Node.js** e **npm**

### Passo a passo

```bash
# 1. Clonar e entrar na pasta
git clone https://github.com/klissiadev/receitas-recomendador.git
cd receitas-recomendador

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar os dados (ver data/README.md)

# 5. Rodar a API (terminal 1)
cd src
uvicorn main:app --reload --port 8000

# 6. Rodar a interface (terminal 2, a partir da raiz do projeto)
cd frontend
npm i
npm run dev
```

- API em `http://localhost:8000`
- Documentação interativa (Swagger) em `http://localhost:8000/docs`

Na primeira execução, o treino do SVD pode demorar. Depois disso, o modelo salvo em `svd.pkl` é reutilizado.

## Documentação

- [Relatório do projeto](docs/relatorio.md)
- [Vídeo de demonstração](https://youtu.be/Wcy85ch3wlU)
