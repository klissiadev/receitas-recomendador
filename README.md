# Recomendador de Receitas (Food.com)

Sistema de recomendação por filtragem colaborativa para receitas, desenvolvido como trabalho da disciplina [nome da disciplina].

**Equipe:** [Nome A] e [Nome B]

## Objetivo
Recomendar receitas personalizadas a partir do histórico de avaliações dos usuários, excluindo as receitas já avaliadas, com interface para consultar histórico, ver recomendações e avaliar receitas.

## Estrutura
```
data/        instruções para baixar os dados (CSVs não versionados)
notebooks/   01_eda, 02_modelos, 03_avaliacao
src/         preprocess.py, models.py, evaluate.py
app/         app.py (Streamlit)
docs/        relatorio.md e link do vídeo
```

## Como executar

```bash
# 1. Clonar e entrar na pasta
git clone [URL-DO-REPOSITORIO]
cd receitas-recomendador

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar os dados (ver data/README.md)

# 5. Rodar a interface
streamlit run app/app.py
```

## Documentação
- Relatório: [docs/relatorio.md](docs/relatorio.md)
- Vídeo de demonstração: [link]

> Status: em desenvolvimento.
