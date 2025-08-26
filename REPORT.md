## Relatório Analítico — IMDB (Indicium / Lighthouse)

### 1) Introdução
Análise exploratória, respostas de negócio e modelagem preditiva da nota `IMDB_Rating` utilizando a base `desafio_indicium_imdb.csv`. O objetivo é orientar qual tipo de filme desenvolver e demonstrar a capacidade de construir um modelo de previsão de nota.

### 2) Dados e Limpeza
- Registros: 999 filmes; 20 colunas após parsing.
- Limpezas: remoção de `Unnamed: 0`; `Runtime` → `Runtime_min` (minutos); `Gross` → `Gross_usd` (numérico); extração de `Primary_Genre` a partir de `Genre`.
- Nulos relevantes: `Released_Year` (1), `Meta_score` (157), `Gross_usd` (169). Demais colunas completas.

### 3) EDA e Principais Insights
- Distribuição: `IMDB_Rating` concentrada em notas altas (mediana > 7.5). `No_of_Votes` com cauda longa (usar escala logarítmica nas análises).
- Correlações: `Gross_usd` associa-se positivamente a `No_of_Votes`, `Meta_score` e a anos mais recentes (`Released_Year`).
- Gênero: boxplots indicam que gêneros dramáticos/biográficos/crime tendem a ter notas elevadas no IMDB, enquanto ação/aventura/sci-fi alavancam faturamento.
- Diretores/Atores e faturamento: maiores `Gross` médios (com ≥3 filmes) para diretores como Anthony Russo, J.J. Abrams, James Cameron, David Yates, Peter Jackson; e atores/“Star1” como Joe Russo, Robert Downey Jr., Elijah Wood, Mark Hamill, Daniel Radcliffe.
- Overview (texto): TF-IDF + SVD (2D) mostra separação por gênero; termos como “justice”, “crime”, “empire”, “jewish”, “sauron” surgem entre os mais associados a notas mais altas numa regressão linear simples sobre TF-IDF.

Hipóteses:
- Maior `Meta_score` e base de fãs (proxy `No_of_Votes`) elevam faturamento e reconhecimento.
- Franquias/IPs conhecidas e gêneros “Action/Adventure/Sci-Fi” tendem a maior `Gross`.
- Gêneros dramáticos e temas humanos complexos influenciam positivamente a nota IMDB.

### 4) Respostas às Perguntas de Negócio
1) Recomendação genérica de filme:
   - Critério: `IMDB_Rating * log10(1 + No_of_Votes)` para balancear qualidade e popularidade.
   - Top exemplos retornados: The Dark Knight; The Godfather; Pulp Fiction; Inception; The Lord of the Rings: The Return of the King.
   - Recomendação: The Dark Knight (nota alta, altíssimo número de votos, ampla aceitação).

2) Principais fatores para alta expectativa de faturamento:
   - Quantitativos: `No_of_Votes` (alcance/base), `Meta_score` (qualidade crítica), `Released_Year` (mercado/distribuição recente).
   - Categóricos: franquias e equipes criativas recorrentes (direção/elenco) elevam `Gross` médio.
   - Gêneros: “Action/Adventure/Sci-Fi” e franquias de grande apelo tendem a maior teto de faturamento.

3) Insights da coluna `Overview` e inferência de gênero:
   - Texto carrega sinal semântico suficiente para separar gêneros (TF-IDF + redução de dimensionalidade).
   - É possível inferir gênero com modelos de classificação baseados em TF-IDF (baseline viável; não detalhado aqui para manter o foco na regressão da nota).

### 5) Modelagem para Prever `IMDB_Rating`
- Tipo de problema: regressão.
- Features consideradas:
  - Numéricas: `Runtime_min`, `Meta_score`, `No_of_Votes`, `Released_Year`, `Gross_usd`.
  - Categóricas: `Certificate`, `Primary_Genre` (One-Hot com `min_frequency` para reduzir dimensionalidade).
  - Texto (variante mais rica): `Overview` com TF-IDF (modelo avançado testado).
- Pipelines avaliados:
  - Simples (entrega final enxuta): `ColumnTransformer(num+cat)` + `Ridge`.
  - Avançado (para comparação): `num+cat+TF-IDF(Overview)` + `RandomForestRegressor` (melhor desempenho observado).
- Métrica de performance: RMSE (principal), com MAE e R² reportados; validação hold-out e CV k=5.

Resultados observados (execução do notebook):
- Hold-out (modelo avançado – RF):
  - RMSE ≈ 0.206
  - MAE ≈ 0.162
  - R² ≈ 0.354
- Validação cruzada k=5:
  - RF: RMSE ≈ 0.196 ± 0.006
  - Ridge: RMSE ≈ 0.207 ± 0.013

Interpretação:
- O RF capturou não linearidades e interações, superando o Ridge. O Ridge, por sua vez, é mais simples e interpretável, com desempenho competitivo e pipeline mais enxuto (bom para deploy rápido).

### 6) Predição do Exemplo (Enunciado)
Filme: The Shawshank Redemption (1994) — Predição estimada: ~8.767 (modelo avançado). Valor plausível e condizente com o padrão do dataset.

### 7) Recomendação Executiva
- Para maximizar faturamento: priorizar gêneros “Action/Adventure/Sci-Fi” ancorados em franquias/IPs, com diretores/atores de histórico comprovado e alto investimento em divulgação (aumenta `No_of_Votes`).
- Para maximizar nota: priorizar roteiros dramáticos/biográficos/crime com forte apelo crítico; manter `Meta_score` elevado.
- Estratégia híbrida: franquia de grande apelo com foco em qualidade crítica (roteiro/direção), equilibrando `Gross` e notas.

### 8) Reprodutibilidade
- Notebook: `lh_indicium_imdb.ipynb` (Colab/local).
- Dependências: `requirements.txt`.
- Modelo salvo: `best_imdb_model.pkl` (após reexecutar treino sem lambdas não serializáveis).
- Instruções de execução e detalhes adicionais no `README.md`.


