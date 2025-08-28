## [Lighthouse] Desafio Ciência de Dados 2025-11 — IMDB

### Como executar no Google Colab
- Envie para o Colab os arquivos `desafio_indicium_imdb.csv` e `lh_indicium_imdb.ipynb`.
- Abra o notebook e rode as células na ordem. Se faltar algum pacote, descomente a célula de instalação.

### Como executar localmente
1. Python 3.10+
2. Crie um virtualenv e instale dependências:
```
pip install -r requirements.txt
```
3. Abra o `lh_indicium_imdb.ipynb` (Jupyter/VSCode) e execute as células.

### Saídas
- EDA com gráficos e hipóteses.
- Respostas das perguntas de negócio (2a–2c) no próprio notebook.
- Modelos comparados (Ridge e RandomForest), métricas (RMSE/MAE/R2) e validação cruzada.
- Predição para o exemplo do desafio.
- Modelo salvo em `best_imdb_model.pkl`.

### Estrutura dos dados
Veja o dicionário no enunciado. O notebook faz parsing de `Runtime` para minutos e de `Gross` para número, além de TF-IDF em `Overview`.


### O QUE FALTA?
- Fazer com que o modelo pegue dados categóricos
- Fazer análise da pergunta 2b
