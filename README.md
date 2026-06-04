# Machine Learning Teaching Labs

Projeto didatico com scripts simples para ensinar machine learning com Python.

## Como executar

```bash
pip install -r requirements.txt
python src/iris_lab.py --output-dir reports
```

## Saidas

- `reports/iris_metrics.csv`
- `reports/iris_confusion_matrix.csv`

## Conteudo

O primeiro laboratorio usa o dataset Iris do scikit-learn para mostrar:

- separacao treino/teste;
- treinamento de regressao logistica;
- predicao;
- acuracia;
- matriz de confusao.

## Estrutura

```text
src/iris_lab.py
reports/
requirements.txt
```

## Proximos laboratorios

- regressao linear;
- k-NN;
- arvores de decisao;
- validacao cruzada;
- metricas de classificacao;
- perceptron.

## Dados

Usa datasets publicos do scikit-learn ou dados sinteticos.

## Licenca

MIT License.
