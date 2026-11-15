# Plano de aula demonstrativo — Classificação supervisionada com Iris

**Público-alvo:** graduação em Computação/ADS, disciplina introdutória de Aprendizagem de Máquina.  
**Duração estimada:** 50 minutos.  
**Natureza:** material de portfólio docente em elaboração; não comprova aula efetivamente ministrada.  
**Dataset:** Iris, disponibilizado pelo scikit-learn para fins educacionais.

## Objetivos de aprendizagem

Ao final, cada estudante deverá conseguir:
1. Explicar por que separamos treino, validação cruzada e teste;
2. Comparar regressão logística e k-NN em classificação multiclasse;
3. Interpretar acurácia, F1 macro e matriz de confusão;
4. Reconhecer o risco de *data leakage* e como pipelines podem evitá-lo.

## Pré-requisitos

Python 3.11+, pandas, NumPy e scikit-learn. Conhecimentos básicos de programação, arrays e estatística descritiva.

## Roteiro de 50 minutos

| Tempo | Etapa | Atividade |
|---|---|---|
| 0–5 min | Contextualização | Apresentar problema e variáveis do Iris |
| 5–12 min | Leitura dos dados | Identificar classes, atributos e balanço |
| 12–20 min | Separação dos dados | Explicar divisão treino/teste estratificada |
| 20–33 min | Comparação | Executar regressão logística e k-NN em pipelines |
| 33–42 min | Avaliação | Comparar 5-fold CV, F1 macro e matriz de confusão |
| 42–50 min | Aplicação e síntese | Responder questões e debater generalização |

## Demonstração reproduzível

```bash
python -m pip install -r requirements.txt
python src/model_comparison.py --output-dir reports
```

Arquivos gerados: `model_comparison.csv` e duas matrizes de confusão. Comparar a validação cruzada no conjunto de treino com as métricas do conjunto de teste mantido separado.

## Exercícios dos estudantes

A. Alterar `n_neighbors` de 5 para 3; prever o possível efeito antes de executar.  
B. Explicar por que não se deve ajustar o `StandardScaler` usando o conjunto de teste.  
C. Identificar duas classes confundidas na matriz e propor uma hipótese a ser testada.  
D. Debater por que a acurácia sozinha pode enganar quando as classes são desbalanceadas.

## Rubrica de avaliação (10 pontos)

- Separação correta entre treino e teste: 2 pontos.
- Código executável e reprodutível: 2 pontos.
- Interpretação de validação cruzada e métricas: 3 pontos.
- Discussão de *data leakage* e limitações: 2 pontos.
- Clareza e documentação: 1 ponto.

## Limitações e ética

O Iris é um conjunto pequeno e clássico; resultados em Iris não comprovam desempenho em ambientes reais. Não usar este laboratório como validação de IA clínica, pois os dados não são médicos. Para uma avaliação científica, seria necessário definir hipóteses, critérios de validação, incerteza e dados representativos.

## Referências

- Scikit-learn — `load_iris`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- Scikit-learn — `Pipeline`: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- Scikit-learn — `StratifiedKFold`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
