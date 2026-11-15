# Machine Learning Teaching Labs

**Portfólio docente em Computação:** laboratórios reproduzíveis de aprendizado de máquina com Python e materiais de aula.

> Material demonstrativo. Não se trata de uma disciplina oficialmente oferecida ou de comprovação de aulas ministradas.

## Laboratório 1 — Iris: introdução à classificação

```bash
python -m pip install -r requirements.txt
python src/iris_lab.py --output-dir reports
```

O script original usa regressão logística e gera acurácia e matriz de confusão.

## Laboratório 2 — Comparação de modelos e validação cruzada

```bash
python src/model_comparison.py --output-dir reports
```

Este módulo compara **regressão logística** e **k-NN** com:
- Separação estratificada treino/teste, usando o teste apenas no fim;
- `StandardScaler` dentro de pipelines (para evitar vazamento de dados);
- Validação cruzada estratificada de 5 folds usando somente treino;
- Acurácia média e desvio na CV, acurácia de teste, F1 macro e matrizes de confusão.

Saídas: `reports/model_comparison.csv` e duas matrizes de confusão com rótulos de classe.

## Plano de aula para demonstração docente

Consulte **[Plano de aula de 50 minutos](docs/plano_de_aula_iris.md)**, com objetivos mensuráveis, roteiro de atividades, exercícios, rubrica de avaliação e limitações.

## Verificar testes

```bash
python -m pip install pytest
python -m pytest -q
```

O workflow em `.github/workflows/teaching-tests.yml` executa verificações em pull requests.

## Limitações

O Iris é um dataset público e pequeno, adequado a demonstração de conceitos. O laboratório **não** demonstra uso real de IA clínica ou resultados empresariais. Novas aulas sobre regressão, árvores, seleção de hiperparâmetros e visão computacional podem ser adicionadas após validar esta base.

## Licença

MIT — consultar `LICENSE`.
