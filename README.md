# Genetic Algorithm for Feature Selection

A Python implementation of a Genetic Algorithm (GA) for automated feature selection in binary classification tasks.

## Authors
**Mykola Zabulskyi &**
**Pelyno Mykhailo &**
**Yatsko Alina &**
**Havronska Sophia**

## Overview
This project applies a Genetic Algorithm to select the most relevant features from a dataset, reducing dimensionality while maintaining or improving model performance. The algorithm evolves a population of binary chromosomes, where each bit represents whether a feature is included or excluded.

## How It Works
1. **Initialization** — random population of binary chromosomes is generated
2. **Fitness Evaluation** — each chromosome is scored using cross-validated ROC-AUC with a penalty for using too many features
3. **Selection** — tournament selection picks the best candidates
4. **Crossover** — single-point crossover produces new children
5. **Mutation** — random bit flips introduce diversity
6. **Elitism** — best chromosome is always preserved to the next generation
7. **Early Stopping** — halts if no improvement is found for N generations

## Project Structure
├── main.py                  # Genetic algorithm loop and entry point
├── models.py                # Model definition and fitness function
├── genetic_algorithm.py     # GA operators: init, selection, crossover, mutation
├── data.py                  # Dataset loading, preprocessing, train/test split
└── datasets/                # CSV datasets

## Requirements
numpy
pandas
scikit-learn
joblib
How to install?
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Set your dataset path and parameters in `data.py`:
```python
FILENAME = 'datasets/your_dataset.csv'
SAMPLE_SIZE = 10000  # or None for full dataset
```
2. Tune the algorithm parameters in `main.py`:
```python
N_GENERATION = 100
PENALTY = 0.01
```
3. Run:
```bash
python main.py
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `N_GENERATION` | 100 | Number of generations |
| `PENALTY` | 0.01 | Penalty for using too many features |
| `population_size` | 30 | Number of chromosomes per generation |
| `crossover_rate` | 0.8 | Probability of crossover |
| `mutation_rate` | 0.05 | Probability of bit flip |
| `tournament_k` | 3 | Tournament selection size |
| `cv` | 5 | Cross-validation folds |
| `scoring` | roc_auc | Fitness scoring metric |

## Requirements for Dataset
- CSV format
- Target variable must be the **last column**
- Target must be binary: `0` and `1`
- Any categorical columns are automatically encoded

## Example Results
Features: 78 → 29
Baseline (all features):
Accuracy: 0.927 | F1: 0.941 | ROC-AUC: 0.965
GA Model (29 features):
Accuracy: 0.916 | F1: 0.932 | ROC-AUC: 0.965
> 78 features reduced to 29 with no loss in ROC-AUC
