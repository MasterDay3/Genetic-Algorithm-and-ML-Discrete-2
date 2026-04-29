import numpy as np
from sklearn.linear_model import LogisticRegression
from dataset import X_train, X_test, y_train, y_test
from baseline import evaluate_baseline
from baseline import fitness_function

from genetic_algorithm import (
    initialize_population,
    tournament_selection,
    crossover,
    mutation
)

#  два ключових параметри, підбирати супер акуратно і малнькими кроками
N_GENERATION = 100
PENALTY = 0.05




def genetic_algorithm(
    X_train,
    y_train,
    feature_names,
    model=None,
    population_size: int = 30,
   # N_GENERATION: int = 40,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    tournament_k: int = 3,
   # PENALTY: float = 0.01,
    cv: int = 5,
    scoring: str = "roc_auc",
    verbose: bool = True,
):

    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=42)

    X_np = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    y_np = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    n_features = X_np.shape[1]
    feature_names = list(feature_names)

    population = initialize_population(population_size, n_features)
    best_chromosome = population[0].copy()
    best_fitness = -np.inf
    history = []

    for gen in range(N_GENERATION):
        fitness_scores = np.array([
            fitness_function(ch, X_np, y_np, model, PENALTY, cv, scoring)
            for ch in population
        ])

        gen_best_idx = np.argmax(fitness_scores)

        if fitness_scores[gen_best_idx] > best_fitness:
            best_fitness = fitness_scores[gen_best_idx]
            best_chromosome = population[gen_best_idx].copy()
        n_selected = int(best_chromosome.sum())
        if (gen+1) % 2 == 0:
            color = '\033[31m'
        else: color = '\033[32m'
        if verbose:
            print(
                f"{color}Gen {gen+1:>3}/{N_GENERATION} | "
                f"best_fitness={best_fitness:.4f} | "
                f"avg_fitness={fitness_scores.mean():.4f} | "
                f"features={n_selected}/{n_features}{color}"
            )

        new_population = [best_chromosome.copy()]
        while len(new_population) < population_size:
            p1 = tournament_selection(population, fitness_scores, tournament_k)
            p2 = tournament_selection(population, fitness_scores, tournament_k)

            child1, child2 = crossover(p1, p2, crossover_rate)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = np.array(new_population)
    selected_features = [
        feature_names[i]
        for i, bit in enumerate(best_chromosome)
        if bit == 1
    ]
    return best_chromosome, selected_features, history


if __name__ == "__main__":

    base_model = LogisticRegression(max_iter=1000, random_state=42)
    acc_base, f1_base, auc_base = evaluate_baseline(
    base_model, X_train, y_train, X_test, y_test
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    best_chromosome, selected_features, history = genetic_algorithm(
        X_train,
        y_train,
        X_train.columns,
        model=model
    )

    print("Selected features:", selected_features)

    eval_model = LogisticRegression(max_iter=1000, random_state=42)
    acc, f1, auc = evaluate_baseline(
        eval_model,
        X_train[selected_features],
        y_train,
        X_test[selected_features],
        y_test
    )
    print(f"GA Model -> Accuracy: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {auc:.3f}")
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Features: {len(X_train.columns)} → {len(selected_features)}")
    print(f"Selected: {selected_features}")
    print(f"\nBaseline (all features):")
    print(f"  Accuracy: {acc_base:.3f} | F1: {f1_base:.3f} | ROC-AUC: {auc_base:.3f}")
    print(f"\nGA Model ({len(selected_features)} features):")
    print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    print("="*50)
