import numpy as np
from sklearn.linear_model import LogisticRegression
from dataset import X_train, X_test, y_train, y_test
from baseline import evaluate_baseline
from genetic_algorithm import (
    initialize_population,
    fitness_function,
    tournament_selection,
    crossover,
    mutation
)

        
def genetic_algorithm(
    X_train,
    y_train,
    feature_names,
    model=None,
    population_size: int = 30,
    n_generations: int = 40,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.02,
    tournament_k: int = 3,
    penalty: float = 0.01,
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

    for gen in range(n_generations):
        fitness_scores = np.array([
            fitness_function(ch, X_np, y_np, model, penalty, cv, scoring)
            for ch in population
        ])

        gen_best_idx = np.argmax(fitness_scores)

        if fitness_scores[gen_best_idx] > best_fitness:
            best_fitness = fitness_scores[gen_best_idx]
            best_chromosome = population[gen_best_idx].copy()
        n_selected = int(best_chromosome.sum())
        if verbose:
            print(
                f"Gen {gen+1:>3}/{n_generations} | "
                f"best_fitness={best_fitness:.4f} | "
                f"avg_fitness={fitness_scores.mean():.4f} | "
                f"features={n_selected}/{n_features}"
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
