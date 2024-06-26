import random
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX
from controllers import pid, BaseController
import numpy as np
from typing import List, Tuple

model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)


class GeneticAlgorithmPIDOptimizer:
    def __init__(self, population_size: int, generations: int, mutation_rate: float):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initialize_population(self) -> List[Tuple[float, float, float]]:
        return [
            # p=0.1182, i=0.1292, d=0.043
            # p=0.1221, i=0.1448, d=0.0367
            (
                #  random.uniform(0, 0.15),
                # random.uniform(0.05, 0.16),
                # random.uniform(-0.01, 0.07),
                random.uniform(0, 1),
                random.uniform(0.0, 0.5),
                random.uniform(-0.2, 0.2),
            )
            for _ in range(self.population_size)
        ]

    def fitness(
        self, individual: Tuple[float, float, float], test_cases: List[str]
    ) -> float:
        p, i, d = individual
        total_cost = 0

        for test_case in test_cases:
            controller = pid.Controller()
            controller.p, controller.i, controller.d = p, i, d
            sim = TinyPhysicsSimulator(
                model, test_case, controller=controller, debug=False
            )
            cost = sim.rollout()["total_cost"]
            total_cost += cost

        return -total_cost # Negative because we want to maximize fitness (minimize cost)

    def select_parents(
        self, population: List[Tuple[float, float, float]], fitnesses: List[float]
    ) -> List[Tuple[float, float, float]]:
        # Convert fitnesses to selection probabilities (lower cost = higher probability)
        max_fitness = 0.0
        min_fitness = -10000.0
        if( min(fitnesses)<min_fitness ):
            normalized_fitnesses = [
            min_fitness if f<min_fitness else (2.0* (f+1e6 - min_fitness)/(max_fitness-min_fitness))  -1 for f in fitnesses
            ]  # Add small constant to avoid zero probabilities
            normalized_fitnesses = [1/(1 + np.exp(-f/0.0005)) for f in normalized_fitnesses] #sigmoid
        else:
            min_fitness = min(fitnesses)
            normalized_fitnesses = [
            min_fitness if f<min_fitness else (2.0* (f+1e6 - min_fitness)/(max_fitness-min_fitness))  -1 for f in fitnesses
            ] 

        total_fitness = sum(normalized_fitnesses)
        probabilities = [f / total_fitness for f in normalized_fitnesses]
        print(probabilities)

        return random.choices(population, weights=probabilities, k=2)

    def crossover(
        self, parent1: Tuple[float, float, float], parent2: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        child = tuple(
            p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)
        )
        return child

    def mutate(
        self, individual: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        return (
            (
                individual[0] + random.gauss(0, 0.025)
                if random.random() < self.mutation_rate
                else individual[0]
            ),
            (
                individual[1] + random.gauss(0, 0.025)
                if random.random() < self.mutation_rate
                else individual[1]
            ),
            (
                individual[2] + random.gauss(0, 0.007)
                if random.random() < self.mutation_rate
                else individual[2]
            )
        )

    def optimize(self, test_cases: List[str]) -> Tuple[float, float, float]:
        population = self.initialize_population()
        for generation in range(self.generations):
            fitnesses = [self.fitness(ind, test_cases) for ind in population]
            best_individual = population[np.argmax(fitnesses)]
            best_fitness = max(fitnesses)

            print(f"Generation {generation + 1}/{self.generations}")
            print(f"Population: {population}")
            print(
                f"Best individual: p={best_individual[0]:.4f}, i={best_individual[1]:.4f}, d={best_individual[2]:.4f}"
            )
            print(f"Best fitness: {best_fitness:.4f}")
            print(f"Fitness: {fitnesses}")

            new_population = [best_individual]  # Elitism 

            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        final_fitnesses = [self.fitness(ind, test_cases) for ind in population]
        best_individual = population[np.argmax(final_fitnesses)]
        return best_individual


# Usage
if __name__ == "__main__":
    optimizer = GeneticAlgorithmPIDOptimizer(
        population_size=20, generations=10, mutation_rate=0.5
    )
    test_cases = [f"./data/{i:05d}.csv" for i in range(0, 5)]  # 00000 to 00010
    best_p, best_i, best_d = optimizer.optimize(test_cases)

    print(f"Optimized PID parameters:")
    print(f"P: {best_p:.4f}")
    print(f"I: {best_i:.4f}")
    print(f"D: {best_d:.4f}")

    # Final evaluation
    controller = pid.Controller(best_p, best_i, best_d)
    total_cost = 0

    for test_case in test_cases:
        sim = TinyPhysicsSimulator(model, test_case, controller=controller, debug=False)
        cost = sim.rollout()["total_cost"]
        total_cost += cost
        print(f"Test case {test_case}: Cost = {cost:.4f}")

    print(f"Total cost across all test cases: {total_cost:.4f}")
