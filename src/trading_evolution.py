# trading_evolution.py
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

def evaluate_strategy(individual, price_data):
    """
    Dummy evaluation function.
    individual: [weight, buy_threshold, sell_threshold]
    Simulates a very basic trading strategy on historical data.
    """
    weight, buy_threshold, sell_threshold = individual
    cash = 100000
    holdings = 0
    for price in price_data['close']:
        indicator = price * weight  # a simple indicator
        if indicator < buy_threshold and cash >= price:
            cash -= price
            holdings += 1
        elif indicator > sell_threshold and holdings > 0:
            cash += price
            holdings -= 1
    final_value = cash + holdings * price_data['close'].iloc[-1]
    return (final_value,)

def main():
    # Generate dummy price data
    dates = pd.date_range(start="2025-03-01", periods=100)
    prices = np.linspace(100, 150, num=100)
    price_data = pd.DataFrame({'close': prices}, index=dates)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Parameter ranges: weight [0.1, 2.0], buy_threshold [90, 110], sell_threshold [140, 160]
    toolbox.register("attr_weight", random.uniform, 0.1, 2.0)
    toolbox.register("attr_buy_threshold", random.uniform, 90, 110)
    toolbox.register("attr_sell_threshold", random.uniform, 140, 160)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_weight, toolbox.attr_buy_threshold, toolbox.attr_sell_threshold), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_strategy, price_data=price_data)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=50)
    NGEN = 40
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        best = tools.selBest(population, 1)[0]
        print(f"Generation {gen}: Best Fitness = {best.fitness.values[0]}")
    print("Best individual:", best, best.fitness.values)

if __name__ == "__main__":
    main()
