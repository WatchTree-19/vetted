"""An automated strategy search, and what its best result is worth.

An automated research loop (a grid search, a genetic algorithm or an AI
agent writing and backtesting ideas) tries 200 strategies on five years of
daily data and keeps the one with the best Sharpe ratio. None of the 200
has any skill: every return series is noise.

Then the same search again, where one of the 200 ideas is real.

Run:  python examples/strategy_search.py
"""

import numpy as np

import vetted


def search(rng, n_days=1260, n_strategies=200, skilled=None):
    # correlated like real variations on a few ideas: 10 idea families
    family = rng.normal(0, 0.006, (n_days, 10))
    own = rng.normal(0, 0.008, (n_days, n_strategies))
    trials = family[:, np.arange(n_strategies) % 10] + own
    if skilled is not None:
        trials[:, skilled] += 0.002  # a true annualised Sharpe ratio near 3.2
    sharpes = trials.mean(0) / trials.std(0, ddof=1) * np.sqrt(252)
    best = int(np.argmax(sharpes))
    return trials, best, sharpes[best]


def main():
    rng = np.random.default_rng(2026)

    trials, best, sr = search(rng)
    print(f"Search 1: best of 200 skill-less strategies, annualised Sharpe {sr:.2f}\n")
    print(vetted.validate_strategy(trials[:, best], trials, title="Search 1 (no skill)"))

    trials, best, sr = search(rng, skilled=57)
    print(f"\n\nSearch 2: one real idea among 200, picked strategy #{best}, Sharpe {sr:.2f}\n")
    print(vetted.validate_strategy(trials[:, best], trials, title="Search 2 (one real idea)"))


if __name__ == "__main__":
    main()
