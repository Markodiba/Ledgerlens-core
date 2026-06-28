"""Genetic-algorithm adversarial attacker for the red team loop.

The attacker treats the detector as a black box: it only needs a way to turn a
feature vector into a risk score in ``[0, 100]``.  Starting from the feature
vector of a known wash trade, it evolves a population of mutated vectors to
*minimise* that score while keeping every feature inside realistic on-chain
bounds (a wash trade cannot, for example, have ``trade_count < 1`` or
``volume <= 0``).

Unlike the gradient-based attacks in :mod:`detection.adversarial_attack`, the GA
is derivative-free and operates over an arbitrary, caller-supplied feature space
described by ``feature_constraints``, so it can attack any model interface.
"""

from __future__ import annotations

import numpy as np

from detection.red_team import EVASION_THRESHOLD


def evaluate_score(model, feature_dict: dict) -> float:
    """Return the detector's risk score in ``[0, 100]`` for ``feature_dict``.

    ``model`` may be:

    * a callable mapping a feature dict to a 0-100 score (or a 0-1 probability,
      which is rescaled);
    * a dict of fitted sklearn-style classifiers (an ensemble), in which case the
      mean positive-class probability is used; or
    * a single fitted classifier exposing ``predict_proba``.
    """
    if callable(model) and not hasattr(model, "predict_proba"):
        # Callables are expected to return the final 0-100 score directly.
        raw = float(model(feature_dict))
    elif isinstance(model, dict):
        import pandas as pd

        X = pd.DataFrame([feature_dict]).fillna(0.0)
        probs = [
            float(m.predict_proba(X)[:, 1][0])
            for name, m in model.items()
            if name != "temporal_lstm"
        ]
        raw = float(np.mean(probs)) * 100.0 if probs else 0.0
    else:
        import pandas as pd

        X = pd.DataFrame([feature_dict]).fillna(0.0)
        raw = float(model.predict_proba(X)[:, 1][0]) * 100.0

    return max(0.0, min(100.0, raw))


class GeneticAttacker:
    """Mutates feature vectors of known wash trades to minimise the model score.

    Constraints (``feature_constraints``) bound the search to realistic Stellar
    on-chain values.  Each entry is ``{"min": float, "max": float,
    "mutable": bool}``; an immutable feature is pinned to its seed value.
    """

    def __init__(
        self,
        model,
        feature_constraints: dict,
        population_size: int = 50,
        mutation_rate: float = 0.3,
        mutation_scale: float = 0.25,
        elite_frac: float = 0.2,
        seed: int | None = None,
    ):
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        self.model = model
        self.feature_constraints = dict(feature_constraints)
        self.feature_names = list(self.feature_constraints.keys())
        self.population_size = int(population_size)
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)
        self.elite_frac = float(elite_frac)
        self._rng = np.random.default_rng(seed)

        self._mins = np.array(
            [self.feature_constraints[f].get("min", -np.inf) for f in self.feature_names], dtype=float
        )
        self._maxs = np.array(
            [self.feature_constraints[f].get("max", np.inf) for f in self.feature_names], dtype=float
        )
        self._mutable = np.array(
            [bool(self.feature_constraints[f].get("mutable", True)) for f in self.feature_names]
        )
        finite = np.isfinite(self._maxs) & np.isfinite(self._mins)
        self._span = np.where(finite, self._maxs - self._mins, 1.0)
        self._span = np.where(self._span <= 0, 1.0, self._span)

    # -- helpers -----------------------------------------------------------

    def _clip(self, pop: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(pop, self._mins), self._maxs)

    def to_dict(self, vec: np.ndarray) -> dict:
        """Render an aligned feature array back into a ``{name: value}`` dict."""
        return {f: float(v) for f, v in zip(self.feature_names, vec)}

    def _score_row(self, vec: np.ndarray) -> float:
        return evaluate_score(self.model, self.to_dict(vec))

    def _score_population(self, pop: np.ndarray) -> np.ndarray:
        return np.array([self._score_row(row) for row in pop], dtype=float)

    def _mutate(self, pop: np.ndarray, seed: np.ndarray) -> np.ndarray:
        mask = (self._rng.random(pop.shape) < self.mutation_rate) & self._mutable
        noise = self._rng.normal(0.0, self.mutation_scale, size=pop.shape) * self._span
        pop = pop + mask * noise
        # immutable features are always pinned to the seed value
        pop[:, ~self._mutable] = seed[~self._mutable]
        return self._clip(pop)

    # -- public API --------------------------------------------------------

    def evolve(
        self, seed_features: np.ndarray, n_generations: int = 100
    ) -> tuple[np.ndarray, float]:
        """Evolve against the seed and return ``(best_evasion_features, evasion_score)``.

        Stops early as soon as a candidate scores below
        :data:`detection.red_team.EVASION_THRESHOLD`.
        """
        seed = np.asarray(seed_features, dtype=float).ravel()
        if seed.shape[0] != len(self.feature_names):
            raise ValueError(
                f"seed_features has {seed.shape[0]} values but "
                f"{len(self.feature_names)} features are constrained"
            )

        # Seed the population: the seed itself plus mutated copies.
        pop = np.tile(seed, (self.population_size, 1))
        pop = self._mutate(pop, seed)
        pop[0] = self._clip(seed.copy())

        best_vec = self._clip(seed.copy())
        best_score = self._score_row(best_vec)
        n_elite = max(1, int(self.elite_frac * self.population_size))
        # Generation at which the returned best vector was found; exposed for
        # the ``mean_generations_to_evade`` robustness metric.
        self.last_generation = 0

        for gen in range(1, max(1, n_generations) + 1):
            scores = self._score_population(pop)
            order = np.argsort(scores)  # ascending: lower score == more evasive
            pop, scores = pop[order], scores[order]

            if scores[0] < best_score:
                best_score = float(scores[0])
                best_vec = pop[0].copy()
                self.last_generation = gen
            if best_score < EVASION_THRESHOLD:
                return best_vec, best_score

            elites = pop[:n_elite]
            n_children = self.population_size - n_elite
            father = elites[self._rng.integers(0, n_elite, size=n_children)]
            mother = elites[self._rng.integers(0, n_elite, size=n_children)]
            crossover = self._rng.random((n_children, seed.shape[0])) < 0.5
            children = np.where(crossover, father, mother)
            children = self._mutate(children, seed)
            pop = np.vstack([elites, children])

        # Final sweep in case the last generation improved on the best.
        scores = self._score_population(pop)
        i = int(np.argmin(scores))
        if scores[i] < best_score:
            best_score = float(scores[i])
            best_vec = pop[i].copy()
        return best_vec, best_score
