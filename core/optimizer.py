# core/optimizer.py
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Interface de base pour tous les optimiseurs."""

    @abstractmethod
    def step(self, params, gradients):
        """Met à jour les paramètres à partir des gradients."""
        pass


class SGDOptimizer(BaseOptimizer):
    """
    Descente de gradient stochastique (simulation pédagogique).
    Formule : w = w - learning_rate * gradient
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        print(f"  [Optimizer] Initialisation SGD (lr={learning_rate})")

    def step(self, params, gradients):
        if not isinstance(params, list) or not isinstance(gradients, list):
            print("  [Optimizer] ⚠️  params et gradients doivent être des listes")
            return params

        # Mise à jour : w_i ← w_i - lr * g_i
        updated = [
            p - self.learning_rate * g
            for p, g in zip(params, gradients)
        ]
        return updated

    def __repr__(self):
        return f"SGDOptimizer(lr={self.learning_rate})"