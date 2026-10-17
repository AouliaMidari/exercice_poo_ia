from utils.metrics import calculate_accuracy

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self):
        print("  [Pipeline] Évaluation du modèle en cours...")
        # Faux test
        score = calculate_accuracy([1, 0], [1, 0])
        print(f"  [Pipeline] Score du modèle : {score}")
        return score