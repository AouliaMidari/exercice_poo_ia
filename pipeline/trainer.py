class Trainer:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def run_training(self):
        print(f"  [Pipeline] Début de l'entraînement avec {type(self.model).__name__}...")
        # On appelle la méthode train du modèle
        self.model.train(self.data, None) 
        print("  [Pipeline] Entraînement terminé.")
        return self.model