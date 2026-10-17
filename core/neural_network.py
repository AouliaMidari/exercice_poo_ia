from core.model import BaseModel

class NeuralNetworkModel(BaseModel):
    def predict(self, patient_data):
        # Simulation d'un réseau de neurones
        if patient_data[0] > 38.5:
            return 0.99
        return 0.05

    def train(self, X, y):
        print("  [Core] Entraînement du Réseau de Neurones...")