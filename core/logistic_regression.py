from core.model import BaseModel

class LogisticRegressionModel(BaseModel):
    def predict(self, patient_data):
        # Simulation d'une régression logistique
        # patient_data = [temperature, toux, fatigue, age]
        temperature = patient_data[0]
        toux = patient_data[1]
        
        if temperature > 38.0 and toux == 1:
            return 0.90 # Forte probabilité
        return 0.10 # Faible probabilité

    def train(self, X, y):
        print("  [Core] Entraînement de la Régression Logistique...")