from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Interface de base pour tous les modèles d'IA."""
    
    @abstractmethod
    def predict(self, patient_data):
        """Toute classe qui hérite de BaseModel DOIT avoir une méthode predict()."""
        pass