class ClinicalPredictor:
    def __init__(self, trained_model):
        # On reçoit n'importe quel modèle (Logistic, Neural, etc.) tant qu'il a predict()
        self.model = trained_model

    def diagnose(self, patient_data):
        # Appel de la méthode predict()
        prediction = self.model.predict(patient_data)
        
        # Gestion si c'est un tableau
        if isinstance(prediction, (list, tuple)) or hasattr(prediction, '__iter__'):
            val = prediction[0]
        else:
            val = prediction

        if val >= 0.5:
            return "Infecté"
        else:
            return "Sain"