import os
from core.logistic_regression import LogisticRegressionModel
from core.neural_network import NeuralNetworkModel
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from app.interface_clinique import ClinicalPredictor
from app.api import start_api
from utils.preprocessing import clean_data

def main():
    print("=== 1. INITIALISATION DE L'APPLICATION ===")
    start_api()
    
    # --- SCÉNARIO A : AVEC REGRESSION LOGISTIQUE ---
    print("\n=== SCÉNARIO A : Modèle de Régression Logistique ===")
    # 1. Préparation des données
    data = clean_data({"X": [1,2], "y": [0,1]})
    
    # 2. Choix du modèle
    model_lr = LogisticRegressionModel()
    
    # 3. Entraînement (Pipeline)
    trainer = Trainer(model=model_lr, data=data)
    trained_lr = trainer.run_training()
    Evaluator(trained_lr).evaluate()
    
    # 4. Prédiction (App)
    predictor_lr = ClinicalPredictor(trained_model=trained_lr)
    patient_test = [39.0, 1, 0, 45] # Fièvre + Toux
    print(f"  -> Résultat pour P001 : {predictor_lr.diagnose(patient_test)}")


    # --- SCÉNARIO B : AVEC RESEAU DE NEURONES ---
    print("\n=== SCÉNARIO B : Modèle de Réseau de Neurones ===")
    # 1. Même données
    data = clean_data({"X": [1,2], "y": [0,1]})
    
    # 2. Changement de modèle (Polymorphisme en action !)
    model_nn = NeuralNetworkModel()
    
    # 3. Entraînement
    trainer = Trainer(model=model_nn, data=data)
    trained_nn = trainer.run_training()
    
    # 4. Prédiction avec LA MEME classe ClinicalPredictor
    predictor_nn = ClinicalPredictor(trained_model=trained_nn)
    print(f"  -> Résultat pour P002 : {predictor_nn.diagnose([38.8, 0, 1, 60])}")

    print("\n=== EXERCICE TERMINÉ AVEC SUCCÈS ===")

if __name__ == "__main__":
    main()