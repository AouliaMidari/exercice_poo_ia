# main.py
import os
from core.logistic_regression import LogisticRegressionModel
from core.neural_network import NeuralNetworkModel
from core.dataset import Dataset          # ✅ nouveau
from core.optimizer import SGDOptimizer   # ✅ nouveau
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

    # 1. Préparation des données (via Dataset + clean_data)
    dataset = Dataset("data/patient_data.csv").load()
    data = clean_data({"X": dataset.X, "y": dataset.y})

    # 2. Optimiseur (utilisé pour l'entraînement)
    optimizer = SGDOptimizer(learning_rate=0.01)

    # 3. Choix du modèle
    model_lr = LogisticRegressionModel()

    # 4. Entraînement (Pipeline)
    trainer = Trainer(model=model_lr, data=data)
    trained_lr = trainer.run_training()

    # Démo de l'optimiseur sur des paramètres fictifs
    print("  [Demo] Application d'une étape d'optimisation...")
    new_params = optimizer.step(params=[0.5, -0.2], gradients=[0.1, 0.3])
    print(f"  [Demo] Nouveaux paramètres : {new_params}")

    Evaluator(trained_lr).evaluate()

    # 5. Prédiction (App)
    predictor_lr = ClinicalPredictor(trained_model=trained_lr)
    patient_test = [39.0, 1, 0, 45]  # Fièvre + Toux
    print(f"  -> Résultat pour P001 : {predictor_lr.diagnose(patient_test)}")

    # --- SCÉNARIO B : AVEC RESEAU DE NEURONES ---
    print("\n=== SCÉNARIO B : Modèle de Réseau de Neurones ===")

    # 1. Même dataset (réutilisation !)
    data = clean_data({"X": dataset.X, "y": dataset.y})

    # 2. Changement de modèle (Polymorphisme en action !)
    model_nn = NeuralNetworkModel()

    # 3. Entraînement
    trainer = Trainer(model=model_nn, data=data)
    trained_nn = trainer.run_training()

    # 4. Prédiction avec LA MEME classe ClinicalPredictor
    predictor_nn = ClinicalPredictor(trained_model=trained_nn)
    print(f"  -> Résultat pour P002 : {predictor_nn.diagnose([38.8, 0, 1, 60])}")

    # 5. Bonus : split du dataset (démontre la flexibilité)
    (X_train, y_train), (X_test, y_test) = dataset.split(ratio=0.8)

    print("\n=== EXERCICE TERMINÉ AVEC SUCCÈS ===")


if __name__ == "__main__":
    main()